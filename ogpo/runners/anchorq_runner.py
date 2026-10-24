"""Online AnchorQ with cached flow-reversed banks and matched physical budgets.

The default experiment uses prioritized replay and shares gamma**executed_length, successful
trajectory Q updates, and two critic steps per eligible physical step with the
square baseline. No SAC actor, entropy objective, or latent BC loss is added.
Set updates_per_env_step=false and chunk_reward_aggregation=query_sum for the
paper's decision-level convention; selection itself follows Algorithms 1--4.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.runners.dsrl_runner import (
    _build_eval_flags, _reshape_for_utd, _run_evaluation, _sample_success_chunks,
)
from ogpo.runners.online_rl_runner import _state_for_encoder, _wrap_actor_for_frozen_encoder
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.evaluation import _split_obs
from ogpo.utils.sharding import shard_batch


class PriorityIndex:
    """Host sum/min trees; O(log capacity) sampling and priority refresh.

    Alpha/beta are explicit adaptation choices, not claimed paper table values.
    IS weights use the global minimum probability, not the current batch minimum.
    """
    def __init__(self, capacity, alpha=.6, epsilon=1e-6):
        if not 0 <= alpha <= 1 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Invalid PER alpha/epsilon')
        self.width = 1 << (capacity - 1).bit_length()
        self.total = np.zeros(2 * self.width, np.float64)
        self.minimum = np.full(2 * self.width, np.inf)
        self.alpha, self.epsilon, self.maximum = alpha, epsilon, 1.

    def update(self, indices, errors):
        indices, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
        errors = np.asarray(errors, dtype=np.float64)
        if not np.isfinite(errors).all():
            raise FloatingPointError('Non-finite replay priorities')
        merged = np.zeros(len(indices))
        np.maximum.at(merged, inverse, np.abs(errors))
        raw = merged + self.epsilon
        self.maximum = max(self.maximum, float(raw.max()))
        nodes = indices + self.width
        self.total[nodes] = self.minimum[nodes] = raw ** self.alpha
        while nodes[0] > 1:
            nodes = np.unique(nodes // 2)
            self.total[nodes] = self.total[2 * nodes] + self.total[2 * nodes + 1]
            self.minimum[nodes] = np.minimum(self.minimum[2 * nodes], self.minimum[2 * nodes + 1])

    def insert(self, index):
        self.update([index], [max(0., self.maximum - self.epsilon)])

    def sample(self, count, rng, beta):
        if self.total[1] <= 0:
            raise ValueError('Cannot sample empty priority tree')
        masses = (np.arange(count) + rng.random(count)) * self.total[1] / count
        nodes = np.ones(count, dtype=np.int64)
        while nodes[0] < self.width:
            left = 2 * nodes
            right = masses >= self.total[left]
            masses -= np.where(right, self.total[left], 0.)
            nodes = left + right
        weights = (self.total[nodes] / self.minimum[1]) ** (-beta)
        return nodes - self.width, weights.astype(np.float32)


def _configure_references(agent, env):
    """Read normalization from the actual action wrapper, if it is enabled."""
    current, seen = env, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, 'normalize', False):
            return agent.configure_reference_actions(current.action_min, current.action_max)
        current = getattr(current, 'env', None)
    return agent.configure_reference_actions()


def run_anchorq_online_rl(
    agent, env, eval_env, eval_envs, train_dataset, config, logger, save_dir,
    algo_name='anchorq', online_steps=2_000_000, start_training=10_000,
    batch_size=256, horizon_length=4, discount=0.99, utd_online=1,
    utd_q=None, utd_pi=None, utd_v=None, offline_ratio=0.,
    use_success_buffer=False, success_buffer_batch_size=256,
    eval_interval=10_000, n_eval_envs=32, eval_episodes=2,
    video_episodes=0, video_frame_skip=3, plot_q_vs_mc=False,
    run_name='anchorq', env_name='', log_interval=5000, save_interval=500_000,
    preemption_backup_interval=-1, log_enabled=True, buffer_size=2_000_000,
    seed=1, is_resumed=False, start_step=90_000, force_jax_sync=True,
    encode_fn=None, mesh=None, **kwargs,
):
    if algo_name != 'anchorq' or offline_ratio != 0.:
        raise ValueError('AnchorQ requires algo_name=anchorq and offline_ratio=0')
    if is_resumed or kwargs.get('resumed_replay_buffer') is not None:
        raise ValueError('AnchorQ requires a fresh shared-BC run; process resume is unsupported')
    if preemption_backup_interval > 0:
        raise ValueError('AnchorQ process resume is unsupported; use periodic parameter snapshots')
    if any(kwargs.get(key, 0) for key in ('bc_refine_steps', 'q_warmup_steps')) or utd_v:
        raise ValueError('AnchorQ trains only its latent critic')
    if any(u is not None and u != utd_online for u in (utd_q, utd_pi)):
        raise ValueError('Use the matched UTD and critic_updates_per_actor settings')
    if not bool(agent.decoder_loaded):
        raise ValueError('Load the shared BC checkpoint before constructing anchors')
    if (batch_size < 1 or utd_online < 1 or horizon_length < 1
            or online_steps < 1 or start_training < 0 or buffer_size < batch_size):
        raise ValueError('Invalid online training budget or replay capacity')
    if not 0 < discount <= 1:
        raise ValueError('Discount must lie in (0,1]')
    if (use_success_buffer and config.get('use_success_buffer_q', False)
            and success_buffer_batch_size != batch_size):
        raise ValueError('Matched success-buffer Q updates require equal batch sizes')
    per_step = config.get('updates_per_env_step', True)
    aggregation = config.get('chunk_reward_aggregation', 'discounted_sum')
    if aggregation not in ('discounted_sum', 'query_sum'):
        raise ValueError('AnchorQ reward aggregation must be discounted_sum or query_sum')
    if per_step and aggregation != 'discounted_sum':
        raise ValueError('Physical-step matching requires discounted_sum')
    example = train_dataset.sample(())
    action_dim = example['actions'].shape[-1]
    if agent.config['noise_dim'] != action_dim * horizon_length:
        raise ValueError('Environment action width/horizon differs from decoder')
    agent = _configure_references(agent, env)
    count = 6 + int(agent.config['anchorq_include_defer'])
    transition = dict(
        observations=np.asarray(example['observations'], np.float32),
        next_observations=np.asarray(example['observations'], np.float32),
        actions=np.zeros(agent.config['noise_dim'], np.float32),
        next_anchors=np.zeros((count, agent.config['noise_dim']), np.float32),
        rewards=np.float32(0), terminals=np.float32(0), masks=np.float32(1),
        chunk_length=np.int32(horizon_length),
    )
    replay = ReplayBuffer.create(transition, buffer_size,
                                numpy_rng=np.random.default_rng(seed + 30000))
    replay.p_aug = None
    prioritized = agent.config['anchorq_prioritized_replay']
    priorities = PriorityIndex(buffer_size, agent.config['anchorq_per_alpha'],
                               agent.config['anchorq_per_epsilon']) if prioritized else None
    beta_start = agent.config['anchorq_per_beta_start']
    if not 0 <= beta_start <= 1:
        raise ValueError('PER beta must lie in [0,1]')

    # Snapshot the actual normalized reference commands for reproducibility.
    if log_enabled:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / 'anchorq_references.npz', reference_actions=np.asarray(agent.reference_actions))
        (path / 'anchorq_method.json').write_text(json.dumps(dict(
            paper='2610.22852v1, Algorithms 1--4',
            reference_map='six constant single-arm Cartesian deltas; neutral rotation/gripper',
            replay='prioritized' if prioritized else 'uniform',
            random_priors=bool(agent.config['anchorq_prior_scale']),
            prior_scale=agent.config['anchorq_prior_scale'],
            per_alpha=agent.config['anchorq_per_alpha'], per_beta_start=beta_start, latent_actor=False,
            target_aggregation=agent.config['anchorq_target_aggregation'],
            updates_per_env_step=per_step, reward_aggregation=aggregation,
        ), indent=2) + '\n')

    use_full_state = config.get('use_state', 'proprio') == 'full'

    def encode(raw):
        state, images, full = _split_obs(raw)
        if encode_fn is not None and images is not None:
            state = encode_fn(_state_for_encoder(state, full, use_full_state)[None], images[None])[0]
        return np.asarray(state, np.float32)

    def shard(batch):
        return shard_batch(batch, mesh) if mesh is not None and batch is not None else batch

    online_rng = jax.random.PRNGKey(seed + 40000)
    bank_constructions = 0

    def bank(observation):
        nonlocal online_rng, bank_constructions
        online_rng, key = jax.random.split(online_rng)
        result = np.asarray(agent.build_anchor_bank(jnp.asarray(observation), key), np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite flow-reversed anchor bank')
        bank_constructions += 1
        return result

    raw, _ = env.reset(seed=seed + 10000)
    observation = encode(raw)
    anchors = bank(observation)
    steps, decisions, episodes, updates, critic_updates = 0, 0, 0, 0, 0
    pending = []
    log_step, eval_info = start_step, {}
    initial_critic_step = int(agent.critic_network.step)
    initial_decoder_step = int(agent.bc_flow_network.step)
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)

    def log_budget():
        logger.log(dict(env_steps=steps, decisions=decisions,
                        gradient_updates=updates, critic_gradient_updates=critic_updates,
                        actor_gradient_updates=0, decoder_updates=0,
                        anchor_bank_constructions=bank_constructions,
                        reverse_flow_queries=6 * bank_constructions), 'online_budget', step=log_step)

    with tqdm.tqdm(total=online_steps, desc='AnchorQ Online RL', dynamic_ncols=True) as progress:
        while steps < online_steps:
            previous = steps
            online_rng, selection_key = jax.random.split(online_rng)
            epsilon = agent.exploration_epsilon()
            latent, selection_info = agent.select_latent(
                jnp.asarray(observation), jnp.asarray(anchors), selection_key, epsilon)
            decoded = np.asarray(agent.compute_flow_actions(jnp.asarray(observation), noises=latent))
            if not np.isfinite(decoded).all():
                raise FloatingPointError('Non-finite decoded action')
            rewards, terminated, done, info = [], False, False, {}
            for action in decoded.reshape(horizon_length, action_dim):
                raw_next, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                steps += 1
                progress.update(1)
                done = bool(terminated or truncated)
                if done or steps == online_steps:
                    break
            decisions += 1
            agent = agent.replace(decision_step=jnp.asarray(decisions, jnp.int32))
            log_step = start_step + steps
            next_observation = encode(raw_next)
            # True terminals do not bootstrap; truncations keep the final bank.
            next_anchors = np.zeros_like(anchors) if terminated else bank(next_observation)
            reward = (sum(discount ** i * r for i, r in enumerate(rewards))
                      if aggregation == 'discounted_sum' else sum(rewards))
            pending.append(dict(
                observations=observation, next_observations=next_observation,
                actions=np.asarray(latent, np.float32), next_anchors=next_anchors,
                rewards=np.float32(reward), masks=np.float32(not terminated),
                terminals=np.float32(done),
                chunk_length=np.int32(len(rewards) if aggregation == 'discounted_sum' else 1)))
            if done or steps == online_steps:
                success = float(info.get('success', 0.)) if done else 0.
                for item in pending:
                    item['is_success'] = success if use_success_buffer else 0.
                    if priorities is not None:
                        priorities.insert(replay.pointer)
                    replay.add_transition(item)
                pending = []
                if done:
                    episodes += 1
                    if steps < online_steps:
                        reset_index = episodes if per_step else steps
                        raw, _ = env.reset(seed=seed + 10000 + reset_index)
                        observation = encode(raw)
                        anchors = bank(observation)
            else:
                observation, anchors = next_observation, next_anchors

            logger.log(dict(total_env_steps=steps, epsilon=float(epsilon),
                            particle_active=float(agent.config['anchorq_particle_count'] > 0),
                            latent_rms=float(np.sqrt(np.mean(np.asarray(latent) ** 2))),
                            decoded_gripper_abs=float(np.abs(decoded.reshape(horizon_length, action_dim)[:, -1]).mean()),
                            successful_replay_chunks=int(replay._traj_success_mask[:replay.size].sum()),
                            **{k: float(v) for k, v in selection_info.items()}), 'env', step=log_step)
            if steps <= start_training:
                continue
            eligible = max(0, steps - max(previous, start_training))
            utd = utd_online * eligible if per_step else utd_online
            if utd == 0:
                continue
            if replay.size < batch_size * utd:
                raise RuntimeError('AnchorQ replay cannot supply matched updates; increase warmup')
            batch = (None if prioritized else
                     _reshape_for_utd(replay.sample(batch_size * utd), utd, batch_size))
            success_batch, success_flag = None, False
            if use_success_buffer and config.get('use_success_buffer_q', False):
                successful = replay._traj_success_mask[:replay.size]
                available = (replay['chunk_length'][:replay.size][successful].sum()
                             if per_step else successful.sum())
                required = (success_buffer_batch_size * utd_online + horizon_length
                            if per_step else success_buffer_batch_size * utd)
                if available >= required:
                    if not prioritized:
                        success_batch = _reshape_for_utd(
                            _sample_success_chunks(replay, success_buffer_batch_size * utd),
                            utd, success_buffer_batch_size)
                    success_flag = True
            if prioritized:
                extra = config.get('critic_updates_per_actor', 1) == 2 or (
                    success_flag and config.get('use_success_buffer_q', False))
                beta = beta_start + (1. - beta_start) * min(steps / online_steps, 1.)
                metrics = []
                for _ in range(utd):
                    for extra_step in range(1 + int(extra)):
                        if extra_step and success_flag:
                            # Keep the baseline's extra successful-trajectory update.
                            candidates = np.flatnonzero(replay._traj_success_mask[:replay.size])
                            indices = replay.numpy_rng.choice(candidates, batch_size, replace=True)
                            weights = np.ones(batch_size, np.float32)
                        else:
                            indices, weights = priorities.sample(batch_size, replay.numpy_rng, beta)
                        data = replay.sample(batch_size, idxs=indices)
                        data['weights'] = weights
                        agent, info_update, errors = agent.critic_update(shard(data))
                        # Refresh BEFORE sampling the next optimizer step (Appendix B).
                        priorities.update(indices, np.asarray(errors))
                        metrics.append(info_update)
                update_info = jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *metrics)
                update_info['per_beta'] = beta
            else:
                agent, update_info = agent.batch_update(shard(batch), shard(success_batch), success_flag)
            updates += utd
            extra = config.get('critic_updates_per_actor', 1) == 2 or (
                success_flag and config.get('use_success_buffer_q', False))
            critic_updates += utd * (1 + int(extra))
            if force_jax_sync:
                jax.block_until_ready(agent)
            if log_interval > 0 and steps // log_interval > previous // log_interval:
                logger.log(update_info, 'online_agent', step=log_step)
                log_budget()
            if steps == online_steps or (eval_interval > 0 and steps // eval_interval > previous // eval_interval):
                actor_fn = agent.sample_actions
                if encode_fn is not None:
                    actor_fn = _wrap_actor_for_frozen_encoder(
                        actor_fn, encode_fn,
                        actor_uses_state=config.get('_original_actor_obs', 'state') == 'state',
                        use_full_state=use_full_state)
                eval_info = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot_q_vs_mc, encode_fn=encode_fn)
                logger.log(eval_info, 'eval', step=log_step)
                print(f'[AnchorQ EVAL] env_step={steps} | success={float(eval_info.get("success", 0.)):.4f}')
            if log_enabled and save_interval > 0 and steps // save_interval > previous // save_interval:
                save_agent(agent, save_dir, log_step)

    expected = max(0, online_steps - max(0, start_training)) * utd_online
    if per_step and updates != expected:
        raise RuntimeError(f'AnchorQ physical-step update budget mismatch: {updates} != {expected}')
    if int(agent.critic_network.step) - initial_critic_step != critic_updates:
        raise RuntimeError('AnchorQ critic optimizer steps differ from recorded budget')
    if int(agent.bc_flow_network.step) != initial_decoder_step:
        raise RuntimeError('Frozen BC decoder was updated')
    log_budget()
    save_agent(agent, save_dir, log_step)
    return agent, log_step, eval_info, replay
