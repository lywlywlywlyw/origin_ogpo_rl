"""PSS online collection, fixed response probing, and matched-budget SAC.

The fixed subspace is fitted from offline observations before collection.
Every online query uses the latent actor and Gaussian orthogonal complement.
Replay stores actor z as the SAC action, decoded chunks for the action critic,
and full initial noise for diagnostics.
"""

import hashlib
import json
from pathlib import Path

import jax
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


def _save_probe_artifacts(agent, diagnostics, observations, base_noises, save_dir):
    """Save enough information to reproduce Eq. (4) without another rollout."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / 'pss_subspace.npz', observations=observations,
             base_noises=base_noises, **diagnostics)
    decoder_hash = hashlib.sha256()
    for leaf in jax.tree.leaves(agent.bc_flow_network.params['modules_actor_bc_flow']):
        array = np.ascontiguousarray(leaf)
        decoder_hash.update(str((array.shape, str(array.dtype))).encode())
        decoder_hash.update(array.tobytes())
    metadata = dict(
        method='forward_difference_uncentered_response_gram',
        decoder_parameter_sha256=decoder_hash.hexdigest(),
        dimension=agent.config['noise_dim'], controlled_dimension=agent.config['pss_dim'],
        num_probes=len(observations), delta=agent.config['pss_delta'],
        decoder_queries=len(observations) * (agent.config['noise_dim'] + 1),
        retained_energy=diagnostics['retained_energy'],
        probe_source='offline dataset before online collection',
        reward_aggregation=agent.config['chunk_reward_aggregation'],
        updates_per_env_step=agent.config['updates_per_env_step'],
    )
    (path / 'pss_subspace.json').write_text(json.dumps(metadata, indent=2) + '\n')


def run_pss_online_rl(
    agent, env, eval_env, eval_envs, train_dataset, config, logger, save_dir,
    algo_name='pss', online_steps=2_000_000, start_training=10_000,
    batch_size=256, horizon_length=4, discount=0.99, utd_online=1,
    utd_q=None, utd_pi=None, offline_ratio=0.,
    use_success_buffer=False, success_buffer_batch_size=256,
    eval_interval=10_000, n_eval_envs=32, eval_episodes=2,
    video_episodes=0, video_frame_skip=3, plot_q_vs_mc=False,
    run_name='pss', env_name='', log_interval=5000, save_interval=500_000,
    preemption_backup_interval=-1, log_enabled=True, buffer_size=2_000_000,
    seed=1, is_resumed=False, start_step=90_000, force_jax_sync=True,
    encode_fn=None, mesh=None, **kwargs,
):
    """Run PSS with the same physical-step budget as the existing baselines.

    Matched mode uses discounted chunk sums and gamma**executed_length, with
    physical-step UTD accrued at chunk boundaries, exactly as matched DSRL.
    ``query_sum`` + updates_per_env_step=false supports the paper's reward sum
    and one gamma per query. The supplied shared-checkpoint config uses matched
    mode; its SAC regularization intentionally follows the existing baseline.

    Exact process resume is not implemented: unfinished trajectories, simulator
    state and external PRNG streams must all be saved together to support it.
    """
    if algo_name != 'pss' or offline_ratio != 0:
        raise ValueError('PSS requires algo_name=pss and offline_ratio=0')
    if is_resumed or kwargs.get('resumed_replay_buffer') is not None:
        raise ValueError('PSS currently supports fresh shared-checkpoint runs, not resume')
    if preemption_backup_interval > 0:
        raise ValueError('PSS process resume is unsupported; use regular model snapshots')
    if kwargs.get('bc_refine_steps', 0) or kwargs.get('q_warmup_steps', 0):
        raise ValueError('PSS does not perform BC refinement or separate Q pretraining')
    if plot_q_vs_mc:
        raise ValueError('Q-vs-MC plotting is not implemented in the PSS runner')
    if not bool(agent.decoder_loaded):
        raise ValueError('PSS requires an explicitly restored offline BC flow checkpoint')
    if bool(agent.basis_ready):
        raise ValueError('Fresh runs must probe the restored checkpoint before online collection')
    per_step = config.get('updates_per_env_step', True)
    aggregation = config.get('chunk_reward_aggregation', 'discounted_sum')
    if aggregation not in ('discounted_sum', 'query_sum'):
        raise ValueError('PSS reward aggregation must be discounted_sum or query_sum')
    if per_step and aggregation != 'discounted_sum':
        raise ValueError('Matched physical-step budgets require discounted_sum')
    if any(u is not None and u != utd_online for u in (utd_q, utd_pi)):
        raise ValueError('PSS requires equal actor/critic UTD; use critic_updates_per_actor')
    n_probes, k = agent.config['pss_num_probes'], agent.config['pss_dim']
    dim = agent.config['noise_dim']
    if start_training < 0 or online_steps < 1:
        raise ValueError('Require nonnegative warmup and positive online_steps')
    minimum_capacity = batch_size * utd_online * horizon_length if per_step else batch_size
    if batch_size <= 0 or utd_online < 1 or buffer_size < minimum_capacity:
        raise ValueError('Invalid batch/UTD or insufficient replay capacity')

    example = train_dataset.sample(())
    action_dim = example['actions'].shape[-1]
    if dim != action_dim * horizon_length:
        raise ValueError('PSS decoder action width/horizon do not match the environment')
    transition_example = dict(
        observations=np.asarray(example['observations'], dtype=np.float32),
        next_observations=np.asarray(example['observations'], dtype=np.float32),
        actions=np.zeros(k, np.float32), initial_noise=np.zeros(dim, np.float32),
        true_actions=np.zeros(dim, np.float32),
        rewards=np.float32(0), terminals=np.float32(0), masks=np.float32(1),
        chunk_length=np.int32(horizon_length),
    )
    replay = ReplayBuffer.create(transition_example, buffer_size,
                                numpy_rng=np.random.default_rng(seed + 30000))
    # No image augmentation: observations are state or frozen embeddings.
    replay.p_aug = None
    use_full_state = config.get('use_state', 'proprio') == 'full'

    def encode(raw):
        state, images, full = _split_obs(raw)
        if encode_fn is not None and images is not None:
            return np.asarray(encode_fn(
                _state_for_encoder(state, full, use_full_state)[None], images[None])[0])
        return np.asarray(state, dtype=np.float32)

    def shard(batch):
        return shard_batch(batch, mesh) if mesh is not None and batch is not None else batch

    def learner_batch(batch):
        # Diagnostics must not enlarge the JIT update graph or learner action space.
        return {key: batch[key] for key in
                ('observations', 'next_observations', 'actions', 'true_actions', 'rewards',
                 'masks', 'terminals', 'chunk_length')}

    # Fit the frozen decoder's response basis without collecting from a different policy.
    probe_rng = np.random.default_rng(seed + 50000)
    probe_indices = probe_rng.choice(train_dataset.size, size=n_probes,
                                     replace=train_dataset.size < n_probes)
    probe_obs = np.asarray(train_dataset.sample(n_probes, idxs=probe_indices)['observations'])
    base_noises = probe_rng.standard_normal((n_probes, dim)).astype(np.float32)
    agent, diagnostics = agent.fit_subspace(probe_obs, base_noises)
    _save_probe_artifacts(agent, diagnostics, probe_obs, base_noises, save_dir)
    print(f'PSS: k={k}/{dim}, response energy={diagnostics["retained_energy"]:.4f}, '
          'offline probes complete before online collection')

    raw, _ = env.reset(seed=seed + 10000)
    observation = encode(raw)
    online_rng = jax.random.PRNGKey(seed + 40000)
    pending = []
    steps, episodes, actor_updates, critic_updates = 0, 0, 0, 0
    log_step, eval_info = start_step, {}
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    initial_actor_step, initial_critic_step = int(agent.actor_network.step), int(agent.critic_network.step)
    initial_noise_critic_step = int(agent.noise_critic_network.step)
    distill_steps = agent.config['noise_critic_grad_steps']

    def log_budget():
        logger.log(dict(env_steps=steps, gradient_updates=actor_updates,
                        critic_gradient_updates=critic_updates,
                        noise_critic_gradient_updates=actor_updates * distill_steps, decoder_updates=0,
                        probe_env_steps=probe_env_steps,
                        probe_decoder_queries=n_probes * (dim + 1)),
                   'online_budget', step=log_step)

    probe_env_steps = 0
    with tqdm.tqdm(total=online_steps, desc='PSS Online RL', dynamic_ncols=True) as progress:
        while steps < online_steps:
            previous = steps
            # Keep the same policy and independent RNG streams throughout collection.
            online_rng, actor_key, complement_key = jax.random.split(online_rng, 3)
            z = agent.sample_noise(observation, actor_key)
            full_noise = agent.reconstruct_noise(z, complement_key)
            decoded = np.asarray(agent.compute_flow_actions(observation, noises=full_noise))
            rewards, terminated, done, info = [], False, False, {}
            for action in decoded.reshape(horizon_length, action_dim):
                raw_next, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                steps += 1
                progress.update(1)
                done = bool(terminated or truncated)
                if done or steps == online_steps:
                    break
            log_step = start_step + steps
            next_observation = encode(raw_next)
            reward = (sum(discount ** i * r for i, r in enumerate(rewards))
                      if aggregation == 'discounted_sum' else sum(rewards))
            # DSRL's SAC target uses gamma**chunk_length. A query uses exponent 1.
            pending.append(dict(
                observations=observation, next_observations=next_observation,
                actions=np.asarray(z, dtype=np.float32),
                initial_noise=np.asarray(full_noise, dtype=np.float32),
                true_actions=decoded.reshape(-1).astype(np.float32),
                rewards=np.float32(reward), terminals=np.float32(done),
                masks=np.float32(not terminated),
                chunk_length=np.int32(len(rewards) if aggregation == 'discounted_sum' else 1)))
            if done:
                success = float(info.get('success', 0.))
                for item in pending:
                    item['is_success'] = success if use_success_buffer else 0.
                    replay.add_transition(item)
                pending = []
                episodes += 1
                # Match the completed physical-step baseline's reset seed schedule.
                reset_index = episodes if per_step else steps
                raw, _ = env.reset(seed=seed + 10000 + reset_index)
                observation = encode(raw)
            else:
                observation = next_observation

            env_info = {key: value for key, value in info.items() if key.startswith('distance')}
            logger.log(dict(env_info, total_env_steps=steps), 'env', step=log_step)
            if steps <= start_training:
                continue
            agent._require_basis()
            eligible = max(0, steps - max(previous, start_training))
            utd = utd_online * eligible if per_step else utd_online
            minimum_replay = batch_size * utd if per_step else batch_size
            if replay.size < minimum_replay:
                raise RuntimeError('PSS replay cannot supply the requested update budget; '
                                   'refusing to silently skip optimizer updates')
            batch = learner_batch(replay.sample(batch_size * utd))
            batch = _reshape_for_utd(batch, utd, batch_size)
            success_batch, success_flag = None, False
            if use_success_buffer:
                successful = replay._traj_success_mask[:replay.size]
                available = (replay['chunk_length'][:replay.size][successful].sum()
                             if per_step else successful.sum())
                required = (success_buffer_batch_size * utd_online + horizon_length
                            if per_step else success_buffer_batch_size * utd)
                if available >= required:
                    success_batch = learner_batch(_sample_success_chunks(replay, success_buffer_batch_size * utd))
                    success_batch = _reshape_for_utd(success_batch, utd, success_buffer_batch_size)
                    success_flag = True
            agent, update_info = agent.batch_update(shard(batch), shard(success_batch), success_flag)
            actor_updates += utd
            extra_critic = config.get('critic_updates_per_actor', 1) == 2 or (
                success_flag and config.get('use_success_buffer_q', False))
            critic_updates += utd * (1 + int(extra_critic))
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
                    video_frame_skip, run_name, False, encode_fn=encode_fn)
                logger.log(eval_info, 'eval', step=log_step)
                print(f'[PSS EVAL] env_step={steps} | success={float(eval_info.get("success", 0.)):.4f}')
            if log_enabled and save_interval > 0 and steps // save_interval > previous // save_interval:
                save_agent(agent, save_dir, log_step)

    # Retain the final incomplete episode without assigning a terminal outcome.
    for item in pending:
        replay.add_transition(item)
    if per_step:
        expected = max(0, online_steps - max(0, start_training)) * utd_online
        if actor_updates != expected:
            raise RuntimeError(f'Actor update budget mismatch: {actor_updates} != {expected}')
    if int(agent.actor_network.step) - initial_actor_step != actor_updates:
        raise RuntimeError('Recorded actor budget differs from actual optimizer steps')
    if int(agent.critic_network.step) - initial_critic_step != critic_updates:
        raise RuntimeError('Recorded critic budget differs from actual optimizer steps')
    if int(agent.noise_critic_network.step) - initial_noise_critic_step != actor_updates * distill_steps:
        raise RuntimeError('Recorded distillation budget differs from actual optimizer steps')
    if config.get('critic_updates_per_actor', 1) == 2 and critic_updates != 2 * actor_updates:
        raise RuntimeError('Critic update budget mismatch')
    log_budget()
    save_agent(agent, save_dir, log_step)
    return agent, log_step, eval_info, replay
