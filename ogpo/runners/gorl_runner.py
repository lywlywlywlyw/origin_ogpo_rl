"""GoRL Algorithm 1 with shared BC initialization and a strict physical-step budget.

All fresh decoder-collection interactions count toward online_steps; no replay warmup.
Only evaluation uses separate environments. No offline data enters online losses.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.runners.dsrl_runner import _build_eval_flags, _run_evaluation
from ogpo.runners.online_rl_runner import _state_for_encoder, _wrap_actor_for_frozen_encoder
from ogpo.utils.evaluation import _split_obs


def sequence_batches(data, unroll_length, minibatch_size, rng):
    """Shuffle contiguous sequences, retaining time order and masking padding."""
    count = len(data['observations'])
    sequences = int(np.ceil(count / unroll_length))
    total = sequences * unroll_length
    fields = ('observations', 'next_observations', 'actions', 'old_log_probs',
              'rewards', 'masks', 'truncations', 'chunk_length')
    padded = {key: np.concatenate((data[key], np.repeat(data[key][-1:], total-count, axis=0)))
              for key in fields}
    padded['valid'] = (np.arange(total) < count).astype(np.float32)
    arrays = {key: value.reshape((sequences, unroll_length, *value.shape[1:]))
              for key, value in padded.items()}
    order = rng.permutation(sequences)
    # Smaller last batches are budget-boundary batches, not a new PPO setting.
    for start in range(0, sequences, minibatch_size):
        ids = order[start:start + minibatch_size]
        yield {key: jnp.asarray(value[ids].swapaxes(0, 1)[None])
               for key, value in arrays.items()}


def stage_boundaries(total_steps, weights):
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
    boundaries = np.rint(total_steps * cumulative / cumulative[-1]).astype(int)
    boundaries[-1] = total_steps
    return boundaries


def _stack(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _decoder_split(data, fraction, rng):
    """Hold out complete episodes whenever more than one episode is available."""
    episodes = np.unique(data['episode_id'])
    if len(episodes) > 1:
        episodes = rng.permutation(episodes)
        n = min(len(episodes) - 1, max(1, int(len(episodes) * fraction)))
        validation = np.flatnonzero(np.isin(data['episode_id'], episodes[:n]))
        train = np.flatnonzero(~np.isin(data['episode_id'], episodes[:n]))
    else:
        indices = rng.permutation(len(data['observations']))
        n = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * fraction))))
        validation, train = indices[:n], indices[n:]
    if not len(train) or not len(validation):
        raise ValueError('Fresh decoder collection must contain at least two transitions')
    return train, validation


def run_gorl_online_rl(agent, env, eval_env, eval_envs, train_dataset, config,
    logger, save_dir, algo_name='gorl', online_steps=2_000_000, start_training=0,
    batch_size=256, horizon_length=4, discount=.99, offline_ratio=0.,
    use_success_buffer=False, eval_interval=10000, n_eval_envs=32, eval_episodes=2,
    video_episodes=0, video_frame_skip=3, plot_q_vs_mc=False, run_name='gorl',
    env_name='', log_interval=5000, save_interval=500000,
    preemption_backup_interval=-1, log_enabled=True, seed=1, is_resumed=False,
    start_step=90000, force_jax_sync=True, encode_fn=None, mesh=None, **kwargs):
    if algo_name != 'gorl' or offline_ratio != 0:
        raise ValueError('GoRL requires algo_name=gorl and offline_ratio=0')
    if is_resumed or kwargs.get('resumed_replay_buffer') is not None or preemption_backup_interval > 0:
        raise ValueError('GoRL supports fresh checkpoint initialization, not exact process resume')
    if kwargs.get('bc_refine_steps', 0) or kwargs.get('q_warmup_steps', 0):
        raise ValueError('Use GoRL stage refinement, not generic BC/Q pretraining')
    if use_success_buffer:
        raise ValueError('GoRL PPO uses on-policy data; disable success_buffer.use_success_buffer')
    if mesh is not None:
        raise ValueError('GoRL currently supports training.gpu_mode=single')
    if plot_q_vs_mc:
        raise ValueError('GoRL has V(s), not the Q(s,a) expected by generic Q plots')
    if not bool(agent.decoder_loaded):
        raise ValueError('Restore the shared offline decoder before starting GoRL')
    if start_training != 0:
        raise ValueError('GoRL has no replay warmup; set buffer.start_training=0')
    if online_steps <= 0 or batch_size <= 0:
        raise ValueError('Invalid GoRL interaction or minibatch budget')
    if agent.config['noise_dim'] != train_dataset.sample(())['actions'].shape[-1] * horizon_length:
        raise ValueError('GoRL action chunk width does not match the environment')
    if config.get('chunk_reward_aggregation') != 'discounted_sum':
        raise ValueError('Matched GoRL requires discounted_sum chunk rewards')
    c = agent.config
    boundaries = stage_boundaries(online_steps, c['gorl_stage_weights'])
    collection_steps = c['gorl_collection_steps']
    stage_starts = np.concatenate(([0], boundaries[:-1]))
    for index, (start, end) in enumerate(zip(stage_starts, boundaries)):
        reserve = collection_steps if index < len(boundaries) - 1 else 0
        if end - start <= reserve + horizon_length:
            raise ValueError('GoRL stage budget too small for PPO and fresh collection')
    if hasattr(logger, 'enable_optimizer_metrics'):
        logger.enable_optimizer_metrics('gorl_decoder', 'gorl_decoder/update')
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    decoder_hash = hashlib.sha256()
    for leaf in jax.tree.leaves(agent.bc_flow_network.params):
        array = np.ascontiguousarray(leaf)
        decoder_hash.update(str((array.shape, str(array.dtype))).encode())
        decoder_hash.update(array.tobytes())
    manifest = dict(method='GoRL-FM: latent PPO + fixed-prior CFM',
        initial_decoder_sha256=decoder_hash.hexdigest(), stage_boundaries=boundaries.tolist(),
        collection_steps_per_boundary=collection_steps, online_steps=online_steps,
        collection_included_in_online_steps=True, offline_training_steps=0,
        decoder_initialization=c['gorl_decoder_initialization'],
        decoder_steps_per_boundary=c['gorl_decoder_steps'],
        latent_policy='reference fresh Gaussian, softplus scale capped at 10',
        evaluation_policy='stochastic composite policy, last parameters',
        reward_aggregation='discounted_sum', discount=discount,
        value_function='reference single V(s)',
        observation_normalization='cumulative PPO; initial BC identity; later decoder train-split stats',
        ppo_requested_sequences=c['gorl_ppo_batch_size'] * c['gorl_num_minibatches'],
        rollout_physical_step_cap=c['gorl_rollout_max_env_steps'],
        collection_requested_episodes=c['gorl_collection_episodes'])
    (output / 'gorl_protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')

    use_full = config.get('use_state', 'proprio') == 'full'
    def encode(raw):
        state, images, full = _split_obs(raw)
        if encode_fn is not None and images is not None:
            state = encode_fn(_state_for_encoder(state, full, use_full)[None], images[None])[0]
        return np.asarray(state, np.float32)

    rng = jax.random.PRNGKey(seed + 40000)
    sample_rng = np.random.default_rng(seed + 60000)
    raw, _ = env.reset(seed=seed + 10000)
    observation = encode(raw)
    steps = episodes = encoder_updates = value_updates = decoder_updates = 0
    encoder_env_steps = collection_env_steps = warmup_env_steps = 0
    phase, stage = 'encoder', 0
    best_agent, best_return = None, -np.inf
    last_eval_step = -1
    stage_records = []
    final_eval = {}
    flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    action_dim = c['action_dim']

    def budget():
        return dict(env_steps=steps, encoder_env_steps=encoder_env_steps,
            decoder_collection_env_steps=collection_env_steps, warmup_env_steps=warmup_env_steps,
            encoder_gradient_updates=encoder_updates, value_gradient_updates=value_updates,
            decoder_updates=decoder_updates, stage=stage)

    def evaluate():
        actor_fn = agent.sample_actions
        if encode_fn is not None:
            actor_fn = _wrap_actor_for_frozen_encoder(actor_fn, encode_fn,
                actor_uses_state=True, use_full_state=use_full)
        return _run_evaluation(agent, eval_env, eval_envs, flags, action_dim,
            actor_fn, start_step + steps, eval_episodes, video_episodes,
            video_frame_skip, run_name, False, encode_fn=encode_fn)

    def record_evaluation():
        nonlocal final_eval, best_agent, best_return, last_eval_step
        if steps == last_eval_step:
            return
        final_eval = evaluate()
        last_eval_step = steps
        logger.log(final_eval, 'eval', step=start_step + steps)
        if phase == 'encoder':
            if 'return' not in final_eval:
                raise ValueError('GoRL best-agent selection requires evaluation return')
            score = float(final_eval['return'])
            if score > best_return:
                best_return, best_agent = score, agent

    def collect_query(limit):
        nonlocal rng, observation, steps, episodes, final_eval
        nonlocal encoder_env_steps, collection_env_steps, warmup_env_steps
        rng, key = jax.random.split(rng)
        z, log_prob, value = agent.sample_latent(observation, key)
        action = np.asarray(agent.compute_flow_actions(observation, noises=z))
        action = action.reshape(horizon_length, action_dim)
        rewards, terminated, done, info = [], False, False, {}
        previous = steps
        old_observation = observation
        for a in action[:min(horizon_length, limit - steps)]:
            next_raw, reward, terminated, truncated, info = env.step(a)
            rewards.append(float(reward))
            steps += 1
            progress.update(1)
            done = bool(terminated or truncated)
            if done:
                break
        next_observation = encode(next_raw)
        next_value = float(agent.values(next_observation))
        length = len(rewards)
        row = dict(observations=old_observation, next_observations=next_observation,
            actions=np.asarray(z, np.float32), old_log_probs=np.float32(log_prob),
            values=np.float32(value), next_values=np.float32(next_value),
            true_actions=action.reshape(-1).astype(np.float32),
            action_valid=np.repeat(np.arange(horizon_length) < length, action_dim).astype(np.float32),
            rewards=np.float32(sum(discount ** i * r for i, r in enumerate(rewards))),
            masks=np.float32(not terminated), dones=np.float32(done),
            truncations=np.float32(truncated),
            chunk_length=np.int32(length), episode_id=np.int32(episodes))
        if phase == 'decoder_collection':
            collection_env_steps += length
        elif phase == 'warmup':
            warmup_env_steps += length
        else:
            encoder_env_steps += length
        if done:
            episodes += 1
            raw, _ = env.reset(seed=seed + 10000 + episodes)
            observation = encode(raw)
        else:
            observation = next_observation
        log_step = start_step + steps
        logger.log(dict(total_env_steps=steps, stage=stage,
                        phase={'warmup': 0, 'encoder': 1, 'decoder_collection': 2}[phase]), 'env', step=log_step)
        if log_interval > 0 and steps // log_interval > previous // log_interval:
            logger.log(budget(), 'online_budget', step=log_step)
        if (phase == 'decoder_collection' and eval_interval > 0
                and steps // eval_interval > previous // eval_interval):
            record_evaluation()
        if log_enabled and save_interval > 0 and steps // save_interval > previous // save_interval:
            save_agent(agent, save_dir, log_step)
        return row

    def optimize_rollout(rows):
        nonlocal agent, encoder_updates, value_updates
        data = _stack(rows)
        batches_per_epoch = 0
        for _ in range(c['gorl_ppo_epochs']):
            batches_per_epoch = 0
            for batches in sequence_batches(data, c['gorl_unroll_length'],
                                            c['gorl_ppo_batch_size'], sample_rng):
                agent, metrics = agent.ppo_update(batches)
                encoder_updates += 1
                value_updates += 1
                batches_per_epoch += 1
        # No change to observation statistics while evaluating likelihood ratios.
        agent = agent.update_observation_stats(jnp.asarray(data['observations']))
        metrics = dict(metrics, rollout_queries=len(rows),
                       sequence_count=int(np.ceil(len(rows) / c['gorl_unroll_length'])),
                       minibatches_per_epoch=batches_per_epoch)
        if force_jax_sync:
            jax.block_until_ready(agent)
        logger.log(metrics, 'online_agent', step=start_step + steps)

    with tqdm.tqdm(total=online_steps, desc='GoRL Online RL', dynamic_ncols=True) as progress:
        for stage, stage_end in enumerate(boundaries):
            agent = agent.reset_encoder(stage)
            best_agent, best_return = None, -np.inf
            # Reset only after fitting the preceding decoder; never mix rollouts
            # collected under different encoder or decoder parameter versions.
            reserve = collection_steps if stage < len(boundaries) - 1 else 0
            encoder_end = int(stage_end) - reserve
            phase = 'encoder'
            while steps < encoder_end:
                rows = []
                limit = min(encoder_end, steps + c['gorl_rollout_max_env_steps'])
                if eval_interval > 0:
                    limit = min(limit, (steps // eval_interval + 1) * eval_interval)
                requested = c['gorl_unroll_length'] * c['gorl_ppo_batch_size'] * c['gorl_num_minibatches']
                previous = steps
                while steps < limit and len(rows) < requested:
                    rows.append(collect_query(limit))
                optimize_rollout(rows)
                if eval_interval > 0 and steps // eval_interval > previous // eval_interval:
                    record_evaluation()
            if not reserve:
                continue
            # Reuse only evaluations from the shared cadence; no extra episodes
            # to select the stage's best encoder. Tiny test schedules may have none.
            if best_agent is not None:
                agent = best_agent.replace(rng=agent.rng)
            phase = 'decoder_collection'
            episodes += 1
            raw, _ = env.reset(seed=seed + 10000 + episodes)
            observation = encode(raw)
            first_episode = episodes
            collection_start = steps
            rows = []
            while steps < stage_end and episodes - first_episode < c['gorl_collection_episodes']:
                rows.append(collect_query(int(stage_end)))
            data = _stack(rows)
            # Every target is a fresh action from the selected stage policy. Latents
            # are retained for provenance but are not inputs to the CFM loss.
            np.savez_compressed(output / f'gorl_decoder_rollout_stage{stage}.npz', **data)
            split_rng = np.random.default_rng(seed + 80000 + stage)
            decoder_rng = np.random.default_rng(seed + 90000 + stage)
            train, validation = _decoder_split(data, c['gorl_validation_fraction'], split_rng)
            np.savez(output / f'gorl_decoder_split_stage{stage}.npz', train=train, validation=validation)
            stage_records.append(dict(stage=stage, collection_steps=steps-collection_start,
                collection_queries=len(rows), collection_completed_episodes=episodes-first_episode,
                collection_policy='best_evaluated' if best_agent is not None else 'last_unevaluated',
                best_evaluation_return=best_return if best_agent is not None else None))
            agent = agent.reset_decoder(data['observations'][train])
            decoder_batch_size = c['gorl_decoder_batch_size']
            # Reference fixed held-out indices, multiple batches and fresh eval noise.
            val_rng = np.random.default_rng(seed + 70000 + stage)
            val_indices = val_rng.choice(validation,
                size=(c['gorl_decoder_eval_batches'], decoder_batch_size), replace=True)
            fields = ('observations', 'true_actions', 'action_valid')
            val_batch = {key: jnp.asarray(data[key][val_indices]) for key in fields}
            val_key = jax.random.PRNGKey(seed + 70000 + stage)
            def validate(key):
                losses = [float(agent.decoder_validation(
                    {k: v[i] for k, v in val_batch.items()}, jax.random.fold_in(key, i)))
                    for i in range(c['gorl_decoder_eval_batches'])]
                return float(np.mean(losses))
            best_loss, best_decoder = validate(val_key), agent.bc_flow_network
            if not np.isfinite(best_loss):
                raise FloatingPointError('Non-finite initial GoRL decoder validation loss')
            for update in range(1, c['gorl_decoder_steps'] + 1):
                indices = decoder_rng.choice(train, size=decoder_batch_size, replace=True)
                batch = {key: jnp.asarray(data[key][indices]) for key in fields}
                agent, metrics = agent.decoder_update(batch)
                decoder_updates += 1
                if update % c['gorl_decoder_eval_interval'] == 0 or update == c['gorl_decoder_steps']:
                    val_key, evaluation_key = jax.random.split(val_key)
                    loss = validate(evaluation_key)
                    if not np.isfinite(loss):
                        raise FloatingPointError('Non-finite GoRL decoder validation loss')
                    if loss < best_loss:
                        best_loss, best_decoder = loss, agent.bc_flow_network
                    logger.log(dict(flow_loss=float(metrics['flow_loss']), validation_loss=loss,
                        best_validation_loss=best_loss, stage=stage, update=decoder_updates),
                        'gorl_decoder', step=start_step + steps)
            agent = agent.replace(bc_flow_network=best_decoder)
            stage_output = output / f'gorl_stage{stage}_refined'
            stage_output.mkdir(parents=True, exist_ok=True)
            # Save a usable prior+refined-decoder policy, not the old encoder
            # whose latent coordinates were optimized for the previous map.
            # Next loop iteration reproduces exactly this reset from agent.rng.
            save_agent(agent.reset_encoder(stage + 1), stage_output, start_step + steps)
            logger.log(budget(), 'online_budget', step=start_step + steps)

    if steps != online_steps or steps != encoder_env_steps + collection_env_steps + warmup_env_steps:
        raise RuntimeError('GoRL physical interaction budget mismatch')
    expected_decoder_updates = (len(boundaries) - 1) * c['gorl_decoder_steps']
    if decoder_updates != expected_decoder_updates:
        raise RuntimeError('GoRL decoder update budget mismatch')
    # Report the final updated policy once, including non-aligned final steps.
    record_evaluation()
    logger.log(budget(), 'online_budget', step=start_step + steps)
    save_agent(agent, save_dir, start_step + steps)
    manifest.update(actual_budget=budget(), stages=stage_records, final_eval={k: float(v) for k, v in final_eval.items()
                                                      if np.isscalar(v)})
    (output / 'gorl_protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return agent, start_step + steps, final_eval, None
