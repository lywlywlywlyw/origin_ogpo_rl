"""In-memory offline preparation and frozen-decoder latent IQL for GRACE."""
import jax
import numpy as np
import optax
import tqdm

from ogpo.utils.datasets import Dataset


def with_demo_boundaries(dataset):
    demo_ends = getattr(dataset, 'demo_ends', None)
    if demo_ends is None:
        return dataset
    result = Dataset.create(**dict(dataset, terminals=np.array(demo_ends, copy=True)),
                            numpy_rng=dataset.numpy_rng)
    result.frame_stack = dataset.frame_stack
    return result


def prepare_iql_transitions(dataset, max_episode_length=None):
    """Loaded robomimic rewards are -1/0 (or already shaped -1/+5).

    Keep each episode through its first success or time limit. Episode boundaries
    prevent chunk crossing; only success disables the Bellman bootstrap.
    The source arrays/files are never mutated.
    """
    if not dataset.size:
        raise ValueError('Latent IQL requires nonempty offline data')
    dataset = with_demo_boundaries(dataset)
    kept, terminals = [], []
    start = 0
    for end in np.union1d(dataset.terminal_locs, [dataset.size - 1]):
        limit = min(end + 1, start + max_episode_length) if max_episode_length else end + 1
        successes = np.flatnonzero(np.asarray(dataset['rewards'][start:limit]) > -.5)
        stop = start + int(successes[0]) + 1 if len(successes) else limit
        rows = np.arange(start, stop)
        kept.extend(rows)
        terminals.extend([0.] * (len(rows) - 1) + [1.])
        start = int(end) + 1
    rows = np.asarray(kept)
    # sample() preserves any observation frame stacking configured by the loader.
    raw = dataset.sample(len(rows), idxs=rows)
    success = np.asarray(raw['rewards']) > -.5
    return Dataset.create(
        observations=np.array(raw['observations'], copy=True),
        next_observations=np.array(raw['next_observations'], copy=True),
        actions=np.array(raw['actions'], copy=True),
        rewards=np.where(success, 5., -1.).astype(np.float32),
        masks=(~success).astype(np.float32),
        terminals=np.asarray(terminals, dtype=np.float32))


def build_latent_iql_dataset(agent, dataset, config, seed, max_episode_length=None,
                             *, batch_size=None, purpose='Latent IQL'):
    """Shared in-memory inverse data preparation for IQL and online demo replay."""
    from ogpo.runners.grace_runner import _offline_q_chunks
    prepared = prepare_iql_transitions(dataset, max_episode_length)
    horizon = config['horizon_length'] if config['action_chunking'] else 1
    batch_size = config['iql_batch_size'] if batch_size is None else batch_size
    output = None
    for offset in tqdm.trange(0, prepared.size, batch_size, desc='GRACE offline inversion'):
        starts = np.arange(offset, min(offset + batch_size, prepared.size))
        chunks, _ = _offline_q_chunks(prepared, starts, horizon, config['discount'], 'discounted_sum')
        z = np.asarray(agent.inverse_fm_batch(chunks['observations'], chunks['true_actions']))
        if not np.isfinite(z).all():
            raise ValueError(f'Non-finite offline inverse latents at offset {offset}')
        chunks['actions'] = z
        chunks['episode_id'] = np.searchsorted(prepared.terminal_locs, starts)
        if output is None:
            output = {k: np.empty((prepared.size, *v.shape[1:]), dtype=v.dtype)
                      for k, v in chunks.items()}
        for k, v in chunks.items():
            output[k][offset:offset + len(starts)] = v
    print(f'{purpose}: {dataset.size} loaded rows -> {prepared.size} pre-success/time-limit '
          f'rows; z RMS={np.sqrt(np.mean(output["actions"] ** 2)):.4g}; '
          'step rewards=-1/+5; success mask=0, timeout mask=1')
    return Dataset.create(**output, numpy_rng=np.random.default_rng(seed + 60000))


def split_iql_indices(data, fraction, seed):
    """Hold out whole trajectories: overlapping action chunks must not leak."""
    episodes = np.unique(data['episode_id'])
    if len(episodes) < 2:
        raise ValueError('IQL validation requires at least two offline trajectories')
    rng = np.random.default_rng(seed + 70000)
    count = min(len(episodes) - 1, max(1, int(len(episodes) * fraction)))
    held_out = rng.choice(episodes, count, replace=False)
    mask = np.isin(data['episode_id'], held_out)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def pretrain_latent_iql(agent, dataset, seed, logger, log_step, maybe_shard,
                       max_episode_length=None):
    config = agent.config
    data = build_latent_iql_dataset(agent, dataset, config, seed, max_episode_length)
    train_indices, validation_indices = split_iql_indices(data, config['iql_validation_fraction'], seed)
    rng = np.random.default_rng(seed + 80000)
    validation_indices = rng.choice(validation_indices, min(len(validation_indices),
        config['iql_validation_batches'] * config['iql_batch_size']), replace=False)
    validation_batch = data.sample(len(validation_indices), idxs=validation_indices)
    best_state, best_score, best_step, stale = None, float('inf'), 0, 0
    validation_metrics = {}
    # Match GoRL's offline optimizer settings, then restore fresh online optimizer
    # states around the learned parameters (including SAC's untrained temperature).
    online_actor_tx, online_critic_tx = agent.actor_network.tx, agent.critic_network.tx
    def optimizer(lr):
        return optax.chain(optax.clip_by_global_norm(config['iql_clip_grad_norm']), optax.adam(lr))
    def reset(state, tx):
        return state.replace(tx=tx, opt_state=tx.init(state.params), step=1)
    agent = agent.replace(actor_network=reset(agent.actor_network, optimizer(config['iql_actor_lr'])),
                          critic_network=reset(agent.critic_network, optimizer(config['iql_critic_lr'])))
    for step in tqdm.trange(1, config['iql_steps'] + 1, desc='GRACE offline latent IQL'):
        indices = rng.choice(train_indices, config['iql_batch_size'], replace=True)
        batch = maybe_shard(data.sample(config['iql_batch_size'], idxs=indices))
        agent, metrics = agent.pretrain_latent_iql(batch)
        validate = step % config['iql_validation_interval'] == 0 or step == config['iql_steps']
        should_stop = False
        if validate:
            validation_metrics = dict(jax.device_get(agent.iql_validation_metrics(validation_batch)))
            score = float(validation_metrics['validation_score'])
            if not np.isfinite(score):
                raise ValueError(f'Non-finite IQL validation score at step {step}')
            # Match GoRL: every strict improvement is retained, including before
            # min_steps. Its declared min_delta is not used by its IQL loop.
            if score < best_score:
                best_score, best_step, stale = score, step, 0
                best_state = (agent.actor_network, agent.critic_network, agent.value_network)
            elif step >= config['iql_early_stopping_min_steps']:
                stale += 1
            should_stop = (step >= config['iql_early_stopping_min_steps']
                           and stale >= config['iql_early_stopping_patience'])
        if step == 1 or step % 1000 == 0 or validate:
            selection = (dict(best_iql_step=best_step, best_validation_score=best_score,
                              stale_validations=stale) if best_state is not None else {})
            logger.log(dict(metrics, **validation_metrics, **selection, updates=step,
                            dataset_size=data.size, train_size=len(train_indices),
                            validation_size=len(validation_indices), early_stopped=int(should_stop)),
                       'latent_iql', step=log_step + step)
        if should_stop:
            break
    jax.block_until_ready(agent)
    actor, critic, value = best_state
    print(f'IQL completed {step} updates (early_stopped={should_stop}); '
          f'restored best step {best_step}, validation score={best_score:.6g}')
    agent = agent.replace(actor_network=reset(actor, online_actor_tx),
                          critic_network=reset(critic, online_critic_tx), value_network=value)
    return agent, step + 1
