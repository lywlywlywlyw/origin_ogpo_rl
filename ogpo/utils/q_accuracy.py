"""Isolated CPU Monte Carlo evaluation of paired latent/action Q predictions.

A fixed first action chunk is essential: policy-only rollouts estimate V(s),
not Q(s,a). Time limits are ignored during diagnostic continuation because the
training TD masks bootstrap through truncations. Only true terminals stop G.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np


def select_accuracy_states(dataset_path, count, seed):
    """Sample all four temporal quarters of successful demos, without duplicates."""
    import h5py
    if isinstance(count, bool) or not isinstance(count, int) or count < 4:
        raise ValueError('accuracy_s_num must be an integer >= 4')
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, 'r') as f:
        demos = []
        for name in sorted(f['data']):
            group = f['data'][name]
            if 'states' not in group or 'model_file' not in group.attrs:
                continue
            successes = np.flatnonzero(np.asarray(group['rewards']).reshape(-1) > .5)
            if len(successes) and successes[0] + 1 >= 4:
                demos.append((name, int(successes[0]) + 1))
        if not demos:
            raise ValueError('Q accuracy requires successful demos with simulator states and model_file')
        bank, used = [], set()
        for i in range(count):
            stage = i % 4
            candidates = []
            for index in rng.permutation(len(demos)):
                name, length = demos[index]
                lo, hi = stage * length // 4, (stage + 1) * length // 4
                times = [t for t in range(lo, hi) if (name, t) not in used]
                if times:
                    candidates.append((name, length, int(rng.choice(times))))
            if not candidates:
                raise ValueError(f'Not enough distinct accuracy states in stage {stage}')
            # Prefer distinct demos within each stage when possible.
            stage_demos = {row['demo'] for row in bank if row['stage'] == stage}
            name, length, timestep = next((c for c in candidates if c[0] not in stage_demos), candidates[0])
            group = f['data'][name]
            model = group.attrs['model_file']
            if isinstance(model, bytes):
                model = model.decode()
            ep_meta = group.attrs.get('ep_meta', '{}')
            if isinstance(ep_meta, bytes):
                ep_meta = ep_meta.decode()
            bank.append(dict(demo=name, timestep=timestep, stage=stage,
                             progress=timestep / length, model=model, ep_meta=ep_meta,
                             states=np.asarray(group['states'][timestep]).copy(),
                             demo_action=np.asarray(group['actions'][timestep]).copy()))
            used.add((name, timestep))
    return bank


def restore_accuracy_state(env, state, seed):
    """Restore model XML as well as qpos/qvel; reset controller and wrapper clocks."""
    env.reset(seed=seed)
    env.env.reset_to({key: state[key] for key in ('states', 'model', 'ep_meta')})
    env.t = 0
    env.t_succ = None
    env.episode_return = 0.
    env.episode_length = 0
    return np.asarray(env.get_observation(), dtype=np.float32)


def rollout_return(agent, env, observation, first_actions, rng, discount, max_steps):
    """Fixed a first, then the snapshot's stochastic policy at chunk boundaries."""
    import jax
    action_dim = env.action_space.shape[-1]
    queue = list(np.asarray(first_actions).reshape(-1, action_dim))
    value, weight, hit_time_limit = 0., 1., False
    for step in range(max_steps):
        if not queue:
            rng, key = jax.random.split(rng)
            queue = list(np.asarray(agent.sample_actions(observation, rng=key)).reshape(-1, action_dim))
        observation, reward, terminated, truncated, _ = env.step(queue.pop(0))
        value += weight * float(reward)
        weight *= discount
        hit_time_limit |= bool(truncated)
        if terminated:
            return value, step + 1, False, hit_time_limit
        # Artificial time limits are nonterminal in DSRL's Bellman objective.
    return value, max_steps, True, hit_time_limit


def accuracy_metrics(qz, qa, returns, stages):
    means = np.mean(returns, axis=1)
    sem = np.std(returns, axis=1, ddof=1) / np.sqrt(returns.shape[1])
    metrics = dict(mc_return_mean=float(means.mean()), mc_se_mean=float(sem.mean()),
                   mc_se_max=float(sem.max()), states=len(means),
                   rollouts_per_state=returns.shape[1])
    for name, predictions in [('q_sz', qz), ('q_sa', qa)]:
        error = predictions - means
        metrics.update({f'{name}/mae': float(np.abs(error).mean()),
                        f'{name}/rmse': float(np.sqrt(np.square(error).mean())),
                        f'{name}/bias': float(error.mean()),
                        f'{name}/prediction_mean': float(predictions.mean()),
                        f'{name}/mc_noise_corrected_mse': float(np.mean(error ** 2 - sem ** 2))})
        for stage in np.unique(stages):
            mask = stages == stage
            metrics[f'{name}/stage_{stage}/mae'] = float(np.abs(error[mask]).mean())
            metrics[f'{name}/stage_{stage}/bias'] = float(error[mask].mean())
    for i in range(len(means)):
        metrics.update({f'state_{i}/mc_mean': float(means[i]),
                        f'state_{i}/mc_se': float(sem[i]),
                        f'state_{i}/q_sz': float(qz[i]), f'state_{i}/q_sa': float(qa[i])})
    return metrics


def _install_snapshot(agent, snapshot):
    import jax
    for name, params in snapshot.items():
        state = getattr(agent, name)
        merged = dict(state.params)
        merged.update(jax.device_put(params))
        agent = agent.replace(**{name: state.replace(params=merged)})
    return agent


def _accuracy_worker(conn, spec, initial_snapshot):
    env = None
    try:
        # The parent sets CPU env vars before spawning (before any JAX import).
        if hasattr(os, 'sched_getaffinity'):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, set(available[:spec['cpu_count']]))
        import jax
        import jax.numpy as jnp
        from envs.robomimic_utils import make_env, _check_dataset_exists
        from ogpo.agents.jgcw import JGCWAgent
        config = spec['config']
        agent = JGCWAgent.create(spec['seed'], jnp.zeros(tuple(config['ob_dims'])),
                                 jnp.zeros(config['action_dim']), config)
        agent = _install_snapshot(agent, initial_snapshot)
        env = make_env(spec['env_name'], seed=spec['seed'],
                       post_success_steps=spec['post_success_steps'])
        bank = select_accuracy_states(_check_dataset_exists(spec['env_name']),
                                      spec['state_count'], spec['seed'])
        observations = np.stack([restore_accuracy_state(env, row, spec['seed'] + i)
                                 for i, row in enumerate(bank)])
        if observations.shape[1:] != tuple(config['ob_dims']):
            raise ValueError('Q accuracy currently supports unencoded Robomimic low_dim observations only')
        noises = np.asarray(agent.sample_noise(observations, rng=jax.random.PRNGKey(spec['seed'] + 1)))
        actions = np.asarray(agent.noise_to_actions_public(observations, noises))
        stages = np.asarray([row['stage'] for row in bank])
        out = Path(spec['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'fixed_bank.npz', observations=observations, noises=noises,
                            actions=actions, sim_states=np.stack([row['states'] for row in bank]),
                            stages=stages, models=np.asarray([row['model'] for row in bank]))
        manifest = [{k: v for k, v in row.items() if k not in ('states', 'model', 'demo_action')}
                    for row in bank]
        (out / 'fixed_bank.json').write_text(json.dumps(manifest, indent=2))
        (out / 'evaluation_config.json').write_text(json.dumps(spec, indent=2))
        conn.send(('ready', {'stage_counts': np.bincount(stages, minlength=4).tolist()}))
        while True:
            message = conn.recv()
            if message is None:
                break
            snapshot_step, snapshot = message
            started = time.monotonic()
            agent = _install_snapshot(agent, snapshot)
            qz = agent.critic_network.select('critic')(observations, actions=noises)
            qa = agent.action_critic_network.select('critic')(observations, actions=actions)
            aggregate = np.min if config['q_agg'] == 'min' else np.mean
            qz, qa = aggregate(np.asarray(qz), axis=0), aggregate(np.asarray(qa), axis=0)
            returns = np.zeros((len(bank), spec['repeats']))
            cutoffs = time_limits = sim_steps = 0
            restore_max_error = 0.
            for i, state in enumerate(bank):
                for repeat in range(spec['repeats']):
                    # Common random seeds across snapshots reduce comparison noise;
                    # repeats still have distinct independent policy RNG streams.
                    rollout_seed = spec['seed'] + 10000 + i * spec['repeats'] + repeat
                    observation = restore_accuracy_state(env, state, rollout_seed)
                    restore_max_error = max(restore_max_error,
                                            float(np.abs(observation - observations[i]).max()))
                    if restore_max_error > 1e-4:
                        raise ValueError(f'Unstable simulator state restoration: {restore_max_error}')
                    value, steps, cutoff, time_limit = rollout_return(
                        agent, env, observation, actions[i], jax.random.PRNGKey(rollout_seed),
                        config['discount'], spec['rollout_steps'])
                    returns[i, repeat] = value
                    sim_steps += steps
                    cutoffs += int(cutoff)
                    time_limits += int(time_limit)
            metrics = accuracy_metrics(qz, qa, returns, stages)
            metrics.update(snapshot_step=snapshot_step, duration_seconds=time.monotonic() - started,
                           simulator_steps=sim_steps, cutoff_fraction=cutoffs / returns.size,
                           time_limit_crossed_fraction=time_limits / returns.size,
                           restore_max_error=restore_max_error,
                           # Wrapper rewards are in [-1, 5]. Unobserved tail bound.
                           cutoff_tail_bound=5 * config['discount'] ** spec['rollout_steps'] / (1 - config['discount']))
            np.savez_compressed(out / f'snapshot_{snapshot_step}.npz', q_sz=qz, q_sa=qa,
                                returns=returns, stages=stages)
            (out / f'snapshot_{snapshot_step}.json').write_text(json.dumps(metrics, indent=2))
            conn.send(('result', metrics))
    except BaseException:
        try:
            conn.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.env.env.close()
            except AttributeError:
                env.close()
        conn.close()


class AsyncQAccuracy:
    """One CPU worker, at most one outstanding snapshot, parent-only W&B logging."""
    def __init__(self, agent, config, env_name, save_dir, seed, post_success_steps=0):
        if not config.get('train_q_sa', False):
            raise ValueError('cal_q_accuracy requires train_q_sa=true')
        if not env_name.endswith('-low_dim') or config.get('encoder') is not None:
            raise ValueError('Q accuracy currently requires a Robomimic low_dim environment')
        if config.get('chunk_reward_aggregation') != 'discounted_sum' or config.get('entropy_backup', False):
            raise ValueError('Q accuracy requires discounted_sum chunk rewards and entropy_backup=false')
        if not 0 < config['discount'] < 1:
            raise ValueError('Q accuracy requires 0 < discount < 1')
        count, repeats = config.get('accuracy_s_num', 16), config.get('accuracy_a_num', 8)
        if isinstance(count, bool) or not isinstance(count, int) or count < 4:
            raise ValueError('accuracy_s_num must be an integer >= 4')
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
            raise ValueError('accuracy_a_num must be an integer >= 2 (needed for MC standard errors)')
        self.interval = config.get('accuracy_interval', 100000)
        self.timeout = config.get('accuracy_timeout', 3600)
        rollout_steps = config.get('accuracy_rollout_steps', 1000)
        cpu_count = config.get('accuracy_cpu_count', 2)
        for name, value in [('accuracy_interval', self.interval), ('accuracy_timeout', self.timeout),
                            ('accuracy_rollout_steps', rollout_steps), ('accuracy_cpu_count', cpu_count)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        context = mp.get_context('spawn')
        self.conn, child = context.Pipe()
        self.busy = False
        self.last_step = None
        self.submitted_at = None
        spec = dict(config=dict(agent.config), seed=seed + 73000, env_name=env_name,
                    post_success_steps=post_success_steps, state_count=count, repeats=repeats,
                    rollout_steps=rollout_steps, cpu_count=cpu_count,
                    output_dir=str(Path(save_dir).resolve() / 'q_accuracy'))
        self.process = context.Process(target=_accuracy_worker,
                                       args=(child, spec, self.snapshot(agent, include_decoder=True)),
                                       name='jgcw-q-accuracy', daemon=True)
        # Spawn reimports __main__. Set these before it can import JAX, and
        # restore parent settings immediately after the child's environment copy.
        overrides = dict(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', WANDB_MODE='disabled')
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            self.process.start()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            child.close()
        try:
            kind, payload = self._wait_message()
            if kind != 'ready':
                raise RuntimeError(f'Q accuracy initialization failed: {payload}')
            print(f"Q accuracy bank ready: {count} states, four-stage counts {payload['stage_counts']}, {repeats} rollouts/state", flush=True)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def snapshot(agent, include_decoder=False):
        import jax
        names = ['actor_network', 'critic_network', 'action_critic_network']
        if include_decoder:
            names.append('bc_flow_network')
        result = {}
        for name in names:
            # Only online modules are used for evaluation. No optimizer states.
            params = {k: v for k, v in getattr(agent, name).params.items()
                      if not k.startswith('modules_target_')}
            result[name] = jax.device_get(params)
        return result

    def _wait_message(self):
        started = self.submitted_at if self.busy else time.monotonic()
        while not self.conn.poll(1.):
            if not self.process.is_alive():
                raise RuntimeError(f'Q accuracy worker exited ({self.process.exitcode})')
            if time.monotonic() - started > self.timeout:
                raise TimeoutError('Q accuracy worker timed out')
        return self.conn.recv()

    def poll(self, logger, log_step):
        if not self.busy:
            return
        if self.conn.poll():
            kind, payload = self.conn.recv()
            self.busy = False
            if kind != 'result':
                raise RuntimeError(f'Q accuracy worker failed: {payload}')
            # Log on the current training step; retain the evaluated snapshot's
            # original step explicitly, avoiding out-of-order W&B writes.
            logger.log(payload, 'q_accuracy', step=log_step)
        elif not self.process.is_alive():
            raise RuntimeError(f'Q accuracy worker exited ({self.process.exitcode})')
        elif time.monotonic() - self.submitted_at > self.timeout:
            raise TimeoutError('Q accuracy evaluation exceeded accuracy_timeout')

    def submit(self, agent, snapshot_step):
        if self.busy:
            return False
        self.conn.send((snapshot_step, self.snapshot(agent)))
        self.busy, self.last_step = True, snapshot_step
        self.submitted_at = time.monotonic()
        return True

    def finish(self, agent, snapshot_step, logger, log_step):
        if self.busy:
            kind, payload = self._wait_message()
            self.busy = False
            if kind != 'result':
                raise RuntimeError(f'Q accuracy worker failed: {payload}')
            logger.log(payload, 'q_accuracy', step=log_step)
        if self.last_step != snapshot_step:
            self.submit(agent, snapshot_step)
            kind, payload = self._wait_message()
            self.busy = False
            if kind != 'result':
                raise RuntimeError(f'Q accuracy worker failed: {payload}')
            logger.log(payload, 'q_accuracy', step=log_step)

    def close(self):
        if self.process.is_alive():
            if not self.busy:
                try:
                    self.conn.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.conn.close()
