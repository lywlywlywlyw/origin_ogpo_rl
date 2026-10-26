"""Spawned frozen-decoder augmentation worker and sole shared-replay writer."""
import multiprocessing as mp
import os
from queue import Empty, Full
import time
import traceback
from types import SimpleNamespace

import numpy as np

from ogpo.utils.shared_replay import SharedReplay


def make_augmentation_kernel(config, model_def, params):
    """Build only the frozen decoder, without actor/critics/optimizer copies."""
    import jax
    from ogpo.agents.jgcw import JGCWAgent
    params = jax.tree.map(jax.numpy.asarray, params)
    def select(name):
        return lambda *args: model_def.apply({'params': params}, *args, name=name)
    decoder = SimpleNamespace(config=config, bc_flow_network=SimpleNamespace(select=select))
    decoder.inverse_fm_batch = lambda *args: JGCWAgent.inverse_fm_batch.__wrapped__(decoder, *args)
    return jax.jit(lambda obs, actions, keys, anchors:
                   JGCWAgent.augment_z.__wrapped__(decoder, obs, actions, keys, anchors))


def _worker(queue, conn, replay_spec, spec, initial):
    replay = None
    try:
        cpu_count = spec['config'].get('aug_worker_cpu_count', 4)
        if hasattr(os, 'sched_getaffinity'):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, available[:cpu_count])
        import jax
        replay = SharedReplay(replay_spec, writer=True)
        if initial is not None:
            # Restore cached data only in the designated writer.
            for start in range(0, initial.size, 128):
                rows = [{key: value[i] for key, value in initial.items()}
                        for i in range(start, min(initial.size, start + 128))]
                replay.write_batch(rows, [None] * len(rows))
            with replay._lock:
                replay._meta[1] = initial.pointer
        config = spec['config']
        augment = make_augmentation_kernel(config, spec['model_def'], spec['params'])
        base_key = jax.random.fold_in(jax.random.PRNGKey(spec['seed'] + 50000), spec['resumed_step'])
        make_keys = jax.jit(jax.vmap(lambda episode, chunk: jax.random.fold_in(
            jax.random.fold_in(base_key, episode), chunk)))
        conn.send(('ready', str(jax.devices()[0])))
        pending_control = None
        batches, transitions, compute_seconds = 0, 0, 0.
        while True:
            message = pending_control if pending_control is not None else queue.get()
            pending_control = None
            kind, payload = message
            if kind == 'stop':
                break
            if kind == 'episode':
                replay.finish_episode(*payload)
                continue
            if kind == 'barrier':
                conn.send(('stats', dict(batches=batches, augmented_transitions=transitions,
                                         compute_seconds=compute_seconds)))
                conn.send(('barrier', payload))
                continue
            if kind != 'transition':
                raise ValueError(f'Unknown augmentation message: {kind}')
            tasks = [payload]
            # Coalesce queued transitions without waiting for an episode or
            # accumulating unbounded work. Do not cross an ordered control message.
            while len(tasks) < config['aug_worker_batch_size']:
                try:
                    kind, payload = queue.get_nowait()
                except Empty:
                    break
                if kind != 'transition':
                    pending_control = (kind, payload)
                    break
                tasks.append(payload)
            count = len(tasks)
            # At most log2(max_batch)+1 shapes, not one compilation per episode length.
            bucket = min(config['aug_worker_batch_size'], 1 << (count - 1).bit_length())
            padded = tasks + [tasks[-1]] * (bucket - count)
            obs = np.stack([t[2]['observations'] for t in padded]).astype(np.float32)
            actions = np.stack([t[2]['true_actions'] for t in padded]).astype(np.float32)
            anchors = np.stack([t[2]['actions'] for t in padded]).astype(np.float32)
            # Keys belong to transitions, independent of scheduling and padding.
            if not all(np.isfinite(value).all() for value in (obs, actions, anchors)):
                raise FloatingPointError('Non-finite source data rejected before augmentation')
            keys = make_keys(np.asarray([t[0] for t in padded], dtype=np.uint32),
                             np.asarray([t[1] for t in padded], dtype=np.uint32))
            started = time.perf_counter()
            z_aug = np.asarray(augment(obs, actions, keys, anchors), dtype=np.float32)[:count]
            compute_seconds += time.perf_counter() - started
            if not np.isfinite(z_aug).all():
                raise FloatingPointError('Non-finite z_aug rejected before shared replay insertion')
            rows = [dict(task[2], z_aug=z) for task, z in zip(tasks, z_aug)]
            replay.write_batch(rows, [task[0] for task in tasks])
            batches += 1
            transitions += count
            if batches % 16 == 1:
                conn.send(('stats', dict(batches=batches, augmented_transitions=transitions,
                                         compute_seconds=compute_seconds)))
    except BaseException:
        try:
            conn.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if replay is not None:
            replay.close()
        conn.close()


class AsyncAugmentationReplay:
    """Main process sends raw transitions and reads shared, completed rows."""

    def __init__(self, agent, example, capacity, seed, config, resumed_step=0, initial=None):
        import jax
        config = {**dict(agent.config), **dict(config)}
        defaults = dict(aug_worker_batch_size=8, aug_worker_queue_size=128,
                        aug_worker_cpu_count=4, aug_worker_timeout=600,
                        aug_worker_device='auto', aug_worker_cuda_visible_devices=None)
        for key, value in defaults.items():
            config.setdefault(key, value)
        for key in ('aug_worker_batch_size', 'aug_worker_queue_size', 'aug_worker_cpu_count',
                    'aug_worker_timeout'):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        device = config['aug_worker_device']
        if device == 'auto':
            device = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'
        if device not in ('cpu', 'cuda'):
            raise ValueError('aug_worker_device must be auto, cpu, or cuda')
        context = mp.get_context('spawn')
        self.replay = SharedReplay.create(example, capacity, context, seed + 30000)
        if initial is not None:
            self.replay.numpy_rng.bit_generator.state = initial.numpy_rng.bit_generator.state
        self.queue = context.Queue(maxsize=config['aug_worker_queue_size'])
        self.conn, child = context.Pipe(duplex=False)
        self.timeout = config['aug_worker_timeout']
        self.initial_size = initial.size if initial is not None else 0
        self.submitted = 0
        self.stats = dict(batches=0, augmented_transitions=0, compute_seconds=0.)
        self.backend = 'initializing'
        self.closed = False
        self._barrier = 0
        self._acknowledged = -1
        self.wait_seconds = 0.
        spec = dict(config=config, model_def=agent.bc_flow_network.model_def,
                    params=jax.device_get(agent.bc_flow_network.params), seed=seed,
                    resumed_step=resumed_step)
        self.process = context.Process(target=_worker,
            args=(self.queue, child, self.replay.spec, spec, initial),
            name='jgcw-augmentation-writer', daemon=True)
        overrides = dict(JAX_PLATFORMS=device, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                         OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                         WANDB_MODE='disabled')
        if config['aug_worker_cuda_visible_devices'] is not None:
            overrides['CUDA_VISIBLE_DEVICES'] = str(config['aug_worker_cuda_visible_devices'])
        if device == 'cpu':
            overrides['CUDA_VISIBLE_DEVICES'] = ''
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            self.process.start()
            self.replay.health_check = self.check
        except BaseException:
            self.replay.health_check = None
            self.replay.unlink()
            self.replay.close()
            self.queue.close()
            self.conn.close()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            child.close()

    def check(self):
        while self.conn.poll():
            try:
                kind, payload = self.conn.recv()
            except EOFError:
                raise RuntimeError(f'JGCW augmentation worker exited (code {self.process.exitcode})') from None
            if kind == 'error':
                raise RuntimeError(f'JGCW augmentation worker failed:\n{payload}')
            if kind == 'ready':
                self.backend = payload
                print(f'JGCW augmentation writer ready: pid={self.process.pid}, device={payload}', flush=True)
            elif kind == 'stats':
                self.stats = payload
            elif kind == 'barrier':
                self._acknowledged = payload
        if not self.process.is_alive():
            raise RuntimeError(f'JGCW augmentation worker exited (code {self.process.exitcode})')

    def _send(self, kind, payload):
        deadline = time.monotonic() + self.timeout
        started = time.perf_counter()
        while True:
            self.check()
            try:
                self.queue.put((kind, payload), timeout=.05)
                break
            except Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError('JGCW augmentation queue remained full')
        self.wait_seconds += time.perf_counter() - started

    def submit(self, transition, episode, chunk):
        # Own immutable host copies: Queue's feeder serializes asynchronously.
        row = {key: np.array(value, copy=True) for key, value in transition.items()}
        self._send('transition', (episode, chunk, row))
        self.submitted += 1

    def finish_episode(self, episode, success):
        self._send('episode', (episode, bool(success)))

    def wait_for_size(self, minimum):
        if minimum > self.replay.max_size:
            raise RuntimeError('Shared replay capacity cannot supply the requested update batch')
        deadline = time.monotonic() + self.timeout
        started = time.perf_counter()
        while self.replay.size < minimum:
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for augmented replay rows')
            time.sleep(.005)
        self.wait_seconds += time.perf_counter() - started

    def flush(self):
        """Wait for all preceding transitions/labels; writer then stays idle."""
        token = self._barrier
        self._barrier += 1
        self._send('barrier', token)
        deadline = time.monotonic() + self.timeout
        while self._acknowledged < token:
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out draining JGCW augmentation worker')
            time.sleep(.005)

    def metrics(self):
        self.check()
        return {**self.stats, 'submitted_transitions': self.submitted,
                'published_transitions': self.replay.total_written,
                'pending_transitions': max(0, self.submitted + self.initial_size - self.replay.total_written),
                'queue_and_replay_wait_seconds': self.wait_seconds,
                'worker_pid': self.process.pid}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                try:
                    self.queue.put(('stop', None), timeout=.2)
                except Full:
                    pass
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
        finally:
            self.queue.cancel_join_thread()
            self.queue.close()
            self.conn.close()
            # Unlink the OS name, retain the parent's mapped read view for callers.
            self.replay.health_check = None
            self.replay.unlink()
