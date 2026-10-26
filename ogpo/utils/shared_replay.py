"""Single-writer shared replay with optimistic, version-checked row copies.

Only publication/version metadata is locked. Large field writes and reader copies
run concurrently; a reader retries if any copied row changed during its copy.
"""
from collections.abc import Mapping
from multiprocessing import shared_memory
import os
import shutil
import time

import numpy as np


def _restore_snapshot(fields, capacity, pointer, rng_state, p_aug):
    from ogpo.utils.datasets import ReplayBuffer
    result = ReplayBuffer.create({k: v[0] for k, v in fields.items()}, capacity)
    size = len(next(iter(fields.values())))
    for key, values in fields.items():
        result._dict[key][:size] = values
    result.size, result.pointer = size, pointer
    result._traj_success_mask[:size] = fields['is_success'] > 0
    result.numpy_rng.bit_generator.state = rng_state
    result.p_aug = p_aug
    return result


class SharedReplay(Mapping):
    """Parent read view; the designated child is the only transition writer."""

    @classmethod
    def create(cls, example, capacity, context, seed=0):
        if capacity < 1:
            raise ValueError('Shared replay capacity must be positive')
        example = dict(example)
        example.setdefault('is_success', np.float32(0))
        example.setdefault('score', np.float32(0))
        layout, offset = {}, 0
        for key, value in example.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError('Shared replay requires numeric array fields')
            offset = (offset + 63) // 64 * 64
            shape = (capacity,) + array.shape
            layout[key] = (offset, shape, array.dtype.str)
            offset += int(np.prod(shape)) * array.dtype.itemsize
        offset = (offset + 63) // 64 * 64
        version_offset = offset
        offset += capacity * 8
        if offset > shutil.disk_usage('/dev/shm').free:
            raise MemoryError(f'Shared replay needs {offset / 2**30:.2f} GiB of /dev/shm')
        segment = shared_memory.SharedMemory(create=True, size=offset)
        # POSIX guarantees new pages are zero; do not touch unused replay capacity.
        spec = dict(name=segment.name, layout=layout, capacity=capacity,
                    version_offset=version_offset, meta=context.RawArray('q', 6),
                    lock=context.RLock())
        result = cls(spec, segment=segment, seed=seed)
        result._owner = True
        return result

    def __init__(self, spec, segment=None, seed=0, writer=False):
        self.spec = spec
        self._segment = segment or shared_memory.SharedMemory(name=spec['name'])
        self._owner, self._unlinked = False, False
        self._writer = writer
        self.max_size = spec['capacity']
        self._lock, self._meta = spec['lock'], spec['meta']
        self._arrays = {key: np.ndarray(shape, np.dtype(dtype), self._segment.buf, offset)
                        for key, (offset, shape, dtype) in spec['layout'].items()}
        self._versions = np.ndarray((self.max_size,), np.uint64, self._segment.buf,
                                    spec['version_offset'])
        for array in self._arrays.values():
            array.flags.writeable = writer
        self.numpy_rng = np.random.default_rng(seed)
        self.p_aug = None
        self.rng = None
        self.health_check = None
        self._episodes = {}  # writer-only: episode -> (slot, published version).
        if writer:
            with self._lock:
                if self._meta[5] not in (0, os.getpid()):
                    raise RuntimeError('Shared replay already has a different writer')
                self._meta[5] = os.getpid()

    def __getitem__(self, key):
        # Introspection only. Training must use sample(), which validates copies.
        return self._arrays[key]

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self):
        return len(self._arrays)

    @property
    def size(self):
        with self._lock:
            return int(self._meta[0])

    @property
    def pointer(self):
        with self._lock:
            return int(self._meta[1])

    @property
    def total_written(self):
        with self._lock:
            return int(self._meta[2])

    @property
    def _traj_success_mask(self):
        return self._arrays['is_success'] > 0

    def success_count(self, physical=False):
        with self._lock:
            return int(self._meta[4 if physical else 3])

    def _require_writer(self):
        if not self._writer or self._meta[5] != os.getpid():
            raise PermissionError('Only the augmentation process may write shared replay')

    def add_transition(self, transition):
        self._require_writer()
        self.write_batch([transition], [None])

    def _write_values(self, slots, values):
        # Intentionally outside the metadata lock: readers can copy other rows.
        for key, array in self._arrays.items():
            array[slots] = values[key]

    def write_batch(self, transitions, episodes):
        self._require_writer()
        if len(transitions) > self.max_size:
            # Preserve ring semantics even for tiny test buffers.
            split = self.max_size
            self.write_batch(transitions[:split], episodes[:split])
            self.write_batch(transitions[split:], episodes[split:])
            return
        if not transitions:
            return
        normalized = []
        for row in transitions:
            row = dict(row)
            row.setdefault('is_success', 0.)
            row.setdefault('score', 1. if row['is_success'] > 0 else 1e-9)
            normalized.append(row)
        values = {key: np.asarray([row[key] for row in normalized], dtype=array.dtype)
                  for key, array in self._arrays.items()}
        for key, array in self._arrays.items():
            if values[key].shape != (len(transitions),) + array.shape[1:]:
                raise ValueError(f'Shared replay shape mismatch: {key}')
        with self._lock:
            size, pointer = self._meta[0], self._meta[1]
            slots = (pointer + np.arange(len(transitions))) % self.max_size
            old_success = (self._arrays['is_success'][slots] > 0) & (slots < size)
            old_lengths = (self._arrays['chunk_length'][slots] if 'chunk_length' in self._arrays
                           else np.ones(len(slots)))
            self._versions[slots] += np.uint64(1)  # odd: cannot be read yet.
        self._write_values(slots, values)
        new_success = values['is_success'] > 0
        new_lengths = values.get('chunk_length', np.ones(len(slots)))
        with self._lock:
            self._meta[3] += int(new_success.sum() - old_success.sum())
            self._meta[4] += int(new_lengths[new_success].sum() - old_lengths[old_success].sum())
            self._versions[slots] += np.uint64(1)  # even: all fields now complete.
            self._meta[0] = min(self.max_size, size + len(slots))
            self._meta[1] = int((pointer + len(slots)) % self.max_size)
            self._meta[2] += len(slots)
            versions = self._versions[slots].copy()
        for episode, slot, version in zip(episodes, slots, versions):
            if episode is not None:
                self._episodes.setdefault(episode, []).append((int(slot), int(version)))

    def finish_episode(self, episode, success):
        """Publish trajectory success after its outcome is known, writer only."""
        self._require_writer()
        entries = self._episodes.pop(episode, [])
        with self._lock:
            slots = np.asarray([slot for slot, version in entries
                                if self._versions[slot] == version], dtype=np.int64)
            if not len(slots):
                return
            old = self._arrays['is_success'][slots] > 0
            lengths = (self._arrays['chunk_length'][slots] if 'chunk_length' in self._arrays
                           else np.ones(len(slots)))
            self._versions[slots] += np.uint64(1)
        self._arrays['is_success'][slots] = float(success)
        self._arrays['score'][slots] = 1. if success else 1e-9
        with self._lock:
            self._meta[3] += int(len(slots) * bool(success) - old.sum())
            self._meta[4] += int(lengths.sum() * bool(success) - lengths[old].sum())
            self._versions[slots] += np.uint64(1)

    def _read(self, indices, success_only=False, timeout=60):
        deadline = time.monotonic() + timeout
        indices = np.asarray(indices)
        while True:
            with self._lock:
                if np.any(indices < 0) or np.any(indices >= self._meta[0]):
                    raise IndexError('Shared replay index outside published rows')
                before = self._versions[indices].copy()
            if np.all(before % 2 == 0):
                result = {key: array[indices] for key, array in self._arrays.items()}
                with self._lock:
                    after = self._versions[indices].copy()
                if np.array_equal(before, after):
                    if success_only and not np.all(result['is_success'] > 0):
                        raise ValueError('Success candidates changed during sampling; retry')
                    return result
            if self.health_check is not None:
                self.health_check()
            if time.monotonic() >= deadline:
                raise TimeoutError('Shared replay writer did not publish complete rows')
            time.sleep(.0005)

    def sample(self, batch_size, idxs=None, success_only=False, replace=False):
        if idxs is None:
            if success_only:
                # Selection is optimistic too. _read checks row versions/labels.
                for _ in range(100):
                    candidates = np.flatnonzero(self._arrays['is_success'][:self.size] > 0)
                    if not len(candidates) or (not replace and len(candidates) < batch_size):
                        raise ValueError('Not enough successful transitions available')
                    idxs = self.numpy_rng.choice(candidates, size=batch_size, replace=replace)
                    try:
                        result = self._read(idxs, success_only=True)
                        break
                    except ValueError:
                        continue
                else:
                    raise RuntimeError('Success replay is changing too quickly to sample')
            else:
                size = self.size
                if size == 0:
                    raise ValueError('Cannot sample empty shared replay')
                # Exclude currently claimed slots; copying and validation remain unlocked.
                for _ in range(100):
                    idxs = self.numpy_rng.integers(size, size=batch_size)
                    with self._lock:
                        if np.all(self._versions[idxs] % 2 == 0):
                            break
                result = self._read(idxs)
        else:
            result = self._read(idxs, success_only=success_only)
        if self.p_aug is not None and result['observations'].ndim > 3:
            if self.numpy_rng.random() < self.p_aug:
                import jax
                from ogpo.utils.datasets import Dataset
                if self.rng is None:
                    self.rng = jax.random.PRNGKey(int(self.numpy_rng.integers(2**31)))
                Dataset.augment(self, result, ['observations', 'next_observations'])
        return result

    def sample_success_chunks(self, batch_size):
        return self.sample(batch_size, success_only=True, replace=True)

    def __reduce__(self):
        # Portable checkpoint, not a dependency on this live shm name/worker.
        # The runner drains/pauses the writer before taking a rolling checkpoint.
        with self._lock:
            size, pointer = int(self._meta[0]), int(self._meta[1])
        fields = {key: np.empty((size,) + array.shape[1:], array.dtype)
                  for key, array in self._arrays.items()}
        for start in range(0, size, 1024):
            block = self._read(np.arange(start, min(size, start + 1024)))
            for key in fields:
                fields[key][start:start + len(block[key])] = block[key]
        if size == 0:
            # Keep the example's shape for empty-buffer checkpoint restoration.
            from ogpo.utils.datasets import ReplayBuffer
            empty = ReplayBuffer.create({k: np.zeros(v.shape[1:], v.dtype)
                                         for k, v in self._arrays.items()}, self.max_size)
            return empty.__reduce__()
        return (_restore_snapshot, (fields, self.max_size, pointer,
                                    self.numpy_rng.bit_generator.state, self.p_aug))

    def unlink(self):
        if self._owner and not self._unlinked:
            self._segment.unlink()
            self._unlinked = True

    def close(self):
        segment = getattr(self, '_segment', None)
        if segment is not None:
            self._arrays = {}
            self._versions = None
            segment.close()
            self._segment = None

    def __del__(self):
        try:
            self.unlink()
            self.close()
        except (AttributeError, BufferError, FileNotFoundError):
            pass
