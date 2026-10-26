"""Real spawn readers/writers: no partial rows and no reader lock during data writes."""
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import pytest

from ogpo.utils.shared_replay import SharedReplay


def row(value):
    return dict(observations=np.full(8, value, np.float32),
                next_observations=np.full(8, value, np.float32),
                actions=np.full(2, value, np.float32),
                true_actions=np.full(2, value, np.float32),
                z_aug=np.full((3, 2), value, np.float32),
                rewards=np.float32(value), terminals=np.float32(0), masks=np.float32(1),
                chunk_length=np.int32(4))


def paused_writer(spec, ready, writing, release):
    replay = SharedReplay(spec, writer=True)
    try:
        replay.write_batch([row(1), row(2)], [0, 0])
        ready.set()
        original = replay._write_values
        def partial(slots, values):
            replay._arrays['observations'][slots] = values['observations']
            writing.set()
            assert release.wait(15)
            original(slots, values)
        replay._write_values = partial
        replay.write_batch([row(3)], [1])
    finally:
        replay.close()


def stress_writer(spec, ready, done):
    replay = SharedReplay(spec, writer=True)
    try:
        replay.write_batch([row(i) for i in range(32)], [0] * 32)
        ready.set()
        for i in range(32, 800):
            replay.write_batch([row(i)], [i // 20])
            if i % 20 == 19:
                replay.finish_episode(i // 20, i % 40 == 39)
            time.sleep(.0001)
    finally:
        done.set()
        replay.close()


def assert_aligned(batch):
    marker = batch['rewards']
    for key in ('observations', 'next_observations', 'actions', 'true_actions', 'z_aug'):
        np.testing.assert_array_equal(batch[key], np.broadcast_to(
            marker.reshape((-1,) + (1,) * (batch[key].ndim-1)), batch[key].shape))


def test_reads_continue_during_partial_write_and_overwritten_rows_retry():
    ctx = mp.get_context('spawn')
    replay = SharedReplay.create(row(0), 2, ctx)
    ready, writing, release = ctx.Event(), ctx.Event(), ctx.Event()
    process = ctx.Process(target=paused_writer, args=(replay.spec, ready, writing, release))
    process.start()
    try:
        assert ready.wait(15) and writing.wait(15)
        with pytest.raises(PermissionError):
            replay.add_transition(row(9))
        assert not replay['observations'].flags.writeable
        # Slot 0 contains a deliberately partial overwrite, slot 1 is still readable.
        begin = time.monotonic()
        stable = replay.sample(1, idxs=np.array([1]))
        assert time.monotonic() - begin < 1.
        assert stable['rewards'][0] == 2
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(replay.sample, 1, np.array([0]))
            time.sleep(.05)
            assert not future.done()
            release.set()
            batch = future.result(timeout=15)
        assert batch['rewards'][0] == 3
        assert_aligned(batch)
        process.join(15)
        assert process.exitcode == 0
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
            process.join()
        replay.unlink()
        replay.close()


def test_concurrent_ring_overwrite_never_returns_mixed_fields():
    ctx = mp.get_context('spawn')
    replay = SharedReplay.create(row(0), 32, ctx)
    ready, done = ctx.Event(), ctx.Event()
    process = ctx.Process(target=stress_writer, args=(replay.spec, ready, done))
    process.start()
    try:
        assert ready.wait(15)
        reads = 0
        while not done.is_set() or reads < 100:
            assert_aligned(replay.sample(16))
            reads += 1
        process.join(15)
        assert process.exitcode == 0
        assert replay.size == 32 and replay.total_written == 800
        restored = pickle.loads(pickle.dumps(replay))
        assert restored.size == replay.size and restored.pointer == replay.pointer
        assert_aligned(restored.sample(64))
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        replay.unlink()
        replay.close()


def test_success_labels_ignore_overwritten_episode_rows_and_checkpoint():
    ctx = mp.get_context('spawn')
    replay = SharedReplay.create(row(0), 3, ctx)
    writer = SharedReplay(replay.spec, writer=True)
    try:
        writer.write_batch([row(i) for i in range(5)], [0] * 5)
        writer.finish_episode(0, True)
        assert replay.success_count() == 3 and replay.success_count(physical=True) == 12
        writer.write_batch([row(5), row(6)], [1, 1])
        writer.finish_episode(1, False)
        assert replay.success_count() == 1 and replay.success_count(physical=True) == 4
        batch = replay.sample_success_chunks(20)
        assert_aligned(batch)
        assert np.all(batch['is_success'] == 1) and np.all(batch['rewards'] == 4)
        restored = pickle.loads(pickle.dumps(replay))
        assert restored._traj_success_mask[:restored.size].sum() == 1
    finally:
        writer.close()
        replay.unlink()
        replay.close()
