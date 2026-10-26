"""Spawned frozen-decoder worker: scheduling independence, errors, drain and cleanup."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import multiprocessing as mp
from multiprocessing import shared_memory
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.jgcw import JGCWAgent
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.jgcw_augmentation import AsyncAugmentationReplay


@pytest.fixture(scope='module')
def agent():
    config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  train_q_sa=False, use_aug_z=True, aug_z_num=2, flow_steps=2,
                  inverse_fm_iterations=2, aug_worker_device='cpu', aug_worker_timeout=60)
    return JGCWAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)


def transition(i):
    return dict(observations=np.full(3, i/10, np.float32),
                next_observations=np.full(3, (i+1)/10, np.float32),
                actions=np.full(8, i/100, np.float32),
                true_actions=np.full(8, i/10, np.float32),
                rewards=np.float32(-1), terminals=np.float32(i == 5),
                masks=np.float32(i != 5), chunk_length=np.int32(4))


def example():
    return dict(transition(0), z_aug=np.zeros((2, 8), np.float32))


def test_worker_is_only_writer_and_rng_does_not_depend_on_batching(agent):
    outputs = []
    for size in (1, 4):
        worker = AsyncAugmentationReplay(agent, example(), 5, 3,
            dict(agent.config, aug_worker_batch_size=size, aug_worker_queue_size=2))
        name = worker.replay.spec['name']
        try:
            for i in range(6):
                worker.submit(transition(i), 0, i)
            worker.finish_episode(0, True)
            worker.flush()
            assert worker.process.pid != os.getpid()
            assert worker.replay._meta[5] == worker.process.pid
            assert worker.replay.total_written == 6 and worker.replay.size == 5
            assert worker.replay.success_count(physical=True) == 20
            with pytest.raises(PermissionError):
                worker.replay.add_transition(dict(transition(7), z_aug=np.zeros((2, 8))))
            sampled = worker.replay.sample(5, idxs=np.arange(5))
            order = np.argsort(sampled['observations'][:, 0])
            outputs.append(sampled['z_aug'][order])
            assert np.isfinite(outputs[-1]).all()
        finally:
            worker.close()
        assert not worker.process.is_alive()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
        # Reader views remain usable after normal worker close and OS unlink.
        assert np.isfinite(worker.replay.sample(2)['z_aug']).all()
        worker.replay.close()
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)


def test_worker_error_reaches_parent_and_cleans_up(agent):
    worker = AsyncAugmentationReplay(agent, example(), 5, 3, dict(agent.config))
    try:
        invalid = transition(0)
        invalid['true_actions'][0] = np.nan
        worker.submit(invalid, 0, 0)
        with pytest.raises(RuntimeError, match='Non-finite source data'):
            worker.flush()
        assert worker.replay.size == 0
    finally:
        worker.close()
        worker.replay.close()
    assert not worker.process.is_alive()


def test_invalid_worker_configuration(agent):
    for config in (dict(aug_worker_batch_size=0), dict(aug_worker_queue_size=True),
                   dict(aug_worker_device='invalid')):
        with pytest.raises(ValueError):
            AsyncAugmentationReplay(agent, example(), 5, 3, dict(agent.config, **config))


def test_runner_preserves_physical_update_budget_and_keeps_partial_episode(agent, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from ogpo.runners.jgcw_runner import run_jgcw_online_rl
    from ogpo.utils.datasets import Dataset
    class Env:
        def reset(self, seed):
            self.t = 0
            return np.zeros(3, np.float32), {}
        def step(self, action):
            self.t += 1
            return np.full(3, self.t/10, np.float32), -1., self.t == 9, False, {
                'success': float(self.t == 9)}
    source = Dataset.create(observations=np.zeros((10, 3), np.float32),
                            next_observations=np.zeros((10, 3), np.float32),
                            actions=np.zeros((10, 2), np.float32),
                            rewards=np.zeros(10), terminals=np.zeros(10), masks=np.ones(10))
    config = dict(agent.config, updates_per_env_step=True)
    logs = []
    with patch('ogpo.runners.jgcw_runner.save_agent'), patch(
            'ogpo.runners.jgcw_runner._run_evaluation', return_value={}):
        updated, step, _, replay = run_jgcw_online_rl(
            agent, Env(), None, None, source, config,
            SimpleNamespace(log=lambda values, key, **kwargs: logs.append((key, values))),
            str(tmp_path), 'jgcw', online_steps=21, start_training=12, batch_size=1,
            horizon_length=4, discount=.99, n_eval_envs=1, log_interval=-1,
            eval_interval=-1, save_interval=-1, buffer_size=32,
            use_success_buffer=True, success_buffer_batch_size=1)
    try:
        assert step == 21 and replay.total_written == 7
        assert int(updated.actor_network.step) - int(agent.actor_network.step) == 9
        assert int(updated.critic_network.step) - int(agent.critic_network.step) == 18
        assert replay['chunk_length'][:replay.size].sum() == 21
        assert replay.success_count(physical=True) == 18
        assert not any(child.name == 'jgcw-augmentation-writer' for child in mp.active_children())
        final_budget = [values for key, values in logs if key == 'online_budget'][-1]
        assert final_budget['gradient_updates'] == 9
        assert final_budget['decoder_updates'] == 0
    finally:
        replay.close()
