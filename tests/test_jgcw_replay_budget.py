"""Long chunks must not require distinct replay rows for every UTD update."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ogpo.runners.jgcw_runner import run_jgcw_online_rl
from ogpo.utils.config_loader import load_config
from ogpo.utils.datasets import Dataset


@pytest.mark.parametrize('task,first_episode_length,first_update', [
    ('square', 1000, 10004),
    ('toolhang', 1000, 10008),
    ('transport', 997, 10005),
])
def test_first_online_updates_preserve_budget(task, first_episode_length, first_update, tmp_path):
    cfg = load_config(f'jgcw_ogpo_{task}_shared')
    config = cfg['agent']
    horizon = config['horizon_length']
    batch_size = config['batch_size']

    class Env:
        total = 0
        episode = 0

        def reset(self, seed):
            self.episode += 1
            self.t = 0
            return np.zeros(3, np.float32), {}

        def step(self, action):
            self.total += 1
            self.t += 1
            length = first_episode_length if self.episode == 1 else 1000
            return np.zeros(3, np.float32), -1., self.t == length, False, {'success': 1.}

    env = Env()
    calls = []

    class Agent:
        def sample_noise(self, observations, rng):
            return np.zeros(horizon * 2, np.float32)

        def noise_to_actions_public(self, observations, noise):
            return noise

        def sample_actions(self, **kwargs):
            return np.zeros(horizon * 2, np.float32)

        def batch_update(self, batch, success_batch, success_flag):
            updates = batch['actions'].shape[0]
            assert batch['actions'].shape == (updates, batch_size, horizon * 2)
            assert success_flag
            assert success_batch['actions'].shape == batch['actions'].shape
            calls.append((env.total, updates))
            return self, {}

    dataset = Dataset.create(
        observations=np.zeros((1, 3), np.float32),
        actions=np.zeros((1, 2), np.float32),
        next_observations=np.zeros((1, 3), np.float32),
        rewards=np.zeros(1), terminals=np.zeros(1), masks=np.ones(1))
    records = []
    logger = SimpleNamespace(log=lambda metrics, prefix, **kw: records.append((prefix, metrics)))
    with patch('ogpo.runners.jgcw_runner.save_agent'), patch(
            'ogpo.runners.jgcw_runner._run_evaluation', return_value={}):
        _, step, _, replay = run_jgcw_online_rl(
            Agent(), env, None, None, dataset, config, logger, str(tmp_path), 'jgcw',
            online_steps=10016, start_training=10000, batch_size=batch_size,
            horizon_length=horizon, discount=config['discount'],
            utd_online=cfg['buffer']['utd_online'], n_eval_envs=1,
            use_success_buffer=True, success_buffer_batch_size=batch_size,
            buffer_size=3000, eval_interval=-1, log_interval=-1, save_interval=-1,
            force_jax_sync=False)

    assert step == 10016
    assert calls[0] == (first_update, first_update - 10000)
    assert sum(updates for _, updates in calls) == 16
    if horizon == 8:
        # Reproduce both failures: 1250 stored rows, 2048/1280 requested samples.
        assert replay.size == 1250
        assert replay.size < calls[0][1] * batch_size
    budget = [metrics for prefix, metrics in records if prefix == 'online_budget'][-1]
    assert budget['gradient_updates'] == 16
    assert budget['critic_gradient_updates'] == 16 * config['critic_updates_per_actor']
    assert budget['action_critic_gradient_updates'] == budget['critic_gradient_updates']
