import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.grace import GRACEAgent, demo_buffer_enabled
from ogpo.runners.grace_iql import build_latent_iql_dataset
from ogpo.runners.grace_runner import run_grace_online_rl, _sample_demo_mixed_batch
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.datasets import Dataset


def config():
    return dict(config_to_agent_dict(load_config('grace_ogpo_square_shared')),
                actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), use_demo_buffer=True,
                use_layer_norm=True, use_no_tanh=True, use_offline_iql=False)


def source():
    return Dataset.create(observations=np.arange(30, dtype=np.float32).reshape(10, 3),
        next_observations=np.arange(30, dtype=np.float32).reshape(10, 3) + 1,
        actions=np.full((10, 2), .2, np.float32),
        rewards=np.array([-1, -1, 0, 0, 0, -1, -1, -1, -1, -1], np.float32),
        masks=np.zeros(10, np.float32),
        terminals=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], np.float32))


def test_demo_gate_and_conflicting_iql():
    for norm, no_tanh, demo in product((False, True), repeat=3):
        assert demo_buffer_enabled(dict(use_layer_norm=norm, use_no_tanh=no_tanh,
                                       use_demo_buffer=demo)) == (norm and no_tanh and demo)
    assert not demo_buffer_enabled(dict(use_layer_norm=True, use_no_tanh=True))
    with pytest.raises(ValueError, match='incompatible'):
        GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), dict(config(), use_offline_iql=True))
    with pytest.raises(ValueError, match='even'):
        _sample_demo_mixed_batch(None, None, 3, 2)


@pytest.mark.parametrize('norm,no_tanh,demo', [(False, False, False),
                                             (True, True, False), (True, True, True)])
def test_demo_actor_is_sac_and_distillation_is_gaussian_only(norm, no_tanh, demo):
    agent = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2),
        dict(config(), use_layer_norm=norm, use_no_tanh=no_tanh, use_demo_buffer=demo))
    data = build_latent_iql_dataset(agent, source(), agent.config, 0, 4).sample(8)
    key = jax.random.PRNGKey(51)
    gaussian = jax.random.normal(key, data['actions'].shape, dtype=data['actions'].dtype)
    decoded = agent.noise_to_actions_public(data['observations'], gaussian, noises_constrained=True)
    target = agent._aggregate_q(agent.action_critic_network.select('critic')(
        data['observations'], actions=decoded))
    q = agent.critic_network.select('critic')(data['observations'], actions=gaussian)
    loss, info = agent.critic_loss(data, agent.critic_network.params, key)
    np.testing.assert_allclose(loss, jnp.mean((q - target) ** 2), rtol=1e-6)
    assert 'actor_distillation_count' not in info
    changed = dict(data, actions=np.full_like(data['actions'], 1000.))
    # Neither the SAC objective nor Gaussian distillation uses cached demo z as
    # a behavior-cloning target or a substitute for freshly sampled actor z.
    np.testing.assert_allclose(agent.actor_loss(data, agent.actor_network.params, key)[0],
                               agent.actor_loss(changed, agent.actor_network.params, key)[0])
    np.testing.assert_allclose(loss, agent.critic_loss(changed, agent.critic_network.params, key)[0])
    scanned = jax.tree.map(lambda x: x[None], data)
    updated, metrics = agent.batch_update(scanned)
    assert 'critic/distillation_loss' in metrics and 'action_critic/loss' in metrics
    assert int(updated.critic_network.step) == int(agent.critic_network.step) + 2
    assert int(updated.action_critic_network.step) == int(agent.action_critic_network.step) + 2
    for x, y in zip(jax.tree.leaves(agent.bc_flow_network), jax.tree.leaves(updated.bc_flow_network)):
        np.testing.assert_array_equal(x, y)
    assert updated.value_network is None


def test_runner_builds_demo_before_online_and_mixes_every_utd_batch():
    events, batches = [], []
    class Env:
        max_episode_length = 4
        def reset(self, seed):
            events.append('reset')
            self.steps = 0
            return np.full(3, 1000., np.float32), {}
        def step(self, action):
            self.steps += 1
            return np.full(3, 1000. + self.steps, np.float32), -1., False, self.steps == 5, {'success': 0.}
    class Agent:
        def inverse_fm_batch(self, obs, actions):
            events.append('inverse')
            return actions + 100.
        def sample_noise(self, **kwargs):
            return np.zeros(8)
        def noise_to_actions_public(self, obs, noise):
            return np.zeros(8)
        def sample_actions(self, **kwargs):
            return np.zeros(8)
        def batch_update(self, batch, success_batch, success_flag):
            assert success_batch is None and not success_flag
            batches.append(batch)
            return self, {'critic/distillation_loss': 0.}
    raw = source()
    rewards = raw['rewards'].copy()
    cfg = config()
    env = Env()
    with patch('ogpo.runners.grace_runner.save_agent'), \
         patch('ogpo.runners.grace_runner._run_evaluation', return_value={'success': 0.}), \
         patch('ogpo.runners.grace_iql.pretrain_latent_iql', side_effect=AssertionError('IQL ran')):
        run_grace_online_rl(Agent(), env, env, None, raw, cfg,
            SimpleNamespace(log=lambda *args, **kwargs: None), '/unused', 'grace',
            online_steps=127, start_training=100, batch_size=4, horizon_length=4,
            discount=.99, utd_online=2, n_eval_envs=1, use_success_buffer=True,
            buffer_size=100, force_jax_sync=False, save_interval=-1,
            log_interval=-1, eval_interval=-1)
    assert events[0] == 'inverse' and batches
    assert sum(b['observations'].shape[0] for b in batches) == 56
    for b in batches:
        assert b['observations'].shape[1] == 4
        assert np.all(b['observations'][:, :2] >= 1000.)
        assert np.all(b['observations'][:, 2:] < 1000.)
        np.testing.assert_allclose(b['actions'][:, 2:], 100.2)
        np.testing.assert_allclose(b['true_actions'][:, 2:], .2)
        assert 'episode_id' not in b
    np.testing.assert_array_equal(raw['rewards'], rewards)
