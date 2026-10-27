"""OGPO collection and offline-success seeding for GRACE."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.grace import GRACEAgent
from ogpo.agents.jgcw import JGCWAgent
from ogpo.runners.grace_runner import (
    _build_ogpo_success_dataset, _sample_ogpo_success_batch, run_grace_online_rl,
)
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.datasets import Dataset, ReplayBuffer


def make_agent(**overrides):
    config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  horizon_length=2, flow_steps=2, ogpo_num_samples=4,
                  inverse_fm_batch_size=4, updates_per_env_step=False)
    config.update(overrides)
    return GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)


def source():
    data = Dataset.create(
        observations=np.arange(18, dtype=np.float32).reshape(6, 3) / 10,
        next_observations=np.arange(18, dtype=np.float32).reshape(6, 3) / 10 + .1,
        actions=np.arange(12, dtype=np.float32).reshape(6, 2) / 30,
        rewards=np.array([-1, 0, 0, -1, -1, -1], np.float32),
        masks=np.ones(6, np.float32), terminals=np.array([0, 0, 0, 0, 0, 1], np.float32))
    data.demo_ends = np.array([0, 0, 1, 0, 0, 1], np.float32)
    return data


def test_action_q_selection_keeps_winning_latent_and_action():
    # Action scores deliberately prefer a different z than a latent-Q score would.
    dummy = SimpleNamespace(config={'ogpo_num_samples': 8}, rng=jax.random.PRNGKey(1))
    dummy.sample_noise = lambda obs, key: jax.random.normal(key, (*obs.shape[:-1], 2))
    dummy.noise_to_actions_public = lambda obs, z, **kw: -z
    dummy.action_critic_network = SimpleNamespace(select=lambda name:
        lambda obs, actions: jnp.stack((actions[..., 0], actions[..., 0] + 1)))
    key = jax.random.PRNGKey(7)
    for obs in (jnp.zeros(3), jnp.zeros((3, 3))):
        candidates = jax.vmap(lambda k: dummy.sample_noise(obs, k))(jax.random.split(key, 8))
        winner = jnp.argmin(candidates[..., 0], axis=0)
        expected = jnp.take_along_axis(candidates, winner[None, ..., None], axis=0)[0]
        z, a = GRACEAgent.sample_ogpo_action.__wrapped__(dummy, obs, key)
        np.testing.assert_array_equal(z, expected)
        np.testing.assert_array_equal(a, -expected)


def test_compiled_bon_and_jgcw_inverse():
    agent = make_agent()
    key = jax.random.PRNGKey(17)
    for obs in (jnp.zeros(3), jnp.zeros((2, 3))):
        z, a = agent.sample_ogpo_action(obs, key)
        np.testing.assert_allclose(a, agent.noise_to_actions_public(
            obs, z, noises_constrained=True), atol=1e-6)
        np.testing.assert_allclose(agent.sample_actions(obs, key), a, atol=1e-6)
        keys = jax.random.split(key, agent.config['ogpo_num_samples'])
        proposals = jax.vmap(lambda k: agent.noise_to_actions_public(
            obs, agent.sample_noise(obs, k), noises_constrained=True))(keys)
        scores = jax.vmap(lambda x: jnp.min(agent.action_critic_network.select(
            'target_critic')(obs, actions=x), axis=0))(proposals)
        actual = jnp.min(agent.action_critic_network.select('target_critic')(obs, actions=a), axis=0)
        np.testing.assert_allclose(actual, scores.max(axis=0), atol=1e-6)
    obs, target = jnp.zeros((2, 3)), jnp.full((2, 4), .2)
    z = agent.inverse_ogpo_batch(obs, target)
    np.testing.assert_allclose(z, JGCWAgent.inverse_fm_batch(agent, obs, target), atol=1e-6)
    initial_error = jnp.square(agent.noise_to_actions_public(obs, jnp.zeros_like(z)) - target).sum(-1)
    final_error = jnp.square(agent.noise_to_actions_public(obs, z) - target).sum(-1)
    assert np.all(final_error <= initial_error + 1e-6)


def test_all_offline_rows_and_boundaries_use_jgcw_not_grace_inverse():
    agent, data = make_agent(), source()
    with patch.object(GRACEAgent, 'inverse_fm_batch', side_effect=AssertionError('GRACE inverse used')):
        pool = _build_ogpo_success_dataset(agent, data, agent.config, 1)
    assert pool.size == data.size  # Includes post-success rows and failed episodes.
    np.testing.assert_array_equal(pool['observations'], data['observations'])
    np.testing.assert_array_equal(pool['chunk_length'], [2, 2, 1, 2, 2, 1])
    np.testing.assert_array_equal(pool['true_actions'][2], np.tile(data['actions'][2], 2))
    assert np.isfinite(pool['actions']).all()
    example = {key: value[0] for key, value in pool.items()}
    replay = ReplayBuffer.create(example, 10)
    offline_only = _sample_ogpo_success_batch(replay, pool, 32)
    assert set(offline_only) == set(replay)
    np.testing.assert_array_equal(offline_only['is_success'], 1.)
    for success, marker in ((0., -100.), (1., 100.)):
        row = dict(example, observations=np.full(3, marker), is_success=success, score=success or 1e-9)
        replay.add_transition(row)
    mixed = _sample_ogpo_success_batch(replay, pool, 512)
    assert np.any(mixed['observations'][:, 0] == 100.)
    assert np.any(mixed['observations'][:, 0] < 100.)
    assert not np.any(mixed['observations'][:, 0] == -100.)
    for key in replay:
        assert mixed[key].dtype == replay[key].dtype


def test_online_executes_and_stores_winner_and_trains_without_online_success():
    agent = make_agent()
    class Env:
        def __init__(self):
            self.executed = []
        def reset(self, seed):
            self.steps = 0
            return np.zeros(3, np.float32), {}
        def step(self, a):
            self.executed.append(a)
            self.steps += 1
            return np.zeros(3, np.float32), -1., self.steps == 2, False, {'success': 0.}
    env = Env()
    logger = SimpleNamespace(log=lambda *args, **kwargs: None)
    updates = []
    original_update = GRACEAgent.batch_update
    def update(current, batch, batch_success=None, success_flag=False):
        updates.append((batch_success, success_flag))
        return original_update(current, batch, batch_success, success_flag)
    with patch('ogpo.runners.grace_runner.save_agent'), patch(
            'ogpo.runners.grace_runner._run_evaluation', return_value={}), patch.object(
                GRACEAgent, 'batch_update', update):
        trained, _, _, replay = run_grace_online_rl(
            agent, env, env, None, source(), dict(agent.config), logger, '/unused',
            algo_name='grace', discount=.99, online_steps=4, start_training=0, batch_size=2,
            success_buffer_batch_size=2, horizon_length=2, buffer_size=32,
            n_eval_envs=1, eval_interval=-1, log_interval=-1, save_interval=-1,
            force_jax_sync=False)
    assert updates and all(flag for _, flag in updates)
    assert not replay._traj_success_mask[:replay.size].any()
    online_rng = jax.random.PRNGKey(40000)
    for row in range(replay.size):
        online_rng, key = jax.random.split(online_rng)
        z, a = agent.sample_ogpo_action(jnp.zeros(3), key)
        np.testing.assert_allclose(replay['actions'][row], z, atol=1e-6)
        np.testing.assert_allclose(replay['true_actions'][row], a, atol=1e-6)
    np.testing.assert_allclose(np.asarray(env.executed).reshape(-1, 4), replay['true_actions'][:replay.size])
    for before, after in zip(jax.tree.leaves(agent.bc_flow_network.params),
                             jax.tree.leaves(trained.bc_flow_network.params)):
        np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(x, y) for x, y in zip(
        jax.tree.leaves(agent.actor_network.params), jax.tree.leaves(trained.actor_network.params)))


@pytest.mark.parametrize('overrides,match', [
    ({'ogpo_num_samples': 0}, 'ogpo_num_samples'),
    ({'inverse_fm_radius': float('nan')}, 'inverse_fm_radius'),
    ({'encoder_sphere_limit': True}, 'encoder_sphere_limit'),
    ({'use_layer_norm': True, 'use_no_tanh': True, 'use_offline_iql': True}, 'offline IQL'),
    ({'use_layer_norm': True, 'use_no_tanh': True, 'use_demo_buffer': True}, 'demo-buffer'),
])
def test_invalid_combinations(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_agent(**overrides)
