"""FASTER's winner-only latent-Q regression inside GRACE."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.grace import GRACEAgent
from ogpo.utils.config_loader import load_config, config_to_agent_dict


@pytest.fixture(scope='module')
def agent():
    config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  use_layer_norm=False, encoder_sphere_limit=False,
                  use_encoder_clip=False, use_offline_iql=False, use_demo_buffer=False,
                  use_grrl_distill=False, use_faster_distill=True,
                  faster_distill_num_samples=8, flow_steps=2)
    return GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)


def data():
    key = jax.random.PRNGKey(11)
    obs = jax.random.normal(key, (5, 3))
    return dict(observations=obs, next_observations=obs + .1,
                actions=jnp.zeros((5, 8)), true_actions=jnp.zeros((5, 8)),
                rewards=-jnp.ones(5), masks=jnp.ones(5), chunk_length=jnp.full(5, 4))


def equal_tree(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)


def test_selects_a_winner_per_state_using_latent_q():
    obs = jnp.array([[1.], [-1.], [0.]])  # Max, min, and deterministic tie.
    candidates = jax.random.normal(jax.random.PRNGKey(3), (3, 8, 4))
    network = SimpleNamespace(select=lambda name: lambda obs, actions: jnp.stack(
        [obs[:, 0] * actions[:, 0], obs[:, 0] * actions[:, 0] + 1.]))
    dummy = SimpleNamespace(config={'faster_distill_num_samples': 8},
                            critic_network=network, _aggregate_q=lambda q: q.mean(0))
    actual, info = GRACEAgent._select_faster_distill_noise(
        dummy, obs, jnp.zeros((3, 4)), jax.random.PRNGKey(3))
    indices = jnp.array([jnp.argmax(candidates[0, :, 0]),
                         jnp.argmin(candidates[1, :, 0]), 0])
    np.testing.assert_array_equal(actual, candidates[jnp.arange(3), indices])
    assert info['faster_selected_q_mean'] >= info['faster_candidate_q_mean']


def test_winner_only_mse_gradient_and_decoder_batch(agent):
    batch, key = data(), jax.random.PRNGKey(21)
    candidates = jax.random.normal(key, (5, 8, 8))
    score = agent._aggregate_q(agent.critic_network.select('critic')(
        jnp.repeat(batch['observations'], 8, axis=0), actions=candidates.reshape(-1, 8)))
    selected = candidates[jnp.arange(5), jnp.argmax(score.reshape(5, 8), axis=-1)]
    actions = agent.noise_to_actions_public(batch['observations'], selected, noises_constrained=True)
    target = jax.lax.stop_gradient(agent._aggregate_q(agent.action_critic_network.select('critic')(
        batch['observations'], actions=actions)))
    def expected_loss(p):
        predicted = agent.critic_network.select('critic')(
            batch['observations'], actions=selected, params=p)
        return jnp.mean((predicted - target) ** 2)

    calls = []
    original = GRACEAgent.noise_to_actions_public
    def decode(current, observations, noises, **kwargs):
        calls.append((observations.shape, np.asarray(noises)))
        return original(current, observations, noises, **kwargs)
    with patch.object(GRACEAgent, 'noise_to_actions_public', decode):
        loss, info = agent.critic_loss(batch, agent.critic_network.params, key)
    assert len(calls) == 1
    assert calls[0][0] == (5, 3)  # Only B winners, never B*N candidates.
    np.testing.assert_array_equal(calls[0][1], selected)
    np.testing.assert_allclose(loss, expected_loss(agent.critic_network.params), rtol=1e-6)
    equal_tree(jax.grad(lambda p: agent.critic_loss(batch, p, key)[0])(agent.critic_network.params),
               jax.grad(expected_loss)(agent.critic_network.params))
    assert info['faster_selected_q_mean'] >= info['faster_candidate_q_mean']
    for field in ('action_critic_network', 'bc_flow_network', 'actor_network'):
        state = getattr(agent, field)
        def external_loss(p):
            current = agent.replace(**{field: state.replace(params=p)})
            return current.critic_loss(batch, agent.critic_network.params, key)[0]
        for leaf in jax.tree.leaves(jax.grad(external_loss)(state.params)):
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_single_candidate_and_disabled_compatibility(agent):
    batch, key = data(), jax.random.PRNGKey(13)
    single = agent.replace(config=agent.config.copy(dict(faster_distill_num_samples=1)))
    disabled = agent.replace(config=agent.config.copy(dict(use_faster_distill=False)))
    legacy_config = dict(disabled.config)
    legacy_config.pop('use_faster_distill')
    legacy_config.pop('faster_distill_num_samples')
    legacy = disabled.replace(config=legacy_config)
    expected = legacy.critic_loss(batch, legacy.critic_network.params, key)
    equal_tree(expected, disabled.critic_loss(batch, disabled.critic_network.params, key))
    np.testing.assert_allclose(single.critic_loss(batch, single.critic_network.params, key)[0],
                               expected[0], rtol=1e-6)


def test_updates_include_extra_success_step_and_freeze_decoder(agent):
    batch = jax.tree.map(lambda x: x[None], data())
    for success in (False, True):
        updated, metrics = agent.batch_update(batch, batch if success else None, success)
        assert int(updated.critic_network.step) == int(agent.critic_network.step) + 2
        assert np.isfinite(metrics['critic/distillation_loss'])
        assert 'critic/faster_selected_q_mean' in metrics
        equal_tree(updated.bc_flow_network, agent.bc_flow_network)


def test_configuration_validation(agent):
    for overrides, message in ((dict(use_grrl_distill=True), 'mutually exclusive'),
                               (dict(faster_distill_num_samples=0), 'positive integer'),
                               (dict(faster_distill_num_samples=True), 'positive integer'),
                               (dict(faster_distill_num_samples=1.5), 'positive integer')):
        with pytest.raises(ValueError, match=message):
            GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), dict(agent.config, **overrides))
    assert load_config('grace_ogpo_square_shared')['agent']['use_faster_distill'] is False
