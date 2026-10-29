"""Adaptive Eq. (9) weights with JGCW's existing mean-Q SAC objective."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.jgcw import JGCWAgent
from ogpo.agents.dsrl import DSRLBestOfNAgent
from test_jgcw_klactor import make_agent, make_batch


def test_adaptive_weights_and_stop_gradient():
    gradients = jnp.array([[[1., 0.], [2., 0.]],
                           [[1., 0.], [-1., 0.]],
                           [[1., 0.], [0., 1.]],
                           [[0., 0.], [0., 0.]]])
    weights, scores = JGCWAgent._vmfer_weights(gradients)
    np.testing.assert_allclose(weights, np.exp([1., 0., 1 / np.sqrt(2), 0.]), rtol=1e-6)
    np.testing.assert_allclose(scores, [1., 0., 1 / np.sqrt(2), 0.], atol=1e-6)
    np.testing.assert_array_equal(jax.grad(
        lambda x: JGCWAgent._vmfer_weights(x)[0].sum())(gradients), 0.)


def test_actor_loss_and_gradient_match_detached_weighted_sac():
    agent = make_agent(use_vmfer=True, use_klactor_clip=False, lr_actor=3e-4)
    batch = jax.tree.map(lambda x: x[0], make_batch())
    key = jax.random.PRNGKey(12)
    _, sample_key = jax.random.split(key)
    dist = agent.actor_network.select('actor')(batch['observations'])
    z, _ = dist.sample_and_log_prob(seed=sample_key)
    critic = agent.critic_network.select('critic')
    # Independent full batch Jacobian: extract each sample's own derivative.
    jacobian = jax.jacrev(lambda actions: critic(batch['observations'], actions=actions))(z)
    per_sample = jnp.stack([jacobian[:, b, b, :] for b in range(z.shape[0])])
    weights, _ = agent._vmfer_weights(per_sample)

    def expected(params):
        policy = agent.actor_network.select('actor')(batch['observations'], params=params)
        actions, log_probs = policy.sample_and_log_prob(seed=sample_key)
        q = critic(batch['observations'], actions=actions).mean(axis=0)
        temp = agent.actor_network.select('temp')()
        sac = jnp.mean(weights * (temp * log_probs - q)) / weights.mean()
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        return sac + agent.actor_network.select('temp')(params=params) * (
            entropy - agent.config['target_entropy'])

    value, info = agent.actor_loss(batch, agent.actor_network.params, key)
    np.testing.assert_allclose(value, expected(agent.actor_network.params), rtol=1e-6)
    actual_grad = jax.grad(lambda p: agent.actor_loss(batch, p, key)[0])(agent.actor_network.params)
    expected_grad = jax.grad(expected)(agent.actor_network.params)
    for actual, wanted in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
        np.testing.assert_allclose(actual, wanted, rtol=2e-5, atol=2e-6)
    assert float(info['vmfer_weight_min']) > 0
    # Flag off exactly preserves the existing SAC objective.
    disabled = agent.replace(config=dict(agent.config, use_vmfer=False))
    actual, _ = disabled.actor_loss(batch, disabled.actor_network.params, key)
    wanted, _ = DSRLBestOfNAgent.actor_loss(disabled, batch, disabled.actor_network.params, key)
    np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize('ratio_clip,q_first', [(False, False), (False, True), (True, False)])
def test_training_updates_with_success_bc(ratio_clip, q_first):
    agent = make_agent(use_vmfer=True, use_klactor_clip=ratio_clip,
                       first_train_q=q_first, lr_actor=3e-4)
    batch = make_batch()
    updated, info = agent.batch_update(batch, batch, True)
    assert 'actor/vmfer_weight_mean' in info
    assert np.isfinite(float(info['actor/actor_loss']))
    assert int(updated.critic_network.step) > int(agent.critic_network.step)
    assert int(updated.actor_network.step) > int(agent.actor_network.step)
    assert float(info['actor_bc_loss']) >= 0
    if ratio_clip:
        assert int(updated.temperature_network.step) > int(agent.temperature_network.step)
