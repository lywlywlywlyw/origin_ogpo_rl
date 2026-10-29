"""Sampled-ratio actor backtracking under the legacy use_klactor_clip flag."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import unittest

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.jgcw import JGCWAgent
from ogpo.utils.config_loader import config_to_agent_dict, load_config


def make_agent(**overrides):
    config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
    config.update(use_vmfer=False, actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  train_q_sa=True, use_qa_tdtarget=True, use_aug_z=False,
                  use_klactor_clip=True, optimizer_actor='adam', lr_actor=.5)
    config.update(overrides)
    return JGCWAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)


def make_batch():
    rng = np.random.default_rng(9)
    return dict(observations=jnp.asarray(rng.normal(size=(2, 3, 3)), dtype=jnp.float32),
                next_observations=jnp.asarray(rng.normal(size=(2, 3, 3)), dtype=jnp.float32),
                actions=jnp.asarray(rng.normal(size=(2, 3, 8)), dtype=jnp.float32),
                true_actions=jnp.asarray(rng.uniform(-1, 1, size=(2, 3, 8)), dtype=jnp.float32),
                rewards=-jnp.ones((2, 3)), masks=jnp.ones((2, 3)),
                chunk_length=jnp.full((2, 3), 4))


def assert_tree_equal(test, left, right):
    test.assertEqual(jax.tree.structure(left), jax.tree.structure(right))
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(a, b)


def proposal_loss(agent, batch, gradient_scale=1., samples=None):
    base = agent._actor_base_distribution(
        batch['observations'], agent.actor_network.params['modules_actor'])
    if samples is None:
        samples = base.sample(seed=jax.random.PRNGKey(17))
    old_log_probs = base.log_prob(samples)

    def loss(params):
        return -gradient_scale * params['modules_actor']['mean_net']['bias'].sum(), {
            'actor/entropy': jnp.float32(0.),
            'actor/_clip_samples': samples, 'actor/_clip_log_probs': old_log_probs}
    return loss


@jax.jit
def bias_step(agent, batch, gradient_scale=1.):
    agent, actor, info = agent._apply_actor_loss(
        proposal_loss(agent, batch, gradient_scale), batch)
    return agent.replace(actor_network=actor), info


class RatioActorTests(unittest.TestCase):
    def test_check_samples_match_sac_draw_and_joint_transformed_ratio(self):
        agent = make_agent()
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(123)
        _, info = agent.actor_total_loss(batch, agent.actor_network.params, key)
        samples = info['actor/_clip_samples']
        old_log_probs = info['actor/_clip_log_probs']
        old = agent.actor_network.select('actor')(batch['observations'])
        _, sample_rng = jax.random.split(key)
        z, sac_log_probs = old.sample_and_log_prob(seed=sample_rng)
        np.testing.assert_array_equal(z, old.bijector.forward(samples))
        np.testing.assert_allclose(old_log_probs, old.distribution.log_prob(samples), atol=1e-6)
        np.testing.assert_allclose(info['actor/entropy'], -sac_log_probs.mean(), atol=1e-6)
        params = dict(agent.actor_network.params)
        actor_params = dict(params['modules_actor'])
        actor_params['mean_net'] = dict(actor_params['mean_net'], bias=jnp.full(8, .003))
        params['modules_actor'] = actor_params
        new = agent.actor_network.select('actor')(batch['observations'], params=params)
        log_ratio = new.distribution.log_prob(samples) - old_log_probs
        np.testing.assert_allclose(log_ratio, new.log_prob(z) - old.log_prob(z), atol=5e-6)
        self.assertEqual(log_ratio.shape, (3,))

    def test_full_accept_backtrack_and_rollback_keep_temperature_independent(self):
        batch = jax.tree.map(lambda x: x[0], make_batch())
        initial = make_agent()
        results = []
        for multiplier, backtracks, expected in ((0., 10, 'full'),
                                                (1., 10, 'scaled'),
                                                (1., 0, 'rejected')):
            agent = initial.replace(config=dict(initial.config,
                actor_ratio_clip_max_backtracks=backtracks))
            proposed, _ = agent.actor_network.apply_loss_fn(proposal_loss(agent, batch, multiplier))
            updated, info = bias_step(agent, batch, multiplier)
            results.append(updated)
            self.assertGreaterEqual(float(info['actor/ratio_after_min']), .99 - 1e-7)
            self.assertLessEqual(float(info['actor/ratio_after_max']), 1.01 + 1e-7)
            self.assertEqual(float(info['actor/ratio_after_outside_fraction']), 0.)
            self.assertEqual(int(updated.temperature_network.step), 2)
            self.assertNotEqual(float(updated.actor_network.select('temp')()),
                                float(initial.actor_network.select('temp')()))
            self.assertFalse(any('/kl_' in key or '_clip_samples' in key for key in info))
            old = initial.actor_network.params['modules_actor']
            new = updated.actor_network.params['modules_actor']
            scale = float(info['actor/update_scale'])
            if expected == 'rejected':
                self.assertEqual(scale, 0.)
                self.assertEqual(float(info['actor/rejection_rate']), 1.)
                self.assertEqual(float(info['actor/ratio_after_min']), 1.)
                assert_tree_equal(self, new, old)
                assert_tree_equal(self, updated.actor_network.opt_state,
                                  initial.actor_network.opt_state)
                self.assertEqual(int(updated.actor_network.step), int(initial.actor_network.step))
            else:
                self.assertEqual(int(updated.actor_network.step), int(initial.actor_network.step) + 1)
                assert_tree_equal(self, updated.actor_network.opt_state, proposed.opt_state)
                if expected == 'full':
                    self.assertEqual(scale, 1.)
                else:
                    self.assertGreater(scale, 0.)
                    self.assertLess(scale, 1.)
                    self.assertGreater(float(info['actor/ratio_before_outside_fraction']), 0.)
                wanted = jax.tree.map(lambda a, b: a + scale * (b - a),
                                      old, proposed.params['modules_actor'])
                for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(wanted)):
                    np.testing.assert_allclose(a, b, atol=1e-7)
                old_dist = initial._actor_base_distribution(batch['observations'], old)
                samples = old_dist.sample(seed=jax.random.PRNGKey(17))
                new_dist = updated._actor_base_distribution(batch['observations'], new)
                ratios = jnp.exp(new_dist.log_prob(samples) - old_dist.log_prob(samples))
                self.assertTrue(np.all(np.asarray(ratios) >= .99 - 1e-6))
                self.assertTrue(np.all(np.asarray(ratios) <= 1.01 + 1e-6))
        for result in results[1:]:
            assert_tree_equal(self, result.temperature_network, results[0].temperature_network)

    def test_any_violating_sample_rejects_even_when_mean_ratio_is_inside(self):
        agent = make_agent(lr_actor=.02, actor_ratio_clip_max_backtracks=0)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        batch = dict(batch, observations=jnp.zeros_like(batch['observations']))
        samples = jnp.array([[1.] * 8, [-1.] * 8, [0.] * 8])
        loss_fn = proposal_loss(agent, batch, samples=samples)
        _, _, info = jax.jit(lambda a: a._apply_actor_loss(loss_fn, batch))(agent)
        self.assertGreater(float(info['actor/ratio_before_mean']), .99)
        self.assertLess(float(info['actor/ratio_before_mean']), 1.01)
        self.assertGreater(float(info['actor/ratio_before_outside_fraction']), 0.)
        self.assertEqual(float(info['actor/update_rejected']), 1.)

    def test_nonfinite_proposal_is_rejected(self):
        agent = make_agent(actor_ratio_clip_max_backtracks=2)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        _, actor, info = jax.jit(lambda a: a._apply_actor_loss(
            proposal_loss(a, batch, jnp.nan), batch))(agent)
        self.assertEqual(float(info['actor/update_rejected']), 1.)
        assert_tree_equal(self, actor.params['modules_actor'], agent.actor_network.params['modules_actor'])
        assert_tree_equal(self, actor.opt_state, agent.actor_network.opt_state)
        self.assertTrue(np.isfinite(actor.select('temp')()))

    def test_saturated_tanh_and_checkpoint(self):
        initial = make_agent()
        batch = jax.tree.map(lambda x: x[0], make_batch())
        params = dict(initial.actor_network.params)
        actor_params = dict(params['modules_actor'])
        actor_params['mean_net'] = dict(actor_params['mean_net'], bias=jnp.full(8, 100.))
        params['modules_actor'] = actor_params
        initial = initial.replace(actor_network=initial.actor_network.replace(params=params))
        np.testing.assert_array_equal(initial.actor_network.select('actor')(
            batch['observations']).sample(seed=jax.random.PRNGKey(17)), jnp.ones((3, 8)))
        first, info = bias_step(initial, batch)
        self.assertTrue(np.isfinite(info['actor/ratio_after_mean']))
        second, _ = bias_step(first, batch)
        self.assertEqual(int(second.actor_clip_update_count), 2)
        state = flax.serialization.to_state_dict(second)
        self.assertFalse(any('reference' in key for key in state))
        restored = flax.serialization.from_state_dict(make_agent(), state)
        assert_tree_equal(self, flax.serialization.to_state_dict(restored), state)

    def test_unchanged_policy_ratio_is_one_with_tiny_std_and_saturated_tanh(self):
        agent = make_agent()
        batch = jax.tree.map(lambda x: x[0], make_batch())
        params = dict(agent.actor_network.params)
        actor_params = dict(params['modules_actor'])
        actor_params['mean_net'] = dict(actor_params['mean_net'], bias=jnp.full(8, 100.))
        actor_params['log_std_net'] = jax.tree.map(
            jnp.zeros_like, actor_params['log_std_net'])
        actor_params['log_std_net']['bias'] = jnp.full(8, -20.)
        params['modules_actor'] = actor_params
        agent = agent.replace(actor_network=agent.actor_network.replace(params=params))

        def step(a):
            def zero_loss(p):
                value, info = a.actor_total_loss(batch, p, jax.random.PRNGKey(123))
                return value * 0., info
            return a._apply_actor_loss(zero_loss, batch)

        _, _, info = jax.jit(step)(agent)
        self.assertEqual(float(info['actor/update_scale']), 1.)
        self.assertEqual(float(info['actor/ratio_after_min']), 1.)
        self.assertEqual(float(info['actor/ratio_after_max']), 1.)

    def test_rejection_preserves_critic_budget_all_orderings_and_success_paths(self):
        batches = make_batch()
        successes = dict(batches, actions=batches['actions'] + .3)
        for q_first in (False, True):
            for success in (None, False, True):
                with self.subTest(q_first=q_first, success=success):
                    initial = make_agent(first_train_q=q_first, actor_ratio_clip_max_backtracks=0,
                                         critic_updates_per_actor=2)
                    updated, info = initial.batch_update(
                        batches, successes if success is not None else None, bool(success))
                    assert_tree_equal(self, updated.actor_network.params['modules_actor'],
                                      initial.actor_network.params['modules_actor'])
                    assert_tree_equal(self, updated.actor_network.opt_state,
                                      initial.actor_network.opt_state)
                    self.assertEqual(int(updated.actor_network.step), 1)
                    self.assertEqual(int(updated.critic_network.step), 5)
                    self.assertEqual(int(updated.action_critic_network.step), 5)
                    self.assertEqual(int(updated.temperature_network.step), 3)
                    self.assertEqual(int(updated.actor_clip_reject_count), 2)
                    self.assertEqual(float(info['actor/update_rejected']), 1.)
                    self.assertEqual(float(info['actor/ratio_after_mean']), 1.)
                    if success is True:
                        self.assertGreater(float(info['actor_bc_loss']), 0.)

    def test_inactive_constraint_matches_original_sac_and_success_bc(self):
        batches = make_batch()
        successes = dict(batches, actions=batches['actions'] + .7)
        for q_first in (False, True):
            for success in (None, True):
                with self.subTest(q_first=q_first, success=success):
                    enabled = make_agent(first_train_q=q_first, lr_actor=3e-4,
                                         actor_ratio_clip_epsilon=.99)
                    disabled = make_agent(first_train_q=q_first, lr_actor=3e-4,
                                          use_klactor_clip=False)
                    args = (batches, successes if success is not None else None, bool(success))
                    actual, info = enabled.batch_update(*args)
                    expected, _ = disabled.batch_update(*args)
                    self.assertEqual(float(info['actor/update_scale']), 1.)
                    for field in ('actor_network', 'critic_network', 'action_critic_network'):
                        for left, right in zip(jax.tree.leaves(getattr(actual, field).params),
                                               jax.tree.leaves(getattr(expected, field).params)):
                            np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
