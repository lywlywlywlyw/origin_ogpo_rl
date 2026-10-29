"""KL-constrained SAC updates: backtracking, rollback, temperature and scan paths."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import unittest

import distrax
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.jgcw import JGCWAgent
from ogpo.utils.config_loader import config_to_agent_dict, load_config


def make_agent(**overrides):
    config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
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


def bias_loss(params):
    # A deterministic Adam proposal with a large change in Gaussian means.
    return -params['modules_actor']['mean_net']['bias'].sum(), {
        'actor/entropy': jnp.float32(0.)}


@jax.jit
def bias_step(agent, batch):
    agent, actor, info = agent._apply_actor_loss(bias_loss, batch)
    return agent.replace(actor_network=actor), info


class KLActorTests(unittest.TestCase):
    def test_analytic_kl_matches_distrax_and_sums_dimensions(self):
        old = distrax.MultivariateNormalDiag(
            loc=jnp.array([[0., 1.], [2., -1.]]),
            scale_diag=jnp.array([[1., 2.], [.5, 1.]]))
        new = distrax.MultivariateNormalDiag(
            loc=jnp.array([[.4, 1.2], [1.5, .3]]),
            scale_diag=jnp.array([[2., 1.], [.7, .8]]))
        actual = JGCWAgent._gaussian_kl(old, new)
        np.testing.assert_allclose(actual, old.kl_divergence(new), rtol=1e-6)
        self.assertEqual(actual.shape, (2,))
        np.testing.assert_array_equal(JGCWAgent._gaussian_kl(old, old), jnp.zeros(2))

    def test_full_accept_backtrack_and_rollback_keep_temperature_independent(self):
        batch = jax.tree.map(lambda x: x[0], make_batch())
        initial = make_agent()
        proposed, _ = initial.actor_network.apply_loss_fn(bias_loss)
        results = []
        for threshold, backtracks, expected in ((100., 10, 'full'),
                                               (.01, 10, 'scaled'),
                                               (.01, 0, 'rejected')):
            agent = initial.replace(config=dict(initial.config,
                klactor_clip_threshold=threshold, klactor_clip_max_backtracks=backtracks))
            updated, info = bias_step(agent, batch)
            results.append(updated)
            self.assertGreater(float(info['actor/kl_before']), .01)
            self.assertLessEqual(float(info['actor/kl_after']), threshold)
            self.assertEqual(int(updated.temperature_network.step), 2)
            self.assertNotEqual(float(updated.actor_network.select('temp')()),
                                float(initial.actor_network.select('temp')()))
            np.testing.assert_allclose(info['actor/kl_reference'], info['actor/kl_after'],
                                       rtol=3e-5, atol=2e-7)
            old = initial.actor_network.params['modules_actor']
            new = updated.actor_network.params['modules_actor']
            scale = float(info['actor/update_scale'])
            if expected == 'rejected':
                self.assertEqual(scale, 0.)
                self.assertEqual(float(info['actor/rejection_rate']), 1.)
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
                    self.assertGreater(float(info['actor/backtrack_steps']), 0.)
                wanted = jax.tree.map(lambda a, b: a + scale * (b - a),
                                      old, proposed.params['modules_actor'])
                for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(wanted)):
                    np.testing.assert_allclose(a, b, atol=1e-7)
        for result in results[1:]:
            assert_tree_equal(self, result.temperature_network, results[0].temperature_network)

    def test_nonfinite_proposal_is_rejected(self):
        agent = make_agent(klactor_clip_max_backtracks=2)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        def loss(params):
            value, info = bias_loss(params)
            return value * jnp.nan, info
        _, actor, info = jax.jit(lambda a: a._apply_actor_loss(loss, batch))(agent)
        self.assertEqual(float(info['actor/update_rejected']), 1.)
        assert_tree_equal(self, actor.params['modules_actor'], agent.actor_network.params['modules_actor'])
        assert_tree_equal(self, actor.opt_state, agent.actor_network.opt_state)
        self.assertTrue(np.isfinite(actor.select('temp')()))

    def test_reference_captures_first_update_and_survives_checkpoint(self):
        initial = make_agent()
        batch = jax.tree.map(lambda x: x[0], make_batch())
        # Simulate loading actor parameters after create(), before online training.
        params = dict(initial.actor_network.params)
        actor_params = dict(params['modules_actor'])
        actor_params['mean_net'] = dict(actor_params['mean_net'], bias=jnp.ones(8))
        params['modules_actor'] = actor_params
        initial = initial.replace(actor_network=initial.actor_network.replace(params=params))
        first, _ = bias_step(initial, batch)
        assert_tree_equal(self, first.klactor_reference_params, actor_params)
        second, info = bias_step(first, batch)
        assert_tree_equal(self, second.klactor_reference_params, actor_params)
        self.assertGreater(float(info['actor/kl_reference']), float(info['actor/kl_after']))
        restored = flax.serialization.from_state_dict(
            make_agent(), flax.serialization.to_state_dict(second))
        assert_tree_equal(self, flax.serialization.to_state_dict(restored),
                          flax.serialization.to_state_dict(second))

    def test_rejection_preserves_critic_budget_all_orderings_and_success_paths(self):
        batches = make_batch()
        successes = dict(batches, actions=batches['actions'] + .3)
        for q_first in (False, True):
            for success in (None, False, True):
                with self.subTest(q_first=q_first, success=success):
                    initial = make_agent(first_train_q=q_first, klactor_clip_threshold=1e-20,
                                         klactor_clip_max_backtracks=0,
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
                    self.assertEqual(int(updated.klactor_reject_count), 2)
                    self.assertEqual(float(info['actor/update_rejected']), 1.)
                    self.assertEqual(float(info['actor/kl_after']), 0.)
                    self.assertTrue(np.isfinite(info['actor/pre_tanh_std_mean']))
                    if success is True:
                        self.assertGreater(float(info['actor_bc_loss']), 0.)

    def test_inactive_constraint_matches_original_sac_and_success_bc(self):
        batches = make_batch()
        successes = dict(batches, actions=batches['actions'] + .7)
        for q_first in (False, True):
            for success in (None, True):
                with self.subTest(q_first=q_first, success=success):
                    enabled = make_agent(first_train_q=q_first, lr_actor=3e-4,
                                         klactor_clip_threshold=1e6)
                    disabled = make_agent(first_train_q=q_first, lr_actor=3e-4,
                                          use_klactor_clip=False)
                    args = (batches, successes if success is not None else None, bool(success))
                    actual, info = enabled.batch_update(*args)
                    expected, _ = disabled.batch_update(*args)
                    self.assertEqual(float(info['actor/update_scale']), 1.)
                    for field in ('actor_network', 'critic_network', 'action_critic_network'):
                        a = getattr(actual, field).params
                        b = getattr(expected, field).params
                        for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                            np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
