"""Independent-Q, causal action coverage and isolated Monte Carlo diagnostics."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.jgcw import JGCWAgent
from ogpo.runners.jgcw_runner import (
    run_jgcw_online_rl, _cache_augmented_latents, _sample_success_chunks,
)
from ogpo.utils.action_coverage import ActionCoverage
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.datasets import Dataset
from ogpo.utils.main_helper import LoggingHelper, training_log_prefixes
from ogpo.utils.log_utils import CsvLogger
from ogpo.utils.q_accuracy import (
    AsyncQAccuracy, accuracy_metrics, rollout_return, select_accuracy_states,
)


def make_agent(train=True, qa_tdtarget=False, distill=False):
    config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  train_q_sa=train, use_qa_tdtarget=qa_tdtarget, use_qa_distill=distill,
                  first_train_q=False, use_klactor_clip=False,
                  cal_q_accuracy=False, cal_action_coverage=False, use_aug_z=False)
    return JGCWAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)


def make_batch():
    rng = np.random.default_rng(9)
    return dict(observations=jnp.asarray(rng.normal(size=(2, 3, 3)), dtype=jnp.float32),
                next_observations=jnp.asarray(rng.normal(size=(2, 3, 3)), dtype=jnp.float32),
                actions=jnp.asarray(rng.normal(size=(2, 3, 8)), dtype=jnp.float32),
                true_actions=jnp.asarray(rng.uniform(-1, 1, size=(2, 3, 8)), dtype=jnp.float32),
                rewards=-jnp.ones((2, 3)), masks=jnp.ones((2, 3)),
                chunk_length=jnp.full((2, 3), 4))


def assert_equal(test, left, right):
    a, b = jax.tree.leaves(left), jax.tree.leaves(right)
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


class IndependentQTests(unittest.TestCase):
    def test_training_preserves_dsrl_exactly_with_optional_qa(self):
        batch = make_batch()
        for enabled in (False, True):
            initial = make_agent(enabled)
            baseline = DSRLBestOfNAgent.create(1, jnp.zeros(3), jnp.zeros(2), dict(initial.config))
            for success in (None, False, True):
                updated, info = initial.batch_update(batch, batch if success is not None else None,
                                                     bool(success))
                expected, _ = baseline.batch_update(batch, batch if success is not None else None,
                                                     bool(success))
                for field in ('rng', 'actor_network', 'critic_network', 'bc_flow_network'):
                    assert_equal(self, getattr(updated, field), getattr(expected, field))
                if enabled:
                    self.assertEqual(int(updated.action_critic_network.step),
                                     int(initial.action_critic_network.step) + 4)
                    self.assertTrue(np.isfinite(info['action_critic/loss']))
                else:
                    self.assertIsNone(updated.action_critic_network)
                    self.assertNotIn('action_critic/loss', info)

    def test_qa_uses_executed_actions_and_its_own_bootstrap(self):
        agent = make_agent()
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(4)
        loss, _ = agent.action_critic_loss(batch, agent.action_critic_network.params, key)
        changed = dict(batch, actions=batch['actions'] + 100)
        same, _ = agent.action_critic_loss(changed, agent.action_critic_network.params, key)
        np.testing.assert_array_equal(loss, same)
        changed = dict(batch, true_actions=batch['true_actions'] + 100)
        different, _ = agent.action_critic_loss(changed, agent.action_critic_network.params, key)
        self.assertNotAlmostEqual(float(loss), float(different), places=5)
        for terminal in (False, True):
            b = dict(batch, masks=jnp.zeros_like(batch['masks']) if terminal else batch['masks'])
            _, info = agent.action_critic_loss(b, agent.action_critic_network.params, key)
            if terminal:
                np.testing.assert_allclose(info['target_q_mean'], b['rewards'].mean())

    def test_latent_qa_target_matches_action_target_and_ignores_latent_target(self):
        initial = make_agent(qa_tdtarget=True)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        batch = dict(batch, chunk_length=jnp.array([1, 2, 4]), masks=jnp.array([0., 1., 1.]))
        key = jax.random.PRNGKey(4)
        for aggregation in ('min', 'mean'):
            for entropy in (False, True):
                current = initial.replace(config=dict(initial.config, q_agg=aggregation,
                                                      entropy_backup=entropy))
                loss, info = current.critic_loss(batch, current.critic_network.params, key)
                _, qa_info = current.action_critic_loss(batch, current.action_critic_network.params, key)
                target, next_z, log_probs = current._action_td_target(batch, key)
                _, sample_rng = jax.random.split(key)
                expected_z, expected_log_probs = current.actor_network.select('actor')(
                    batch['next_observations']).sample_and_log_prob(seed=sample_rng)
                np.testing.assert_array_equal(next_z, expected_z)
                next_a = current.noise_to_actions_public(batch['next_observations'], expected_z)
                qs = current.action_critic_network.select('target_critic')(
                    batch['next_observations'], actions=next_a)
                expected = (qs.min(0) if aggregation == 'min' else qs.mean(0))
                if entropy:
                    expected -= current.actor_network.select('temp')() * jnp.clip(expected_log_probs, -100, 100)
                expected = jnp.clip(batch['rewards'] + .99 ** batch['chunk_length'] *
                                    batch['masks'] * expected, -500., 500.)
                np.testing.assert_allclose(target, expected, atol=1e-6)
                self.assertEqual(float(target[0]), float(batch['rewards'][0]))
                np.testing.assert_array_equal(info['target_q_mean'], qa_info['target_q_mean'])
                q = current.critic_network.select('critic')(
                    batch['observations'], actions=batch['actions'])
                np.testing.assert_allclose(loss, jnp.square(q - expected).mean(), atol=1e-6)
                params = dict(current.critic_network.params)
                params['modules_target_critic'] = jax.tree.map(
                    lambda x: x + 100, params['modules_target_critic'])
                changed = current.replace(critic_network=current.critic_network.replace(params=params))
                same, _ = changed.critic_loss(batch, params, key)
                np.testing.assert_array_equal(loss, same)

    def test_target_q_clip_switch_covers_latent_and_action_targets(self):
        batch = jax.tree.map(lambda x: x[0], make_batch())
        # Terminal rewards isolate both bounds from bootstrap estimates.
        batch = dict(batch, rewards=jnp.array([-900., 800., -20.]),
                     masks=jnp.zeros_like(batch['masks']))
        key = jax.random.PRNGKey(4)
        for train_qa, qa_tdtarget in ((False, False), (True, False), (True, True)):
            initial = make_agent(train=train_qa, qa_tdtarget=qa_tdtarget)
            self.assertFalse(initial.config['use_no_targetq_clip'])
            for disabled in (False, True):
                current = initial.replace(config=dict(initial.config, use_no_targetq_clip=disabled))
                expected = batch['rewards'] if disabled else jnp.clip(batch['rewards'], -500., 500.)
                loss, info = current.critic_loss(batch, current.critic_network.params, key)
                q = current.critic_network.select('critic')(
                    batch['observations'], actions=batch['actions'])
                np.testing.assert_allclose(loss, jnp.square(q - expected).mean(), rtol=1e-6)
                np.testing.assert_allclose(info['target_q_mean'], expected.mean())
                if train_qa:
                    target, _, _ = current._action_td_target(batch, key)
                    np.testing.assert_array_equal(target, expected)
                    _, action_info = current.action_critic_loss(
                        batch, current.action_critic_network.params, key)
                    np.testing.assert_allclose(action_info['target_q_mean'], expected.mean())

    def test_qa_tdtarget_keeps_augmented_consistency_loss(self):
        current = make_agent(qa_tdtarget=True)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(12)
        td, _ = current.critic_loss(batch, current.critic_network.params, key)
        augmented = current.replace(config=dict(current.config, use_aug_z=True, aug_z_num=2))
        batch['z_aug'] = jnp.repeat(batch['actions'][:, None], 2, axis=1) + .1
        loss, info = augmented.critic_loss(batch, augmented.critic_network.params, key)
        np.testing.assert_array_equal(info['td_loss'], td)
        np.testing.assert_allclose(loss, td + info['aug_z_loss'], atol=1e-6)
        self.assertGreater(float(info['aug_z_loss']), 0.)

    def test_qa_target_is_detached_and_flag_is_inactive_without_qa(self):
        current = make_agent(qa_tdtarget=True)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(3)
        def loss_with_qa(params):
            a = current.replace(action_critic_network=current.action_critic_network.replace(params=params))
            return a.critic_loss(batch, a.critic_network.params, key)[0]
        grads = jax.grad(loss_with_qa)(current.action_critic_network.params)
        for grad in jax.tree.leaves(grads):
            np.testing.assert_array_equal(grad, jnp.zeros_like(grad))
        no_qa = make_agent(train=False, qa_tdtarget=True)
        actual, _ = no_qa.critic_loss(batch, no_qa.critic_network.params, key)
        expected, _ = DSRLBestOfNAgent.critic_loss(no_qa, batch, no_qa.critic_network.params, key)
        np.testing.assert_array_equal(actual, expected)

    def test_qa_bootstrap_updates_regular_and_success_paths(self):
        initial = make_agent(qa_tdtarget=True)
        batch = jax.tree.map(lambda x: x[:1], make_batch())
        for success in (None, False, True):
            updated, info = initial.batch_update(batch, batch if success is not None else None,
                                                 bool(success))
            self.assertEqual(int(updated.critic_network.step), int(initial.critic_network.step) + 2)
            self.assertEqual(int(updated.action_critic_network.step),
                             int(initial.action_critic_network.step) + 2)
            self.assertTrue(np.isfinite(info['critic/critic_loss']))
            assert_equal(self, initial.bc_flow_network, updated.bc_flow_network)

    def test_first_train_q_updates_critics_before_actor_with_same_budget(self):
        initial = make_agent(qa_tdtarget=True, distill=True)
        self.assertFalse(initial.config['first_train_q'])
        batches = jax.tree.map(lambda x: x[:1], make_batch())
        successes = dict(batches, rewards=batches['rewards'] + 3.)
        batch = jax.tree.map(lambda x: x[0], batches)
        success_batch = jax.tree.map(lambda x: x[0], successes)
        for count, success in ((1, None), (2, None), (1, False), (1, True),
                               (2, False), (2, True)):
            current = initial.replace(config=dict(initial.config, first_train_q=True,
                                                   critic_updates_per_actor=count))
            keys = jax.random.split(current.rng, 3 if success is None else 4)
            new_rng, actor_rng, critic_rng = keys[:3]
            extra_rng = jax.random.fold_in(critic_rng, 1) if success is None else keys[3]
            extra = count == 2 or bool(success)
            chosen = success_batch if success else batch
            updates = [(batch, critic_rng)] + ([(chosen, extra_rng)] if extra else [])

            # Reference ordering: finish Qa before Qz consumes its targets/teacher.
            expected = current
            for data, key in updates:
                expected, _ = expected._action_step(data, key)
            for data, key in updates:
                critic, _ = expected.critic_network.apply_loss_fn(
                    lambda params: expected.critic_total_loss(data, params, key))
                expected.target_update(critic, 'critic')
                expected = expected.replace(critic_network=critic)

            # Actor gradients must use the newly trained Qz, not the old Qz.
            def actor_loss(params):
                loss, info = expected.actor_total_loss(batch, params, actor_rng)
                if success:
                    loss += expected.actor_bc_loss(success_batch, params, extra_rng)[0]
                return loss, info
            actor, expected_info = expected.actor_network.apply_loss_fn(actor_loss)
            expected = expected.replace(actor_network=actor, rng=new_rng)
            updated, info = current.batch_update(
                batches, successes if success is not None else None, bool(success))
            # Fused JIT and eager reference updates may round float32 differently.
            for field in ('rng', 'actor_network', 'critic_network', 'action_critic_network',
                          'bc_flow_network'):
                actual_leaves = jax.tree.leaves(getattr(updated, field))
                expected_leaves = jax.tree.leaves(getattr(expected, field))
                self.assertEqual(len(actual_leaves), len(expected_leaves))
                for actual, wanted in zip(actual_leaves, expected_leaves):
                    np.testing.assert_allclose(actual, wanted, rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(info['actor/q_actor_mean'],
                                       expected_info['actor/q_actor_mean'], rtol=2e-5, atol=2e-7)
            old_info = current.actor_total_loss(batch, current.actor_network.params, actor_rng)[1]
            self.assertNotAlmostEqual(float(info['actor/q_actor_mean']),
                                      float(old_info['actor/q_actor_mean']), places=7)
            self.assertEqual(int(updated.actor_network.step), int(current.actor_network.step) + 1)
            self.assertEqual(int(updated.critic_network.step),
                             int(current.critic_network.step) + len(updates))
            self.assertEqual(int(updated.action_critic_network.step),
                             int(current.action_critic_network.step) + len(updates))

    def test_distillation_matches_gaussian_online_teacher_and_student_gradients(self):
        current = make_agent(qa_tdtarget=True, distill=True)
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(21)
        distill_key = jax.random.fold_in(key, 919)
        for aggregation in ('mean', 'min'):
            a = current.replace(config=dict(current.config, q_agg=aggregation))
            baseline = a.replace(config=dict(a.config, use_qa_distill=False))
            td, _ = baseline.critic_loss(batch, baseline.critic_network.params, key)
            z = jax.random.normal(distill_key, batch['actions'].shape, dtype=batch['actions'].dtype)
            actions = a.noise_to_actions_public(batch['observations'], z)
            teacher = a.action_critic_network.select('critic')(batch['observations'], actions=actions)
            teacher = teacher.mean(0) if aggregation == 'mean' else teacher.min(0)
            def expected(params):
                td_value, _ = baseline.critic_loss(batch, params, key)
                student = a.critic_network.select('critic')(
                    batch['observations'], actions=z, params=params)
                return td_value + jnp.mean((student - jax.lax.stop_gradient(teacher)) ** 2)
            loss, info = a.critic_loss(batch, a.critic_network.params, key)
            np.testing.assert_allclose(loss, expected(a.critic_network.params), atol=1e-6)
            np.testing.assert_allclose(loss, td + info['distillation_loss'], atol=1e-6)
            np.testing.assert_array_equal(info['distill_noise_mean'], z.mean())
            np.testing.assert_array_equal(info['distill_noise_std'], z.std())
            actual_grad = jax.grad(lambda p: a.critic_loss(batch, p, key)[0])(a.critic_network.params)
            expected_grad = jax.grad(expected)(a.critic_network.params)
            for x, y in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
                np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-5)
        def teacher_loss(params):
            a = current.replace(action_critic_network=current.action_critic_network.replace(params=params))
            return a.critic_loss(batch, a.critic_network.params, key)[0]
        for g in jax.tree.leaves(jax.grad(teacher_loss)(current.action_critic_network.params)):
            np.testing.assert_array_equal(g, jnp.zeros_like(g))

    def test_distillation_requires_both_prerequisites(self):
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(22)
        for train, target in ((False, False), (False, True), (True, False)):
            enabled = make_agent(train=train, qa_tdtarget=target, distill=True)
            disabled = enabled.replace(config=dict(enabled.config, use_qa_distill=False))
            actual = enabled.critic_loss(batch, enabled.critic_network.params, key)
            expected = disabled.critic_loss(batch, disabled.critic_network.params, key)
            assert_equal(self, actual, expected)

    def test_distillation_adds_to_augmentation_and_runs_in_success_updates(self):
        current = make_agent(qa_tdtarget=True, distill=True)
        current = current.replace(config=dict(current.config, use_aug_z=True, aug_z_num=2))
        batch = jax.tree.map(lambda x: x[:1], make_batch())
        batch['z_aug'] = jnp.repeat(batch['actions'][:, :, None], 2, axis=2) + .1
        single = jax.tree.map(lambda x: x[0], batch)
        loss, info = current.critic_loss(single, current.critic_network.params, jax.random.PRNGKey(23))
        np.testing.assert_allclose(loss, info['td_loss'] + info['distillation_loss'] + info['aug_z_loss'],
                                   atol=1e-6)
        for success in (None, False, True):
            updated, metrics = current.batch_update(batch, batch if success is not None else None,
                                                    bool(success))
            self.assertTrue(np.isfinite(metrics['critic/distillation_loss']))
            self.assertEqual(int(updated.critic_network.step), int(current.critic_network.step) + 2)
            assert_equal(self, current.bc_flow_network, updated.bc_flow_network)

    def test_validation_and_shared_config(self):
        config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
        self.assertEqual(config['agent_name'], 'jgcw')
        base = config_to_agent_dict(load_config('jgcw'))
        self.assertEqual(base['agent_name'], 'jgcw')
        self.assertFalse(base['train_q_sa'])
        self.assertTrue(config['train_q_sa'])
        self.assertIsInstance(config['use_qa_tdtarget'], bool)
        self.assertFalse(config['use_qa_distill'])
        self.assertFalse(config['use_aug_z'])
        self.assertIn('online_budget', training_log_prefixes('jgcw', 100))
        self.assertIn('augmentation', training_log_prefixes('jgcw', 100))
        config.update(train_q_sa=False, cal_q_accuracy=True)
        with self.assertRaisesRegex(ValueError, 'train_q_sa'):
            JGCWAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)

    def test_runner_records_physical_actions_and_cleans_up(self):
        agent = make_agent()
        config = dict(agent.config, cal_action_coverage=True, coverage_k=2,
                      coverage_sample_interval=1, coverage_rebuild_interval=2)
        class Env:
            def reset(self, seed):
                self.t = 0
                return np.zeros(3), {}
            def step(self, action):
                self.t += 1
                return np.full(3, self.t), -1., self.t == 5, False, {'success': 0.}
        dataset = Dataset.create(observations=np.zeros((10, 3)), actions=np.zeros((10, 2)),
                                 next_observations=np.zeros((10, 3)), rewards=np.zeros(10),
                                 terminals=np.zeros(10), masks=np.ones(10))
        with tempfile.TemporaryDirectory() as directory:
            prefixes = ['eval', 'env', 'online_budget', 'online_agent', 'action_coverage']
            logger = LoggingHelper({k: CsvLogger(str(Path(directory) / (k + '.csv'))) for k in prefixes})
            with patch('ogpo.runners.jgcw_runner.save_agent'), patch(
                    'ogpo.runners.jgcw_runner._run_evaluation', return_value={}):
                _, step, _, _ = run_jgcw_online_rl(
                    agent, Env(), Env(), None, dataset, config, logger, directory, 'jgcw',
                    online_steps=12, start_training=100, batch_size=3, horizon_length=4,
                    discount=.99, n_eval_envs=1, log_interval=4, eval_interval=-1,
                    save_interval=-1, buffer_size=30)
            logger.close()
            self.assertEqual(step, 12)
            import csv
            rows = list(csv.DictReader((Path(directory) / 'action_coverage.csv').open()))
            self.assertEqual(int(rows[-1]['history_size']), 12)
            self.assertIn('action_span/1', rows[-1])
            self.assertNotIn('action_span/2', rows[-1])


class AugmentedZTests(unittest.TestCase):
    def test_gaussian_augmentation_and_safeguarded_inverse(self):
        dummy = SimpleNamespace(
            config=dict(flow_steps=4, aug_z_num=3, aug_z_sigma=.02,
                        inverse_fm_radius=10., inverse_fm_batch_size=4),
            bc_flow_network=SimpleNamespace(select=lambda name: lambda obs, x, t: jnp.ones_like(x) * .2))
        dummy.inverse_fm_batch = lambda *args: JGCWAgent.inverse_fm_batch.__wrapped__(dummy, *args)
        obs = jnp.zeros((2, 3))
        actions = jnp.full((2, 8), .4)
        key = jax.random.PRNGKey(13)
        actual = JGCWAgent.augment_z.__wrapped__(dummy, obs, actions, key)
        perturbed = actions[:, None, :] + .02 * jax.random.normal(key, (2, 3, 8))
        np.testing.assert_allclose(actual, perturbed - .2, atol=1e-6)

    def test_inverse_handles_noncontractive_discrete_flow(self):
        # Picard x <- y - 2*x diverges; the discrete map is still invertible.
        dummy = SimpleNamespace(config=dict(flow_steps=1, inverse_fm_iterations=16),
            bc_flow_network=SimpleNamespace(select=lambda name: lambda obs, x, t: 2*x + t))
        obs = jnp.zeros((3, 1))
        latents = jnp.linspace(-.1, .1, 24).reshape(3, 8)
        recovered = JGCWAgent.inverse_fm_batch.__wrapped__(dummy, obs, 3*latents)
        np.testing.assert_allclose(recovered, latents, atol=1e-6)

    def test_inverse_bounds_unreachable_targets_and_preserves_zero_noise(self):
        dummy = SimpleNamespace(config=dict(flow_steps=2, inverse_fm_iterations=16),
            bc_flow_network=SimpleNamespace(select=lambda name: lambda obs, x, t: .2*x + t))
        obs = jnp.zeros((2, 1))
        anchor = jnp.full((2, 3), .2)
        def decode(z):
            return jnp.clip((z * 1.1) * 1.1 + .25, -1, 1)
        exact = decode(anchor)
        recovered = JGCWAgent.inverse_fm_batch.__wrapped__(dummy, obs, exact, anchor)
        np.testing.assert_allclose(recovered, anchor, atol=1e-6)
        target = jnp.full_like(exact, 2.)
        recovered = JGCWAgent.inverse_fm_batch.__wrapped__(dummy, obs, target, anchor)
        self.assertTrue(np.isfinite(recovered).all())
        self.assertTrue(np.all(np.abs(recovered) <= 1.))
        self.assertTrue(np.all(np.linalg.norm(recovered-anchor, axis=-1) <= .500001))
        self.assertTrue(np.all(np.linalg.norm(decode(recovered)-target, axis=-1)
                               <= np.linalg.norm(exact-target, axis=-1)))

    def test_loss_and_gradient_match_detached_original_q(self):
        agent = make_agent(False)
        agent = agent.replace(config=dict(agent.config, use_aug_z=True, aug_z_num=2))
        batch = jax.tree.map(lambda x: x[0], make_batch())
        key = jax.random.PRNGKey(7)
        z = agent.augment_z(batch['observations'], batch['true_actions'],
                            jax.random.fold_in(key, 918))
        self.assertEqual(z.shape, (3, 2, 8))
        batch['z_aug'] = z
        q = agent.critic_network.select('critic')(
            batch['observations'], actions=batch['actions'])
        def expected_loss(params):
            td, _ = DSRLBestOfNAgent.critic_loss(agent, batch, params, key)
            q_aug = agent.critic_network.select('critic')(
                jnp.repeat(batch['observations'], 2, axis=0),
                actions=z.reshape(-1, 8), params=params).reshape(2, 3, 2)
            return td + jnp.mean(jnp.square(q_aug - q[..., None]))
        actual, info = agent.critic_loss(batch, agent.critic_network.params, key)
        np.testing.assert_allclose(actual, expected_loss(agent.critic_network.params), atol=1e-6)
        np.testing.assert_allclose(actual, info['td_loss'] + info['aug_z_loss'])
        self.assertGreater(float(info['aug_z_loss']), 0.)
        actual_grad = jax.grad(lambda p: agent.critic_loss(batch, p, key)[0])(
            agent.critic_network.params)
        expected_grad = jax.grad(expected_loss)(agent.critic_network.params)
        for x, y in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-5)

    def test_augmented_updates_cover_regular_and_success_paths(self):
        initial = make_agent(False)
        initial = initial.replace(config=dict(initial.config, use_aug_z=True))
        batch = jax.tree.map(lambda x: x[:1], make_batch())
        obs, actions = batch['observations'][0], batch['true_actions'][0]
        batch['z_aug'] = initial.augment_z(obs, actions, jax.random.PRNGKey(17))[None]
        success_batch = dict(batch, true_actions=batch['true_actions'] + .1,
                             rewards=batch['rewards'] + .5,
                             z_aug=initial.augment_z(obs, actions + .1, jax.random.PRNGKey(18))[None])
        for success in (None, False, True):
            replay_success = success_batch if success is not None else None
            with patch.object(JGCWAgent, 'augment_z', side_effect=AssertionError('Training must use cache')):
                updated, info = initial.batch_update(batch, replay_success, bool(success))
            self.assertEqual(int(updated.critic_network.step),
                             int(initial.critic_network.step) + 2)
            self.assertTrue(np.isfinite(info['critic/aug_z_loss']))
            assert_equal(self, initial.bc_flow_network, updated.bc_flow_network)
            # Compare every extra critic step to an explicit augmented-loss update.
            single = initial.replace(config=dict(initial.config, critic_updates_per_actor=1,
                                                  use_success_buffer_q=False))
            first, _ = single.batch_update(batch, replay_success, bool(success))
            rng = (jax.random.fold_in(jax.random.split(initial.rng, 3)[2], 1)
                   if success is None else jax.random.split(initial.rng, 4)[3])
            sample = jax.tree.map(lambda x: x[0], success_batch if success else batch)
            state, _ = first.critic_network.apply_loss_fn(
                lambda p: first.critic_total_loss(sample, p, rng))
            first.target_update(state, 'critic')
            # Fused scan and standalone updates may differ by float32 rounding.
            for x, y in zip(jax.tree.leaves(updated.critic_network), jax.tree.leaves(state)):
                np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-5)

    def test_cache_padding_order_and_independent_batches(self):
        calls = []
        def augment(obs, actions, rng, anchors):
            calls.append((np.asarray(obs), np.asarray(actions), np.asarray(rng)))
            return actions[:, None, :] + obs[:, None, :1]
        dummy = SimpleNamespace(augment_z=augment)
        transitions = [dict(observations=np.full(3, i), true_actions=np.full(8, i * 10.), actions=np.zeros(8))
                       for i in range(5)]
        _cache_augmented_latents(dummy, transitions, jax.random.PRNGKey(20), 2)
        self.assertEqual(len(calls), 3)
        for obs, actions, _ in calls:
            self.assertEqual(obs.shape, (2, 3))
            self.assertEqual(actions.shape, (2, 8))
            self.assertEqual(obs.dtype, np.float32)
        self.assertFalse(np.array_equal(calls[0][2], calls[1][2]))
        np.testing.assert_array_equal(calls[-1][0][0], calls[-1][0][1])
        self.assertEqual(len(transitions), 5)
        for i, transition in enumerate(transitions):
            np.testing.assert_array_equal(transition['z_aug'], np.full((1, 8), i * 11.))
            self.assertEqual(transition['z_aug'].dtype, np.float32)

    def test_cache_rejects_nonfinite_inverse_before_insertion(self):
        transitions = [dict(observations=np.zeros(3), true_actions=np.zeros(8), actions=np.zeros(8))]
        for invalid in (float('nan'), float('inf')):
            dummy = SimpleNamespace(augment_z=lambda obs, actions, rng, anchors: np.full((2, 1, 8), invalid))
            with self.assertRaisesRegex(FloatingPointError, 'non-finite z_aug'):
                _cache_augmented_latents(dummy, transitions, jax.random.PRNGKey(1), 2)
            self.assertNotIn('z_aug', transitions[0])

    def test_runner_caches_once_and_preserves_replay_alignment(self):
        import pickle
        agent = make_agent(False)
        config = dict(agent.config, use_aug_z=True, aug_z_num=2, updates_per_env_step=False)
        agent = agent.replace(config=config)
        class Env:
            def reset(self, seed):
                self.episode = getattr(self, 'episode', 0) + 1
                self.t = 0
                return np.array([self.episode, self.t, 0], dtype=np.float32), {}
            def step(self, action):
                self.t += 1
                return (np.array([self.episode, self.t, 0], dtype=np.float32),
                        -1., self.t == 9, False, {'success': 1.})
        dataset = Dataset.create(observations=np.zeros((10, 3), dtype=np.float32),
                                 actions=np.zeros((10, 2), dtype=np.float32),
                                 next_observations=np.zeros((10, 3), dtype=np.float32),
                                 rewards=np.zeros(10), terminals=np.zeros(10), masks=np.ones(10))
        with tempfile.TemporaryDirectory() as directory, patch.object(
                JGCWAgent, 'augment_z', side_effect=AssertionError('Main process must not invert')), patch(
                'ogpo.runners.jgcw_runner.save_agent'), patch(
                'ogpo.runners.jgcw_runner._run_evaluation', return_value={}):
            updated, step, _, replay = run_jgcw_online_rl(
                agent, Env(), None, None, dataset, config, SimpleNamespace(log=lambda *a, **kw: None),
                directory, 'jgcw', online_steps=18, start_training=9, batch_size=3,
                horizon_length=4, discount=.99, n_eval_envs=1, log_interval=-1,
                eval_interval=-1, save_interval=-1, buffer_size=5,
                use_success_buffer=True, success_buffer_batch_size=3)
        self.assertEqual(step, 18)
        self.assertEqual(int(updated.critic_network.step), int(agent.critic_network.step) + 6)
        self.assertEqual(replay.size, 5)  # Six chunks exercise ring-buffer overwrite.
        base_key = jax.random.PRNGKey(50000)
        base_key = jax.random.fold_in(base_key, 0)
        for sampled in (replay.sample(12), _sample_success_chunks(replay, 12),
                        pickle.loads(pickle.dumps(replay)).sample(12)):
            keys = jnp.stack([jax.random.fold_in(jax.random.fold_in(
                base_key, int(obs[0]) - 1), int(obs[1]) // 4) for obs in sampled['observations']])
            expected = agent.augment_z(jnp.asarray(sampled['observations'], dtype=jnp.float32),
                jnp.asarray(sampled['true_actions'], dtype=jnp.float32), keys,
                jnp.asarray(sampled['actions'], dtype=jnp.float32))
            np.testing.assert_allclose(sampled['z_aug'], expected, atol=1e-5, rtol=1e-5)
        replay.close()
        assert_equal(self, agent.bc_flow_network, updated.bc_flow_network)

    def test_enabled_loss_requires_cache(self):
        agent = make_agent(False)
        agent = agent.replace(config=dict(agent.config, use_aug_z=True))
        batch = jax.tree.map(lambda x: x[0], make_batch())
        with self.assertRaisesRegex(ValueError, 'cached z_aug'):
            agent.critic_loss(batch, agent.critic_network.params, jax.random.PRNGKey(1))
        batch['z_aug'] = jnp.zeros((3, 2, 8))
        with self.assertRaisesRegex(ValueError, 'shape'):
            agent.critic_loss(batch, agent.critic_network.params, jax.random.PRNGKey(1))

    def test_invalid_augmentation_config(self):
        base = dict(make_agent(False).config)
        for options, error in [({'inverse_fm_radius': 0}, 'inverse_fm_radius'),
                               ({'inverse_fm_radius': float('nan')}, 'inverse_fm_radius'),
                               ({'inverse_fm_batch_size': 0}, 'inverse_fm_batch_size'),
                               ({'inverse_fm_iterations': 0}, 'inverse_fm_iterations'),
                               ({'inverse_fm_iterations': 1.5}, 'inverse_fm_iterations'),
                               ({'inverse_fm_iterations': True}, 'inverse_fm_iterations'),
                               ({'aug_z_num': 0}, 'aug_z_num'),
                               ({'aug_z_num': 1.5}, 'aug_z_num'),
                               ({'aug_z_num': True}, 'aug_z_num'),
                               ({'aug_z_sigma': -.1}, 'aug_z_sigma'),
                               ({'aug_z_sigma': float('nan')}, 'aug_z_sigma'),
                               ({'use_aug_z': True, 'noise_chunk_length': 1}, 'noise_chunk_length')]:
            with self.assertRaisesRegex(ValueError, error):
                JGCWAgent.create(1, jnp.zeros(3), jnp.zeros(2), dict(base, **options))


class CoverageTests(unittest.TestCase):
    def test_exact_knn_matches_all_history_before_and_after_rebuilds(self):
        rng = np.random.default_rng(3)
        monitor = ActionCoverage(np.array([2., .4, 1.]), 2, k=4,
                                 sample_interval=1, rebuild_interval=7)
        states, actions = [], []
        for _ in range(40):
            observation, action = rng.normal(size=3), rng.normal(size=2)
            result = monitor.observe(observation, action)
            if len(states) >= 4:
                nearest = np.argsort(np.linalg.norm(np.asarray(states) - observation / monitor.scale, axis=1))[:4]
                local = np.asarray(actions)[nearest]
                expected = np.linalg.norm(local - action, axis=1).min()
                self.assertAlmostEqual(result['novelty'], expected, places=5)
                self.assertAlmostEqual(result['local_cov_trace'], np.trace(np.cov(local, rowvar=False)), places=5)
            states.append(np.asarray(observation / monitor.scale, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
        self.assertEqual(monitor.size, 40)
        self.assertEqual(monitor.queries, 36)
        self.assertEqual(monitor.metrics()['window_query_count'], 36)
        empty = monitor.metrics()
        self.assertEqual(empty['window_query_count'], 0)
        self.assertIn('novelty_mean', empty)

    def test_no_self_match_duplicate_rate_and_action_range(self):
        monitor = ActionCoverage(np.ones(1), 1, k=2, sample_interval=1, epsilon=.1)
        self.assertIsNone(monitor.observe([0], [0]))
        self.assertIsNone(monitor.observe([2], [1]))
        novel = monitor.observe([1], [.5])
        self.assertAlmostEqual(novel['novelty'], .5)
        duplicate = monitor.observe([0], [0])
        self.assertEqual(duplicate['duplicate'], 1.)
        metrics = monitor.metrics()
        self.assertEqual(metrics['duplicate_rate'], .5)
        self.assertEqual(metrics['action_span/0'], 1.)


class AccuracyTests(unittest.TestCase):
    def test_fixed_first_chunk_discount_and_nonterminal_time_limit(self):
        class Agent:
            def sample_actions(self, observations, rng):
                return np.array([7., 8.])
        class Env:
            action_space = type('Space', (), {'shape': (1,)})()
            def __init__(self):
                self.actions = []
            def step(self, action):
                self.actions.append(float(action[0]))
                return np.zeros(1), -1., len(self.actions) == 5, len(self.actions) >= 2, {}
        env = Env()
        result = rollout_return(Agent(), env, np.zeros(1), [3., 4.], jax.random.PRNGKey(0), .9, 10)
        self.assertEqual(env.actions, [3., 4., 7., 8., 7.])
        self.assertAlmostEqual(result[0], -sum(.9 ** i for i in range(5)))
        self.assertEqual(result[1:], (5, False, True))

    def test_mc_errors_and_standard_errors(self):
        returns = np.array([[1., 3.], [2., 4.]])
        metrics = accuracy_metrics(np.array([3., 2.]), np.array([2., 3.]), returns, np.array([0, 1]))
        self.assertEqual(metrics['q_sz/rmse'], 1.)
        self.assertEqual(metrics['q_sa/mae'], 0.)
        self.assertAlmostEqual(metrics['mc_se_mean'], 1.)
        self.assertEqual(metrics['q_sz/bias'], 0.)

    def test_four_stage_sampling_distinct_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demos.hdf5'
            with h5py.File(path, 'w') as f:
                for i in range(8):
                    g = f.create_group(f'data/demo_{i}')
                    g.create_dataset('states', data=np.arange(40).reshape(20, 2))
                    g.create_dataset('actions', data=np.zeros((20, 1)))
                    g.create_dataset('rewards', data=np.array([0.] * 19 + [1.]))
                    g.attrs['model_file'] = '<xml/>'
            bank = select_accuracy_states(path, 24, 8)
            np.testing.assert_array_equal(np.bincount([r['stage'] for r in bank]), [6] * 4)
            pairs = [(r['demo'], r['timestep']) for r in bank]
            self.assertEqual(len(set(pairs)), 24)
            self.assertEqual(pairs, [(r['demo'], r['timestep']) for r in select_accuracy_states(path, 24, 8)])

    def test_worker_lifecycle_in_runner_and_error_cleanup(self):
        from types import SimpleNamespace
        agent = make_agent()
        config = dict(agent.config, cal_q_accuracy=True)
        fake = SimpleNamespace(submit=lambda *a: True, finish=lambda *a: None)
        events = []
        fake.close = lambda: events.append('closed')
        # Exercise timeout-free lifecycle through a real runner call with a fake
        # evaluator and simulated failure during environment reset.
        class BrokenEnv:
            def reset(self, seed):
                raise RuntimeError('test reset failure')
        dataset = Dataset.create(observations=np.zeros((2, 3)), actions=np.zeros((2, 2)),
                                 terminals=np.zeros(2), rewards=np.zeros(2), masks=np.ones(2))
        with patch('ogpo.utils.q_accuracy.AsyncQAccuracy', return_value=fake):
            with self.assertRaisesRegex(RuntimeError, 'test reset failure'):
                run_jgcw_online_rl(agent, BrokenEnv(), None, None, dataset, config, None, '/unused',
                                   'jgcw', online_steps=1, start_training=10, batch_size=1,
                                   horizon_length=4, discount=.99)
        self.assertEqual(events, ['closed'])


if __name__ == '__main__':
    unittest.main()
