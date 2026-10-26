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


def make_agent(train=True):
    config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  train_q_sa=train, cal_q_accuracy=False, cal_action_coverage=False, use_aug_z=False)
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

    def test_validation_and_shared_config(self):
        config = config_to_agent_dict(load_config('jgcw_ogpo_square_shared'))
        self.assertEqual(config['agent_name'], 'jgcw')
        base = config_to_agent_dict(load_config('jgcw'))
        self.assertEqual(base['agent_name'], 'jgcw')
        self.assertFalse(base['train_q_sa'])
        self.assertFalse(config['train_q_sa'])
        self.assertTrue(config['use_aug_z'])
        self.assertIn('online_budget', training_log_prefixes('jgcw', 100))
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
    def test_gaussian_augmentation_and_grace_inverse(self):
        class ConstantFlow:
            def select(self, name):
                return lambda obs, x, t: jnp.ones_like(x) * .2
        dummy = SimpleNamespace(
            config=dict(flow_steps=4, aug_z_num=3, aug_z_sigma=.02),
            bc_flow_network=ConstantFlow())
        dummy.inverse_fm_batch = lambda obs, a: JGCWAgent.inverse_fm_batch.__wrapped__(dummy, obs, a)
        obs = jnp.zeros((2, 3))
        actions = jnp.full((2, 8), .999)
        key = jax.random.PRNGKey(13)
        actual = JGCWAgent.augment_z.__wrapped__(dummy, obs, actions, key)
        perturbed = actions[:, None, :] + .02 * jax.random.normal(key, (2, 3, 8))
        self.assertGreater(float(perturbed.max()), 1.)
        np.testing.assert_allclose(actual, perturbed - .2, atol=1e-6)
        from ogpo.agents.grace import GRACEAgent
        np.testing.assert_array_equal(
            dummy.inverse_fm_batch(obs, actions),
            GRACEAgent.inverse_fm_batch.__wrapped__(dummy, obs, actions))

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
        def augment(obs, actions, rng):
            calls.append((np.asarray(obs), np.asarray(actions), np.asarray(rng)))
            return actions[:, None, :] + obs[:, None, :1]
        dummy = SimpleNamespace(augment_z=augment)
        transitions = [dict(observations=np.full(3, i), true_actions=np.full(8, i * 10.))
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
        cached = []
        original = JGCWAgent.augment_z
        def record(current, obs, actions, rng):
            z = original(current, obs, actions, rng)
            cached.append((np.asarray(obs), np.asarray(actions), np.asarray(z)))
            return z
        with tempfile.TemporaryDirectory() as directory, patch.object(
                JGCWAgent, 'augment_z', record), patch('ogpo.runners.jgcw_runner.save_agent'), patch(
                'ogpo.runners.jgcw_runner._run_evaluation', return_value={}):
            updated, step, _, replay = run_jgcw_online_rl(
                agent, Env(), None, None, dataset, config, SimpleNamespace(log=lambda *a, **kw: None),
                directory, 'jgcw', online_steps=18, start_training=9, batch_size=3,
                horizon_length=4, discount=.99, n_eval_envs=1, log_interval=-1,
                eval_interval=-1, save_interval=-1, buffer_size=5,
                use_success_buffer=True, success_buffer_batch_size=3)
        self.assertEqual(step, 18)
        self.assertEqual(int(updated.critic_network.step), int(agent.critic_network.step) + 6)
        self.assertEqual(len(cached), 2)  # One insertion batch per episode; none in training.
        self.assertEqual(replay.size, 5)  # Six chunks exercise ring-buffer overwrite.
        expected = {tuple(obs): (action, z) for observations, actions, latents in cached
                    for obs, action, z in zip(observations, actions, latents)}
        for sampled in (replay.sample(12), _sample_success_chunks(replay, 12),
                        pickle.loads(pickle.dumps(replay)).sample(12)):
            for obs, action, z in zip(sampled['observations'], sampled['true_actions'], sampled['z_aug']):
                expected_action, expected_z = expected[tuple(obs)]
                np.testing.assert_array_equal(action, expected_action)
                np.testing.assert_array_equal(z, expected_z)
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
        for options, error in [({'aug_z_num': 0}, 'aug_z_num'),
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
