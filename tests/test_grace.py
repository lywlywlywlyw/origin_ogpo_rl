"""GRACE action-Q distillation and per-sample input-gradient diagnostics."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.grace import GRACEAgent
from ogpo.runners.grace_runner import run_grace_online_rl
from ogpo.utils.config_loader import load_config, config_to_agent_dict


def make_pair():
    config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2)
    dsrl = DSRLBestOfNAgent.create(1, jnp.zeros(3), jnp.zeros(2), dict(config))
    grace = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), dict(config))
    return dsrl, grace


def batch():
    rng = np.random.default_rng(8)
    return dict(observations=jnp.asarray(rng.normal(size=(2, 3, 3)), dtype=jnp.float32),
                next_observations=jnp.asarray(rng.normal(size=(2, 3, 3)), dtype=jnp.float32),
                actions=jnp.asarray(rng.normal(size=(2, 3, 8)), dtype=jnp.float32),
                true_actions=jnp.asarray(rng.uniform(-1, 1, size=(2, 3, 8)), dtype=jnp.float32),
                rewards=-jnp.ones((2, 3)), masks=jnp.ones((2, 3)),
                chunk_length=jnp.full((2, 3), 4))


def assert_tree_equal(test, a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)


class GraceTests(unittest.TestCase):
    def test_encoder_normal_init_is_standard_normal_and_trainable(self):
        _, baseline = make_pair()
        config = dict(baseline.config, encoder_normal_init=True, action_magnitude=.5)
        grace = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        observations = jax.random.normal(jax.random.PRNGKey(31), (128, 3)) * 10
        dist = grace.actor_network.select('actor')(observations)
        np.testing.assert_array_equal(dist.mean(), np.zeros((128, 8)))
        np.testing.assert_array_equal(dist.stddev(), np.ones((128, 8)))
        for head in ('mean_net', 'log_std_net'):
            for leaf in jax.tree.leaves(grace.actor_network.params['modules_actor'][head]):
                np.testing.assert_array_equal(leaf, np.zeros_like(leaf))
        key = jax.random.PRNGKey(32)
        samples = grace.sample_noise(observations, key)
        self.assertTrue(np.any(np.abs(samples) > 1.))
        np.testing.assert_array_equal(samples, dist.sample(seed=key))
        expected_log_probs = -.5 * jnp.sum(samples ** 2 + jnp.log(2 * jnp.pi), axis=-1)
        np.testing.assert_allclose(dist.log_prob(samples), expected_log_probs, rtol=1e-6)
        np.testing.assert_array_equal(dist.mode(), np.zeros((128, 8)))
        updated, metrics = grace.batch_update(batch())
        self.assertTrue(np.isfinite(metrics['actor/total_loss']))
        self.assertTrue(any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(
            updated.actor_network.params['modules_actor']['mean_net'])))

    def test_encoder_normal_init_disabled_preserves_original_policy(self):
        dsrl, grace = make_pair()
        self.assertFalse(grace.config['encoder_normal_init'])
        observations = jax.random.normal(jax.random.PRNGKey(33), (128, 3))
        key = jax.random.PRNGKey(34)
        np.testing.assert_array_equal(grace.sample_noise(observations, key),
                                      dsrl.sample_noise(observations, key))
        self.assertTrue(np.all(np.abs(grace.sample_noise(observations, key)) <= 1.))
        config = dict(grace.config)
        config.pop('encoder_normal_init')
        default = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        np.testing.assert_array_equal(default.sample_noise(observations, key),
                                      grace.sample_noise(observations, key))

    def test_config_matches_dsrl(self):
        dsrl = load_config('dsrl_ogpo_square_shared')
        grace = load_config('grace_ogpo_square_shared')
        for key in dsrl:
            if key not in ('agent', 'experiment'):
                self.assertEqual(dsrl[key], grace[key], key)
        for key, value in dsrl['agent'].items():
            if key != 'agent_name':
                self.assertEqual(value, grace['agent'][key], key)
        from ogpo.utils.main_helper import training_log_prefixes
        self.assertIn('online_budget', training_log_prefixes('grace', 100))

    def test_bounded_decoder_noise_and_unchanged_evaluation(self):
        _, grace = make_pair()
        self.assertFalse(grace.config['decoder_add_noise'])
        self.assertEqual(grace.config['decoder_noise_std'], .09)
        self.assertEqual(grace.config['decoder_noise_clip'], 3.)
        actions = jnp.tile(jnp.array([-2., -.95, 0., .95, 2.]), 200)
        key = jax.random.PRNGKey(17)
        np.testing.assert_array_equal(grace.add_decoder_noise(actions, key), actions)
        noisy = grace.replace(config=grace.config.copy(dict(
            decoder_add_noise=True, decoder_noise_clip=.5)))
        center = jnp.clip(actions, -1., 1.)
        raw = center + .09 * jax.random.normal(key, center.shape)
        expected = jnp.clip(jnp.clip(raw, center - .045, center + .045), -1., 1.)
        actual = noisy.add_decoder_noise(actions, key)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertTrue(np.all(np.abs(actual) <= 1.))
        self.assertGreater(float(jnp.max(jnp.abs(actual - center))), .044)
        np.testing.assert_array_equal(actual, noisy.add_decoder_noise(actions, key))
        self.assertFalse(np.array_equal(actual, noisy.add_decoder_noise(actions, jax.random.fold_in(key, 1))))
        obs = jnp.zeros(3)
        np.testing.assert_array_equal(grace.sample_actions(obs, key), noisy.sample_actions(obs, key))
        zero = noisy.replace(config=noisy.config.copy(dict(decoder_noise_std=0.)))
        np.testing.assert_array_equal(zero.add_decoder_noise(actions, key), center)
        for name, value in (('decoder_noise_std', -.1), ('decoder_noise_std', float('nan')),
                            ('decoder_noise_clip', 0.)):
            with self.assertRaisesRegex(ValueError, name):
                GRACEAgent.create(1, obs, jnp.zeros(2), dict(grace.config, **{name: value}))

    def test_online_decoder_noise_is_executed_and_stored(self):
        _, grace = make_pair()
        grace = grace.replace(config=grace.config.copy(dict(decoder_add_noise=True)))
        class Env:
            def __init__(self):
                self.executed = []
            def reset(self, seed):
                self.steps = 0
                return np.zeros(3), {}
            def step(self, action):
                self.executed.append(np.asarray(action))
                self.steps += 1
                return np.zeros(3), -1., self.steps == 5, False, {'success': 0.}
        class Dataset:
            def sample(self, shape):
                return dict(observations=np.zeros(3), actions=np.zeros(2))
        class Logger:
            def log(self, *args, **kwargs):
                pass
        env = Env()
        with patch('ogpo.runners.grace_runner.save_agent'), patch(
                'ogpo.runners.grace_runner._run_evaluation', return_value={'success': 0.}):
            _, _, _, replay = run_grace_online_rl(
                grace, env, env, None, Dataset(), dict(grace.config), Logger(), '/unused',
                algo_name='grace', discount=.99, seed=1,
                online_steps=10, start_training=20, batch_size=1, horizon_length=4,
                n_eval_envs=1, eval_interval=-1, log_interval=-1, save_interval=-1,
                force_jax_sync=False, buffer_size=100)
        self.assertEqual(len(env.executed), 10)
        online_rng = jax.random.PRNGKey(40001)
        position = 0
        for index in range(replay.size):
            online_rng, latent_rng = jax.random.split(online_rng)
            z = jax.random.normal(latent_rng, (8,))
            clean = grace.noise_to_actions_public(jnp.zeros(3), z)
            online_rng, noise_rng = jax.random.split(online_rng)
            expected = grace.add_decoder_noise(clean, noise_rng)
            np.testing.assert_allclose(replay['true_actions'][index], expected, atol=1e-7)
            np.testing.assert_array_equal(replay['actions'][index], z)
            self.assertFalse(np.array_equal(clean, expected))
            length = int(replay['chunk_length'][index])
            np.testing.assert_allclose(env.executed[position:position + length],
                                       np.asarray(expected).reshape(4, 2)[:length], atol=1e-7)
            position += length
        self.assertEqual(position, 10)

    def test_distillation_updates_and_frozen_decoder(self):
        dsrl, grace = make_pair()
        self.assertEqual(grace.config['gradient_delta_a'], 1.)
        self.assertEqual(grace.config['gradient_delta_z'], 1.)
        # Restore the same offline decoder into both algorithms.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bc.pkl'
            with path.open('wb') as f:
                pickle.dump({'agent': {'actor_network': {'params': {
                    'modules_actor': dsrl.bc_flow_network.params['modules_actor_bc_flow']}}}}, f)
            dsrl = dsrl.restore_bc_flow_params(path)
            grace = grace.restore_bc_flow_params(path)
        frozen = grace.bc_flow_network
        data = batch()
        for success in (False, True):
            grace, metrics = grace.batch_update(data, data if success else None, success)
            self.assertTrue(np.isfinite(metrics['critic/distillation_loss']))
            np.testing.assert_allclose(metrics['critic/critic_loss'],
                                       metrics['critic/distillation_loss'])
            assert_tree_equal(self, frozen, grace.bc_flow_network)
            self.assertEqual(int(grace.action_critic_network.step), int(grace.critic_network.step))
            for key in ('grad_z_norm', 'grad_a_norm', 'ratio'):
                self.assertTrue(np.isfinite(metrics['sensitivity/' + key]))
        self.assertEqual(int(grace.critic_network.step), 9)  # 2 scans * 2 Q steps * 2 calls + init

    def test_gaussian_distillation_target_and_stopped_teacher_gradient(self):
        _, grace = make_pair()
        data = jax.tree.map(lambda x: x[0], batch())
        rng = jax.random.PRNGKey(19)
        z = jax.random.normal(rng, data['actions'].shape, dtype=data['actions'].dtype)
        a = grace.noise_to_actions_public(data['observations'], z)
        target = grace._aggregate_q(grace.action_critic_network.select('critic')(
            data['observations'], actions=a))
        q = grace.critic_network.select('critic')(
            data['observations'], actions=z)
        loss, info = grace.critic_loss(data, grace.critic_network.params, rng)
        np.testing.assert_allclose(loss, jnp.mean(jnp.square(q - target)), rtol=1e-6)
        np.testing.assert_allclose(info['target_q_mean'], target.mean(), rtol=1e-6)

        # Replay noise and TD fields do not supply the distillation target.
        changed = dict(data, actions=jnp.full_like(data['actions'], 100.),
                       true_actions=jnp.zeros_like(data['true_actions']),
                       rewards=data['rewards'] * 100., masks=jnp.zeros_like(data['masks']),
                       next_observations=data['next_observations'] * 100.)
        changed_loss, _ = grace.critic_loss(changed, grace.critic_network.params, rng)
        np.testing.assert_allclose(changed_loss, loss, rtol=1e-6)

        def teacher_loss(params):
            teacher = grace.replace(action_critic_network=
                                    grace.action_critic_network.replace(params=params))
            return teacher.critic_loss(data, grace.critic_network.params, rng)[0]
        grads = jax.grad(teacher_loss)(grace.action_critic_network.params)
        for leaf in jax.tree.leaves(grads):
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

    def test_latent_step_uses_updated_online_action_critic(self):
        _, grace = make_pair()
        data = jax.tree.map(lambda x: x[0], batch())
        rng = jax.random.PRNGKey(23)
        action_rng, latent_rng = jax.random.split(rng)
        action_state, _ = grace.action_critic_network.apply_loss_fn(
            lambda p: grace.action_critic_loss(data, p, action_rng))
        grace.target_update(action_state, 'critic')
        teacher = grace.replace(action_critic_network=action_state)
        expected, _ = grace.critic_network.apply_loss_fn(
            lambda p: teacher.critic_total_loss(data, p, latent_rng))
        grace.target_update(expected, 'critic')
        updated, _ = grace._critic_step(data, rng)
        assert_tree_equal(self, updated.action_critic_network, action_state)
        assert_tree_equal(self, updated.critic_network, expected)

    def test_analytic_per_sample_norms_and_mean_of_ratios(self):
        class Network:
            def __init__(self, scale):
                self.scale = scale
            def select(self, name):
                return lambda obs, actions: jnp.stack([
                    self.scale * obs[:, 0] * actions.sum(-1),
                    3 * self.scale * obs[:, 0] * actions.sum(-1)])
        dummy = SimpleNamespace(critic_network=Network(1.), action_critic_network=Network(2.),
            config=dict(q_agg='mean', gradient_delta_z=3., gradient_delta_a=4.,
                        gradient_ratio_epsilon=.1))
        dummy._aggregate_q = lambda x: GRACEAgent._aggregate_q(dummy, x)
        data = dict(observations=jnp.array([[1.], [2.]]), actions=jnp.ones((2, 2)),
                    true_actions=jnp.ones((2, 3)))
        z = np.array([2., 4.]) * np.sqrt(2)
        a = np.array([4., 8.]) * np.sqrt(3)
        for agg in ('mean', 'min'):
            dummy.config['q_agg'] = agg
            factor = 1 if agg == 'mean' else .5
            result = GRACEAgent.input_gradient_metrics(dummy, data)
            np.testing.assert_allclose(result['sensitivity/grad_z_norm'], (factor*z).mean())
            np.testing.assert_allclose(result['sensitivity/grad_a_norm'], (factor*a).mean())
            np.testing.assert_allclose(result['sensitivity/ratio'],
                np.mean(4*factor*a / (3*factor*z + .1)), rtol=1e-6)

    def test_extra_update_metrics_are_included(self):
        _, grace = make_pair()
        data = jax.tree.map(lambda x: x[0], batch())
        rng = jax.random.PRNGKey(9)
        once, first = grace._critic_step(data, rng)
        _, second = once._critic_step(data, rng)
        updated, combined = grace._extra_critic_update(once, data, None, False, rng, first)
        self.assertEqual(int(updated.action_critic_network.step), 3)
        for key in first:
            expected = ((first[key] + second[key]) / 2
                        if key.startswith(('sensitivity/', 'action_critic/')) else first[key])
            np.testing.assert_allclose(combined[key], expected, rtol=1e-6, atol=1e-6)

    def test_real_offline_checkpoint(self):
        cfg = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
        path = Path(load_config('grace_ogpo_square_shared')['checkpoint']['restore_bc_flow_path'])
        if not path.exists():
            self.skipTest('Shared offline checkpoint unavailable')
        dsrl = DSRLBestOfNAgent.create(1, jnp.zeros(23), jnp.zeros(7), dict(cfg))
        grace = GRACEAgent.create(1, jnp.zeros(23), jnp.zeros(7), dict(cfg))
        dsrl, grace = dsrl.restore_bc_flow_params(path), grace.restore_bc_flow_params(path)
        obs, z = jnp.ones((2, 23)), jnp.zeros((2, 28))
        np.testing.assert_allclose(dsrl.noise_to_actions_public(obs,z),
                                   grace.noise_to_actions_public(obs,z), atol=1e-6)

    def test_runner_budget_and_dsrl_logging_semantics(self):
        class Env:
            total = 0
            def reset(self, seed):
                self.steps = 0
                return np.zeros(2), {}
            def step(self, action):
                self.steps += 1
                self.total += 1
                return np.zeros(2), -1., self.steps == 5, False, {'success': 0.}
        class Dataset:
            def sample(self, shape):
                return dict(observations=np.zeros(2), actions=np.zeros(1))
        class Agent:
            updates = 0
            def sample_noise(self, **kwargs):
                return np.zeros(4)
            def noise_to_actions_public(self, obs, noise):
                return np.zeros(4)
            def sample_actions(self, **kwargs):
                return np.zeros(4)
            def batch_update(self, batch, *args):
                count = batch['actions'].shape[0]
                values = np.arange(self.updates+1, self.updates+count+1)
                self.updates += count
                return self, {'sensitivity/ratio': values.mean(), 'actor/actor_loss': values.mean()}
        class Logger:
            def __init__(self):
                self.records = []
            def log(self, values, prefix, step):
                self.records.append((prefix, dict(values), step))
        env, agent, logger = Env(), Agent(), Logger()
        with patch('ogpo.runners.grace_runner.save_agent'), patch(
                'ogpo.runners.grace_runner._run_evaluation', return_value={'success': 0.}):
            _, step, _, replay = run_grace_online_rl(agent, env, env, None, Dataset(),
                dict(encoder=None, noise_chunk_length=-1, updates_per_env_step=True,
                     chunk_reward_aggregation='discounted_sum', critic_updates_per_actor=2),
                logger, '/unused', 'grace', online_steps=47, start_training=20,
                batch_size=1, horizon_length=4, discount=.99, n_eval_envs=1,
                eval_interval=10, log_interval=10, save_interval=-1,
                force_jax_sync=False, buffer_size=100, start_step=90000)
        self.assertEqual(step, 90047)
        self.assertEqual(agent.updates, 28)
        metrics = [v for k,v,_ in logger.records if k == 'online_agent']
        self.assertEqual(len(metrics), 3)
        self.assertEqual([v['actor/actor_loss'] for v in metrics], [1., 11., 21.])
        self.assertEqual([v['sensitivity/ratio'] for v in metrics], [1., 11., 21.])
        self.assertEqual([step for k, _, step in logger.records if k == 'online_agent'],
                         [90020, 90030, 90040])
        self.assertIn('true_actions', replay)
        budget = [v for k,v,_ in logger.records if k == 'online_budget'][-1]
        self.assertEqual(budget['critic_gradient_updates'], 56)


if __name__ == '__main__':
    unittest.main()
