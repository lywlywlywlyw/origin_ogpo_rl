"""Numerical PSS geometry, frozen checkpoint interoperability and budget tests."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flax
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.pss import PSSAgent, estimate_response_subspace
from ogpo.runners.pss_rl_runner import run_pss_online_rl
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.main_helper import training_log_prefixes


def make_agent(small=True):
    config = config_to_agent_dict(load_config('pss_ogpo_square_shared'))
    if small:
        config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8),
                      num_qs=2, pss_dim=2, pss_num_probes=2)
    return PSSAgent.create(1, jnp.zeros(23), jnp.zeros(7), config)


def fitted_small_agent():
    agent = make_agent()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'bc.pkl'
        with path.open('wb') as file:
            pickle.dump({'agent': {'actor_network': {'params': {
                'modules_actor': agent.bc_flow_network.params['modules_actor_bc_flow']}}}}, file)
        agent = agent.restore_bc_flow_params(path)
    return agent.fit_subspace(np.zeros((2, 23)),
                             np.random.default_rng(3).normal(size=(2, 28)))[0]


class ResponseGeometryTests(unittest.TestCase):
    def test_linear_decoder_gram_orientation_and_maximum_response(self):
        # Non-square decoder and nonzero constant responses expose transposes
        # and erroneous centering of the response matrix.
        matrix = np.array([[3., 0., 0.], [0., 2., 0.]], np.float32)
        calls = []
        def decode(obs, noise):
            calls.append(len(noise))
            return noise @ matrix.T + obs[:, :2]
        observations = np.ones((4, 2), np.float32)
        base = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
        basis, values, gram = estimate_response_subspace(decode, observations, base,
                                                        delta=.05, batch_size=2)
        np.testing.assert_allclose(gram, matrix.T @ matrix, atol=3e-5)
        np.testing.assert_allclose(values, [9., 4., 0.], atol=3e-5)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(abs(basis[:, 0]), [1., 0., 0.], atol=1e-6)
        self.assertEqual(sum(calls), 4 * (3 + 1))
        self.assertLessEqual(max(calls), 2)

    def test_independent_complement_and_gaussian_prior(self):
        agent = make_agent()
        rotation, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(28, 28)))
        agent = agent.replace(steering_basis=jnp.asarray(rotation, dtype=jnp.float32),
                              basis_ready=jnp.asarray(True))
        n = 10000
        z = jax.random.normal(jax.random.PRNGKey(1), (n, 2))
        w1 = np.asarray(agent.reconstruct_noise(z, jax.random.PRNGKey(2)))
        w2 = np.asarray(agent.reconstruct_noise(z, jax.random.PRNGKey(3)))
        np.testing.assert_allclose(agent.project_noise(w1), z, atol=2e-6)
        np.testing.assert_allclose((w1 - w2) @ rotation[:, :2], 0., atol=2e-6)
        self.assertGreater(np.std(w1 - w2), .5)
        np.testing.assert_allclose(w1.mean(0), 0., atol=.04)
        np.testing.assert_allclose(np.cov(w1.T), np.eye(28), atol=.06)
        full = agent.replace(config=agent.config.copy({'pss_dim': 28}))
        np.testing.assert_allclose(full.reconstruct_noise(jnp.ones(28), jax.random.PRNGKey(4)),
                                   rotation @ np.ones(28), atol=1e-6)

    def test_require_loaded_checkpoint_and_fixed_basis(self):
        agent = make_agent()
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            agent.fit_subspace(np.zeros((2, 23)), np.zeros((2, 28)))
        with self.assertRaisesRegex(ValueError, 'basis'):
            agent.sample_actions(jnp.zeros(23), jax.random.PRNGKey(1))
        fitted = fitted_small_agent()
        with self.assertRaisesRegex(ValueError, 'refit'):
            fitted.fit_subspace(np.zeros((2, 23)), np.zeros((2, 28)))
        with self.assertRaisesRegex(ValueError, 'frozen'):
            fitted.bc_update({})


class CheckpointAndUpdateTests(unittest.TestCase):
    def test_same_real_ogpo_and_dsrl_decoder_outputs(self):
        config = load_config('pss_ogpo_square_shared')
        path = Path(config['checkpoint']['restore_bc_flow_path'])
        if not path.exists():
            self.skipTest('Real offline checkpoint not available')
        pss = make_agent(False).restore_bc_flow_params(path)
        dsrl = DSRLBestOfNAgent.create(1, jnp.zeros(23), jnp.zeros(7),
            config_to_agent_dict(load_config('dsrl_ogpo_square_shared'))).restore_bc_flow_params(path)
        with path.open('rb') as file:
            original = pickle.load(file)['agent']['actor_network']['params']['modules_actor']
        for actual, expected in zip(jax.tree.leaves(pss.bc_flow_network.params['modules_actor_bc_flow']),
                                     jax.tree.leaves(original)):
            np.testing.assert_array_equal(actual, expected)
        observations = jax.random.normal(jax.random.PRNGKey(1), (3, 23))
        noise = jax.random.normal(jax.random.PRNGKey(2), (3, 28))
        np.testing.assert_allclose(pss.compute_flow_actions(observations, noises=noise),
                                   dsrl.compute_flow_actions(observations, noises=noise), atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            # DSRL serialization layout, including a different online actor, must
            # still import only the frozen decoder.
            dsrl_path = Path(directory) / 'dsrl.pkl'
            with dsrl_path.open('wb') as file:
                pickle.dump({'agent': flax.serialization.to_state_dict(dsrl)}, file)
            imported = pss.restore_bc_flow_params(dsrl_path)
            for x, y in zip(jax.tree.leaves(pss.bc_flow_network.params),
                            jax.tree.leaves(imported.bc_flow_network.params)):
                np.testing.assert_array_equal(x, y)

    def test_online_updates_preserve_decoder_basis_and_checkpoint_roundtrip(self):
        agent = fitted_small_agent()
        batch = dict(observations=jnp.zeros((2, 3, 23)),
                     next_observations=jnp.ones((2, 3, 23)),
                     actions=jnp.zeros((2, 3, 2)), rewards=-jnp.ones((2, 3)),
                     masks=jnp.ones((2, 3)), chunk_length=jnp.full((2, 3), 4))
        for success in (False, True):
            updated, info = agent.batch_update(batch, batch if success else None, success)
            self.assertEqual(int(updated.actor_network.step) - int(agent.actor_network.step), 2)
            self.assertEqual(int(updated.critic_network.step) - int(agent.critic_network.step), 4)
            self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(info)))
            for x, y in zip(jax.tree.leaves(agent.bc_flow_network), jax.tree.leaves(updated.bc_flow_network)):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(agent.steering_basis, updated.steering_basis)
        restored = flax.serialization.from_state_dict(make_agent(), flax.serialization.to_state_dict(updated))
        np.testing.assert_array_equal(restored.steering_basis, updated.steering_basis)
        self.assertTrue(bool(restored.basis_ready))
        observations, key = jnp.zeros((3, 23)), jax.random.PRNGKey(17)
        np.testing.assert_array_equal(restored.sample_actions(observations, key),
                                      updated.sample_actions(observations, key))
        self.assertEqual(restored.sample_actions(jnp.zeros(23), key, deterministic=True).shape, (28,))
        with self.assertRaisesRegex(ValueError, 'k-dimensional'):
            agent.batch_update(dict(batch, actions=jnp.zeros((2, 3, 28))))

    def test_matched_configuration_and_logging(self):
        pss, dsrl = load_config('pss_ogpo_square_shared'), load_config('dsrl_ogpo_square_shared')
        for section in ('env', 'training', 'buffer', 'eval', 'intervals', 'bc',
                        'success_buffer', 'dataset', 'checkpoint'):
            self.assertEqual(pss[section], dsrl[section], section)
        for key, value in dsrl['agent'].items():
            if key != 'agent_name':
                self.assertEqual(pss['agent'][key], value, key)
        self.assertIn('online_budget', training_log_prefixes('pss', 2_000_000))


class RunnerTests(unittest.TestCase):
    def test_warmup_probes_relabeling_terminals_and_actual_update_budgets(self):
        class Env:
            total = 0
            seeds = []
            def reset(self, seed):
                self.seeds.append(seed)
                self.steps = 0
                return np.zeros(23), {}
            def step(self, action):
                assert self.steps < 5
                self.steps += 1
                self.total += 1
                terminated = self.steps == 5
                return np.full(23, self.steps / 10.), -1., terminated, False, {'success': float(terminated)}
        class Dataset:
            def sample(self, _):
                return dict(observations=np.zeros(23), actions=np.zeros(7))
        class Logger:
            def __init__(self):
                self.records = []
            def log(self, data, prefix, step):
                self.records.append((prefix, dict(data), step))

        agent = make_agent().replace(decoder_loaded=jnp.asarray(True))
        env, logger = Env(), Logger()
        frozen = jax.tree.map(np.asarray, agent.bc_flow_network)
        with tempfile.TemporaryDirectory() as directory, patch(
                'ogpo.runners.pss_rl_runner._run_evaluation', return_value={'success': .5}):
            updated, step, _, replay = run_pss_online_rl(
                agent, env, env, None, Dataset(), agent.config, logger, directory,
                online_steps=47, start_training=20, batch_size=1, horizon_length=4,
                discount=.99, use_success_buffer=True, success_buffer_batch_size=1,
                eval_interval=20, n_eval_envs=1, log_interval=10, save_interval=-1,
                buffer_size=100, seed=1)
            metadata = json.loads((Path(directory) / 'pss_subspace.json').read_text())
            self.assertEqual(metadata['decoder_queries'], 2 * (28 + 1))
            self.assertTrue((Path(directory) / 'params_90047.pkl').exists())
        self.assertEqual(env.total, 47)
        self.assertEqual(step, 90047)
        budget = [record for prefix, record, _ in logger.records if prefix == 'online_budget'][-1]
        self.assertEqual(budget['gradient_updates'], 28)
        self.assertEqual(budget['critic_gradient_updates'], 56)
        self.assertEqual(int(updated.actor_network.step) - int(agent.actor_network.step), 28)
        self.assertEqual(int(updated.critic_network.step) - int(agent.critic_network.step), 56)
        np.testing.assert_allclose(replay['actions'][:replay.size],
            replay['initial_noise'][:replay.size] @ np.asarray(updated.steering_basis[:, :2]), atol=1e-6)
        self.assertIn(1, replay['chunk_length'][:replay.size])
        for length, reward in zip(replay['chunk_length'][:replay.size], replay['rewards'][:replay.size]):
            self.assertAlmostEqual(reward, -sum(.99 ** i for i in range(int(length))), places=5)
        for x, y in zip(jax.tree.leaves(frozen), jax.tree.leaves(updated.bc_flow_network)):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(env.seeds[:3], [10001, 10002, 10003])


if __name__ == '__main__':
    unittest.main()
