"""GRACE parity with DSRL and per-sample input-gradient diagnostics."""
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

    def test_exact_dsrl_updates_and_frozen_decoder(self):
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
            dsrl, dsrl_metrics = dsrl.batch_update(data, data if success else None, success)
            grace, metrics = grace.batch_update(data, data if success else None, success)
            for key, value in dsrl_metrics.items():
                np.testing.assert_allclose(metrics[key], value, atol=1e-6, rtol=1e-6,
                                           err_msg=f'Legacy metric changed: {key}')
            assert_tree_equal(self, dsrl.actor_network, grace.actor_network)
            assert_tree_equal(self, dsrl.critic_network, grace.critic_network)
            assert_tree_equal(self, frozen, grace.bc_flow_network)
            self.assertEqual(int(grace.action_critic_network.step), int(grace.critic_network.step))
            for key in ('grad_z_norm', 'grad_a_norm', 'ratio'):
                self.assertTrue(np.isfinite(metrics['sensitivity/' + key]))
        self.assertEqual(int(grace.critic_network.step), 9)  # 2 scans * 2 Q steps * 2 calls + init

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
