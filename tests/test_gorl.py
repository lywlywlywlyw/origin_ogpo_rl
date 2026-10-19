"""GoRL invariants: shared BC, alternating gradients, likelihoods and budgets."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
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
from ogpo.agents.gorl import GoRLAgent, reference_gae
from ogpo.runners.gorl_runner import (run_gorl_online_rl,
                                       stage_boundaries, _decoder_split)
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.main_helper import training_log_prefixes


def create(small=True, **changes):
    config = config_to_agent_dict(load_config('gorl_ogpo_square_shared'))
    if small:
        config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                      gorl_samples_per_action=2, gorl_ppo_epochs=2,
                      gorl_unroll_length=3, gorl_ppo_batch_size=2,
                      gorl_rollout_max_env_steps=12, gorl_decoder_batch_size=2,
                      gorl_decoder_eval_batches=2, gorl_stage_weights=(1, 1),
                      gorl_collection_steps=12, gorl_decoder_steps=3,
                      gorl_decoder_eval_interval=1)
    config.update(changes)
    return GoRLAgent.create(1, jnp.zeros(3 if small else 23), jnp.zeros(1 if small else 7), config)


def same_tree(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


class GoRLTests(unittest.TestCase):
    def test_real_shared_checkpoint_and_decoder_matches_dsrl(self):
        config = load_config('gorl_ogpo_square_shared')
        path = Path(config['checkpoint']['restore_bc_flow_path'])
        if not path.exists():
            self.skipTest('Actual shared checkpoint unavailable')
        agent = create(False).restore_bc_flow_params(path)
        dsrl_config = config_to_agent_dict(load_config('dsrl_ogpo_square_shared'))
        dsrl = DSRLBestOfNAgent.create(1, jnp.zeros(23), jnp.zeros(7), dsrl_config)
        dsrl = dsrl.restore_bc_flow_params(path)
        same_tree(agent.bc_flow_network.params, dsrl.bc_flow_network.params)
        obs = jax.random.normal(jax.random.PRNGKey(8), (2, 23))
        noise = jax.random.normal(jax.random.PRNGKey(9), (2, 28))
        np.testing.assert_allclose(agent.compute_flow_actions(obs, noises=noise),
            dsrl.compute_flow_actions(obs, noises=noise), atol=1e-6)
        self.assertTrue(bool(agent.decoder_loaded))
        source = config_to_agent_dict(load_config('dsrl_ogpo_square_shared'))
        dsrl_small = DSRLBestOfNAgent.create(1, jnp.zeros(3), jnp.zeros(1),
            {**source, 'actor_hidden_dims': (8, 8), 'value_hidden_dims': (8, 8), 'num_qs': 2})
        small = create()
        # Actor parameter structure matches DSRL, despite Gaussian vs tanh output.
        self.assertEqual(jax.tree.structure(small.actor_network.params['modules_actor']),
                         jax.tree.structure(dsrl_small.actor_network.params['modules_actor']))

    def test_prior_reset_and_alternating_frozen_parameter_groups(self):
        agent = create().replace(decoder_loaded=jnp.asarray(True))
        obs = jnp.array([[1., 2., 3.], [-2., 4., 1.]])
        prior = agent.actor_network.select('actor')(obs)
        self.assertTrue(np.isfinite(prior.mean()).all())
        self.assertTrue((np.asarray(prior.stddev()) > 0).all())
        z, log_prob, _ = agent.sample_latent(obs, jax.random.PRNGKey(5))
        np.testing.assert_allclose(agent.sample_actions(obs, rng=jax.random.PRNGKey(5)),
                                   agent.compute_flow_actions(obs, noises=z), atol=1e-6)
        batch = dict(observations=obs[None], next_observations=obs[None],
            actions=z[None], old_log_probs=log_prob[None], rewards=jnp.array([[2., -2.]]),
            masks=jnp.zeros((1, 2)), truncations=jnp.zeros((1, 2)),
            chunk_length=jnp.ones((1, 2)), valid=jnp.ones((1, 2)))
        updated, info = agent.ppo_update(jax.tree.map(lambda x: x[None], batch))
        same_tree(agent.bc_flow_network, updated.bc_flow_network)
        self.assertEqual(int(updated.actor_network.step), int(agent.actor_network.step) + 1)
        self.assertGreater(float(info['approximate_kl']), -1e-6)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(agent.actor_network.params), jax.tree.leaves(updated.actor_network.params))))
        data = dict(observations=obs, true_actions=jnp.full((2, 4), .5), action_valid=jnp.ones((2, 4)))
        refined, info = updated.decoder_update(data)
        same_tree(updated.actor_network, refined.actor_network)
        same_tree(updated.critic_network, refined.critic_network)
        self.assertEqual(int(refined.bc_flow_network.step), int(updated.bc_flow_network.step) + 1)
        reset = refined.reset_encoder(1)
        same_tree(refined.bc_flow_network, reset.bc_flow_network)
        np.testing.assert_array_equal(reset.obs_mean, np.zeros(3))
        np.testing.assert_array_equal(reset.obs_std, np.ones(3))
        self.assertEqual(int(reset.actor_network.step), 1)
        restored = flax.serialization.from_state_dict(create(), flax.serialization.to_state_dict(reset))
        same_tree(reset, restored)

    def test_fixed_prior_cfm_formula_independent_of_rollout_latents(self):
        agent = create()
        obs, actions = jnp.zeros((2, 3)), jnp.full((2, 4), .25)
        data = dict(observations=obs, true_actions=actions,
                    action_valid=jnp.array([[1., 1., 0., 0.], [1., 1., 1., 1.]]))
        key = jax.random.PRNGKey(10)
        nk, tk = jax.random.split(key)
        noise = jax.random.normal(nk, (2, 2, 4))
        t = jax.random.uniform(tk, (2, 2, 1))
        x = (1. - t) * noise + t * actions[:, None]
        pred = agent.bc_flow_network.select('actor_bc_flow')(jnp.zeros((2, 2, 3)), x, t)
        expected = ((pred - (actions[:, None] - noise)) ** 2 * data['action_valid'][:, None]).sum() / 12.
        actual, _ = agent.decoder_loss(data, agent.bc_flow_network.params, key)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        changed, _ = agent.decoder_loss({**data, 'actions': jnp.ones((2, 4)) * 99},
                                        agent.bc_flow_network.params, key)
        np.testing.assert_array_equal(actual, changed)
        fresh = agent.reset_decoder(obs)
        probe = jnp.array([[.2, -.3, .4, -.5], [.1, .2, -.2, .3]])
        np.testing.assert_allclose(fresh.compute_flow_actions(obs, noises=probe), probe, atol=1e-6)
        same_tree(agent.actor_network, fresh.actor_network)

    def test_gae_termination_timeout_and_chunk_discount(self):
        rewards = np.array([1., 2., 3.])
        values = np.array([.2, .3, .4])
        next_values = np.array([.3, 9., 8.])
        masks = np.array([1., 1., 0.])
        dones = np.array([0., 1., 1.])
        lengths = np.array([4, 2, 1])
        ret, adv = reference_gae(jnp.asarray(rewards), jnp.asarray(values),
            jnp.asarray(next_values), jnp.asarray(.9 ** lengths * masks),
            jnp.array([0., 1., 0.]), jnp.ones(3), .8)
        expected_returns = np.array([1. + .9 ** 4 * .3, .3, 3.])
        expected_adv = expected_returns - values
        np.testing.assert_allclose(ret, expected_returns, rtol=1e-6)
        np.testing.assert_allclose(adv, expected_adv, rtol=1e-6)
        self.assertEqual(stage_boundaries(2_000_000, [2, 2, 1, 1])[-1], 2_000_000)

    def test_tiny_runner_all_collection_counted_and_alternation_order(self):
        class Env:
            def reset(self, seed):
                self.steps = 0
                return np.zeros(3), {}
            def step(self, action):
                self.steps += 1
                return np.full(3, self.steps / 10), -1., False, self.steps == 5, {'success': 0.}
        class Dataset:
            def sample(self, _):
                return dict(observations=np.zeros(3), actions=np.zeros(1))
        class Logger:
            def __init__(self): self.records = []
            def log(self, data, prefix, step): self.records.append((prefix, dict(data), step))
        agent = create().replace(decoder_loaded=jnp.asarray(True))
        env, logger = Env(), Logger()
        with tempfile.TemporaryDirectory() as directory, patch(
            'ogpo.runners.gorl_runner._run_evaluation', return_value={'success': 0., 'return': -5.}):
            updated, step, _, _ = run_gorl_online_rl(agent, env, env, None, Dataset(),
                agent.config, logger, directory, online_steps=72, start_training=0,
                batch_size=2, horizon_length=4, eval_interval=10, n_eval_envs=1,
                log_interval=10, save_interval=-1)
            self.assertEqual(step, 90072)
            budget = [data for prefix, data, _ in logger.records if prefix == 'online_budget'][-1]
            self.assertEqual(budget['env_steps'], 72)
            self.assertEqual(budget['warmup_env_steps'], 0)
            self.assertEqual(budget['decoder_collection_env_steps'], 12)
            self.assertEqual(budget['encoder_env_steps'], 60)
            self.assertEqual(budget['decoder_updates'], 3)
            self.assertEqual(budget['encoder_gradient_updates'], budget['value_gradient_updates'])
            self.assertEqual(int(updated.stage), 1)
            self.assertTrue((Path(directory) / 'gorl_protocol.json').exists())
            with (Path(directory) / 'gorl_stage0_refined' / 'params_90036.pkl').open('rb') as f:
                boundary = pickle.load(f)['agent']
            self.assertEqual(int(boundary['stage']), 1)
            self.assertEqual(int(boundary['actor_network']['step']), 1)
            data = dict(np.load(Path(directory) / 'gorl_decoder_rollout_stage0.npz'))
            self.assertEqual(data['chunk_length'].sum(), 12)
            train, val = _decoder_split(data, .1, np.random.default_rng(1))
            self.assertFalse(set(data['episode_id'][train]) & set(data['episode_id'][val]))
            self.assertEqual(len([1 for p, _, _ in logger.records if p == 'gorl_decoder']), 3)

    def test_config_comparison_contract_and_fail_closed(self):
        gorl, dsrl = load_config('gorl_ogpo_square_shared'), load_config('dsrl_ogpo_square_shared')
        for section in ('env', 'eval', 'dataset', 'checkpoint', 'intervals'):
            self.assertEqual(gorl[section], dsrl[section], section)
        self.assertEqual(gorl['training']['online_steps'], dsrl['training']['online_steps'])
        for key in ('actor_hidden_dims', 'horizon_length', 'discount',
                    'flow_steps', 'time_embedding', 'time_embedding_dim', 'batch_size'):
            self.assertEqual(gorl['agent'][key], dsrl['agent'][key], key)
        self.assertFalse(gorl['success_buffer']['use_success_buffer'])
        self.assertIn('gorl_decoder', training_log_prefixes('gorl', 2_000_000))
        agent = create()
        with self.assertRaisesRegex(ValueError, 'on-policy'):
            agent.batch_update({})
        with self.assertRaisesRegex(ValueError, 'Restore'):
            run_gorl_online_rl(agent, None, None, None, None, agent.config, None, '/tmp')


if __name__ == '__main__':
    unittest.main()
