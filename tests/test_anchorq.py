"""CPU regressions for AnchorQ's equations, shared decoder, and online budget."""

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.anchorq import AnchorQAgent, cardinal_reference_actions, value_tilted_barycenters
from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.runners.anchorq_runner import PriorityIndex, run_anchorq_online_rl
from ogpo.utils.config_loader import config_to_agent_dict, load_config
from ogpo.utils.main_helper import training_log_prefixes


def make_agent(**overrides):
    cfg = config_to_agent_dict(load_config('anchorq_ogpo_square_shared'))
    cfg.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
               anchorq_particle_count=2, anchorq_particle_warmup_decisions=0,
               anchorq_query_batch_size=16)
    cfg.update(overrides)
    return AnchorQAgent.create(1, jnp.zeros(23), jnp.zeros(7), cfg)


class ConstantFlow(nn.Module):
    @nn.compact
    def __call__(self, observations, actions, times, is_encoded=True):
        drift = self.param('drift', nn.initializers.constant(0.1), (actions.shape[-1],))
        return jnp.broadcast_to(drift, actions.shape)


class AnchorQTests(unittest.TestCase):
    def test_prioritized_replay_distribution_and_weights(self):
        tree = PriorityIndex(5, alpha=1., epsilon=1e-6)
        for i in range(5):
            tree.insert(i)
        tree.update(np.arange(5), np.arange(1., 6.))
        idx, weights = tree.sample(150000, np.random.default_rng(8), beta=1.)
        expected = (np.arange(1., 6.) + 1e-6) / (15. + 5e-6)
        np.testing.assert_allclose(np.bincount(idx) / len(idx), expected, atol=2e-5)
        np.testing.assert_allclose(weights, (1. + 1e-6) / (idx + 1. + 1e-6), rtol=1e-6)
        tree.update([1, 1], [2., 9.])
        self.assertAlmostEqual(tree.total[tree.width + 1], 9.000001)
        tree.insert(0)  # ring-buffer overwrite takes current maximum priority
        self.assertAlmostEqual(tree.total[tree.width], 9.000001)
        self.assertAlmostEqual(tree.total[1], tree.total[tree.width:].sum())
        self.assertAlmostEqual(tree.minimum[1], 3.000001)

    def test_barycenter_and_uncertainty_equations(self):
        particles = jnp.array([[[0.], [1.], [2.]]])
        values = jnp.array([[[0., 1., 2.]], [[0., 1., 2.]]])
        refined, weights, beta = value_tilted_barycenters(particles, values, beta=1., adaptive=False)
        expected = np.exp([0., 1., 2.]) / np.exp([0., 1., 2.]).sum()
        np.testing.assert_allclose(weights[0], expected, rtol=1e-6)
        self.assertAlmostEqual(float(refined[0, 0]), float(expected @ np.arange(3)), places=6)
        _, uniform, _ = value_tilted_barycenters(particles, values, beta=0., adaptive=False)
        np.testing.assert_allclose(uniform, 1/3)
        # Std over heads = 2, median over the cloud = 2, beta = 1/2.
        heads = values + jnp.array([-2., 2.])[:, None, None]
        _, _, beta = value_tilted_barycenters(particles, heads, adaptive=True)
        np.testing.assert_allclose(beta, .5)
        _, _, beta = value_tilted_barycenters(particles, values, adaptive=True, uncertainty_floor=.1)
        np.testing.assert_allclose(beta, 10.)

    def test_cardinal_commands_and_reverse_direction(self):
        refs = cardinal_reference_actions(7, 4).reshape(6, 4, 7)
        np.testing.assert_array_equal(refs[:, 0, :3], np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]))
        np.testing.assert_array_equal(refs[..., 3:], 0.)
        agent = make_agent()
        model = ModuleDict({'actor_bc_flow': ConstantFlow()})
        params = model.init(jax.random.PRNGKey(5), actor_bc_flow=(jnp.zeros(23), jnp.zeros(28), jnp.zeros(1)))['params']
        agent = agent.replace(bc_flow_network=TrainState.create(model, params))
        obs, actions = jnp.zeros((2, 23)), jnp.full((2, 28), .25)
        latent = agent.reverse_flow_actions(obs, actions)
        np.testing.assert_allclose(latent, .15, atol=1e-6)
        np.testing.assert_allclose(agent.compute_flow_actions(obs, noises=latent), actions, atol=1e-6)
        # Discrete reverse integration evaluates decreasing t; it is not forward flow.
        np.testing.assert_allclose(agent.reverse_flow_actions(obs[0], actions[0]), latent[0])

    def test_refinement_from_first_decision_and_fresh_defer(self):
        agent = make_agent(anchorq_particle_warmup_decisions=10)
        obs = jnp.zeros((2, 23))
        bank = agent.build_anchor_bank(obs, jax.random.PRNGKey(1))
        other = agent.build_anchor_bank(obs, jax.random.PRNGKey(2))
        self.assertEqual(bank.shape, (2, 7, 28))
        np.testing.assert_allclose(bank[:, :6], other[:, :6])
        self.assertFalse(np.array_equal(bank[:, -1], other[:, -1]))
        first = agent.refine_anchors(obs, bank, jax.random.PRNGKey(3))
        active = agent.replace(decision_step=jnp.asarray(10))
        refined = active.refine_anchors(obs, bank, jax.random.PRNGKey(3))
        np.testing.assert_array_equal(first, refined)
        self.assertTrue(np.isfinite(refined).all())
        self.assertGreater(float(jnp.linalg.norm(refined - bank)), 0.)
        selected, _ = active.select_latent(obs, bank, jax.random.PRNGKey(4))
        self.assertEqual(selected.shape, (2, 28))
        self.assertEqual(active.sample_actions(obs[0], jax.random.PRNGKey(4)).shape, (28,))
        self.assertAlmostEqual(float(agent.exploration_epsilon()), 1.)
        self.assertAlmostEqual(float(agent.replace(decision_step=jnp.asarray(10000)).exploration_epsilon()), .05, places=6)

    def test_candidate_batching_and_normalization(self):
        agent = make_agent()
        obs, noise = jnp.ones((3, 23)), jnp.zeros((3, 7, 3, 28))
        actual = agent._candidate_qs(obs, noise)
        direct = agent.critic_values(
            jnp.broadcast_to(obs[:, None, None, :], (3, 7, 3, 23)), noise)
        np.testing.assert_allclose(actual, direct, atol=1e-6)
        configured = agent.configure_reference_actions(-np.ones(7), np.ones(7))
        np.testing.assert_array_equal(configured.reference_actions, agent.reference_actions)

    def test_td_targets_cached_anchors_and_frozen_decoder(self):
        agent = make_agent(anchorq_target_aggregation='min')
        obs = jnp.ones((2, 23))
        banks = agent.build_anchor_bank(obs, jax.random.PRNGKey(3))
        batch = dict(observations=obs, next_observations=obs,
                     actions=jnp.zeros((2, 28)), next_anchors=banks,
                     rewards=jnp.full(2, 5.), masks=jnp.zeros(2), chunk_length=jnp.full(2, 4))
        with patch.object(AnchorQAgent, 'build_anchor_bank', side_effect=AssertionError('must use cache')):
            _, info = agent.critic_loss(batch, agent.critic_network.params, jax.random.PRNGKey(9))
        self.assertAlmostEqual(float(info['target_q_mean']), 5.)
        # With two heads, REDQ must take their minimum, including frozen priors.
        next_latent, _ = agent.select_latent(
            obs, banks, jax.random.split(jax.random.PRNGKey(9))[0], epsilon=0.)
        expected_target = 5. + .99 ** 4 * agent.critic_values(obs, next_latent, target=True).min(axis=0)
        _, bootstrapped = agent.critic_loss(
            dict(batch, masks=jnp.ones(2)), agent.critic_network.params, jax.random.PRNGKey(9))
        self.assertAlmostEqual(float(bootstrapped['target_q_mean']), float(expected_target.mean()), places=5)
        stacked = jax.tree.map(lambda x: x[None], batch)
        updated, info = agent.batch_update(stacked, stacked, True)
        self.assertEqual(int(updated.critic_network.step), int(agent.critic_network.step) + 2)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(updated.critic_network.params)))
        for old, new in zip(jax.tree.leaves(agent.bc_flow_network), jax.tree.leaves(updated.bc_flow_network)):
            np.testing.assert_array_equal(old, new)
        self.assertFalse(hasattr(updated, 'actor_network'))
        self.assertIsNone(updated.prior_network.tx)
        for old, new in zip(jax.tree.leaves(agent.prior_network.params), jax.tree.leaves(updated.prior_network.params)):
            np.testing.assert_array_equal(old, new)
        residual = agent.critic_network.select('critic')(obs, actions=batch['actions'])
        prior = agent.prior_network.select('critic')(obs, actions=batch['actions'])
        np.testing.assert_allclose(agent.critic_values(obs, batch['actions']), residual + 2 * prior)
        _, zero_info, errors = agent.critic_update(dict(batch, weights=jnp.zeros(2)))
        self.assertEqual(float(zero_info['critic_loss']), 0.)
        self.assertEqual(errors.shape, (2,))
        self.assertTrue(np.all(np.asarray(errors) > 0))

    def test_checkpoint_import_and_matching_network_initialization(self):
        agent = make_agent()
        base = DSRLBestOfNAgent.create(1, jnp.zeros(23), jnp.zeros(7), dict(agent.config))
        for actual, expected in zip(jax.tree.leaves(agent.critic_network.params), jax.tree.leaves(base.critic_network.params)):
            np.testing.assert_array_equal(actual, expected)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'bc.pkl'
            params = base.bc_flow_network.params['modules_actor_bc_flow']
            with file.open('wb') as f:
                pickle.dump({'agent': {'actor_network': {'params': {'modules_actor': params}}}}, f)
            loaded = agent.restore_bc_flow_params(file)
            self.assertTrue(bool(loaded.decoder_loaded))
            for actual, expected in zip(jax.tree.leaves(loaded.bc_flow_network.params['modules_actor_bc_flow']), jax.tree.leaves(params)):
                np.testing.assert_array_equal(actual, expected)

    def test_shared_config_and_logging(self):
        baseline = load_config('dsrl_ogpo_square_shared')
        actual = load_config('anchorq_ogpo_square_shared')
        for group in ['training', 'buffer', 'env', 'eval', 'intervals', 'checkpoint', 'success_buffer']:
            self.assertEqual(actual[group], baseline[group])
        for key, value in baseline['agent'].items():
            if key != 'agent_name':
                self.assertEqual(actual['agent'][key], value)
        self.assertIn('online_budget', training_log_prefixes('anchorq', 2_000_000))
        self.assertEqual(load_config('anchorq')['agent']['anchorq_target_aggregation'], 'min')

    def test_real_shared_checkpoint_decoder_outputs(self):
        cfg = load_config('anchorq_ogpo_square_shared')
        checkpoint = Path(cfg['checkpoint']['restore_bc_flow_path'])
        if not checkpoint.is_file():
            self.skipTest('Shared square BC checkpoint is not available')
        from ogpo.networks import ActorVectorField
        from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
        from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
        agent = AnchorQAgent.create(1, jnp.zeros(23), jnp.zeros(7), config_to_agent_dict(cfg))
        loaded = agent.restore_bc_flow_params(checkpoint)
        with checkpoint.open('rb') as f:
            params = pickle.load(f)['agent']['actor_network']['params']['modules_actor']
        for actual, expected in zip(jax.tree.leaves(loaded.bc_flow_network.params['modules_actor_bc_flow']),
                                    jax.tree.leaves(params)):
            np.testing.assert_array_equal(actual, expected)
        model = ActorVectorField(hidden_dims=(512,) * 4, action_dim=28, layer_norm=False,
                                 time_embedding=SinusoidalTimeEmbedding(embed_dim=32))
        obs = jax.random.normal(jax.random.PRNGKey(71), (2, 23))
        noise = jax.random.normal(jax.random.PRNGKey(72), (2, 28))
        expected = sample_flow_actions_ode(
            actor_fn=lambda o, a, t, **kw: model.apply({'params': params}, o, a, t),
            observations=obs, rng=None, flow_steps=10, act_dim=28,
            clip_intermediate=False, noises=noise)
        np.testing.assert_allclose(loaded.compute_flow_actions(obs, noises=noise), expected,
                                   atol=1e-5, rtol=1e-5)
        banks = loaded.build_anchor_bank(obs, jax.random.PRNGKey(73))
        self.assertTrue(np.isfinite(banks).all())
        # Compare the actual GRACE inverse implementation on exactly the same
        # frozen checkpoint/state. It only accesses config and bc_flow_network;
        # no GRACE actor or critic is involved in this numerical operation.
        from ogpo.agents.grace import GRACEAgent
        actions = jnp.clip(noise, -1., 1.)
        anchorq_inverse = loaded.reverse_flow_actions(obs, actions)
        grace_inverse = GRACEAgent.inverse_fm_batch(loaded, obs, actions)
        np.testing.assert_allclose(anchorq_inverse, grace_inverse, atol=1e-5, rtol=1e-5)
        # AnchorQ also broadcasts over six reference commands. Flattening those
        # leading axes must agree with GRACE's strictly 2D batch interface.
        ref_obs = jnp.broadcast_to(obs[:, None, :], (2, 6, 23))
        ref_actions = jnp.broadcast_to(loaded.reference_actions, (2, 6, 28))
        reference_inverse = loaded.reverse_flow_actions(ref_obs, ref_actions)
        grace_reference_inverse = GRACEAgent.inverse_fm_batch(
            loaded, ref_obs.reshape(-1, 23), ref_actions.reshape(-1, 28)).reshape(2, 6, 28)
        np.testing.assert_allclose(reference_inverse, grace_reference_inverse, atol=1e-5, rtol=1e-5)
        print('AnchorQ/GRACE inverse max absolute error:',
              float(jnp.max(jnp.abs(anchorq_inverse - grace_inverse))),
              '(reference bank:',
              float(jnp.max(jnp.abs(reference_inverse - grace_reference_inverse))), ')')

    def test_runner_physical_budget_terminal_truncation_and_cache(self):
        class Environment:
            total = 0
            episode = -1
            def reset(self, seed=None):
                self.t = 0
                self.episode += 1
                return np.zeros(23, np.float32), {}
            def step(self, action):
                self.total += 1
                self.t += 1
                end = self.t == 7
                terminal = end and self.episode % 2 == 0
                return np.full(23, self.t, np.float32), (5. if terminal else -1.), terminal, end and not terminal, {'success': float(terminal)}
        class Dataset:
            def sample(self, shape):
                return dict(observations=np.zeros(23, np.float32), actions=np.zeros(7, np.float32))
        class Logger:
            def __init__(self):
                self.rows = []
            def log(self, data, prefix, step):
                self.rows.append((prefix, data))
        agent = make_agent(anchorq_particle_count=0).replace(decoder_loaded=jnp.asarray(True))
        env, logger = Environment(), Logger()
        with tempfile.TemporaryDirectory() as directory, patch(
                'ogpo.runners.anchorq_runner.save_agent'), patch(
                'ogpo.runners.anchorq_runner._run_evaluation', return_value={'success': 0.}):
            result, step, _, replay = run_anchorq_online_rl(
                agent, env, env, None, Dataset(), dict(agent.config), logger, directory,
                online_steps=47, start_training=20, batch_size=1, horizon_length=4,
                use_success_buffer=True, success_buffer_batch_size=1,
                n_eval_envs=1, eval_interval=10, log_interval=10, save_interval=-1,
                buffer_size=100, start_step=90000)
        self.assertEqual(env.total, 47)
        self.assertEqual(step, 90047)
        self.assertEqual(int(result.critic_network.step) - int(agent.critic_network.step), 2 * (47 - 20))
        budget = [data for prefix, data in logger.rows if prefix == 'online_budget'][-1]
        self.assertEqual(budget['critic_gradient_updates'], 54)
        self.assertEqual(budget['actor_gradient_updates'], 0)
        self.assertLess(budget['anchor_bank_constructions'], 2 * budget['decisions'])
        size = replay.size
        terminal = replay['masks'][:size] == 0
        truncated = (replay['terminals'][:size] == 1) & ~terminal
        self.assertTrue(terminal.any() and truncated.any())
        np.testing.assert_array_equal(replay['next_anchors'][:size][terminal], 0)
        self.assertTrue(np.any(replay['next_anchors'][:size][truncated] != 0))
        self.assertEqual(int(replay['chunk_length'][:size].sum()), 47)
        self.assertIn(1, replay['chunk_length'][:size])
        for i in range(size):
            length = int(replay['chunk_length'][i])
            rewards = [-1.] * length
            if terminal[i]:
                rewards[-1] = 5.
            expected = sum(.99 ** t * reward for t, reward in enumerate(rewards))
            self.assertAlmostEqual(float(replay['rewards'][i]), expected, places=5)


if __name__ == '__main__':
    unittest.main()
