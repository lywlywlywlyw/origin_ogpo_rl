"""CPU checks for shared BC initialization and physical-step training budgets."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.networks import ActorVectorField
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
from ogpo.runners.dsrl_runner import run_dsrl_online_rl
from ogpo.utils.config_loader import load_config, config_to_agent_dict

ROOT = Path(__file__).resolve().parents[1]


def make_agent(small=False):
    config = config_to_agent_dict(load_config('dsrl_ogpo_square_shared'))
    if small:
        config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2)
    return DSRLBestOfNAgent.create(1, jnp.zeros(23), jnp.zeros(7), config)


class SharedCheckpointTests(unittest.TestCase):
    def test_real_ogpo_checkpoint_and_ode_outputs(self):
        config = load_config('dsrl_ogpo_square_shared')
        path = Path(config['checkpoint']['restore_bc_flow_path'])
        if not path.exists():
            self.skipTest('Completed OGPO run is not available')
        with path.open('rb') as f:
            saved = pickle.load(f)['agent']['actor_network']['params']['modules_actor']
        before = make_agent()
        agent = before.restore_bc_flow_params(path)
        self.assertIs(before.actor_network, agent.actor_network)
        self.assertIs(before.critic_network, agent.critic_network)
        loaded = agent.bc_flow_network.params['modules_actor_bc_flow']
        for a, b in zip(jax.tree.leaves(saved), jax.tree.leaves(loaded)):
            np.testing.assert_array_equal(a, b)
        reference = ActorVectorField(hidden_dims=(512,) * 4, action_dim=28,
            layer_norm=False, time_embedding=SinusoidalTimeEmbedding(embed_dim=32))
        obs = jax.random.normal(jax.random.PRNGKey(9), (3, 23))
        noise = jax.random.normal(jax.random.PRNGKey(10), (3, 28))
        expected = sample_flow_actions_ode(
            actor_fn=lambda o, a, t, **kw: reference.apply({'params': saved}, o, a, t),
            observations=obs, rng=None, flow_steps=10, act_dim=28,
            clip_intermediate=False, noises=noise)
        np.testing.assert_allclose(agent.compute_flow_actions(obs, noises=noise),
                                   expected, atol=1e-5, rtol=1e-5)

    def test_reject_mismatch_and_freeze_during_online_update(self):
        agent = make_agent(small=True)
        params = agent.bc_flow_network.params['modules_actor_bc_flow']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bc.pkl'
            with path.open('wb') as f:
                pickle.dump({'agent': {'actor_network': {'params': {'modules_actor': params}}}}, f)
            restored = agent.restore_bc_flow_params(path)
            batch = dict(observations=jnp.zeros((1, 2, 23)),
                next_observations=jnp.ones((1, 2, 23)), actions=jnp.zeros((1, 2, 28)),
                rewards=-jnp.ones((1, 2)), masks=jnp.ones((1, 2)),
                chunk_length=jnp.full((1, 2), 4))
            updated, _ = restored.batch_update(batch)
            for a, b in zip(jax.tree.leaves(restored.bc_flow_network),
                            jax.tree.leaves(updated.bc_flow_network)):
                np.testing.assert_array_equal(a, b)
            # The completed OGPO run also uses a conditional extra SB critic step.
            enabled = restored.replace(config=restored.config.copy({'use_success_buffer_q': True}))
            updated_sb, _ = enabled.batch_update(batch, batch, True)
            self.assertEqual(int(updated_sb.critic_network.step), int(enabled.critic_network.step) + 2)
            updated_rb, _ = restored.batch_update(batch)
            self.assertEqual(int(updated_rb.critic_network.step), int(restored.critic_network.step) + 2)
            self.assertEqual(int(updated_sb.actor_network.step), int(enabled.actor_network.step) + 1)
            for a, b in zip(jax.tree.leaves(enabled.bc_flow_network),
                            jax.tree.leaves(updated_sb.bc_flow_network)):
                np.testing.assert_array_equal(a, b)
            with path.open('wb') as f:
                pickle.dump({'agent': {'actor_network': {'params': {'modules_actor': {'bad': np.zeros(1)}}}}}, f)
            with self.assertRaisesRegex(ValueError, 'structure/shapes'):
                agent.restore_bc_flow_params(path)

    def test_config_matches_completed_run(self):
        path = ROOT/'exp/ogpo/OGPO/square/square-mh-low_dim/ogpo_squareseed1/flags.json'
        if not path.exists():
            self.skipTest('Completed OGPO run is not available')
        import json
        source = json.loads(path.read_text())
        target = load_config('dsrl_ogpo_square_shared')
        for section in ('env', 'buffer', 'eval', 'intervals', 'dataset', 'success_buffer'):
            self.assertEqual(source[section], target[section], section)
        self.assertEqual(target['training']['online_steps'], 2_000_000)
        self.assertEqual(target['training']['bc_pi_steps'], 0)
        self.assertFalse(target['checkpoint']['resume'])
        self.assertEqual(target['checkpoint']['ep_resume'], 90000)
        self.assertEqual(target['agent']['critic_updates_per_actor'], 2)
        self.assertEqual(target['agent']['lr_critic'], source['agent']['ppo_lr'])


class BudgetTests(unittest.TestCase):
    def test_partial_chunks_terminal_reset_and_update_budget(self):
        class Env:
            total = 0
            def reset(self, seed):
                self.steps = 0
                return np.zeros(2), {}
            def step(self, action):
                assert self.steps < 5, 'Stepped after terminal without resetting'
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
                self.updates += batch['actions'].shape[0]
                return self, {'loss': 0.}
        class Logger:
            records = []
            def log(self, values, prefix, step):
                self.records.append((prefix, dict(values), step))
        env, agent, logger = Env(), Agent(), Logger()
        with patch('ogpo.runners.dsrl_runner.save_agent'), patch(
                'ogpo.runners.dsrl_runner._run_evaluation', return_value={'success': 0.}):
            _, step, _, buffer = run_dsrl_online_rl(
                agent, env, env, None, Dataset(),
                dict(encoder=None, noise_chunk_length=-1, updates_per_env_step=True,
                     chunk_reward_aggregation='discounted_sum', critic_updates_per_actor=2), logger,
                '/unused', 'dsrl', online_steps=47, start_training=20,
                batch_size=1, horizon_length=4, discount=.99, utd_online=1,
                n_eval_envs=1, eval_interval=10, log_interval=10,
                save_interval=-1, force_jax_sync=False, buffer_size=100, start_step=90000)
        self.assertEqual(env.total, 47)
        self.assertEqual(step, 90047)
        self.assertEqual(agent.updates, 47 - 20 + 1)
        budget = [v for k, v, _ in logger.records if k == 'online_budget'][-1]
        self.assertEqual(budget['gradient_updates'], agent.updates)
        self.assertEqual(budget['env_steps'], 47)
        self.assertEqual(budget['critic_gradient_updates'], 2 * agent.updates)
        self.assertIn(1, buffer['chunk_length'][:buffer.size])
        for length, reward in zip(buffer['chunk_length'][:buffer.size], buffer['rewards'][:buffer.size]):
            self.assertAlmostEqual(reward, -sum(.99**t for t in range(int(length))), places=5)


class IntegrationRegressionTests(unittest.TestCase):
    def test_real_csv_logger_and_no_wandb_resume(self):
        from types import SimpleNamespace
        from ogpo.utils.main_helper import setup_experiment_logging, training_log_prefixes
        prefixes = training_log_prefixes('dsrl', 2_000_000)
        with tempfile.TemporaryDirectory() as directory:
            old_checkpoint_lookup = unittest.mock.Mock(side_effect=AssertionError('Must not resume'))
            with patch('ogpo.utils.main_helper.setup_wandb') as init_wandb, \
                 patch('wandb.run', SimpleNamespace(id='fresh', project='test-shared')), \
                 patch.dict(os.environ, {'OGPO_EXP_ROOT': directory}):
                savedir, logger, run_id = setup_experiment_logging(
                    seed=1, project='test-shared', run_group='square', wandb_name='shared',
                    save_dir='exp/ogpo', env_name='square-mh-low_dim', prefixes=prefixes,
                    checkpoint_dir_fn=old_checkpoint_lookup, config={'checkpoint': {'resume': False}})
                # Exercise the real assertion and CSV writing; bypass only remote W&B.
                logger.wandb_logger = None
                logger.log({'env_steps': 10000, 'gradient_updates': 1,
                            'critic_gradient_updates': 2, 'decoder_updates': 0},
                           'online_budget', step=100000)
                logger.close()
                self.assertTrue((Path(savedir)/'online_budget.csv').is_file())
                self.assertEqual(run_id, 'fresh')
                old_checkpoint_lookup.assert_not_called()
                self.assertNotIn('resume', init_wandb.call_args.kwargs)

    def test_ogpo_cross_episode_sequences_are_masked_out_of_td_loss(self):
        from ogpo.utils.datasets import Dataset
        from ogpo.agents.modules.q_helper import compute_td_target, compute_td_loss
        data = Dataset.create(
            observations=np.zeros((4, 2)), actions=np.zeros((4, 1)),
            next_observations=np.zeros((4, 2)), rewards=np.array([-1., 5., -1., -1.]),
            masks=np.array([1., 0., 1., 0.]), terminals=np.array([0., 1., 0., 1.]))
        batch = data.sample_sequence(1, 4, .99)
        target = compute_td_target(jnp.asarray(batch['rewards']),
                                   jnp.asarray(batch['masks']), jnp.ones(1), .99, 4)
        loss, _ = compute_td_loss(jnp.zeros((2, 1)), target,
                                 jnp.asarray(batch['valid'][:, -1]))
        self.assertEqual(float(loss), 0.)

    def test_success_sampling_at_physical_step_threshold(self):
        from ogpo.utils.datasets import ReplayBuffer
        from ogpo.runners.dsrl_runner import _sample_success_chunks
        transition = dict(observations=np.zeros(2), actions=np.zeros(4),
                          rewards=np.array(1.), chunk_length=np.array(4),
                          terminals=np.array(0.), masks=np.array(1.),
                          next_observations=np.zeros(2))
        replay = ReplayBuffer.create(transition, 100, numpy_rng=np.random.default_rng(1))
        for i in range(65):
            replay.add_transition(dict(transition, is_success=1., score=1.))
        # 65 chunks = 260 env steps; OGPO's successful-data threshold is met.
        # One four-step group of updates still needs 4 * 256 = 1024 samples.
        sampled = _sample_success_chunks(replay, 1024)
        self.assertEqual(sampled['actions'].shape, (1024, 4))
        np.testing.assert_array_equal(sampled['is_success'], np.ones(1024))


if __name__ == '__main__':
    unittest.main()
