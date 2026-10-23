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
from ogpo.runners.grace_runner import (
    run_grace_online_rl, _offline_q_chunks, _build_q_pretrain_dataset,
)
from ogpo.utils.datasets import Dataset
from ogpo.utils.config_loader import load_config, config_to_agent_dict


def make_pair():
    config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                  encoder_sphere_limit=True, use_layer_norm=False, use_encoder_clip=False)
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
    def test_mixed_distillation_gate_pairing_and_stopped_gradients(self):
        key = jax.random.PRNGKey(31)
        obs = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
        replay = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) + 100.
        data = dict(observations=obs, actions=replay)

        class ActorNetwork:
            def __init__(self, offset):
                self.offset = offset

            def select(self, name):
                def distribution(observations):
                    values = jnp.broadcast_to(observations[:, :1], (len(observations), 4))
                    return SimpleNamespace(sample=lambda seed: values + self.offset)
                return distribution

        class QNetwork:
            def __init__(self, scale):
                self.scale = scale

            def select(self, name):
                return lambda observations, actions, params=None: (
                    self.scale * actions.sum(axis=-1) + observations[:, 0])[None]

        def evaluate(no_tanh, layer_norm, offset=20., teacher_scale=2., inputs=data):
            captured = {}
            def decode(observations, noises, noises_constrained):
                captured['noises'] = noises
                return noises
            dummy = SimpleNamespace(
                config=dict(use_no_tanh=no_tanh, use_layer_norm=layer_norm),
                actor_network=ActorNetwork(offset), critic_network=QNetwork(1.),
                action_critic_network=QNetwork(teacher_scale),
                constrain_encoder_output=lambda x: x, constrain_actor_output=lambda x: x,
                noise_to_actions_public=decode, _aggregate_q=lambda q: q.mean(axis=0))
            loss, info = GRACEAgent.critic_loss(dummy, inputs, None, key)
            return loss, info, captured['noises']

        gaussian = jax.random.normal(key, replay.shape)
        for no_tanh, layer_norm in ((False, False), (True, False), (False, True)):
            _, info, noises = evaluate(no_tanh, layer_norm)
            np.testing.assert_array_equal(noises, gaussian)
            self.assertNotIn('actor_distillation_q_gap', info)

        loss, info, noises = evaluate(True, True)
        compiled_loss = jax.jit(lambda: evaluate(True, True)[0])()
        np.testing.assert_allclose(compiled_loss, loss, rtol=1e-6)
        indices = np.asarray(jax.random.permutation(jax.random.fold_in(key, 1), 8))
        replay_rows, actor_rows, prior_rows = indices[:2], indices[2:4], indices[4:]
        self.assertEqual(noises.shape, replay.shape)
        np.testing.assert_array_equal(noises[replay_rows], replay[replay_rows])
        np.testing.assert_array_equal(noises[prior_rows], gaussian[prior_rows])
        np.testing.assert_array_equal(noises[actor_rows],
                                      np.broadcast_to(np.asarray(obs[actor_rows, :1]) + 20., (2, 4)))
        np.testing.assert_allclose(loss, jnp.mean(noises.sum(axis=-1) ** 2))
        gap = -noises[actor_rows].sum(axis=-1)
        np.testing.assert_allclose(info['actor_distillation_q_gap'], gap.mean())
        np.testing.assert_allclose(info['actor_distillation_q_abs_gap'], jnp.abs(gap).mean())
        self.assertEqual(int(info['actor_distillation_count']), 2)
        self.assertEqual(float(jax.grad(lambda x: evaluate(True, True, offset=x)[0])(20.)), 0.)
        self.assertEqual(float(jax.grad(lambda x: evaluate(True, True, teacher_scale=x)[0])(2.)), 0.)
        replay_grad = jax.grad(lambda x: evaluate(
            True, True, inputs=dict(data, actions=x))[0])(replay)
        np.testing.assert_array_equal(replay_grad, jnp.zeros_like(replay))

    def test_inverse_direction_and_branching(self):
        class ConstantFlow:
            def select(self, name):
                return lambda obs, x, t: jnp.ones_like(x) * .2
        _, agent = make_pair()
        # Test the numerical methods against an exactly invertible vector field.
        dummy = SimpleNamespace(config=dict(flow_steps=4, increase_num=2, noise_add_start=2),
                                bc_flow_network=ConstantFlow())
        obs = jnp.zeros((3, 3))
        actions = jnp.full((3, 8), .3)
        z = GRACEAgent.inverse_fm_batch.__wrapped__(dummy, obs, actions)
        np.testing.assert_allclose(z, .1, atol=1e-6)
        rng = jax.random.PRNGKey(23)
        actual = GRACEAgent.augment_actions_from_z.__wrapped__(dummy, obs, z, rng)
        x = z
        for i in range(4):
            x = x + .05
            if i >= 2:
                x = jnp.repeat(x, 2, axis=0)
                rng, key = jax.random.split(rng)
                x += jax.random.normal(key, x.shape)
        np.testing.assert_allclose(actual, jnp.clip(x, -1, 1).reshape(3, 4, 8), atol=1e-6)
        # Exercise the real decoder's compiled methods too.
        real_z = agent.inverse_fm_batch(obs, actions)
        real_a = agent.augment_actions_from_z(obs, real_z, rng)
        self.assertEqual(real_a.shape, (3, 4, 8))
        self.assertTrue(np.isfinite(real_a).all())
        self.assertTrue(np.all(np.abs(real_a) <= 1.))

    def test_q_pretrain_chunks_rewards_and_frozen_states(self):
        _, agent = make_pair()
        dataset = Dataset.create(
            observations=np.arange(18, dtype=np.float32).reshape(6, 3),
            next_observations=np.arange(18, dtype=np.float32).reshape(6, 3) + .5,
            actions=np.arange(12, dtype=np.float32).reshape(6, 2) / 20,
            rewards=np.array([-1, -1, 0, -1, -1, 0], dtype=np.float32),
            terminals=np.array([0, 0, 1, 0, 0, 1], dtype=np.float32),
            masks=np.array([1, 1, 0, 1, 1, 0], dtype=np.float32))
        chunks, failures = _offline_q_chunks(dataset, np.array([0, 2, 3, 5]), 4, .99,
                                              'discounted_sum')
        np.testing.assert_array_equal(chunks['chunk_length'], [3, 1, 3, 1])
        np.testing.assert_allclose(chunks['rewards'], [-1.99 + .99**2 * 5, 5,
                                                     -1.99 + .99**2 * 5, 5])
        np.testing.assert_allclose(failures, [-2.9701, -1, -2.9701, -1])
        max_chunks, max_failures = _offline_q_chunks(
            dataset, np.array([0, 2, 3, 5]), 4, .99, 'max')
        np.testing.assert_array_equal(max_chunks['rewards'], 5.)
        np.testing.assert_array_equal(max_failures, -1.)
        shaped_dataset = Dataset.create(**dict(dataset, rewards=np.where(
            dataset['rewards'] == 0., 5., dataset['rewards']).astype(np.float32)))
        shaped_chunks, _ = _offline_q_chunks(
            shaped_dataset, np.array([0, 2, 3, 5]), 4, .99, 'discounted_sum')
        np.testing.assert_array_equal(shaped_chunks['rewards'], chunks['rewards'])
        failure_chunk, _ = _offline_q_chunks(dataset, np.array([0]), 1, .99, 'max')
        np.testing.assert_array_equal(failure_chunk['rewards'], -1.)
        np.testing.assert_array_equal(chunks['next_observations'], dataset['next_observations'][[2,2,5,5]])
        np.testing.assert_array_equal(chunks['true_actions'][1].reshape(4, 2),
                                      np.tile(dataset['actions'][2], (4, 1)))
        expanded = _build_q_pretrain_dataset(agent, dataset, dict(agent.config), 2, 1)
        self.assertEqual(expanded.size, 30)
        all_chunks, _ = _offline_q_chunks(dataset, np.arange(6), 4, .99, 'discounted_sum')
        np.testing.assert_array_equal(expanded['true_actions'][::5], all_chunks['true_actions'])
        np.testing.assert_allclose(expanded['rewards'][::5], all_chunks['rewards'])
        self.assertEqual(expanded['rewards'][10], 5.)  # original success-step sample
        self.assertEqual(expanded['rewards'][11], -1.)  # augmented success-step sample
        np.testing.assert_array_equal(expanded['masks'][::5], all_chunks['masks'])
        np.testing.assert_array_equal(expanded['terminals'][::5], all_chunks['terminals'])
        synthetic = np.flatnonzero(np.arange(expanded.size) % 5 != 0)
        np.testing.assert_array_equal(expanded['masks'][synthetic], 0.)
        np.testing.assert_array_equal(expanded['terminals'][synthetic], 1.)
        synthetic_batch = expanded.sample(len(synthetic), idxs=synthetic)
        loss_rng = jax.random.PRNGKey(23)
        loss, metrics = agent.action_critic_loss(
            synthetic_batch, agent.action_critic_network.params, loss_rng)
        predicted = agent.action_critic_network.select('critic')(
            synthetic_batch['observations'], actions=synthetic_batch['true_actions'])
        np.testing.assert_allclose(loss, jnp.mean(
            (predicted - synthetic_batch['rewards'][None]) ** 2), rtol=1e-6)
        np.testing.assert_allclose(metrics['action_critic/target_q_mean'],
                                   synthetic_batch['rewards'].mean(), rtol=1e-6)
        changed_next = dict(synthetic_batch,
                            next_observations=synthetic_batch['next_observations'] + 100.)
        changed_loss, _ = agent.action_critic_loss(
            changed_next, agent.action_critic_network.params, loss_rng)
        np.testing.assert_allclose(changed_loss, loss, rtol=1e-6)
        # One pretraining update is precisely the online action-Q TD/target update.
        data = expanded.sample(4, idxs=np.arange(4))
        new_rng, loss_rng = jax.random.split(agent.rng)
        expected, _ = agent.action_critic_network.apply_loss_fn(
            lambda p: agent.action_critic_loss(data, p, loss_rng))
        agent.target_update(expected, 'critic')
        updated, info = agent.pretrain_action_q(data)
        assert_tree_equal(self, expected, updated.action_critic_network)
        for field in ('actor_network', 'bc_flow_network', 'critic_network'):
            assert_tree_equal(self, getattr(agent, field), getattr(updated, field))
        self.assertTrue(np.isfinite(info['action_critic/loss']))
        np.testing.assert_array_equal(updated.rng, new_rng)
        class Env:
            def reset(self, seed):
                return np.zeros(3, dtype=np.float32), {}
            def step(self, action):
                return np.zeros(3, dtype=np.float32), -1., True, False, {'success': 0.}
        class Logger:
            def log(self, *args, **kwargs):
                pass
        enabled = agent.replace(config=agent.config.copy(dict(pretrain_qnet=True)))
        with patch('ogpo.runners.grace_runner.save_agent'), patch(
                'ogpo.runners.grace_runner._run_evaluation', return_value={}):
            continued, log_step, _, _ = run_grace_online_rl(
                enabled, Env(), Env(), None, dataset, dict(enabled.config), Logger(), '/unused',
                'grace', online_steps=1, start_training=20, batch_size=4,
                horizon_length=4, discount=.99, n_eval_envs=1,
                eval_interval=-1, log_interval=-1, save_interval=-1, buffer_size=100)
        self.assertEqual(log_step, 10)  # 8 Q updates + final Q log + 1 env step
        self.assertEqual(int(continued.action_critic_network.step),
                         int(agent.action_critic_network.step) + 8)
        for field in ('actor_network', 'bc_flow_network', 'critic_network'):
            assert_tree_equal(self, getattr(agent, field), getattr(continued, field))
        for name, value in [('increase_num', 0), ('noise_add_start', -1),
                            ('pretrain_qnet_epochs', 0), ('increase_num', 1.5),
                            ('noise_add_start', 11)]:
            with self.assertRaisesRegex(ValueError, name):
                GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2),
                                  dict(agent.config, pretrain_qnet=True, **{name: value}))

    def test_encoder_clip_sampling_losses_and_prior(self):
        config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
        config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                      use_layer_norm=False, use_encoder_clip=True,
                      encoder_sphere_limit=False, action_magnitude=2.)
        agent = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        obs = jax.random.normal(jax.random.PRNGKey(7), (128, 3))
        key = jax.random.PRNGKey(9)
        dist = agent.actor_network.select('actor')(obs)
        raw = dist.sample(seed=key)
        clipped = jnp.clip(raw, -1., 1.)
        self.assertTrue(bool(jnp.any(jnp.abs(raw) > 1.)))
        np.testing.assert_allclose(agent.sample_noise(obs, key), clipped, atol=1e-6)
        np.testing.assert_allclose(agent.sample_noise(obs, key, deterministic=True),
                                   jnp.clip(dist.mode(), -1., 1.), atol=1e-6)
        np.testing.assert_allclose(agent.sample_actions(obs, key),
            agent.noise_to_actions_public(obs, clipped, noises_constrained=True), atol=1e-6)

        data = jax.tree.map(lambda x: x[0], batch())
        actor_dist = agent.actor_network.select('actor')(data['observations'])
        _, sample_key = jax.random.split(key)
        raw_samples, raw_log_probs = actor_dist.sample_and_log_prob(seed=sample_key)
        q = agent._aggregate_q(agent.critic_network.select('critic')(
            data['observations'], actions=jnp.clip(raw_samples, -1., 1.)))
        _, info = agent.actor_loss(data, agent.actor_network.params, key)
        expected_loss = jnp.mean(agent.actor_network.select('temp')() * raw_log_probs - q)
        np.testing.assert_allclose(info['actor_loss'], expected_loss, atol=1e-6)
        _, bc_info = agent.actor_bc_loss(data, agent.actor_network.params, key)
        np.testing.assert_allclose(bc_info['actor_bc_loss_unweighted'], jnp.mean(
            (jnp.clip(actor_dist.mode(), -1., 1.) - data['actions']) ** 2), atol=1e-6)

        next_dist = agent.actor_network.select('actor')(data['next_observations'])
        next_z, next_log_probs = next_dist.sample_and_log_prob(seed=sample_key)
        next_a = agent.noise_to_actions_public(
            data['next_observations'], jnp.clip(next_z, -1., 1.), noises_constrained=True)
        next_q = agent._aggregate_q(agent.action_critic_network.select('target_critic')(
            data['next_observations'], actions=next_a))
        target = jnp.clip(data['rewards'] + config['discount'] ** data['chunk_length'] *
                          data['masks'] * next_q, -500., 500.)
        predicted = agent.action_critic_network.select('critic')(
            data['observations'], actions=data['true_actions'])
        loss, _ = agent.action_critic_loss(data, agent.action_critic_network.params, key)
        np.testing.assert_allclose(loss, jnp.mean((predicted-target) ** 2), atol=1e-6)

        # The clip switch belongs to the encoder, not Gaussian warmup/distillation.
        prior = jax.random.normal(key, data['actions'].shape)
        _, prior_info = agent.critic_loss(data, agent.critic_network.params, key)
        np.testing.assert_allclose(prior_info['noise_std'], prior.std(), atol=1e-6)
        np.testing.assert_allclose(agent.constrain_encoder_output(raw), raw, atol=1e-6)
        for loss_fn, params in ((agent.actor_loss, agent.actor_network.params),
                                (agent.actor_bc_loss, agent.actor_network.params)):
            gradients = jax.grad(lambda p: loss_fn(data, p, key)[0])(params)
            self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(gradients)))

    def test_encoder_clip_disabled_and_sphere_conflict(self):
        config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
        config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8), num_qs=2,
                      use_layer_norm=False, encoder_sphere_limit=False, use_encoder_clip=False)
        agent = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        legacy = dict(config)
        legacy.pop('use_encoder_clip')
        legacy = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), legacy)
        obs = jnp.ones((4, 3))
        key = jax.random.PRNGKey(21)
        np.testing.assert_array_equal(agent.sample_noise(obs, key), legacy.sample_noise(obs, key))
        config['use_encoder_clip'] = True
        clip_agent = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        assert_tree_equal(self, agent.actor_network.params, clip_agent.actor_network.params)
        assert_tree_equal(self, agent.bc_flow_network.params, clip_agent.bc_flow_network.params)
        config['encoder_sphere_limit'] = True
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)

    def test_encoder_sphere_matches_lps_and_sampling_paths(self):
        _, grace = make_pair()
        self.assertTrue(grace.config['encoder_sphere_limit'])
        observations = jax.random.normal(jax.random.PRNGKey(31), (128, 3)) * 10
        key = jax.random.PRNGKey(32)
        dist = grace.actor_network.select('actor')(observations)
        raw = dist.sample(seed=key)
        # No tanh or action_magnitude rescaling before the LPS projection.
        np.testing.assert_allclose(dist.mean(), dist.mode())
        expected = raw / jnp.sqrt(jnp.sum(raw ** 2, axis=-1, keepdims=True) + 1e-6) * jnp.sqrt(8.)
        samples = grace.sample_noise(observations, key)
        np.testing.assert_allclose(samples, expected, atol=1e-6)
        np.testing.assert_allclose(jnp.linalg.norm(samples, axis=-1), jnp.sqrt(8.), atol=2e-5)
        self.assertTrue(np.any(np.abs(samples) > 1.))
        np.testing.assert_allclose(grace.sample_actions(observations, key),
            grace.noise_to_actions_public(observations, expected, noises_constrained=True), atol=1e-6)
        mode = grace.constrain_encoder_output(dist.mode())
        np.testing.assert_allclose(grace.sample_noise(observations, deterministic=True), mode, atol=1e-6)
        np.testing.assert_allclose(grace.sample_actions(observations, deterministic=True),
            grace.noise_to_actions_public(observations, mode, noises_constrained=True), atol=1e-6)
        zero = grace.constrain_encoder_output(jnp.zeros((2, 8)))
        np.testing.assert_array_equal(zero, np.zeros((2, 8)))
        for head in ('mean_net', 'log_std_net'):
            self.assertTrue(any(np.any(np.asarray(leaf) != 0) for leaf in
                jax.tree.leaves(grace.actor_network.params['modules_actor'][head])))

    def test_all_decoder_entry_points_use_sphere_prior(self):
        _, grace = make_pair()
        obs = jnp.ones((3, 3))
        key = jax.random.PRNGKey(41)
        raw = 7 * jax.random.normal(key, (3, 8))
        expected = grace.constrain_encoder_output(raw)
        # Inspect the actual ODE starting point, not the clipped decoded action.
        with jax.disable_jit(), patch('ogpo.agents.grace.sample_flow_actions_ode',
                side_effect=lambda **kwargs: kwargs['noises']):
            np.testing.assert_allclose(grace.compute_flow_actions(obs, raw), expected)
            np.testing.assert_allclose(grace.noise_to_actions_public(obs, raw), expected)
            prior = grace.constrain_encoder_output(jax.random.normal(key, (3, 8)))
            np.testing.assert_allclose(grace.compute_flow_actions(obs, rng=key), prior)
            np.testing.assert_allclose(grace.sample_actions_normal(obs, key), prior)

    def test_encoder_sphere_used_by_actor_and_td_target(self):
        _, grace = make_pair()
        grace = grace.replace(config=dict(grace.config, delete_sac_entropy=False))
        data = jax.tree.map(lambda x: x[0], batch())
        rng = jax.random.PRNGKey(32)
        _, key = jax.random.split(rng)
        dist = grace.actor_network.select('actor')(data['observations'])
        raw, log_probs = dist.sample_and_log_prob(seed=key)
        z = grace.constrain_encoder_output(raw)
        q = grace._aggregate_q(grace.critic_network.select('critic')(
            data['observations'], actions=z))
        _, info = grace.actor_loss(data, grace.actor_network.params, rng)
        expected = jnp.mean(grace.actor_network.select('temp')() * log_probs - q)
        np.testing.assert_allclose(info['actor_loss'], expected, rtol=1e-6)
        next_z = grace.sample_noise(data['next_observations'], key)
        next_a = grace.noise_to_actions_public(
            data['next_observations'], next_z, noises_constrained=True)
        next_q = grace._aggregate_q(grace.action_critic_network.select('target_critic')(
            data['next_observations'], actions=next_a))
        target = jnp.clip(data['rewards'] + .99 ** data['chunk_length'] * data['masks'] * next_q,
                          -500., 500.)
        _, info = grace.action_critic_loss(data, grace.action_critic_network.params, rng)
        np.testing.assert_allclose(info['action_critic/target_q_mean'], target.mean(), rtol=1e-6)
        mode = grace.constrain_encoder_output(dist.mode())
        loss, _ = grace.actor_bc_loss(data, grace.actor_network.params, rng)
        np.testing.assert_allclose(loss, grace.config['bc_coeff'] *
                                   jnp.mean((mode - data['actions']) ** 2), rtol=1e-6)
        updated, metrics = grace.batch_update(batch())
        self.assertTrue(np.isfinite(metrics['actor/total_loss']))
        self.assertTrue(all(np.isfinite(np.asarray(leaf)).all() for leaf in
                            jax.tree.leaves(updated.actor_network.params)))

    def test_delete_sac_entropy_independent_of_sphere(self):
        _, baseline = make_pair()
        data = jax.tree.map(lambda x: x[0], batch())
        rng = jax.random.PRNGKey(32)
        for sphere in (False, True):
            for delete_entropy in (False, True):
                with self.subTest(sphere=sphere, delete_entropy=delete_entropy):
                    config = dict(baseline.config, encoder_sphere_limit=sphere,
                                  delete_sac_entropy=delete_entropy)
                    agent = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
                    loss_fn = lambda p: agent.actor_loss(data, p, rng)
                    (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        agent.actor_network.params)
                    temp_grads = jax.tree.leaves(grads['modules_temp'])
                    updated, _ = agent.actor_network.apply_loss_fn(loss_fn)
                    if delete_entropy:
                        np.testing.assert_allclose(info['actor_loss'], -info['q_actor_mean'])
                        self.assertEqual(float(info['temp_loss']), 0.)
                        self.assertTrue(all(np.all(np.asarray(g) == 0) for g in temp_grads))
                        assert_tree_equal(self, updated.params['modules_temp'],
                                          agent.actor_network.params['modules_temp'])
                    else:
                        np.testing.assert_allclose(info['actor_loss'],
                            info['temp'] * info['log_probs_mean'] - info['q_actor_mean'],
                            rtol=1e-5)
                        self.assertTrue(any(np.any(np.asarray(g) != 0) for g in temp_grads))

    def test_encoder_sphere_disabled_preserves_original_policy(self):
        dsrl, baseline = make_pair()
        # This legacy equivalence check explicitly requires the tanh actor.
        config = dict(baseline.config, encoder_sphere_limit=False, use_no_tanh=False)
        grace = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        observations = jax.random.normal(jax.random.PRNGKey(33), (128, 3))
        key = jax.random.PRNGKey(34)
        np.testing.assert_array_equal(grace.sample_noise(observations, key),
                                      dsrl.sample_noise(observations, key))
        self.assertTrue(np.all(np.abs(grace.sample_noise(observations, key)) <= 1.))
        config.pop('encoder_sphere_limit')
        default = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        np.testing.assert_array_equal(default.sample_noise(observations, key),
                                      grace.sample_noise(observations, key))

    def test_layer_norm_online_networks_and_offline_checkpoint(self):
        _, template = make_pair()
        config = dict(template.config, encoder_sphere_limit=False,
                      actor_layer_norm=False, layer_norm=False)
        baseline = GRACEAgent.create(9, jnp.zeros(3), jnp.zeros(2), config)
        agent = GRACEAgent.create(9, jnp.zeros(3), jnp.zeros(2),
                                 dict(config, use_layer_norm=True, num_qs=3))
        self.assertEqual(agent.config['num_qs'], 10)
        self.assertEqual(agent.config['actor_hidden_dims'], baseline.config['actor_hidden_dims'])
        self.assertEqual(agent.config['value_hidden_dims'], baseline.config['value_hidden_dims'])

        def norm_scales(params):
            return [value for path, value in jax.tree_util.tree_flatten_with_path(params)[0]
                    if any('LayerNorm' in str(getattr(key, 'key', key)) for key in path)
                    and getattr(path[-1], 'key', None) == 'scale']

        self.assertFalse(norm_scales(baseline.actor_network.params))
        self.assertFalse(norm_scales(baseline.critic_network.params))
        self.assertEqual([v.shape for v in norm_scales(agent.actor_network.params)],
                         [(8,), (8,)])
        for network in (agent.critic_network, agent.action_critic_network):
            for module in ('modules_critic', 'modules_target_critic'):
                self.assertEqual([v.shape for v in norm_scales(network.params[module])],
                                 [(10, 8), (10, 8)])
            self.assertEqual(network.select('critic')(
                jnp.zeros((3, 3)), actions=jnp.zeros((3, 8))).shape, (10, 3))
        self.assertFalse(norm_scales(agent.bc_flow_network.params))
        assert_tree_equal(self, agent.bc_flow_network.params, baseline.bc_flow_network.params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'offline.pkl'
            with path.open('wb') as f:
                pickle.dump({'agent': {'actor_network': {'params': {
                    'modules_actor': baseline.bc_flow_network.params['modules_actor_bc_flow']}}}}, f)
            agent = agent.restore_bc_flow_params(path)
        frozen = agent.bc_flow_network
        # Both the ordinary and success-BC paths must keep the pretrained flow frozen.
        for success in (False, True):
            data = batch()
            agent, info = agent.batch_update(data, data if success else None, success)
            self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in info.values()))
            assert_tree_equal(self, agent.bc_flow_network, frozen)
        self.assertEqual(int(agent.action_critic_network.step), int(agent.critic_network.step))

    def test_serl_subsampled_td_target_and_full_ensemble_actor(self):
        _, template = make_pair()
        agent = GRACEAgent.create(9, jnp.zeros(3), jnp.zeros(2),
            dict(template.config, use_layer_norm=True, q_agg='min'))
        rng = jax.random.PRNGKey(73)
        # Match reference randint sampling (with replacement), shared across the batch.
        qs = jnp.arange(30, dtype=jnp.float32).reshape(10, 3)
        _, subsample_rng = jax.random.split(rng)
        indices = jax.random.randint(subsample_rng, (2,), 0, 10)
        np.testing.assert_array_equal(agent._aggregate_td_target_q(qs, rng),
                                      qs[indices].min(axis=0))
        np.testing.assert_allclose(agent._aggregate_q(qs), qs.mean(axis=0))

        data = jax.tree.map(lambda x: x[0], batch())
        target_rng, sample_rng = jax.random.split(rng)
        next_z = agent.sample_noise(data['next_observations'], sample_rng)
        next_a = agent.noise_to_actions_public(data['next_observations'], next_z,
                                              noises_constrained=True)
        next_qs = agent.action_critic_network.select('target_critic')(
            data['next_observations'], actions=next_a)
        next_q = agent._aggregate_td_target_q(next_qs, target_rng)
        target = jnp.clip(data['rewards'] + .99 ** data['chunk_length'] * data['masks'] * next_q,
                          -500., 500.)
        predicted = agent.action_critic_network.select('critic')(
            data['observations'], actions=data['true_actions'])
        loss, info = agent.action_critic_loss(data, agent.action_critic_network.params, rng)
        np.testing.assert_allclose(loss, jnp.mean((predicted - target[None]) ** 2), rtol=1e-5)
        np.testing.assert_allclose(info['action_critic/target_q_mean'], target.mean(), atol=1e-6)
        dist = agent.actor_network.select('actor')(data['observations'])
        z, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        actor_qs = agent.critic_network.select('critic')(
            data['observations'], actions=agent.constrain_encoder_output(z))
        _, info = agent.actor_loss(data, agent.actor_network.params, rng)
        expected = (agent.actor_network.select('temp')() * log_probs - actor_qs.mean(axis=0)).mean()
        np.testing.assert_allclose(info['actor_loss'], expected, rtol=1e-5)
        # Latent-Q distillation also uses the full online action-Q ensemble mean.
        z = agent.constrain_encoder_output(jax.random.normal(rng, data['actions'].shape))
        actions = agent.noise_to_actions_public(data['observations'], z, noises_constrained=True)
        teacher = agent.action_critic_network.select('critic')(
            data['observations'], actions=actions).mean(axis=0)
        _, info = agent.critic_loss(data, agent.critic_network.params, rng)
        np.testing.assert_allclose(info['target_q_mean'], teacher.mean(), atol=1e-6)

    def test_layer_norm_disabled_preserves_legacy_q_aggregation(self):
        _, agent = make_pair()
        qs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        rng = jax.random.PRNGKey(1)
        for mode in ('mean', 'min'):
            legacy = agent.replace(config=dict(agent.config, q_agg=mode))
            expected = qs.mean(axis=0) if mode == 'mean' else qs.min(axis=0)
            np.testing.assert_array_equal(legacy._aggregate_td_target_q(qs, rng), expected)
        config = dict(agent.config)
        config.pop('use_layer_norm')
        default = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
        self.assertFalse(default.config['use_layer_norm'])
        self.assertEqual(default.config['num_qs'], 2)
        assert_tree_equal(self, default.actor_network.params, agent.actor_network.params)
        assert_tree_equal(self, default.critic_network.params, agent.critic_network.params)

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
            z = grace.constrain_encoder_output(jax.random.normal(latent_rng, (8,)))
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

    def test_prior_distillation_target_and_stopped_teacher_gradient(self):
        _, grace = make_pair()
        data = jax.tree.map(lambda x: x[0], batch())
        rng = jax.random.PRNGKey(19)
        z = grace.constrain_encoder_output(jax.random.normal(
            rng, data['actions'].shape, dtype=data['actions'].dtype))
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
