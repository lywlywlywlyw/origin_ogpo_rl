import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from types import SimpleNamespace
from unittest.mock import patch
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.grace import GRACEAgent
from ogpo.runners.grace_iql import (prepare_iql_transitions, build_latent_iql_dataset,
                                    pretrain_latent_iql, split_iql_indices)
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.utils.datasets import Dataset


def agent():
    config = config_to_agent_dict(load_config('grace_ogpo_square_shared'))
    config.update(use_ogpo_trick=False, actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8),
                  use_no_tanh=True, use_layer_norm=True, encoder_sphere_limit=False,
                  use_offline_iql=True,
                  use_demo_buffer=False,
                  iql_steps=2, iql_batch_size=4)
    return GRACEAgent.create(2, jnp.zeros(3), jnp.zeros(2), config)


def dataset():
    # First episode has two post-success rows; second times out without success.
    return Dataset.create(observations=np.arange(30, dtype=np.float32).reshape(10, 3),
                          next_observations=np.arange(30, dtype=np.float32).reshape(10, 3) + 1,
                          actions=np.ones((10, 2), np.float32) * .2,
                          rewards=np.array([-1, -1, 0, 0, 0, -1, -1, -1, -1, -1], np.float32),
                          terminals=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], np.float32),
                          masks=np.zeros(10, np.float32))


def equal_tree(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_in_memory_success_timeout_and_chunk_targets():
    source = dataset()
    before = {k: np.array(v) for k, v in source.items()}
    prepared = prepare_iql_transitions(source, max_episode_length=4)
    np.testing.assert_array_equal(prepared['observations'][:, 0], [0, 3, 6, 15, 18, 21, 24])
    np.testing.assert_array_equal(prepared['rewards'], [-1, -1, 5, -1, -1, -1, -1])
    np.testing.assert_array_equal(prepared['masks'], [1, 1, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(prepared['terminals'], [0, 0, 1, 0, 0, 0, 1])
    dummy = SimpleNamespace(inverse_fm_batch=lambda obs, actions: actions + 2)
    config = dict(horizon_length=4, action_chunking=True, discount=.9, iql_batch_size=4)
    data = build_latent_iql_dataset(dummy, source, config, 0, 4)
    np.testing.assert_allclose(data['rewards'], [-1-.9+5*.9**2, -1+5*.9, 5,
                                               -1-.9-.9**2-.9**3, -1-.9-.9**2, -1-.9, -1])
    np.testing.assert_array_equal(data['chunk_length'], [3, 2, 1, 4, 3, 2, 1])
    np.testing.assert_array_equal(data['masks'], [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_allclose(data['actions'], 2.2)
    for k in source:
        np.testing.assert_array_equal(source[k], before[k])


def test_repeated_success_dones_are_not_new_demonstrations():
    original = dataset()
    raw = Dataset.create(**dict(original, terminals=np.array(
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 1], np.float32)))
    raw.demo_ends = original['terminals']
    data = prepare_iql_transitions(raw, 4)
    np.testing.assert_array_equal(data['observations'][:, 0], [0, 3, 6, 15, 18, 21, 24])
    assert (data['rewards'] == 5).sum() == 1
    np.testing.assert_array_equal(raw['terminals'], [0, 0, 1, 1, 1, 0, 0, 0, 0, 1])


def test_iql_then_online_updates_only_expected_networks():
    initial = agent()
    logs = []
    logger = SimpleNamespace(log=lambda metrics, *args, **kwargs: logs.append(metrics))
    trained, steps = pretrain_latent_iql(initial, dataset(), 0, logger, 0, lambda b: b, 4)
    assert steps == 3 and len(logs) == 2
    assert all(np.isfinite(np.asarray(v)).all() for log in logs for v in log.values())
    equal_tree(initial.bc_flow_network, trained.bc_flow_network)
    equal_tree(initial.action_critic_network, trained.action_critic_network)
    equal_tree(initial.actor_network.params['modules_temp'], trained.actor_network.params['modules_temp'])
    for name in ('actor_network', 'critic_network', 'value_network'):
        assert any(not np.array_equal(x, y) for x, y in zip(
            jax.tree.leaves(getattr(initial, name).params), jax.tree.leaves(getattr(trained, name).params)))
    assert trained.actor_network.tx is initial.actor_network.tx
    assert trained.critic_network.tx is initial.critic_network.tx
    data = build_latent_iql_dataset(trained, dataset(), trained.config, 0, 4).sample(4)
    # No decoded actions or action critic should be needed by online latent TD.
    data.pop('true_actions')
    rng = jax.random.PRNGKey(5)
    with patch.object(GRACEAgent, 'action_critic_loss', side_effect=AssertionError('action Q used')), \
         patch.object(GRACEAgent, 'noise_to_actions_public', side_effect=AssertionError('decoder used')):
        updated, info = jax.jit(GRACEAgent._update)(trained, data)
    equal_tree(trained.bc_flow_network, updated.bc_flow_network)
    equal_tree(trained.action_critic_network, updated.action_critic_network)
    equal_tree(trained.value_network, updated.value_network)
    assert 'critic/td_loss' in info and not any('distillation' in k for k in info)
    # Terminal targets must be independent of next-state value.
    terminal = dict(data, masks=jnp.zeros(4))
    loss, _ = updated.latent_td_loss(terminal, updated.critic_network.params, rng)
    qs = updated.critic_network.select('critic')(data['observations'], actions=data['actions'])
    np.testing.assert_allclose(loss, jnp.mean((qs - data['rewards']) ** 2), rtol=1e-6)


def test_gate_requires_all_three_flags():
    config = dict(agent().config)
    for no_tanh, norm, offline in product((False, True), repeat=3):
        enabled = no_tanh and norm and offline
        candidate = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2),
            dict(config, use_no_tanh=no_tanh, use_layer_norm=norm, use_offline_iql=offline,
                 iql_steps=2 if enabled else 0))
        assert candidate.latent_iql_enabled == enabled
        assert (candidate.value_network is not None) == enabled
    config.pop('use_offline_iql')
    missing = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2), config)
    assert not missing.latent_iql_enabled and missing.value_network is None


def test_iql_disabled_restores_action_q_and_gaussian_distillation():
    initial = agent()
    disabled = GRACEAgent.create(1, jnp.zeros(3), jnp.zeros(2),
                                dict(initial.config, use_offline_iql=False))
    data = build_latent_iql_dataset(initial, dataset(), initial.config, 0, 4).sample(4)
    with patch.object(GRACEAgent, 'latent_td_loss', side_effect=AssertionError('latent TD used')):
        updated, info = jax.jit(GRACEAgent._update)(disabled, data)
    assert 'critic/distillation_loss' in info
    assert 'critic/td_loss' not in info
    assert 'critic/actor_distillation_count' not in info
    assert 'action_critic/loss' in info
    assert any(not np.array_equal(x, y) for x, y in zip(
        jax.tree.leaves(disabled.action_critic_network.params),
        jax.tree.leaves(updated.action_critic_network.params)))
    equal_tree(disabled.bc_flow_network, updated.bc_flow_network)
    assert updated.value_network is None


def test_trajectory_holdout_has_no_overlapping_chunks():
    ids = np.repeat(np.arange(20), 8)
    train, val = split_iql_indices({'episode_id': ids}, .05, 3)
    assert len(train) == 152 and len(val) == 8
    assert not set(ids[train]) & set(ids[val])
    again = split_iql_indices({'episode_id': ids}, .05, 3)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(val, again[1])


@pytest.mark.parametrize('scores,min_steps,max_steps,expected_updates,stopped', [
    ([3., 2., 4., 4., 4.], 4, 10, 5, True),
    ([3., 2., 4.], 100, 3, 3, False),
])
def test_early_stopping_minimum_patience_and_best_state_restore(
        scores, min_steps, max_steps, expected_updates, stopped):
    initial = agent()
    initial = initial.replace(config=initial.config.copy(add_or_replace=dict(
        iql_steps=max_steps, iql_validation_interval=1, iql_early_stopping_min_steps=min_steps,
        iql_early_stopping_patience=2)))
    states, logs, training_rows, validation_rows = [], [], [], []
    # Best step 2 is retained before min_steps; stale at step 3 must not count.
    scores = iter(scores)
    def update(current, batch):
        training_rows.extend(np.asarray(batch['observations'])[:, 0])
        changes = {}
        for name in ('actor_network', 'critic_network', 'value_network'):
            network = getattr(current, name)
            changes[name] = network.replace(params=jax.tree.map(lambda x: x + .01, network.params))
        result = current.replace(**changes)
        states.append(result)
        return result, {'critic_loss': 0.}
    def validate(current, batch):
        validation_rows.append(np.array(batch['observations']))
        return {'validation_score': next(scores)}
    with patch.object(GRACEAgent, 'pretrain_latent_iql', update), \
         patch.object(GRACEAgent, 'iql_validation_metrics', validate):
        selected, steps = pretrain_latent_iql(initial, dataset(), 0,
            SimpleNamespace(log=lambda metrics, *args, **kwargs: logs.append(metrics)),
            10, lambda b: b, 4)
    assert len(states) == expected_updates and steps == expected_updates + 1
    assert logs[-1]['best_iql_step'] == 2
    assert logs[-1]['stale_validations'] == (2 if stopped else 0)
    assert logs[-1]['early_stopped'] == int(stopped)
    for name in ('actor_network', 'critic_network', 'value_network'):
        equal_tree(getattr(selected, name).params, getattr(states[1], name).params)
    equal_tree(initial.bc_flow_network, selected.bc_flow_network)
    equal_tree(initial.action_critic_network, selected.action_critic_network)
    assert not set(training_rows) & set(validation_rows[0][:, 0])
    for rows in validation_rows:
        np.testing.assert_array_equal(rows, validation_rows[0])


def test_validation_score_matches_gorl_with_chunk_discount():
    current = agent()
    data = build_latent_iql_dataset(current, dataset(), current.config, 0, 4)
    batch = data.sample(data.size, idxs=np.arange(data.size))
    info = current.iql_validation_metrics(batch)
    q = current.critic_network.select('critic')(batch['observations'], actions=batch['actions'])
    target_q = current.critic_network.select('target_critic')(
        batch['observations'], actions=batch['actions']).min(axis=0)
    v = current.value_network.select('value')(batch['observations'])
    next_v = current.value_network.select('value')(batch['next_observations'])
    target = batch['rewards'] + .99 ** batch['chunk_length'] * batch['masks'] * next_v
    td = jnp.mean((q - target) ** 2)
    vl = jnp.mean(jnp.where(target_q > v, .8, .2) * (target_q - v) ** 2)
    nll = -current.actor_network.select('actor')(batch['observations']).log_prob(batch['actions']).mean() / 8
    expected = (jnp.sqrt(td) / jnp.maximum(jnp.abs(target).mean(), 1.)
                + jnp.sqrt(vl) / jnp.maximum(jnp.abs(target_q).mean(), 1.) + .01 * nll)
    np.testing.assert_allclose(info['validation_score'], expected, rtol=1e-5)
