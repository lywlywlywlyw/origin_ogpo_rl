"""DSRL-NA's reference objectives, two-loop schedule and checkpoint compatibility."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.agents.dsrlna import DSRLNAAgent
from ogpo.runners.dsrlna_runner import run_dsrlna_online_rl
from ogpo.utils.config_loader import load_config, config_to_agent_dict


REMOVED = ('use_ogpo_trick', 'use_faster_distill', 'use_grrl_distill', 'pretrain_qnet',
           'use_layer_norm', 'use_demo_buffer', 'use_offline_iql', 'use_no_tanh',
           'encoder_sphere_limit', 'use_encoder_clip', 'delete_sac_entropy', 'decoder_add_noise')


def cfg(**overrides):
    result = config_to_agent_dict(load_config('dsrlna_ogpo_square_shared'))
    result.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8),
                  decoder_hidden_dims=(8, 8), num_train_envs=1, horizon_length=2,
                  flow_steps=2, noise_critic_updates_per_actor=0.5)
    result.update(overrides)
    return result


def agent(**overrides):
    return DSRLNAAgent.create(1, jnp.zeros(3), jnp.zeros(2), cfg(**overrides))


def batches(count):
    rng = np.random.default_rng(5)
    return dict(observations=rng.normal(size=(count, 3, 3)).astype(np.float32),
                next_observations=rng.normal(size=(count, 3, 3)).astype(np.float32),
                actions=rng.uniform(-1, 1, (count, 3, 4)).astype(np.float32),
                rewards=-np.ones((count, 3), np.float32), masks=np.ones((count, 3), np.float32),
                chunk_length=np.full((count, 3), 2, np.int32))


def equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_clean_config_and_independent_registration():
    config = load_config('dsrlna_ogpo_square_shared')
    assert config['agent']['agent_name'] == 'dsrlna'
    assert config['success_buffer']['use_success_buffer'] is False
    assert config['agent']['noise_critic_updates_per_actor'] == 0.5
    assert config['buffer']['utd_online'] == 1
    assert config['agent']['num_train_envs'] == 1
    assert config['buffer']['start_training'] == 10000
    assert config['agent']['discount'] == .99
    assert config['eval']['n_eval_envs'] * config['eval']['eval_episodes'] == 64
    assert config['intervals']['eval_interval'] == 10000
    assert 'DSRLNAReferenceEnv' not in Path('ogpo/main.py').read_text()
    for name in REMOVED:
        assert name not in config['agent']
        assert name not in Path('ogpo/agents/dsrlna.py').read_text()
        assert name not in Path('ogpo/runners/dsrlna_runner.py').read_text()
    tree = ast.parse(Path('ogpo/main.py').read_text())
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'AGENTS' for t in n.targets)]
    assert any(isinstance(k, ast.Constant) and k.value == 'dsrlna' for k in assignments[0].value.keys)


def test_head_paired_distillation_preserves_disagreement():
    def network(values):
        return SimpleNamespace(select=lambda name: lambda observations, actions, **kwargs:
                               jnp.tile(jnp.array(values)[:, None], (1, len(observations))))
    dummy = SimpleNamespace(config={'noise_dim': 4},
                            noise_to_actions_public=lambda obs, z: z,
                            action_critic_network=network([0., 10.]),
                            critic_network=network([0., 10.]))
    batch = {'observations': jnp.zeros((3, 3))}
    loss, _ = DSRLNAAgent.critic_loss(dummy, batch, {}, jax.random.PRNGKey(2))
    np.testing.assert_allclose(loss, 0.)
    dummy.critic_network = network([10., 0.])
    loss, _ = DSRLNAAgent.critic_loss(dummy, batch, {}, jax.random.PRNGKey(2))
    np.testing.assert_allclose(loss, 100.)


def test_soft_td_target_uses_actual_chunk_length_and_terminal_masks():
    class Dist:
        def sample_and_log_prob(self, seed):
            return jnp.zeros((2, 4)), jnp.array([2., 3.])
    class Critic:
        def select(self, name):
            if name == 'target_critic':
                return lambda *args, **kw: jnp.array([[-2., -3.], [10., 10.]])
            return lambda *args, **kw: jnp.zeros((2, 2))
    dummy = SimpleNamespace(config={'discount': .9},
                            actor_network=SimpleNamespace(select=lambda name: lambda obs: Dist()),
                            action_critic_network=Critic(), noise_to_actions_public=lambda obs, z: z)
    batch = dict(observations=jnp.zeros((2, 3)), next_observations=jnp.zeros((2, 3)),
                 actions=jnp.zeros((2, 4)), masks=jnp.array([1., 0.]),
                 rewards=jnp.array([-1000., -2.]), chunk_length=jnp.array([3, 1]))
    loss, info = DSRLNAAgent.action_critic_loss(dummy, batch, {}, jax.random.PRNGKey(1), .5)
    expected = jnp.array([-1000. + .9 ** 3 * (-2. - .5 * 2.), -2.])
    np.testing.assert_allclose(loss, jnp.square(expected).mean(), rtol=1e-6)
    np.testing.assert_allclose(info['target_q_mean'], expected.mean())


def test_update_blocks_freeze_decoder_and_use_stale_student_during_sac():
    original = agent()
    sac_batch, student_batch = batches(2), batches(1)
    after_sac, _ = jax.lax.scan(lambda a, b: a._sac_step(b), original, sac_batch)
    equal(original.critic_network.params, after_sac.critic_network.params)
    no_student, no_student_info = original.batch_update(batches(1))
    equal(original.critic_network.params, no_student.critic_network.params)
    assert int(no_student.actor_network.step) == int(no_student.action_critic_network.step) == 2
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(no_student_info))
    trained, info = original.batch_update(sac_batch, student_batch)
    assert int(trained.actor_network.step) == 3
    assert int(trained.action_critic_network.step) == 3
    assert int(trained.temperature_network.step) == 3
    assert int(trained.critic_network.step) == 2
    assert 'modules_target_critic' not in trained.critic_network.params
    equal(after_sac.actor_network.params, trained.actor_network.params)
    equal(after_sac.action_critic_network.params, trained.action_critic_network.params)
    equal(original.bc_flow_network.params, trained.bc_flow_network.params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(info))
    assert any(not np.array_equal(x, y) for x, y in zip(
        jax.tree.leaves(original.critic_network.params), jax.tree.leaves(trained.critic_network.params)))
    for obs in (jnp.zeros(3), jnp.zeros((3, 3))):
        z = original.sample_noise(obs, jax.random.PRNGKey(5))
        assert np.all(np.abs(z) <= 1.)
        np.testing.assert_allclose(original.sample_actions(obs, jax.random.PRNGKey(5)),
                                   original.noise_to_actions_public(obs, z), atol=1e-6)
        assert np.isfinite(original.sample_actions(obs, deterministic=True)).all()


class _Native:
    def __init__(self, success_at=None):
        self.steps = 0
        self.success_at = success_at

    def step(self, action):
        self.steps += 1
        return {'state': np.zeros(3, np.float32)}, float(self.steps == self.success_at), False, {}


class _LocalWrapper(gym.Env):
    """Exercise the real shared step implementation with a lightweight robot."""
    action_space = gym.spaces.Box(-1., 1., (2,))
    observation_space = gym.spaces.Box(-1., 1., (3,))

    def __init__(self, success_at=None, horizon=3):
        self.env = _Native(success_at)
        self.max_episode_length = horizon
        self.obs_keys = ['state']
        self.normalize = False
        self.video_writer = None
        self.post_success_steps = 0
        self.env_step = 0

    def reset(self, seed=None, options=None):
        self.env.steps = self.t = self.episode_length = 0
        self.episode_return = 0.
        self.t_succ = None
        return np.zeros(3, np.float32), {}

    def step(self, action):
        from envs.robomimic_utils import RobomimicLowdimWrapper
        return RobomimicLowdimWrapper.step(self, action)


def test_shared_environment_success_bonus_and_time_limit():
    env = _LocalWrapper(success_at=2)
    env.reset()
    _, reward, term, trunc, _ = env.step(np.zeros(2))
    assert reward == -1. and not term and not trunc
    _, reward, term, trunc, info = env.step(np.zeros(2))
    assert reward == 5. and term and not trunc and info['success'] == 1
    env = _LocalWrapper(horizon=1)
    env.reset()
    _, reward, term, trunc, _ = env.step(np.zeros(2))
    assert reward == -1. and not term and trunc


@pytest.mark.parametrize('success_at', [3, None])
def test_runner_budget_short_chunks_warmup_crossing_and_bootstrap(success_at):
    current = agent()
    env = _LocalWrapper(success_at=success_at)
    logger_records = []
    logger = SimpleNamespace(log=lambda data, prefix, step: logger_records.append((prefix, data)))
    source = SimpleNamespace(sample=lambda shape: {'observations': np.zeros(3, np.float32)})
    with patch('ogpo.runners.dsrlna_runner.save_agent'), patch(
            'ogpo.runners.dsrlna_runner._run_evaluation', return_value={'success': 1.}) as evaluation:
        trained, step, _, replay = run_dsrlna_online_rl(
            current, env, env, None, source, dict(current.config), logger, '/unused',
            'dsrlna', online_steps=7, start_training=1, batch_size=3,
            horizon_length=2, discount=current.config['discount'], utd_online=1,
            buffer_size=64, n_eval_envs=1, eval_interval=-1,
            log_interval=-1, save_interval=-1, force_jax_sync=False)
    assert step == 7 and replay.size == 5
    budget = [d for p, d in logger_records if p == 'online_budget'][-1]
    # First chunk crosses warmup; final chunk is shortened by the run budget.
    assert budget['gradient_updates'] == 6
    assert budget['critic_gradient_updates'] == 6
    assert budget['noise_critic_updates'] == 3
    assert budget['decoder_updates'] == 0
    assert int(trained.actor_network.step) == 7
    assert int(trained.action_critic_network.step) == 7
    assert int(trained.critic_network.step) == 4
    assert evaluation.call_count == 2
    np.testing.assert_array_equal(replay['chunk_length'][:5], [2, 1, 2, 1, 1])
    np.testing.assert_array_equal(replay['terminals'][:5], [0, 1, 0, 1, 0])
    expected_masks = [1, 0, 1, 0, 1] if success_at else np.ones(5)
    np.testing.assert_array_equal(replay['masks'][:5], expected_masks)
    np.testing.assert_allclose(replay['rewards'][:5],
                               [-1.99, 5 if success_at else -1, -1.99,
                                5 if success_at else -1, -1])
    equal(current.bc_flow_network.params, trained.bc_flow_network.params)
    _, key = jax.random.split(jax.random.PRNGKey(40000))
    np.testing.assert_allclose(replay['actions'][0],
                               current.sample_actions(jnp.zeros(3), key), atol=1e-6)


def test_actual_ogpo_checkpoint_with_independent_online_architecture():
    config = config_to_agent_dict(load_config('dsrlna_ogpo_square_shared'))
    path = Path(load_config('dsrlna_ogpo_square_shared')['checkpoint']['restore_bc_flow_path'])
    if not path.exists():
        pytest.skip('Shared offline checkpoint unavailable')
    config.update(actor_hidden_dims=(8, 8), value_hidden_dims=(8, 8))
    original = DSRLNAAgent.create(7, jnp.zeros(23), jnp.zeros(7), config)
    restored = original.restore_bc_flow_params(path)
    import pickle
    from ogpo.networks import ActorVectorField
    from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
    with path.open('rb') as stream:
        source = pickle.load(stream)['agent']['actor_network']['params']['modules_actor']
    equal(restored.bc_flow_network.params['modules_actor_bc_flow'], source)
    equal(original.actor_network.params, restored.actor_network.params)
    model = ActorVectorField(hidden_dims=(512, 512, 512, 512), action_dim=28,
                             layer_norm=False, time_embedding=SinusoidalTimeEmbedding(embed_dim=32))
    obs = jnp.ones((2, 23))
    z = jax.random.normal(jax.random.PRNGKey(8), (2, 28))
    x = z
    for index in range(10):
        x = x + model.apply({'params': source}, obs, x, jnp.full((2, 1), index / 10),
                            is_encoded=True) / 10
    np.testing.assert_allclose(restored.noise_to_actions_public(obs, z), jnp.clip(x, -1., 1.), atol=2e-5)


@pytest.mark.parametrize('values', [{'noise_critic_updates_per_actor': 0}, {'num_qs': 0},
                                   {'tau': 0.}, {'lr_noise_critic': -1.}, {'num_train_envs': 4}])
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        agent(**values)
