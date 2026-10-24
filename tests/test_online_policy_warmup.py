"""Replay warmup must delay optimization without changing the collection policy."""
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest

from ogpo.runners.online_rl_runner import _q_warmup_collect
from ogpo.utils.datasets import ReplayBuffer


@pytest.mark.parametrize("algorithm,runner", [
    ("dsrl", "dsrl"), ("dsrl_bon", "dsrl"), ("dsrl_plus_expo", "dsrl"),
    ("jgcw", "jgcw"), ("grace", "grace"),
])
@pytest.mark.parametrize("warmup", [20, 22, 47])
def test_actor_collects_before_and_after_warmup(algorithm, runner, warmup):
    # Short episodes and four-action chunks also exercise a threshold inside a chunk.
    class Env:
        total = 0
        def reset(self, seed):
            self.steps = 0
            return np.zeros(2), {}
        def step(self, action):
            np.testing.assert_allclose(action, .25)
            self.steps += 1
            self.total += 1
            return np.zeros(2), -1., self.steps == 5, False, {"success": 0.}

    class Dataset:
        def sample(self, shape):
            return dict(observations=np.zeros(2), actions=np.zeros(1))

    class Agent:
        updates = 0
        def sample_noise(self, **kwargs):
            return np.full(4, .25)
        def noise_to_actions_public(self, obs, noise):
            return noise
        def sample_actions(self, **kwargs):
            # EXPO must also use its complete online sampler during warmup.
            return dict(actions=np.full(4, .25), noise=np.full(4, .25),
                        base_actions=np.full(4, .125))
        def batch_update(self, batch, *args):
            assert env.total > warmup
            self.updates += batch["actions"].shape[0]
            return self, {"loss": 0.}

    class Logger:
        def log(self, *args, **kwargs):
            pass

    env, agent = Env(), Agent()
    module = import_module(f"ogpo.runners.{runner}_runner")
    run = getattr(module, f"run_{runner}_online_rl")
    with patch.object(module, "save_agent"), patch.object(module, "_run_evaluation", return_value={}):
        _, _, _, replay = run(
            agent, env, env, None, Dataset(),
            dict(encoder=None, noise_chunk_length=-1, updates_per_env_step=True,
                 chunk_reward_aggregation="discounted_sum"), Logger(), "/unused",
            algorithm, online_steps=47, start_training=warmup, batch_size=1,
            horizon_length=4, discount=.99, utd_online=1, n_eval_envs=1,
            eval_interval=-1, log_interval=-1, save_interval=-1,
            force_jax_sync=False, buffer_size=100)
    assert env.total == 47
    assert agent.updates == 47 - warmup
    np.testing.assert_allclose(replay["actions"][:replay.size], .25)


@pytest.mark.parametrize("one_step,best_of_n,expected", [
    (False, 1, 1.), (False, 4, 2.), (True, 1, 3.), (True, 4, 4.),
])
def test_q_warmup_uses_configured_rollout_policy(one_step, best_of_n, expected):
    class Agent:
        config = {"use_one_step_for_rollouts": one_step}
        def sample_actions(self, **kwargs):
            return np.full(1, 1.)
        def sample_actions_BON(self, **kwargs):
            return np.full(1, 2.)
        def sample_actions_one_step(self, **kwargs):
            return np.full(1, 3.)
        def sample_actions_one_step_BON(self, **kwargs):
            return np.full(1, 4.)
    class Env:
        def reset(self, seed):
            return np.zeros(2), {}
        def step(self, action):
            np.testing.assert_allclose(action, expected)
            return np.zeros(2), 0., True, False, {"success": 0.}
    replay = ReplayBuffer.create(dict(
        observations=np.zeros(2), next_observations=np.zeros(2),
        actions=np.zeros(1), rewards=np.float32(0), terminals=np.float32(0),
        masks=np.float32(1)), 10)
    _q_warmup_collect(Agent(), Env(), replay, q_warmup_steps=2, action_dim=1,
                     has_images=False, discount=.99, seed=1,
                     use_success_buffer=False, mc_regression=False, encode_fn=None,
                     best_of_n=best_of_n)
    assert replay.size == 2
