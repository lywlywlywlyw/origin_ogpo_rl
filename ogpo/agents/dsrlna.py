"""DSRL-NA: head-paired action-Q to noise-Q distillation over a frozen flow.

Objectives follow ref_codes/dsrl's pinned stable-baselines3 commit 10e5d311,
with the shared baselines' physical-step discount and update budget.
The flow architecture is independent of the online SAC networks so existing
OGPO offline checkpoints remain usable.
"""
import copy
from functools import partial
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
from ogpo.networks import ActorVectorField
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding


def _torch_kernel(key, shape, dtype=jnp.float32):
    """PyTorch Linear's default uniform kernel, using Flax's (in, out) layout."""
    bound = 1. / jnp.sqrt(shape[0])
    return jax.random.uniform(key, shape, dtype, -bound, bound)


class _Linear(nn.Module):
    features: int

    @nn.compact
    def __call__(self, inputs):
        bound = 1. / jnp.sqrt(inputs.shape[-1])
        bias_init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
            key, shape, dtype, minval=-bound, maxval=bound)
        return nn.Dense(self.features, kernel_init=_torch_kernel, bias_init=bias_init)(inputs)


class _Backbone(nn.Module):
    hidden_dims: tuple

    @nn.compact
    def __call__(self, inputs):
        for width in self.hidden_dims:
            inputs = _Linear(width)(inputs)
            # PyTorch LayerNorm default eps is 1e-5 (Flax's default is 1e-6).
            inputs = nn.LayerNorm(epsilon=1e-5)(inputs)
            inputs = jnp.tanh(inputs)
        return inputs


class _NoiseActor(nn.Module):
    hidden_dims: tuple
    action_dim: int

    @nn.compact
    def __call__(self, observations):
        x = _Backbone(self.hidden_dims)(observations)
        mean = _Linear(self.action_dim, name='mean')(x)
        log_std = jnp.clip(_Linear(self.action_dim, name='log_std')(x), -20., 2.)
        base = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


class _QEnsemble(nn.Module):
    hidden_dims: tuple
    num_qs: int

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate((observations, actions), axis=-1)
        values = []
        for head in range(self.num_qs):
            features = _Backbone(self.hidden_dims, name=f'head_{head}')(x)
            values.append(_Linear(1, name=f'value_{head}')(features)[..., 0])
        return jnp.stack(values, axis=0)


class _LogTemperature(nn.Module):
    initial: float

    @nn.compact
    def __call__(self):
        return self.param('log_alpha', lambda key: jnp.asarray(jnp.log(self.initial)))


class DSRLNAAgent(flax.struct.PyTreeNode):
    rng: Any
    actor_network: Any
    critic_network: Any  # Student Q(s,z), no target student in DSRL-NA.
    action_critic_network: Any
    temperature_network: Any
    bc_flow_network: Any
    config: Any = nonpytree_field()

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_noise(self, observations, rng=None, deterministic=False):
        dist = self.actor_network.select('actor')(observations)
        if deterministic:
            return dist.bijector.forward(dist.distribution.mean())
        return dist.sample(seed=self.rng if rng is None else rng)

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_actions(self, observations, rng=None, deterministic=False):
        return self.noise_to_actions_public(
            observations, self.sample_noise(observations, rng, deterministic))

    @jax.jit
    def sample_actions_normal(self, observations, rng):
        z = jax.random.normal(rng, (*observations.shape[:-1], self.config['noise_dim']))
        return self.noise_to_actions_public(observations, z)

    @jax.jit
    def noise_to_actions_public(self, observations, actor_noise):
        single = observations.ndim == 1
        if single:
            observations, actor_noise = observations[None], actor_noise[None]
        result = sample_flow_actions_ode(
            actor_fn=self.bc_flow_network.select('actor_bc_flow'),
            observations=observations, rng=None, flow_steps=self.config['flow_steps'],
            act_dim=self.config['noise_dim'], noises=actor_noise,
            act_min=-1., act_max=1., clip_intermediate=False, is_encoded=True)
        return result[0] if single else result

    @jax.jit
    def get_q_values(self, observations, actions):
        return jnp.min(self.action_critic_network.select('target_critic')(
            observations, actions=actions), axis=0)

    def action_critic_loss(self, batch, params, rng, alpha):
        dist = self.actor_network.select('actor')(batch['next_observations'])
        next_z, log_prob = dist.sample_and_log_prob(seed=rng)
        next_actions = self.noise_to_actions_public(batch['next_observations'], next_z)
        next_q = jnp.min(self.action_critic_network.select('target_critic')(
            batch['next_observations'], actions=next_actions), axis=0)
        # Match the shared baselines: discount by the actual physical chunk length.
        target = jax.lax.stop_gradient(batch['rewards'] + self.config['discount'] ** batch['chunk_length'] *
                                      batch['masks'] * (next_q - alpha * log_prob))
        values = self.action_critic_network.select('critic')(
            batch['observations'], actions=batch['actions'], params=params)
        loss = .5 * jnp.sum(jnp.mean(jnp.square(values - target), axis=-1))
        return loss, {'loss': loss, 'q_mean': values.mean(), 'target_q_mean': target.mean()}

    def critic_loss(self, batch, params, rng):
        z = jax.random.normal(rng, (batch['observations'].shape[0], self.config['noise_dim']))
        actions = self.noise_to_actions_public(batch['observations'], z)
        # Each student learns its corresponding ONLINE teacher head; no aggregation.
        teacher = jax.lax.stop_gradient(self.action_critic_network.select('critic')(
            batch['observations'], actions=actions))
        student = self.critic_network.select('critic')(
            batch['observations'], actions=z, params=params)
        loss = .5 * jnp.sum(jnp.mean(jnp.square(student - teacher), axis=-1))
        return loss, {'distillation_loss': loss, 'q_mean': student.mean(),
                      'target_q_mean': teacher.mean()}

    def actor_loss(self, batch, params, rng, alpha):
        dist = self.actor_network.select('actor')(batch['observations'], params=params)
        z, log_prob = dist.sample_and_log_prob(seed=rng)
        q = jnp.min(self.critic_network.select('critic')(
            batch['observations'], actions=z), axis=0)
        loss = jnp.mean(alpha * log_prob - q)
        return loss, {'actor_loss': loss, 'entropy': -log_prob.mean(),
                      'q_actor_mean': q.mean(), 'temp': alpha}

    def _action_q_step(self, batch, rng, alpha):
        action_q, info = self.action_critic_network.apply_loss_fn(
            lambda p: self.action_critic_loss(batch, p, rng, alpha))
        params = dict(action_q.params)
        params['modules_target_critic'] = jax.tree.map(
            lambda online, target: self.config['tau'] * online + (1 - self.config['tau']) * target,
            params['modules_critic'], params['modules_target_critic'])
        return self.replace(action_critic_network=action_q.replace(params=params)), info

    def _sac_step(self, batch):
        rng, actor_rng, next_rng = jax.random.split(self.rng, 3)
        _, log_prob = self.actor_network.select('actor')(
            batch['observations']).sample_and_log_prob(seed=actor_rng)
        alpha = jax.lax.stop_gradient(jnp.exp(self.temperature_network.select('temp')()))
        entropy_error = jax.lax.stop_gradient(log_prob + self.config['target_entropy'])
        def temperature_loss(params):
            log_alpha = self.temperature_network.select('temp')(params=params)
            loss = -jnp.mean(log_alpha * entropy_error)
            return loss, {'temp_loss': loss}
        temperature, temp_info = self.temperature_network.apply_loss_fn(temperature_loss)
        current, q_info = self._action_q_step(batch, next_rng, alpha)
        # The student is unchanged throughout this whole block of SAC updates.
        actor, actor_info = self.actor_network.apply_loss_fn(
            lambda p: self.actor_loss(batch, p, actor_rng, alpha))
        return current.replace(rng=rng, actor_network=actor,
                               temperature_network=temperature), {
            **{f'action_critic/{k}': v for k, v in q_info.items()},
            **{f'actor/{k}': v for k, v in actor_info.items()},
            **{f'actor/{k}': v for k, v in temp_info.items()},
        }

    def _distill_step(self, batch):
        rng, key = jax.random.split(self.rng)
        critic, info = self.critic_network.apply_loss_fn(
            lambda p: self.critic_loss(batch, p, key))
        return self.replace(rng=rng, critic_network=critic), {
            f'critic/{k}': v for k, v in info.items()}

    @jax.jit
    def batch_update(self, batch, distill_batch=None):
        """One action-Q step per actor, followed by the allocated student steps."""
        agent, sac_info = jax.lax.scan(lambda a, b: a._sac_step(b), self, batch)
        if distill_batch is None:
            return agent, jax.tree.map(jnp.mean, sac_info)
        agent, distill_info = jax.lax.scan(lambda a, b: a._distill_step(b), agent, distill_batch)
        return agent, jax.tree.map(jnp.mean, {**sac_info, **distill_info})

    def restore_bc_flow_params(self, checkpoint_path):
        import pickle
        with open(checkpoint_path, 'rb') as stream:
            saved = pickle.load(stream)['agent']
        if 'bc_flow_network' in saved:
            source = saved['bc_flow_network']['params']['modules_actor_bc_flow']
        else:
            source = saved['actor_network']['params']['modules_actor']
        target = self.bc_flow_network.params['modules_actor_bc_flow']
        leaves, structure = jax.tree.flatten(source)
        other, other_structure = jax.tree.flatten(target)
        if structure != other_structure or any(np.shape(a) != np.shape(b) for a, b in zip(leaves, other)):
            raise ValueError('Offline checkpoint decoder structure/shapes do not match DSRLNA')
        if not all(np.isfinite(x).all() for x in leaves):
            raise ValueError('Offline checkpoint decoder contains non-finite parameters')
        params = dict(self.bc_flow_network.params)
        params['modules_actor_bc_flow'] = jax.tree.map(jnp.asarray, source)
        return self.replace(bc_flow_network=self.bc_flow_network.replace(params=params))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        if config.get('encoder') is not None or ex_observations.ndim != 1:
            raise ValueError('DSRLNA requires flat, pre-encoded observations')
        if not config.get('action_chunking', True) or config.get('noise_chunk_length', -1) != -1:
            raise ValueError('DSRLNA requires full action-chunk latents')
        defaults = dict(num_qs=2, tau=.005, discount=.99, lr_actor=3e-4,
                        lr_critic=3e-4, lr_noise_critic=3e-4, target_entropy=0.,
                        init_temperature=1., noise_critic_updates_per_actor=0.5,
                        num_train_envs=1, flow_steps=10)
        for key, value in defaults.items():
            config.setdefault(key, value)
        for key in ('num_qs', 'num_train_envs', 'flow_steps', 'horizon_length'):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if config['num_train_envs'] != 1:
            raise ValueError('Matched DSRLNA requires exactly one training environment')
        for key in ('lr_actor', 'lr_critic', 'lr_noise_critic', 'init_temperature', 'noise_critic_updates_per_actor'):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 0 <= config['discount'] <= 1 or not 0 < config['tau'] <= 1:
            raise ValueError('DSRLNA requires discount in [0,1] and tau in (0,1]')
        if not np.isfinite(config['target_entropy']):
            raise ValueError('target_entropy must be finite')
        config['ob_dims'] = ex_observations.shape
        config['action_dim'] = ex_actions.shape[-1]
        config['noise_dim'] = config['action_dim'] * config['horizon_length']
        actor = _NoiseActor(tuple(config['actor_hidden_dims']), config['noise_dim'])
        critic = _QEnsemble(tuple(config['value_hidden_dims']), config['num_qs'])
        time_embedding = SinusoidalTimeEmbedding(embed_dim=config['time_embedding_dim'])
        # Exact legacy flow module layout: Dense/GELU, 4x512, no decoder LN.
        decoder = ActorVectorField(
            hidden_dims=tuple(config['decoder_hidden_dims']), action_dim=config['noise_dim'],
            layer_norm=config.get('decoder_layer_norm', False), time_embedding=time_embedding)
        rng, ak, qk, zk, tk, dk = jax.random.split(jax.random.PRNGKey(seed), 6)
        def state(modules, key, args, rate=None):
            model = ModuleDict(modules)
            return TrainState.create(model, model.init(key, **args)['params'],
                                     optax.adam(rate, eps=1e-8) if rate is not None else None)
        obs, z = ex_observations, jnp.zeros(config['noise_dim'])
        actor_state = state({'actor': actor}, ak, {'actor': (obs,)}, config['lr_actor'])
        q_state = state({'critic': critic, 'target_critic': copy.deepcopy(critic)}, qk,
                        {'critic': (obs, z), 'target_critic': (obs, z)}, config['lr_critic'])
        q_state.params['modules_target_critic'] = q_state.params['modules_critic']
        z_state = state({'critic': copy.deepcopy(critic)}, zk, {'critic': (obs, z)}, config['lr_noise_critic'])
        temp_state = state({'temp': _LogTemperature(config['init_temperature'])}, tk,
                           {'temp': ()}, config['lr_actor'])
        decoder_state = state({'actor_bc_flow': decoder}, dk,
                              {'actor_bc_flow': (obs, z, jnp.zeros(1))})
        return cls(rng, actor_state, z_state, q_state, temp_state, decoder_state,
                   flax.core.FrozenDict(config))
