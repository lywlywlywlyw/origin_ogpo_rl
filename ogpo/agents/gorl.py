"""GoRL-FM: reference latent PPO/CFM with a DSRL-compatible decoder backbone.

Reference: ref_codes/GoRL/src/flow_policy/{ppo,rollouts}.py. Only the
observation/action interface and physical chunk discount adapt to Robomimic.
"""
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.networks import Actor, ActorVectorField
from ogpo.networks.modules.mlp import MLP
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding


class GoRLEncoder(Actor):
    """DSRL backbone/head shapes, reference softplus Gaussian parameterization."""
    policy_scale_cap: float = 10.

    def __call__(self, observations):
        features = self.actor_net(observations)
        mean = self.mean_net(features)
        scale = jnp.minimum(jax.nn.softplus(self.log_std_net(features)) + 1e-3,
                            self.policy_scale_cap)
        return distrax.MultivariateNormalDiag(mean, scale)


class GoRLValue(nn.Module):
    hidden_dims: Any

    @nn.compact
    def __call__(self, observations):
        return MLP((*self.hidden_dims, 1), activations=jax.nn.silu,
                   kernel_init=nn.initializers.lecun_uniform())(observations).squeeze(-1)


def reference_gae(rewards, values, next_values, discounts, truncations, valid, lam):
    """Reference GoRL returns and policy advantages, with padded sequence support.

    Shapes (T,B). When valid=1 and next_values is shifted values plus the final
    bootstrap this is exactly flow_policy.rollouts.compute_gae. Timeouts mask
    the TD residual, advantage and value loss as in the reference.
    """
    mask = (1. - truncations) * valid
    deltas = (rewards + discounts * next_values - values) * mask
    def step(carry, inputs):
        delta, scale = inputs
        carry = delta + scale * carry
        return carry, carry
    _, residual = jax.lax.scan(step, jnp.zeros_like(values[0]),
        (deltas, discounts * lam * mask), reverse=True)
    returns = values + residual
    next_valid = jnp.concatenate((valid[1:], jnp.zeros_like(valid[:1])))
    shifted = jnp.concatenate((returns[1:], next_values[-1:]))
    next_returns = jnp.where(next_valid > 0, shifted, next_values)
    advantages = (rewards + discounts * next_returns - values) * mask
    return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(advantages)


class GoRLAgent(DSRLBestOfNAgent):
    decoder_loaded: Any = None
    stage: Any = None
    obs_count: Any = None
    obs_mean: Any = None
    obs_var_sum: Any = None
    obs_std: Any = None
    decoder_obs_mean: Any = None
    decoder_obs_std: Any = None

    def restore_bc_flow_params(self, checkpoint_path):
        agent = super().restore_bc_flow_params(checkpoint_path)
        # Identity preprocessing is essential for the existing offline weights.
        return agent.replace(decoder_loaded=jnp.asarray(True),
            decoder_obs_mean=jnp.zeros(self.config['ob_dims']),
            decoder_obs_std=jnp.ones(self.config['ob_dims']))

    def reset_encoder(self, stage):
        """Reference default 'fresh': random parameters, fresh value/Adam/stats."""
        rng, ak, vk = jax.random.split(self.rng, 3)
        obs = jnp.zeros(self.config['ob_dims'])
        ap = self.actor_network.model_def.init(ak, actor=(obs,))['params']
        vp = self.critic_network.model_def.init(vk, value=(obs,))['params']
        return self.replace(rng=rng, stage=jnp.asarray(stage),
            actor_network=self.actor_network.replace(params=ap, step=1,
                opt_state=self.actor_network.tx.init(ap)),
            critic_network=self.critic_network.replace(params=vp, step=1,
                opt_state=self.critic_network.tx.init(vp)),
            obs_count=jnp.asarray(0., jnp.float32), obs_mean=jnp.zeros_like(obs),
            obs_var_sum=jnp.zeros_like(obs), obs_std=jnp.ones_like(obs))

    def reset_decoder(self, train_observations):
        """Reference fresh, zero-velocity initialization and train-only obs stats."""
        rng, key = jax.random.split(self.rng)
        obs = jnp.zeros(self.config['ob_dims'])
        params = self.bc_flow_network.model_def.init(key,
            actor_bc_flow=(obs, jnp.zeros(self.config['noise_dim']), jnp.zeros(1)))['params']
        head = f"Dense_{len(self.config['actor_hidden_dims'])}"
        params['modules_actor_bc_flow']['mlp'][head] = jax.tree.map(
            jnp.zeros_like, params['modules_actor_bc_flow']['mlp'][head])
        data = np.asarray(train_observations, np.float64)
        return self.replace(rng=rng, bc_flow_network=self.bc_flow_network.replace(
            params=params, step=1, opt_state=self.bc_flow_network.tx.init(params)),
            decoder_obs_mean=jnp.asarray(data.mean(0), jnp.float32),
            decoder_obs_std=jnp.asarray(data.std(0) + 1e-8, jnp.float32))

    @jax.jit
    def update_observation_stats(self, observations):
        """Cumulative Welford update, installed AFTER all PPO epochs."""
        x = observations.reshape((-1, observations.shape[-1]))
        count = self.obs_count + x.shape[0]
        difference = x - self.obs_mean
        mean = self.obs_mean + difference.sum(0) / count
        var_sum = self.obs_var_sum + (difference * (x - mean)).sum(0)
        return self.replace(obs_count=count, obs_mean=mean, obs_var_sum=var_sum,
                            obs_std=jnp.sqrt(jnp.clip(var_sum / count, 1e-12, 1e12)))

    def policy_distribution(self, observations, params=None):
        return self.actor_network.select('actor')(
            (observations - self.obs_mean) / self.obs_std, params=params)

    @jax.jit
    def values(self, observations):
        return self.critic_network.select('value')((observations - self.obs_mean) / self.obs_std)

    @jax.jit
    def compute_flow_actions(self, observations, noises=None, params=None, rng=None):
        normalized = (observations - self.decoder_obs_mean) / self.decoder_obs_std
        return super().compute_flow_actions(normalized, noises=noises, params=params, rng=rng)

    @jax.jit
    def sample_latent(self, observations, rng):
        z, log_prob = self.policy_distribution(observations).sample_and_log_prob(seed=rng)
        return z, log_prob, self.values(observations)

    @jax.jit
    def sample_actions(self, observations, rng=None, deterministic=False):
        dist = self.policy_distribution(observations)
        z = jnp.where(deterministic, dist.mode(), dist.sample(seed=rng))
        return self.compute_flow_actions(observations, noises=z)

    def batch_update(self, *args, **kwargs):
        raise ValueError('GoRL requires on-policy sequence PPO, not SAC replay')

    def bc_update(self, *args, **kwargs):
        raise ValueError('GoRL initializes from BC; use decoder_update at stage boundaries')

    def ppo_loss(self, batch, params):
        ap, vp = params
        obs = (batch['observations'] - self.obs_mean) / self.obs_std
        nxt = (batch['next_observations'] - self.obs_mean) / self.obs_std
        value = self.critic_network.select('value')(obs, params=vp)
        next_value = self.critic_network.select('value')(nxt, params=vp)
        valid = batch['valid']
        mean = lambda x: (x * valid).sum() / jnp.maximum(valid.sum(), 1.)
        returns, advantages = reference_gae(
            batch['rewards'] * self.config['gorl_reward_scaling'], value, next_value,
            self.config['discount'] ** batch['chunk_length'] * batch['masks'],
            batch['truncations'], valid, self.config['gorl_gae_lambda'])
        centered = advantages - mean(advantages)
        advantages = centered / (jnp.sqrt(mean(centered ** 2)) + 1e-8)
        dist = self.actor_network.select('actor')(obs, params=ap)
        log_ratio = dist.log_prob(batch['actions']) - batch['old_log_probs']
        ratio = jnp.exp(log_ratio)
        eps = jnp.where(self.stage == 0, self.config['gorl_ppo_clip_first'], self.config['gorl_ppo_clip'])
        policy = -mean(jnp.minimum(ratio * advantages,
                                  jnp.clip(ratio, 1 - eps, 1 + eps) * advantages))
        value_loss = self.config['gorl_value_coef'] * mean(
            ((returns - value) * (1 - batch['truncations'])) ** 2)
        entropy = mean(dist.entropy())
        reg_coef = jnp.where(self.stage == 0, self.config['gorl_z_reg_first'], self.config['gorl_z_reg'])
        regularizer = reg_coef * mean(jnp.mean(dist.mean() ** 2 + dist.variance(), -1))
        loss = policy + value_loss - self.config['gorl_entropy_coef'] * entropy + regularizer
        return loss, dict(policy_loss=policy, value_loss=value_loss, entropy=entropy,
                         z_reg_loss=regularizer, approximate_kl=mean(ratio - 1 - log_ratio),
                         clip_fraction=mean((jnp.abs(ratio - 1) > eps).astype(jnp.float32)))

    @jax.jit
    def ppo_update(self, minibatches):
        def update(agent, batch):
            (_, info), grads = jax.value_and_grad(agent.ppo_loss, argnums=1, has_aux=True)(
                batch, (agent.actor_network.params, agent.critic_network.params))
            norm = optax.global_norm(grads)
            scale = jnp.minimum(1., agent.config['gorl_max_grad_norm'] / (norm + 1e-8))
            ga, gv = jax.tree.map(lambda x: x * scale, grads)
            # Independent Adam states with the same step/lr are leafwise identical
            # to the reference's one Adam over the joint actor/value parameter tree.
            agent = agent.replace(actor_network=agent.actor_network.apply_gradients(grads=ga),
                                  critic_network=agent.critic_network.apply_gradients(grads=gv))
            return agent, dict(info, grad_norm_before_clip=norm, grad_norm=norm * scale)
        agent, info = jax.lax.scan(update, self, minibatches)
        return agent, jax.tree.map(lambda x: x.mean(), info)

    def decoder_loss(self, batch, params, rng):
        """Independent Gaussian CFM; forward time preserves the shared BC ODE."""
        nk, tk = jax.random.split(rng)
        action = batch['true_actions']
        obs = (batch['observations'] - self.decoder_obs_mean) / self.decoder_obs_std
        n = self.config['gorl_samples_per_action']
        noise = jax.random.normal(nk, (*action.shape[:-1], n, action.shape[-1]))
        t = jax.random.uniform(tk, (*action.shape[:-1], n, 1))
        x = (1 - t) * noise + t * action[..., None, :]
        obs = jnp.broadcast_to(obs[..., None, :], (*noise.shape[:-1], obs.shape[-1]))
        prediction = self.bc_flow_network.select('actor_bc_flow')(obs, x, t, params=params)
        mask = batch['action_valid'][..., None, :]
        loss = (((prediction - (action[..., None, :] - noise)) ** 2) * mask).sum()
        loss /= jnp.maximum(mask.sum() * n, 1.)
        return loss, {'flow_loss': loss}

    @jax.jit
    def decoder_update(self, batch):
        rng, key = jax.random.split(self.rng)
        decoder, info = self.bc_flow_network.apply_loss_fn(lambda p: self.decoder_loss(batch, p, key))
        return self.replace(bc_flow_network=decoder, rng=rng), info

    @jax.jit
    def decoder_validation(self, batch, rng):
        return self.decoder_loss(batch, self.bc_flow_network.params, rng)[0]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        required = ('gorl_ppo_epochs', 'gorl_unroll_length', 'gorl_ppo_batch_size',
            'gorl_num_minibatches', 'gorl_rollout_max_env_steps', 'gorl_collection_steps',
            'gorl_collection_episodes', 'gorl_decoder_steps', 'gorl_decoder_batch_size',
            'gorl_samples_per_action', 'gorl_decoder_eval_interval', 'gorl_decoder_eval_batches')
        for key in required:
            if not isinstance(config.get(key), int) or config[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if config.get('encoder') is not None or config.get('obs_two_tier', False):
            raise ValueError('GoRL requires flat observations or frozen embeddings')
        if config['noise_chunk_length'] != -1 or not config['action_chunking']:
            raise ValueError('GoRL requires full action-chunk noise')
        if config['gorl_decoder_initialization'] != 'fresh':
            raise ValueError('Reference DMControl GoRL uses fresh Decoder fits')
        if config['lr_actor'] != config['lr_critic']:
            raise ValueError('Reference joint PPO uses the same encoder/value learning rate')
        weights = tuple(config['gorl_stage_weights'])
        if not weights or any(not np.isfinite(x) or x <= 0 for x in weights):
            raise ValueError('Invalid GoRL stage weights')
        if not 0 < config['gorl_validation_fraction'] < 1 or not 0 <= config['gorl_gae_lambda'] <= 1:
            raise ValueError('Invalid GoRL validation fraction or lambda')
        config['gorl_stage_weights'] = weights
        config.update(ob_dims=ex_observations.shape, action_dim=ex_actions.shape[-1],
                      noise_dim=ex_actions.shape[-1] * config['horizon_length'])
        obs, dim = ex_observations, config['noise_dim']
        actor = ModuleDict(dict(actor=GoRLEncoder(hidden_dims=config['actor_hidden_dims'],
            action_dim=dim, layer_norm=config['actor_layer_norm'],
            policy_scale_cap=config['gorl_policy_scale_cap'])))
        value = ModuleDict(dict(value=GoRLValue(tuple(config['value_hidden_dims']))))
        time_embedding = (SinusoidalTimeEmbedding(config.get('time_embedding_dim', 32))
                          if config.get('time_embedding') == 'sinusoidal' else None)
        decoder = ModuleDict(dict(actor_bc_flow=ActorVectorField(
            hidden_dims=config['actor_hidden_dims'], action_dim=dim,
            layer_norm=config['actor_layer_norm'], time_embedding=time_embedding)))
        rng, ak, vk, dk = jax.random.split(jax.random.PRNGKey(seed), 4)
        ap = actor.init(ak, actor=(obs,))['params']
        vp = value.init(vk, value=(obs,))['params']
        dp = decoder.init(dk, actor_bc_flow=(obs, jnp.zeros(dim), jnp.zeros(1)))['params']
        return cls(rng=rng, actor_network=TrainState.create(actor, ap, optax.adam(config['lr_actor'])),
            critic_network=TrainState.create(value, vp, optax.adam(config['lr_critic'])),
            bc_flow_network=TrainState.create(decoder, dp, optax.adam(config['lr_actor_bc_flow'])),
            config=flax.core.FrozenDict(config), decoder_loaded=jnp.asarray(False), stage=jnp.asarray(0),
            obs_count=jnp.asarray(0., jnp.float32), obs_mean=jnp.zeros_like(obs),
            obs_var_sum=jnp.zeros_like(obs), obs_std=jnp.ones_like(obs),
            decoder_obs_mean=jnp.zeros_like(obs), decoder_obs_std=jnp.ones_like(obs))
