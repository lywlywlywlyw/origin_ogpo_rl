"""DSRL latent TD critic with optional augmentation consistency and action Q.

The action critic is diagnostic: neither the actor nor the latent critic uses it.
"""
import math
from typing import Any

import jax
import jax.numpy as jnp

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.modules.flax_utils import TrainState


class JGCWAgent(DSRLBestOfNAgent):
    action_critic_network: Any = None

    @jax.jit
    def inverse_fm_batch(self, observations, actions):
        """GRACE's reverse Euler integration, without constraining recovered z."""
        if observations.shape[0] != actions.shape[0] or actions.ndim != 2:
            raise ValueError('Inverse FM expects equally sized observation/action batches')
        steps = self.config['flow_steps']

        def step(x, i):
            t = jnp.full((x.shape[0], 1), 1. - i / steps)
            velocity = self.bc_flow_network.select('actor_bc_flow')(observations, x, t)
            return x - velocity / steps, None

        return jax.lax.scan(step, actions, jnp.arange(steps))[0]

    @jax.jit
    def augment_z(self, observations, actions, rng):
        """Perturb replay action chunks and invert all B*N samples together."""
        count = self.config['aug_z_num']
        noise = jax.random.normal(
            rng, (actions.shape[0], count, actions.shape[-1]), dtype=actions.dtype)
        # Preserve the requested Gaussian perturbation, including near action bounds.
        augmented = actions[:, None, :] + self.config['aug_z_sigma'] * noise
        repeated_obs = jnp.repeat(observations, count, axis=0)
        latents = self.inverse_fm_batch(
            repeated_obs, augmented.reshape(-1, actions.shape[-1]))
        return jax.lax.stop_gradient(latents.reshape(actions.shape[0], count, -1))

    def critic_loss(self, batch, grad_params, rng):
        td_loss, info = super().critic_loss(batch, grad_params, rng)
        if not self.config.get('use_aug_z', False):
            return td_loss, info

        if 'z_aug' not in batch:
            raise ValueError('use_aug_z requires cached z_aug in the replay batch')
        # Inversion happens once before insertion; Q values remain live predictions.
        z_aug = jax.lax.stop_gradient(batch['z_aug'])
        count = self.config['aug_z_num']
        expected_shape = (batch['actions'].shape[0], count, batch['actions'].shape[-1])
        if z_aug.shape != expected_shape:
            raise ValueError(f'Cached z_aug shape {z_aug.shape} != {expected_shape}')
        q_aug = self.critic_network.select('critic')(
            jnp.repeat(batch['observations'], count, axis=0),
            actions=z_aug.reshape(-1, z_aug.shape[-1]), params=grad_params)
        q_aug = q_aug.reshape(q_aug.shape[0], z_aug.shape[0], count)
        # Match each ensemble member to its original Q(s,z); only z_aug is fitted.
        q = self.critic_network.select('critic')(
            batch['observations'], actions=batch['actions'], params=grad_params)
        consistency_loss = jnp.mean(jnp.square(q_aug - jax.lax.stop_gradient(q)[..., None]))
        total_loss = td_loss + consistency_loss
        return total_loss, {**info, 'td_loss': td_loss,
                            'aug_z_loss': consistency_loss, 'critic_loss': total_loss}

    def action_critic_loss(self, batch, grad_params, rng):
        # Match DSRL's next latent sample and TD convention exactly.
        _, sample_rng = jax.random.split(rng)
        dist = self.actor_network.select('actor')(batch['next_observations'])
        next_z, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        next_a = self.noise_to_actions_public(batch['next_observations'], next_z)
        next_qs = self.action_critic_network.select('target_critic')(
            batch['next_observations'], actions=next_a)
        next_q = (jnp.min(next_qs, axis=0) if self.config['q_agg'] == 'min'
                  else jnp.mean(next_qs, axis=0))
        if self.config['entropy_backup']:
            next_q -= self.actor_network.select('temp')() * jnp.clip(log_probs, -100, 100)
        target = jax.lax.stop_gradient(jnp.clip(
            batch['rewards'] + self.config['discount'] ** batch.get(
                'chunk_length', self.config['horizon_length']) * batch['masks'] * next_q,
            -500., 500.))
        q = self.action_critic_network.select('critic')(
            batch['observations'], actions=batch['true_actions'], params=grad_params)
        loss = jnp.mean(jnp.square(q - target))
        return loss, {'loss': loss, 'q_mean': q.mean(), 'target_q_mean': target.mean()}

    def _action_step(self, batch, rng):
        if self.action_critic_network is None:
            return self, {}
        state, info = self.action_critic_network.apply_loss_fn(
            lambda p: self.action_critic_loss(batch, p, rng))
        self.target_update(state, 'critic')
        return self.replace(action_critic_network=state), {
            f'action_critic/{k}': v for k, v in info.items()}

    @staticmethod
    def _update(agent, batch):
        _, _, rng2 = jax.random.split(agent.rng, 3)
        agent, action_info = agent._action_step(batch, rng2)
        agent, info = DSRLBestOfNAgent._update(agent, batch)
        return agent, {**info, **action_info}

    @staticmethod
    def _update_with_bc(agent, batch_tuple):
        batch, _, _ = batch_tuple
        _, _, rng2, _ = jax.random.split(agent.rng, 4)
        agent, action_info = agent._action_step(batch, rng2)
        agent, info = DSRLBestOfNAgent._update_with_bc(agent, batch_tuple)
        return agent, {**info, **action_info}

    @staticmethod
    def _extra_critic_update(agent, batch, success_batch, success_flag, rng):
        fixed_extra = agent.config.get('critic_updates_per_actor', 1) == 2
        if agent.action_critic_network is not None and (
                fixed_extra or agent.config.get('use_success_buffer_q', False)):
            chosen = jax.lax.cond(success_flag,
                                  lambda: success_batch if success_batch is not None else batch,
                                  lambda: batch)
            agent = jax.lax.cond(jnp.logical_or(fixed_extra, success_flag),
                                 lambda a: a._action_step(chosen, rng)[0], lambda a: a, agent)
        return DSRLBestOfNAgent._extra_critic_update(
            agent, batch, success_batch, success_flag, rng)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        config.setdefault('use_aug_z', False)
        config.setdefault('aug_z_num', 1)
        config.setdefault('aug_z_sigma', 0.02)
        count, sigma = config['aug_z_num'], config['aug_z_sigma']
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('aug_z_num must be a positive integer')
        if (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
                or not math.isfinite(sigma) or sigma < 0):
            raise ValueError('aug_z_sigma must be a finite nonnegative standard deviation')
        if config['use_aug_z'] and config.get('noise_chunk_length', -1) != -1:
            raise ValueError('use_aug_z requires noise_chunk_length=-1 for full-chunk inverse latents')
        config.setdefault('train_q_sa', False)
        config.setdefault('cal_q_accuracy', False)
        if config['cal_q_accuracy'] and not config['train_q_sa']:
            raise ValueError('cal_q_accuracy requires train_q_sa=true')
        if config['cal_q_accuracy'] and config.get('entropy_backup', False):
            raise ValueError('Q accuracy compares reward returns and requires entropy_backup=false')
        base = DSRLBestOfNAgent.create(seed, ex_observations, ex_actions, config)
        action_state = None
        if config['train_q_sa']:
            # Reuse the exact Value architecture and optimizer, with independent
            # parameters, targets and optimizer moments. Preserve DSRL RNGs.
            model = base.critic_network.model_def
            full_actions = (jnp.concatenate([ex_actions] * config['horizon_length'], axis=-1)
                            if config['action_chunking'] else ex_actions)
            params = model.init(jax.random.fold_in(base.rng, 917),
                                critic=(ex_observations, full_actions),
                                target_critic=(ex_observations, full_actions))['params']
            params['modules_target_critic'] = params['modules_critic']
            action_state = TrainState.create(model, params, base.critic_network.tx)
        return cls(rng=base.rng, actor_network=base.actor_network,
                   critic_network=base.critic_network, bc_flow_network=base.bc_flow_network,
                   config=base.config, action_critic_network=action_state)
