"""DSRL latent TD critic with an optional independent action TD critic.

The action critic is diagnostic: neither the actor nor the latent critic uses it.
"""
from typing import Any

import jax
import jax.numpy as jnp

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.modules.flax_utils import TrainState


class JGCWAgent(DSRLBestOfNAgent):
    action_critic_network: Any = None

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
