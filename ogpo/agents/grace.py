import copy
import math
from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import Actor, Value, ActorVectorField, LogParam
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets
from ogpo.agents.modules.maintenance import (
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
)
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
    compute_flow_bc_loss_online,
)
from ogpo.agents.modules.q_helper import (
    get_q_values as _get_q_values,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
)

class GRACEAgent(flax.struct.PyTreeNode):
    """DSRL with action-Q distillation and input-sensitivity diagnostics.

    The actor and latent critic operate in noise space. Encoder outputs are
    unsquashed, tanh-bounded, hard-clipped, or projected onto the LPS sphere of radius sqrt(noise_dim).
    The frozen flow maps the resulting latent to actions.
    The action critic learns chunk returns via TD. The latent critic regresses
    onto its values at decoded latents. With both use_no_tanh and
    use_layer_norm, distillation mixes prior, replay, and actor latents.
    """

    rng: Any
    actor_network: TrainState
    critic_network: TrainState
    bc_flow_network: TrainState
    action_critic_network: TrainState
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        # Use the same latent prior as the decoder, including sphere projection.
        noises = jax.random.normal(rng, batch['actions'].shape, dtype=batch['actions'].dtype)
        noises = self.constrain_encoder_output(noises)
        mixed = (self.config.get('use_no_tanh', False)
                 and self.config.get('use_layer_norm', False))
        # Preserve state/latent pairing and the original batch size. Each quarter
        # rounds down; any remainder stays Gaussian (B=256 gives 128/64/64).
        quarter = batch['actions'].shape[0] // 4
        if mixed and quarter:
            indices = jax.random.permutation(jax.random.fold_in(rng, 1), noises.shape[0])
            replay_indices = indices[:quarter]
            actor_indices = indices[quarter:2 * quarter]
            dist = self.actor_network.select('actor')(batch['observations'][actor_indices])
            actor_noises = self.constrain_actor_output(
                dist.sample(seed=jax.random.fold_in(rng, 2)))
            noises = noises.at[replay_indices].set(batch['actions'][replay_indices])
            noises = noises.at[actor_indices].set(actor_noises)
            # This optimizer step trains only Qz, including when differentiating
            # the loss through the agent's actor/replay inputs for diagnostics.
            noises = jax.lax.stop_gradient(noises)
        actions = self.noise_to_actions_public(
            batch['observations'], noises, noises_constrained=True)
        target_q = jax.lax.stop_gradient(self._aggregate_q(
            self.action_critic_network.select('critic')(
                batch['observations'], actions=actions)))
        q = self.critic_network.select('critic')(
            batch['observations'], actions=noises, params=grad_params)
        critic_loss = jnp.mean(jnp.square(q - target_q))
        info = {
            'critic_loss': critic_loss,
            'distillation_loss': critic_loss,
            'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'noise_mean': noises.mean(), 'noise_std': noises.std(),
        }
        if mixed:
            # Reuse the exact actor rows evaluated above; no extra decoder batch.
            gap = (self._aggregate_q(q)[actor_indices] - target_q[actor_indices]
                   if quarter else jnp.zeros((1,), dtype=q.dtype))
            info.update(
                actor_distillation_q_gap=gap.mean(),
                actor_distillation_q_abs_gap=jnp.abs(gap).mean(),
                actor_distillation_count=jnp.asarray(quarter),
            )
        return critic_loss, info

    def _aggregate_q(self, qs):
        # SERL uses all ensemble members for the actor and online-Q values.
        if self.config.get('use_layer_norm', False):
            return jnp.mean(qs, axis=0)
        return jnp.min(qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(qs, axis=0)

    def _aggregate_td_target_q(self, qs, rng):
        if not self.config.get('use_layer_norm', False):
            return self._aggregate_q(qs)
        # Match SERL/HIL-SERL REDQ: one pair per minibatch, sampled with
        # replacement. Only the TD bootstrap uses the subsampled minimum;
        # all ten critics still regress onto the shared target.
        _, subsample_rng = jax.random.split(rng)
        indices = jax.random.randint(subsample_rng, (2,), 0, self.config['num_qs'])
        return jnp.min(qs[indices], axis=0)

    def action_critic_loss(self, batch, grad_params, rng):
        # Bootstrap Q^A at the next action decoded from the latent actor.
        target_rng, sample_rng = jax.random.split(rng)
        dist = self.actor_network.select('actor')(batch['next_observations'])
        next_z, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        next_z = self.constrain_actor_output(next_z)
        next_a = self.noise_to_actions_public(
            batch['next_observations'], next_z, noises_constrained=True)
        next_qs = self.action_critic_network.select('target_critic')(
            batch['next_observations'], actions=next_a)
        next_q = self._aggregate_td_target_q(next_qs, target_rng)
        if self.config['entropy_backup']:
            next_q -= self.actor_network.select('temp')() * jnp.clip(log_probs, -100, 100)
        target = jnp.clip(batch['rewards'] + self.config['discount'] ** batch.get(
            'chunk_length', self.config['horizon_length']) * batch['masks'] * next_q, -500., 500.)
        q = self.action_critic_network.select('critic')(
            batch['observations'], actions=batch['true_actions'], params=grad_params)
        loss = jnp.mean(jnp.square(q - target))
        return loss, {'action_critic/loss': loss, 'action_critic/q_mean': q.mean(),
                      'action_critic/target_q_mean': target.mean()}

    @jax.jit
    def pretrain_action_q(self, batch):
        """Use exactly the online action-Q TD update, freezing actor/decoder/Qz."""
        new_rng, loss_rng = jax.random.split(self.rng)
        state, info = self.action_critic_network.apply_loss_fn(
            lambda p: self.action_critic_loss(batch, p, loss_rng))
        self.target_update(state, 'critic')
        return self.replace(action_critic_network=state, rng=new_rng), info

    @jax.jit
    def inverse_fm_batch(self, observations, actions):
        """Reverse Euler integration of this decoder (action t=1 to latent t=0).

        Unlike the GoRL reference, OGPO trains its flow from noise at t=0
        to actions at t=1. Do not clip or constrain the recovered latent.
        """
        if observations.shape[0] != actions.shape[0] or actions.ndim != 2:
            raise ValueError('Inverse FM expects equally sized observation/action batches')
        steps = self.config['flow_steps']

        def step(x, i):
            t = jnp.full((x.shape[0], 1), 1. - i / steps)
            velocity = self.bc_flow_network.select('actor_bc_flow')(observations, x, t)
            return x - velocity / steps, None

        return jax.lax.scan(step, actions, jnp.arange(steps))[0]

    @jax.jit
    def augment_actions_from_z(self, observations, latents, rng):
        """Branch after each of the final K Euler steps; return (B, N**K, A)."""
        steps = self.config['flow_steps']
        branch_steps = self.config['noise_add_start']
        increase_num = self.config['increase_num']
        batch_size, action_dim = latents.shape
        x = latents
        obs = observations
        for i in range(steps):
            t = jnp.full((x.shape[0], 1), i / steps)
            velocity = self.bc_flow_network.select('actor_bc_flow')(obs, x, t)
            x = x + velocity / steps
            if i >= steps - branch_steps:
                x = jnp.repeat(x, increase_num, axis=0)
                obs = jnp.repeat(obs, increase_num, axis=0)
                rng, noise_rng = jax.random.split(rng)
                x = x + jax.random.normal(noise_rng, x.shape, dtype=x.dtype)
        return jnp.clip(x, -1., 1.).reshape(batch_size, increase_num ** branch_steps, action_dim)

    def input_gradient_metrics(self, batch):
        """Pre-update online-Q gradients, at paired replay (s, z, a).

        Differentiate the ensemble aggregate before taking each sample's L2
        norm. R is averaged per sample, not a ratio of batch-averaged norms.
        Radii are half the allowed coordinate intervals, not distances to bounds.
        """
        def gradient_norm(network, inputs):
            def sum_q(x):
                return self._aggregate_q(network.select('critic')(
                    batch['observations'], actions=x)).sum()
            # Value networks do not mix samples. Summing Q avoids a 1/B factor.
            gradients = jax.grad(sum_q)(inputs)
            return jnp.linalg.norm(gradients, axis=-1)

        z_norm = gradient_norm(self.critic_network, batch['actions'])
        a_norm = gradient_norm(self.action_critic_network, batch['true_actions'])
        delta_z = self.config['gradient_delta_z']
        delta_a = self.config['gradient_delta_a']
        z_scaled, a_scaled = delta_z * z_norm, delta_a * a_norm
        ratio = a_scaled / (z_scaled + self.config['gradient_ratio_epsilon'])
        return jax.tree.map(jax.lax.stop_gradient, {
            'sensitivity/grad_z_norm': z_norm.mean(),
            'sensitivity/grad_a_norm': a_norm.mean(),
            'sensitivity/scaled_grad_z_norm': z_scaled.mean(),
            'sensitivity/scaled_grad_a_norm': a_scaled.mean(),
            'sensitivity/ratio': ratio.mean(),
        })

    def _critic_step(self, batch, rng):
        metrics = self.input_gradient_metrics(batch)
        action_rng, latent_rng = jax.random.split(rng)
        action_state, action_info = self.action_critic_network.apply_loss_fn(
            lambda p: self.action_critic_loss(batch, p, action_rng))
        self.target_update(action_state, 'critic')
        teacher = self.replace(action_critic_network=action_state)
        latent_state, latent_info = self.critic_network.apply_loss_fn(
            lambda p: teacher.critic_total_loss(batch, p, latent_rng))
        self.target_update(latent_state, 'critic')
        # Give optimizer diagnostics for Q^A their own namespace as well.
        action_info = {k if k.startswith('action_critic/') else f'action_critic/{k}': v
                       for k, v in action_info.items()}
        return self.replace(critic_network=latent_state, action_critic_network=action_state), {
            **latent_info, **action_info, **metrics}

    def actor_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)
        delete_sac_entropy = self.config.get('delete_sac_entropy', False)
        if delete_sac_entropy:
            actions = dist.sample(seed=sample_rng)
            log_probs = jnp.zeros(actions.shape[:-1], dtype=actions.dtype)
        else:
            actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        # In clip mode log_probs belongs to the pre-clip Gaussian: hard clip
        # is not invertible, so the tanh Jacobian correction does not apply.
        actions = self.constrain_actor_output(actions)

        qs = self.critic_network.select('critic')(batch['observations'], actions=actions)
        q = self._aggregate_q(qs)

        temp = self.actor_network.select('temp')()
        if delete_sac_entropy:
            # Disable both actor entropy regularization and temperature learning.
            actor_loss = -jnp.mean(q)
            temp_loss = jnp.asarray(0., dtype=q.dtype)
            entropy = jnp.asarray(0., dtype=q.dtype)
        else:
            actor_loss = jnp.mean(temp * log_probs - q)
            temp_param = self.actor_network.select('temp')(params=grad_params)
            entropy = -jax.lax.stop_gradient(log_probs).mean()
            temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + temp_loss
        return total_loss, {
            'total_loss': total_loss, 'actor_loss': actor_loss, 'temp_loss': temp_loss,
            'log_probs_mean': log_probs.mean(), 'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(), 'temp': temp, 'actor_mode': self.constrain_actor_output(dist.mode()).mean(),
        }

    def actor_total_loss(self, batch, grad_params, rng):
        a_loss, a_info = self.actor_loss(batch, grad_params, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info

    def actor_bc_loss(self, batch, grad_params, rng):
        successful_noise = batch["actions"]
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)
        actor_noise = self.constrain_actor_output(
            dist.mode() if hasattr(dist, 'mode') else dist.mean())
        bc_loss_unweighted = jnp.mean(jnp.square(actor_noise - successful_noise))
        bc_loss = self.config["bc_coeff"] * bc_loss_unweighted

        return bc_loss, {
            'actor_bc_loss': bc_loss,
            'actor_bc_loss_unweighted': bc_loss_unweighted,
            'actor_noise_mean': actor_noise.mean(),
            'actor_noise_std': actor_noise.std(),
            'successful_noise_mean': successful_noise.mean(),
            'successful_noise_std': successful_noise.std(),
            'bc_mse_error': bc_loss_unweighted,
        }

    def critic_total_loss(self, batch, grad_params, rng):
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    def bc_loss(self, batch, grad_params, rng):
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        rng, targets_rng = jax.random.split(rng)
        x_t, vel, t = get_flow_targets(
            batch['observations'],
            batch_actions,
            targets_rng,
        )
        pred = self.bc_flow_network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))
        bc_flow_loss = apply_chunking_mask(
            jnp.square(pred - vel), valid_mask, batch_size,
            self.config["horizon_length"], self.config["action_dim"],
            self.config["action_chunking"]
        )

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }

    def bc_loss_online(self, batch, grad_params, rng):
        def model_fn(obs, x_t, t):
            return self.bc_flow_network.select('actor_bc_flow')(obs, x_t, t, params=grad_params)

        return compute_flow_bc_loss_online(
            batch, rng, model_fn,
            self.config["action_chunking"],
            self.config["horizon_length"],
            self.config["action_dim"],
            bc_coeff=self.config["bc_coeff"],
            action_key="true_actions"
        )

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _extra_critic_update(agent, batch, success_batch, success_flag, rng, first_info):
        fixed_extra = agent.config.get('critic_updates_per_actor', 1) == 2
        if not fixed_extra and not agent.config.get('use_success_buffer_q', False):
            return agent, first_info
        if success_batch is None:
            success_batch = batch
        chosen = jax.lax.cond(success_flag, lambda: success_batch, lambda: batch)
        def update(current):
            current, extra_info = current._critic_step(chosen, rng)
            # Preserve DSRL's first-update metrics. Only new diagnostics
            # include both Q updates.
            info = {
                key: (value + extra_info[key]) / 2
                if key.startswith(('sensitivity/', 'action_critic/')) else value
                for key, value in first_info.items()
            }
            return current, info
        return jax.lax.cond(jnp.logical_or(fixed_extra, success_flag), update,
                            lambda a: (a, first_info), agent)

    @staticmethod
    def _update(agent, batch: dict) -> Tuple['GRACEAgent', dict]:
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        def actor_loss_fn(p):
            return agent.actor_total_loss(batch, p, rng1)
        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent, critic_info = agent._critic_step(batch, rng2)
        agent = agent.replace(actor_network=new_actor_state, rng=new_rng)
        agent, critic_info = agent._extra_critic_update(
            agent, batch, None, False, jax.random.fold_in(rng2, 1), critic_info)
        return agent, {**actor_info, **critic_info}

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['GRACEAgent', dict]:
        new_rng, rng1 = jax.random.split(agent.rng, 2)
        def bc_loss_fn(p):
            return agent.bc_loss(batch, p, rng1)
        new_bc_flow_state, info = agent.bc_flow_network.apply_loss_fn(bc_loss_fn)
        return agent.replace(bc_flow_network=new_bc_flow_state, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)

    @staticmethod
    def _update_with_bc(agent, batch_tuple: Tuple[dict, dict, bool]) -> Tuple['GRACEAgent', dict]:
        batch, batch_success, success_flag = batch_tuple
        new_rng, rng1, rng2, rng3 = jax.random.split(agent.rng, 4)

        def actor_loss_fn(p):
            rl_loss, rl_info = agent.actor_total_loss(batch, p, rng1)

            def add_bc_loss():
                bc_loss, bc_info = agent.actor_bc_loss(batch_success, p, rng3)
                return rl_loss + bc_loss, {**rl_info, **bc_info}

            def no_bc_loss():
                dummy_bc_info = {
                    'actor_bc_loss': 0.0, 'actor_bc_loss_unweighted': 0.0,
                    'actor_noise_mean': 0.0, 'actor_noise_std': 0.0,
                    'successful_noise_mean': 0.0, 'successful_noise_std': 0.0,
                    'bc_mse_error': 0.0,
                }
                return rl_loss, {**rl_info, **dummy_bc_info}

            return jax.lax.cond(success_flag, add_bc_loss, no_bc_loss)

        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent, critic_info = agent._critic_step(batch, rng2)
        agent = agent.replace(actor_network=new_actor_state, rng=new_rng)
        agent, critic_info = agent._extra_critic_update(
            agent, batch, batch_success, success_flag, rng3, critic_info)
        return agent, {**actor_info, **critic_info}

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        if batch_success is None:
            agent, infos = jax.lax.scan(self._update, self, batch)
        else:
            def scan_fn(agent, inputs):
                batch_i, batch_success_i = inputs
                return self._update_with_bc(agent, (batch_i, batch_success_i, success_flag))
            agent, infos = jax.lax.scan(scan_fn, self, (batch, batch_success))
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def constrain_encoder_output(self, latent):
        """Match LPS ActorVectorField's sphere branch, including its epsilon."""
        if not self.config.get('encoder_sphere_limit', False):
            return latent
        sq_sum = jnp.sum(jnp.square(latent), axis=-1, keepdims=True)
        return latent / jnp.sqrt(sq_sum + 1e-6) * jnp.sqrt(latent.shape[-1])

    def constrain_actor_output(self, latent):
        """Constrain actor samples/modes without changing the decoder prior.

        Clip mode uses an unsquashed Gaussian actor and its pre-clip entropy.
        Warmup/teacher Gaussian samples keep their existing prior semantics.
        """
        if self.config.get('use_encoder_clip', False):
            return jnp.clip(latent, -1.0, 1.0)
        return self.constrain_encoder_output(latent)

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_noise(self, observations, rng=None, deterministic=False):
        dist = self.actor_network.select('actor')(observations)
        if deterministic:
            latent = dist.mode() if hasattr(dist, 'mode') else dist.mean()
        else:
            latent = dist.sample(seed=rng)
        return self.constrain_actor_output(latent)

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_actions(self, observations, rng=None, deterministic=False):
        latent = self.sample_noise(observations, rng, deterministic=deterministic)
        return self.noise_to_actions_public(observations, latent, noises_constrained=True)

    @jax.jit
    def add_decoder_noise(self, actions, noise_rng):
        """Bounded LPS-style exploration noise for online collection only."""
        if not self.config.get('decoder_add_noise', False):
            return actions
        center = jnp.clip(actions, -1.0, 1.0)
        std = self.config.get('decoder_noise_std', 0.09)
        sample = center + std * jax.random.normal(noise_rng, center.shape)
        radius = self.config.get('decoder_noise_clip', 3.0) * std
        sample = jnp.clip(sample, center - radius, center + radius)
        return jnp.clip(sample, -1.0, 1.0)

    @partial(jax.jit, static_argnames=('noises_constrained',))
    def noise_to_actions_public(self, observations, actor_noise, noises_constrained=False):
        if not noises_constrained:
            actor_noise = self.constrain_encoder_output(actor_noise)
        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actor_noise = jnp.concatenate([actor_noise] * self.config["horizon_length"], axis=-1)
        return self.compute_flow_actions(observations, actor_noise, noises_constrained=True)

    @jax.jit
    def sample_actions_normal(self, observations, rng=None):
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        actions = self.compute_flow_actions(observations, noises)
        return actions

    def get_q_values(self, observations, actions, is_encoded=False):
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations, actions=actions,
            q_agg='mean' if self.config.get('use_layer_norm', False) else self.config['q_agg'],
            is_encoded=is_encoded,
        )

    @partial(jax.jit, static_argnames=('noises_constrained',))
    def compute_flow_actions(self, observations, noises=None, params=None, rng=None,
                             noises_constrained=False):
        add_batch_dim = False
        if (self.config['encoder'] is not None and observations.ndim == 3) or \
           (self.config['encoder'] is None and observations.ndim == 1):
            observations = observations[None, ...]
            if noises is not None:
                noises = noises[None, ...]
            add_batch_dim = True

        act_dim = self.config['action_dim'] * (
            self.config['horizon_length'] if self.config['action_chunking'] else 1)
        if noises is None:
            noises = jax.random.normal(rng, (observations.shape[0], act_dim))
            noises_constrained = False
        if not noises_constrained:
            noises = self.constrain_encoder_output(noises)

        if self.config['encoder'] is not None:
            if params is not None:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations, params=params)
            else:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations)

        def actor_fn(obs, actions, t, is_encoded=True):
            if params is not None:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded, params=params)
            else:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn, observations=observations, rng=rng,
            flow_steps=self.config['flow_steps'], act_dim=act_dim,
            act_min=-1.0, act_max=1.0, clip_intermediate=False,
            noises=noises, is_encoded=True,
        )

        if add_batch_dim:
            actions = actions[0]

        return actions

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_bc_flow_params(self, checkpoint_path):
        """Import an OGPO BC actor (or DSRL decoder) without its RL states."""
        import pickle
        import numpy as np

        with open(checkpoint_path, 'rb') as f:
            saved = pickle.load(f)['agent']
        if 'bc_flow_network' in saved:
            source = saved['bc_flow_network']['params']['modules_actor_bc_flow']
        else:
            source = saved['actor_network']['params']['modules_actor']
        target = self.bc_flow_network.params['modules_actor_bc_flow']
        source_leaves, source_tree = jax.tree_util.tree_flatten(source)
        target_leaves, target_tree = jax.tree_util.tree_flatten(target)
        if source_tree != target_tree or any(
            np.shape(a) != np.shape(b) for a, b in zip(source_leaves, target_leaves)
        ):
            raise ValueError('BC flow checkpoint structure/shapes do not match the DSRL decoder')
        if not all(np.isfinite(x).all() for x in source_leaves):
            raise ValueError('BC flow checkpoint contains non-finite parameters')
        params = dict(self.bc_flow_network.params)
        params['modules_actor_bc_flow'] = jax.tree.map(jnp.asarray, source)
        # This is weight initialization, not a resume of the offline optimizer.
        state = self.bc_flow_network.replace(
            params=params, opt_state=self.bc_flow_network.tx.init(params))
        print(f'Restored frozen BC flow from {checkpoint_path}')
        return self.replace(bc_flow_network=state)

    def restore_actor_params(self, actor_path):
        import pickle
        with open(actor_path, 'rb') as f:
            load_dict = pickle.load(f)
        agent_dict = load_dict['agent']

        if 'actor_network' in agent_dict:
            src = agent_dict['actor_network']['params']
            if 'modules_actor' in src:
                self.actor_network.params['modules_actor'] = src['modules_actor']
                if 'modules_temp' in src:
                    self.actor_network.params['modules_temp'] = src['modules_temp']
                print(f"Restored DSRL actor from {actor_path}")
            else:
                raise ValueError(f"No modules_actor in actor_network checkpoint: {actor_path}")
        elif 'network' in agent_dict:
            src = agent_dict['network']['params']
            if 'modules_actor' in src:
                self.actor_network.params['modules_actor'] = src['modules_actor']
                print(f"Restored DSRL actor from unified checkpoint: {actor_path}")
            else:
                raise ValueError(f"No modules_actor in network checkpoint: {actor_path}")
        else:
            raise ValueError(f"Unknown checkpoint format in {actor_path}")

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        config.setdefault('use_no_tanh', False)
        config.setdefault('encoder_sphere_limit', False)
        config.setdefault('use_encoder_clip', False)
        if config['use_encoder_clip'] and config['encoder_sphere_limit']:
            raise ValueError('use_encoder_clip and encoder_sphere_limit are mutually exclusive')
        config.setdefault('delete_sac_entropy', False)
        config.setdefault('use_layer_norm', False)
        if config['use_layer_norm']:
            config['num_qs'] = 10
        config.setdefault('pretrain_qnet', False)
        config.setdefault('increase_num', 2)
        config.setdefault('noise_add_start', 2)
        config.setdefault('pretrain_qnet_epochs', 1)
        for name in ('increase_num', 'noise_add_start', 'pretrain_qnet_epochs'):
            if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
                raise ValueError(f'{name} must be a positive integer')
        if config['pretrain_qnet'] and not 1 <= config['noise_add_start'] <= config['flow_steps']:
            raise ValueError('noise_add_start must not exceed flow_steps')
        config.setdefault('decoder_add_noise', False)
        config.setdefault('decoder_noise_std', 0.09)
        config.setdefault('decoder_noise_clip', 3.0)
        for name in ('decoder_noise_std', 'decoder_noise_clip'):
            config[name] = float(config[name])
            if not math.isfinite(config[name]) or config[name] < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if config['decoder_noise_clip'] == 0:
            raise ValueError('decoder_noise_clip must be positive')
        if config.get('critic_updates_per_actor', 1) not in (1, 2):
            raise ValueError('critic_updates_per_actor must be 1 or 2')
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config["noise_chunk_length"] == -1:
            noise_dim = full_action_dim
            ex_noise = full_actions
        elif config["noise_chunk_length"] == 1:
            noise_dim = action_dim * config["noise_chunk_length"]
            ex_noise = jnp.concatenate([ex_actions] * config["noise_chunk_length"], axis=-1)
        else:
            raise ValueError(f"Invalid noise chunk length: {config['noise_chunk_length']}")

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Two-tier obs encoder: splits encoded [image_features, proprio] obs into
        # separate towers so proprio isn't buried in a high-dim flat MLP.
        _two_tier_kwargs = dict(
            obs_two_tier=config.get('obs_two_tier', False),
            two_tier_img_dim=config.get('_two_tier_img_dim', 0),
            two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
            two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
        )

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'] or config['use_layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type=config['critic_loss_type'],
            num_bins=config['num_bins'],
            q_min=config['q_min'],
            q_max=config['q_max'],
            **_two_tier_kwargs,
        )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=noise_dim,
            layer_norm=config['actor_layer_norm'] or config['use_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=not (config['use_no_tanh'] or config['encoder_sphere_limit'] or config['use_encoder_clip']),
            state_dependent_std=True,
            low=-config["action_magnitude"],
            high=config["action_magnitude"],
            **_two_tier_kwargs,
        )

        temp_def = LogParam(init_value=config['init_temperature'])

        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))

        # Keep the frozen decoder architecture independent of this switch so
        # existing OGPO offline checkpoints remain structurally compatible.
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            time_embedding=time_emb_module,
            **_two_tier_kwargs,
        )

        actor_nets = {
            'actor': actor_def,
            'temp': temp_def,
        }
        actor_args = {
            'actor': (ex_observations,),
            'temp': (),
        }

        critic_nets = {
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
        }
        critic_args = {
            'critic': (ex_observations, ex_noise),
            'target_critic': (ex_observations, ex_noise),
        }

        bc_flow_nets = {
            'actor_bc_flow': actor_bc_flow_def,
        }
        bc_flow_args = {
            'actor_bc_flow': (ex_observations, full_actions, ex_times),
        }

        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)
        bc_flow_net_def = ModuleDict(bc_flow_nets)

        actor_lr = float(config['lr_actor'])
        critic_lr = float(config['lr_critic'])
        bc_flow_lr = float(config['lr_actor_bc_flow'])

        actor_opt_type = config['optimizer_actor']
        critic_opt_type = config['optimizer_critic']
        bc_flow_opt_type = config['optimizer_actor_bc_flow']

        clip_norm = float(config.get('clip_grad_norm', 1000.0))

        def get_optimizer(lr, opt_type, weight_decay=1e-4):
            if opt_type == 'adam':
                base = optax.adam(learning_rate=lr)
            elif opt_type == 'adamw':
                base = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
            elif opt_type == 'sgd':
                base = optax.sgd(learning_rate=lr)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
            return optax.chain(optax.clip_by_global_norm(clip_norm), base)

        actor_tx = get_optimizer(actor_lr, actor_opt_type)
        critic_tx = get_optimizer(critic_lr, critic_opt_type, config.get('critic_weight_decay', 1e-4))
        bc_flow_tx = get_optimizer(bc_flow_lr, bc_flow_opt_type)

        rng, actor_rng, critic_rng, bc_flow_rng = jax.random.split(rng, 4)
        actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
        critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
        bc_flow_params = bc_flow_net_def.init(bc_flow_rng, **bc_flow_args)['params']

        actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
        critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        bc_flow_state = TrainState.create(bc_flow_net_def, bc_flow_params, bc_flow_tx)

        critic_state.params[f'modules_target_critic'] = critic_state.params[f'modules_critic']

        # Independent parameters and optimizer; keep all DSRL RNG splits intact.
        action_critic_def = ModuleDict({
            'critic': copy.deepcopy(critic_def),
            'target_critic': copy.deepcopy(critic_def),
        })
        action_params = action_critic_def.init(jax.random.fold_in(critic_rng, 1),
            critic=(ex_observations, full_actions),
            target_critic=(ex_observations, full_actions))['params']
        action_params['modules_target_critic'] = action_params['modules_critic']
        action_critic_state = TrainState.create(action_critic_def, action_params, critic_tx)
        for key, default in (
            ('gradient_delta_a', (config['act_max'] - config['act_min']) / 2),
            ('gradient_delta_z', config['action_magnitude']),
            ('gradient_ratio_epsilon', 1e-8),
        ):
            if config.get(key) is None:
                config[key] = default
            if not 0 < float(config[key]) < float('inf'):
                raise ValueError(f'{key} must be finite and positive')

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config["noise_dim"] = noise_dim

        if config['target_entropy'] is None:
            config['target_entropy'] = -float(noise_dim) / 2

        return cls(rng, actor_state, critic_state, bc_flow_state, action_critic_state, config=flax.core.FrozenDict(**config))
