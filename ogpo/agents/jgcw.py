"""DSRL latent TD critic with optional augmentation consistency and action Q.

The optional action critic can provide the latent TD bootstrap when enabled.
The actor always optimizes the latent critic.
"""
import math
from typing import Any

import jax
import jax.numpy as jnp
import optax

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.modules.flax_utils import TrainState


class JGCWAgent(DSRLBestOfNAgent):
    action_critic_network: Any = None
    temperature_network: Any = None
    actor_clip_update_count: Any = 0
    actor_clip_reject_count: Any = 0

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        agent, info = super().batch_update(batch, batch_success, success_flag)
        if self.config.get('use_klactor_clip', False):
            # The parent averages scan diagnostics. Report the final running
            # rejection fraction rather than averaging intermediate fractions.
            info = {**info, 'actor/rejection_rate': (
                agent.actor_clip_reject_count / jnp.maximum(agent.actor_clip_update_count, 1))}
        return agent, info

    def actor_loss(self, batch, grad_params, rng):
        if not self.config.get('use_klactor_clip', False):
            return super().actor_loss(batch, grad_params, rng)
        # Temperature has its own optimizer. Keep the original SAC objective.
        grad_params = dict(grad_params)
        grad_params['modules_temp'] = jax.tree.map(
            jax.lax.stop_gradient, grad_params['modules_temp'])
        loss, info = super().actor_loss(batch, grad_params, rng)
        # Reproduce SAC's exact reparameterized draw with its existing seed.
        # Keep the pre-tanh sample: the common bijector cancels from the density
        # ratio, so no inverse-tanh of possibly saturated actions is needed.
        _, sample_rng = jax.random.split(rng)
        base = self.actor_network.select('actor')(
            batch['observations'], params=grad_params).distribution
        samples = base.sample(seed=sample_rng)
        # Evaluate both densities at the stored floating-point sample. This
        # keeps an unchanged policy's ratio at one even at very small stds.
        log_probs = base.log_prob(samples)
        return loss, {**info, '_clip_samples': samples, '_clip_log_probs': log_probs}

    def _actor_base_distribution(self, observations, actor_params):
        params = dict(self.actor_network.params, modules_actor=actor_params)
        return self.actor_network.select('actor')(observations, params=params).distribution

    def _apply_actor_loss(self, loss_fn, batch):
        if not self.config.get('use_klactor_clip', False):
            return super()._apply_actor_loss(loss_fn, batch)

        old_state = self.actor_network
        candidate, info = old_state.apply_loss_fn(loss_fn)
        samples = jax.lax.stop_gradient(info.pop('actor/_clip_samples'))
        old_log_probs = jax.lax.stop_gradient(info.pop('actor/_clip_log_probs'))
        observations = batch['observations']
        old_params = old_state.params['modules_actor']
        proposed_params = candidate.params['modules_actor']
        epsilon = self.config['actor_ratio_clip_epsilon']
        log_lower, log_upper = jnp.log1p(-epsilon), jnp.log1p(epsilon)

        def log_ratios(params):
            dist = self._actor_base_distribution(observations, params)
            return dist.log_prob(samples) - old_log_probs

        def acceptable(params, log_ratio):
            finite_params = jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)]))
            return (finite_params & jnp.all(jnp.isfinite(log_ratio))
                    & jnp.all((log_ratio >= log_lower) & (log_ratio <= log_upper)))

        before_log_ratio = log_ratios(proposed_params)

        def needs_backtrack(carry):
            steps, _, params, log_ratio = carry
            return ((steps < self.config['actor_ratio_clip_max_backtracks'])
                    & ~acceptable(params, log_ratio))

        def backtrack(carry):
            steps, scale, _, _ = carry
            scale *= 0.5
            params = jax.tree.map(
                lambda old, proposed: old + scale * (proposed - old),
                old_params, proposed_params)
            return steps + 1, scale, params, log_ratios(params)

        steps, scale, params, log_ratio = jax.lax.while_loop(
            needs_backtrack, backtrack,
            (jnp.int32(0), jnp.float32(1.), proposed_params, before_log_ratio))
        accepted = acceptable(params, log_ratio)
        scaled = candidate.replace(params=dict(candidate.params, modules_actor=params))
        # Accepted backtracking keeps the candidate Adam moments. Full rejection
        # restores parameters, moments and the actor optimizer's step counter.
        actor = jax.lax.cond(accepted, lambda: scaled, lambda: old_state)
        after_log_ratio = jax.lax.cond(
            accepted, lambda: log_ratio, lambda: jnp.zeros_like(log_ratio))

        # Use the same pre-update actor samples/entropy as the original SAC loss.
        # Read temperature from actor_network so parameter-only restores are
        # respected too; temperature optimizer moments remain independent.
        temperature = self.temperature_network.replace(params={
            'modules_temp': old_state.params['modules_temp']})
        entropy_error = jax.lax.stop_gradient(
            info['actor/entropy'] - self.config['target_entropy'])

        def temperature_loss(p):
            return temperature.select('temp')(params=p) * entropy_error, {}

        temperature, _ = temperature.apply_loss_fn(temperature_loss)
        actor = actor.replace(params=dict(
            actor.params, modules_temp=temperature.params['modules_temp']))

        new_dist = self._actor_base_distribution(
            observations, actor.params['modules_actor'])
        count = self.actor_clip_update_count + 1
        rejected = (~accepted).astype(jnp.int32)
        reject_count = self.actor_clip_reject_count + rejected
        for label, values in (('before', before_log_ratio), ('after', after_log_ratio)):
            ratios = jnp.exp(values)
            outside = (~jnp.isfinite(values) | (values < log_lower) | (values > log_upper))
            info.update({
                f'actor/ratio_{label}_mean': ratios.mean(),
                f'actor/ratio_{label}_std': ratios.std(),
                f'actor/ratio_{label}_min': ratios.min(),
                f'actor/ratio_{label}_max': ratios.max(),
                f'actor/ratio_{label}_p01': jnp.quantile(ratios, .01),
                f'actor/ratio_{label}_p99': jnp.quantile(ratios, .99),
                f'actor/ratio_{label}_outside_fraction': outside.astype(jnp.float32).mean(),
                f'actor/log_ratio_{label}_min': values.min(),
                f'actor/log_ratio_{label}_max': values.max(),
            })
        info = {**info,
                'actor/update_scale': jnp.where(accepted, scale, 0.),
                'actor/backtrack_steps': steps,
                'actor/update_rejected': rejected.astype(jnp.float32),
                'actor/rejection_rate': reject_count / count,
                'actor/pre_tanh_std_mean': new_dist.stddev().mean(),
                'actor/pre_tanh_std_p95': jnp.quantile(new_dist.stddev(), .95),
                'actor/temp_after': actor.select('temp')()}
        return self.replace(
            temperature_network=temperature, actor_clip_update_count=count,
            actor_clip_reject_count=reject_count), actor, info

    @jax.jit
    def inverse_fm_batch(self, observations, actions, initial_latents=None):
        """Solve the *executed discrete decoder* inverse with safeguarded LM.

        Plain UniSteer Picard iteration requires dt*L < 1, which our frozen
        decoder need not satisfy. Solve the reconstruction least-squares problem
        instead, including final clipping. Replay z provides the known local
        branch. Unreachable noisy actions receive a bounded best approximation.
        """
        if observations.shape[0] != actions.shape[0] or actions.ndim != 2:
            raise ValueError('Inverse FM expects equally sized observation/action batches')
        initial = jnp.zeros_like(actions) if initial_latents is None else initial_latents
        if initial.shape != actions.shape:
            raise ValueError('Inverse initial latents must match the action batch shape')
        steps = self.config['flow_steps']
        iterations = self.config.get('inverse_fm_iterations', 16)
        radius = self.config.get('inverse_fm_radius', 0.5)
        velocity = self.bc_flow_network.select('actor_bc_flow')

        def decode_one(obs, z):
            def step(x, i):
                t = jnp.full((1, 1), i / steps)
                return x + velocity(obs[None], x[None], t)[0] / steps, None
            raw = jax.lax.scan(step, z, jnp.arange(steps))[0]
            return jnp.clip(raw, -1., 1.)

        decode = jax.vmap(decode_one)
        jacobian = jax.vmap(jax.jacfwd(decode_one, argnums=1))
        # Actor support is [-1,1]. Include the anchor even for external diagnostic
        # callers supplying an out-of-support z, preserving the no-worse guarantee.
        lower, upper = jnp.minimum(initial, -1.), jnp.maximum(initial, 1.)

        def refine(_, x):
            residual = decode(observations, x) - actions
            jac = jacobian(observations, x)
            jt = jnp.swapaxes(jac, -1, -2)
            delta = jnp.linalg.solve(
                jt @ jac + 1e-3 * jnp.eye(actions.shape[-1]),
                jt @ residual[..., None])[..., 0]
            delta /= jnp.maximum(1., jnp.linalg.norm(delta, axis=-1, keepdims=True) / .25)

            # Evaluate all six line-search candidates in one batched decoder
            # pass instead of six serial GPU passes. Keep the same best-residual
            # acceptance rule (including retaining x if every trial is rejected).
            scales = jnp.power(.5, jnp.arange(6))[:, None, None]
            candidates = jnp.clip(x[None] - scales * delta[None], lower, upper)
            offsets = candidates - initial
            candidates = initial + offsets / jnp.maximum(
                1., jnp.linalg.norm(offsets, axis=-1, keepdims=True) / radius)
            decoded = jax.vmap(lambda candidate: decode(observations, candidate))(candidates)
            errors = jnp.sum(jnp.square(decoded - actions), axis=-1)
            errors = jnp.where(jnp.isfinite(errors), errors, jnp.inf)
            winners = jnp.argmin(errors, axis=0)
            rows = jnp.arange(actions.shape[0])
            best = candidates[winners, rows]
            accept = errors[winners, rows] < jnp.sum(jnp.square(residual), axis=-1)
            return jnp.where(accept[:, None], best, x)

        return jax.lax.stop_gradient(jax.lax.fori_loop(0, iterations, refine, initial))

    @jax.jit
    def augment_z(self, observations, actions, rng, initial_latents=None):
        """Perturb chunks once; invert in fixed microbatches to bound AD memory."""
        count = self.config['aug_z_num']
        if rng.ndim == 2:
            noise = jax.vmap(lambda key: jax.random.normal(
                key, (count, actions.shape[-1]), dtype=actions.dtype))(rng)
        else:
            noise = jax.random.normal(
                rng, (actions.shape[0], count, actions.shape[-1]), dtype=actions.dtype)
        augmented = actions[:, None, :] + self.config['aug_z_sigma'] * noise
        anchors = jnp.zeros_like(actions) if initial_latents is None else initial_latents
        repeated_obs = jnp.repeat(observations, count, axis=0)
        targets = augmented.reshape(-1, actions.shape[-1])
        anchors = jnp.repeat(anchors, count, axis=0)
        size = min(self.config.get('inverse_fm_batch_size', 128),
                   1 << (targets.shape[0] - 1).bit_length())
        padding = (-targets.shape[0]) % size

        def blocks(x):
            x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1), mode='edge')
            return x.reshape((-1, size) + x.shape[1:])

        latents = jax.lax.map(
            lambda block: self.inverse_fm_batch(*block),
            (blocks(repeated_obs), blocks(targets), blocks(anchors)))
        latents = latents.reshape(-1, actions.shape[-1])[:targets.shape[0]]
        return jax.lax.stop_gradient(latents.reshape(actions.shape[0], count, -1))

    def _clip_td_target(self, target):
        if self.config.get('use_no_targetq_clip', False):
            return target
        return super()._clip_td_target(target)

    def critic_loss(self, batch, grad_params, rng):
        if self.config.get('train_q_sa', False) and self.config.get('use_qa_tdtarget', False):
            target, next_z, log_probs = self._action_td_target(batch, rng)
            q = self.critic_network.select('critic')(
                batch['observations'], actions=batch['actions'], params=grad_params)
            td_loss = jnp.mean(jnp.square(q - target))
            info = {
                'critic_loss': td_loss,
                'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
                'target_q_mean': target.mean(), 'temp': self.actor_network.select('temp')(),
                'next_actions_mean': next_z.mean(), 'next_actions_min': next_z.min(),
                'next_actions_max': next_z.max(), 'next_actions_std': next_z.std(),
                'rewards_mean': batch['rewards'].mean(),
                'rewards_min': batch['rewards'].min(), 'rewards_max': batch['rewards'].max(),
            }
            if self.config['entropy_backup']:
                info['next_log_probs_mean'] = jnp.clip(log_probs, -100, 100).mean()
        else:
            td_loss, info = super().critic_loss(batch, grad_params, rng)
        total_loss = td_loss
        if (self.config.get('train_q_sa', False) and self.config.get('use_qa_tdtarget', False)
                and self.config.get('use_qa_distill', False)):
            distill_loss, distill_info = self.qa_distillation_loss(
                batch, grad_params, jax.random.fold_in(rng, 919))
            total_loss += distill_loss
            info = {**info, **distill_info, 'td_loss': td_loss, 'critic_loss': total_loss}
        if not self.config.get('use_aug_z', False):
            return total_loss, info

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
        total_loss += consistency_loss
        return total_loss, {**info, 'td_loss': td_loss,
                            'aug_z_loss': consistency_loss, 'critic_loss': total_loss,
                            'aug_z_delta_rms': jnp.sqrt(jnp.mean(jnp.square(
                                z_aug - batch['actions'][:, None, :]))),
                            'aug_z_unchanged_fraction': jnp.mean(jnp.all(
                                z_aug == batch['actions'][:, None, :], axis=-1))}

    def qa_distillation_loss(self, batch, grad_params, rng):
        """GRACE-style prior MSE: online Qa ensemble aggregate teaches every Qz head."""
        noises = jax.random.normal(rng, batch['actions'].shape, dtype=batch['actions'].dtype)
        actions = self.noise_to_actions_public(batch['observations'], noises)
        teacher_qs = self.action_critic_network.select('critic')(
            batch['observations'], actions=actions)
        target = jax.lax.stop_gradient(
            teacher_qs.min(axis=0) if self.config['q_agg'] == 'min' else teacher_qs.mean(axis=0))
        q = self.critic_network.select('critic')(
            batch['observations'], actions=noises, params=grad_params)
        loss = jnp.mean(jnp.square(q - target))
        return loss, {'distillation_loss': loss, 'distill_q_mean': q.mean(),
                      'distill_target_q_mean': target.mean(),
                      'distill_noise_mean': noises.mean(), 'distill_noise_std': noises.std()}

    def _action_td_target(self, batch, rng):
        """Reward + discounted target Qa(s', decoder(s', actor(s'))), detached."""
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
        target = batch['rewards'] + self.config['discount'] ** batch.get(
            'chunk_length', self.config['horizon_length']) * batch['masks'] * next_q
        target = jax.lax.stop_gradient(self._clip_td_target(target))
        return target, next_z, log_probs

    def action_critic_loss(self, batch, grad_params, rng):
        target, _, _ = self._action_td_target(batch, rng)
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
    def _update_q_first(agent, batch, success_batch=None, success_flag=False):
        """Finish all Qa updates, then all Qz updates, then update the actor."""
        with_bc = success_batch is not None
        keys = jax.random.split(agent.rng, 4 if with_bc else 3)
        new_rng, actor_rng, critic_rng = keys[:3]
        extra_rng = keys[3] if with_bc else jax.random.fold_in(critic_rng, 1)

        agent, action_info = agent._action_step(batch, critic_rng)
        agent = agent._extra_action_update(
            agent, batch, success_batch, success_flag, extra_rng)

        critic, critic_info = agent.critic_network.apply_loss_fn(
            lambda p: agent.critic_total_loss(batch, p, critic_rng))
        agent.target_update(critic, 'critic')
        agent = agent.replace(critic_network=critic)
        agent = DSRLBestOfNAgent._extra_critic_update(
            agent, batch, success_batch, success_flag, extra_rng)

        def actor_loss_fn(params):
            loss, info = agent.actor_total_loss(batch, params, actor_rng)
            if not with_bc:
                return loss, info

            def add_bc():
                bc_loss, bc_info = agent.actor_bc_loss(success_batch, params, extra_rng)
                return loss + bc_loss, {**info, **bc_info}

            def skip_bc():
                return loss, {**info, **dict.fromkeys((
                    'actor_bc_loss', 'actor_bc_loss_unweighted',
                    'actor_noise_mean', 'actor_noise_std',
                    'successful_noise_mean', 'successful_noise_std', 'bc_mse_error'), 0.0)}

            return jax.lax.cond(success_flag, add_bc, skip_bc)

        agent, actor, actor_info = agent._apply_actor_loss(actor_loss_fn, batch)
        agent = agent.replace(actor_network=actor, rng=new_rng)
        return agent, {**actor_info, **critic_info, **action_info}

    @staticmethod
    def _update(agent, batch):
        if agent.config.get('first_train_q', False):
            return agent._update_q_first(agent, batch)
        _, _, rng2 = jax.random.split(agent.rng, 3)
        agent, action_info = agent._action_step(batch, rng2)
        agent, info = DSRLBestOfNAgent._update(agent, batch)
        return agent, {**info, **action_info}

    @staticmethod
    def _update_with_bc(agent, batch_tuple):
        if agent.config.get('first_train_q', False):
            return agent._update_q_first(agent, *batch_tuple)
        batch, _, _ = batch_tuple
        _, _, rng2, _ = jax.random.split(agent.rng, 4)
        agent, action_info = agent._action_step(batch, rng2)
        agent, info = DSRLBestOfNAgent._update_with_bc(agent, batch_tuple)
        return agent, {**info, **action_info}

    @staticmethod
    def _extra_action_update(agent, batch, success_batch, success_flag, rng):
        fixed_extra = agent.config.get('critic_updates_per_actor', 1) == 2
        if agent.action_critic_network is not None and (
                fixed_extra or agent.config.get('use_success_buffer_q', False)):
            chosen = jax.lax.cond(success_flag,
                                  lambda: success_batch if success_batch is not None else batch,
                                  lambda: batch)
            agent = jax.lax.cond(jnp.logical_or(fixed_extra, success_flag),
                                 lambda a: a._action_step(chosen, rng)[0], lambda a: a, agent)
        return agent

    @staticmethod
    def _extra_critic_update(agent, batch, success_batch, success_flag, rng):
        agent = agent._extra_action_update(agent, batch, success_batch, success_flag, rng)
        return DSRLBestOfNAgent._extra_critic_update(
            agent, batch, success_batch, success_flag, rng)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        config.setdefault('inverse_fm_iterations', 16)
        config.setdefault('inverse_fm_radius', 0.5)
        config.setdefault('inverse_fm_batch_size', 32)
        radius = config['inverse_fm_radius']
        if (isinstance(radius, bool) or not isinstance(radius, (int, float))
                or not math.isfinite(radius) or radius <= 0):
            raise ValueError('inverse_fm_radius must be finite and positive')
        size = config['inverse_fm_batch_size']
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('inverse_fm_batch_size must be a positive integer')
        iterations = config['inverse_fm_iterations']
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError('inverse_fm_iterations must be a positive integer')
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
        config.setdefault('use_qa_tdtarget', False)
        config.setdefault('first_train_q', False)
        config.setdefault('use_klactor_clip', False)
        config.setdefault('actor_ratio_clip_epsilon', 0.01)
        config.setdefault('actor_ratio_clip_max_backtracks', 10)
        epsilon = config['actor_ratio_clip_epsilon']
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or not 0 < epsilon < 1):
            raise ValueError('actor_ratio_clip_epsilon must be finite and between zero and one')
        backtracks = config['actor_ratio_clip_max_backtracks']
        if isinstance(backtracks, bool) or not isinstance(backtracks, int) or backtracks < 0:
            raise ValueError('actor_ratio_clip_max_backtracks must be a nonnegative integer')
        config.setdefault('use_no_targetq_clip', False)
        config.setdefault('use_qa_distill', False)
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
        actor_state = base.actor_network
        temperature_state = None
        if config['use_klactor_clip']:
            labels = {key: 'actor' if key == 'modules_actor' else 'frozen'
                      for key in actor_state.params}
            actor_tx = optax.multi_transform(
                {'actor': actor_state.tx, 'frozen': optax.set_to_zero()}, labels)
            temperature_state = TrainState.create(
                actor_state.model_def, {'modules_temp': actor_state.params['modules_temp']},
                actor_state.tx)
            actor_state = TrainState.create(actor_state.model_def, actor_state.params, actor_tx)
        return cls(rng=base.rng, actor_network=actor_state,
                   critic_network=base.critic_network, bc_flow_network=base.bc_flow_network,
                   config=base.config, action_critic_network=action_state,
                   temperature_network=temperature_state)
