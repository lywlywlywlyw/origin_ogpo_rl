"""Principal Steering Subspaces (Ni et al., 2026), Sec. III, Eqs. (3)--(5).

Only the latent SAC interface changes: the BC flow architecture and its weights
are identical to the shared OGPO/DSRL decoder. The action critic learns TD
values online; the k-dimensional steering critic only learns by distillation,
following DSRL+EXPO's critic -> distillation -> SAC actor update order.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.dsrl_plus_expo import DSRLEXPOAgent
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.networks import Actor, ActorVectorField, LogParam, Value
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding


def estimate_response_subspace(decode, observations, base_noises, delta=0.05,
                               batch_size=64):
    """Forward-query Eq. (3), uncentered Gram Eq. (4), descending eigenbasis.

    ``decode(s, w)`` must return the normalized, decoder-clipped action chunk.
    Each state uses one independent base noise, shared by its D perturbations.
    Exactly N*(D+1) decoder evaluations are needed, with bounded query batches.
    Host float64 accumulation avoids losing weak eigenvalues to float32 sums.
    """
    observations, base_noises = np.asarray(observations), np.asarray(base_noises)
    if base_noises.ndim != 2 or len(base_noises) == 0:
        raise ValueError('base_noises must have shape (N, D) with N > 0')
    if len(observations) != len(base_noises):
        raise ValueError('Each probe observation needs exactly one base noise')
    if not np.isfinite(delta) or delta <= 0 or batch_size < 1:
        raise ValueError('delta and batch_size must be positive')
    n, dim = base_noises.shape
    perturbations = np.concatenate((np.zeros((1, dim)), np.eye(dim)), axis=0)
    gram = np.zeros((dim, dim), dtype=np.float64)
    # Process each state separately; even D+1 need not fit in one query batch.
    for observation, noise in zip(observations, base_noises):
        perturbed = (noise[None] + delta * perturbations).astype(np.float32)
        outputs = []
        for start in range(0, dim + 1, batch_size):
            stop = min(start + batch_size, dim + 1)
            obs = np.broadcast_to(observation, (stop - start, *observation.shape))
            outputs.append(np.asarray(decode(jnp.asarray(obs),
                                            jnp.asarray(perturbed[start:stop]))))
        outputs = np.concatenate(outputs).reshape(dim + 1, -1).astype(np.float64)
        if not np.isfinite(outputs).all():
            raise ValueError('Non-finite decoded actions during PSS probing')
        response = (outputs[1:] - outputs[:1]) / delta  # (D, M), not (M, D)
        gram += response @ response.T / n
    eigenvalues, basis = np.linalg.eigh((gram + gram.T) * 0.5)
    eigenvalues, basis = np.maximum(eigenvalues[::-1], 0), basis[:, ::-1]
    if eigenvalues.sum() <= np.finfo(np.float64).eps:
        raise ValueError('Frozen decoder has no measurable response on the probe set')
    # Fix arbitrary eigenvector signs for reproducible saved coordinates.
    pivots = np.argmax(np.abs(basis), axis=0)
    basis *= np.where(basis[pivots, np.arange(dim)] < 0, -1., 1.)
    return basis.astype(np.float32), eigenvalues.astype(np.float32), gram


class PSSAgent(DSRLBestOfNAgent):
    """SAC in z-space with a frozen full-dimensional flow decoder.

    ``noise_dim`` always denotes decoder input width D, ``pss_dim`` denotes k.
    Basis and readiness flags are pytree fields so checkpoints preserve them.
    """

    noise_critic_network: TrainState
    steering_basis: Any = None
    response_eigenvalues: Any = None
    basis_ready: Any = None
    decoder_loaded: Any = None

    def restore_bc_flow_params(self, checkpoint_path):
        agent = super().restore_bc_flow_params(checkpoint_path)
        return agent.replace(decoder_loaded=jnp.asarray(True),
                             basis_ready=jnp.asarray(False))

    def fit_subspace(self, observations, base_noises):
        if not bool(self.decoder_loaded):
            raise ValueError('Load the shared offline BC checkpoint before probing')
        if bool(self.basis_ready):
            raise ValueError('PSS basis is fixed online; refusing to refit it')
        basis, eigenvalues, gram = estimate_response_subspace(
            lambda obs, noise: self.compute_flow_actions(obs, noises=noise),
            observations, base_noises,
            delta=self.config['pss_delta'], batch_size=self.config['pss_probe_batch_size'])
        agent = self.replace(steering_basis=jnp.asarray(basis),
                             response_eigenvalues=jnp.asarray(eigenvalues),
                             basis_ready=jnp.asarray(True))
        k = self.config['pss_dim']
        return agent, dict(gram=gram, basis=basis, eigenvalues=eigenvalues,
                          cumulative_energy=np.cumsum(eigenvalues) / eigenvalues.sum(),
                          retained_energy=float(eigenvalues[:k].sum() / eigenvalues.sum()))

    def _require_basis(self):
        if not bool(self.basis_ready):
            raise ValueError('PSS basis has not been fitted from the frozen decoder')

    @jax.jit
    def project_noise(self, full_noise):
        return full_noise @ self.steering_basis[:, :self.config['pss_dim']]

    @jax.jit
    def sample_noise(self, observations, rng=None, deterministic=False):
        dist = self.actor_network.select('actor')(observations)
        return jnp.where(deterministic, dist.mode(), dist.sample(seed=rng))

    @jax.jit
    def reconstruct_noise(self, steering, rng):
        """Eq. (5). Fresh Gaussian complement, independent for each batch row."""
        k, dim = self.config['pss_dim'], self.config['noise_dim']
        complement = jax.random.normal(rng, (*steering.shape[:-1], dim - k))
        return (steering @ self.steering_basis[:, :k].T
                + complement @ self.steering_basis[:, k:].T)

    def noise_to_actions_public(self, observations, actor_noise, rng=None):
        self._require_basis()
        if rng is None:
            raise ValueError('PSS decoding requires an RNG for the fresh complement')
        return self.compute_flow_actions(
            observations, noises=self.reconstruct_noise(actor_noise, rng))

    def sample_actions(self, observations, rng=None, deterministic=False):
        self._require_basis()
        if rng is None:
            raise ValueError('PSS sampling requires an RNG')
        return self._sample_actions(observations, rng, deterministic)

    @jax.jit
    def _sample_actions(self, observations, rng, deterministic=False):
        actor_key, complement_key = jax.random.split(rng)
        z = self.sample_noise(observations, actor_key, deterministic)
        return self.compute_flow_actions(
            observations, noises=self.reconstruct_noise(z, complement_key))

    def critic_loss(self, batch, grad_params, rng):
        """DSRL TD MSE on executed actions, with entropy measured in z-space."""
        sample_rng, complement_rng = jax.random.split(rng)
        next_dist = self.actor_network.select('actor')(batch['next_observations'])
        next_z, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)
        next_actions = self.compute_flow_actions(
            batch['next_observations'],
            noises=self.reconstruct_noise(next_z, complement_rng))
        next_qs = self.critic_network.select('target_critic')(
            batch['next_observations'], actions=next_actions)
        next_q = (jnp.min(next_qs, axis=0) if self.config['q_agg'] == 'min'
                  else jnp.mean(next_qs, axis=0))
        temp = self.actor_network.select('temp')()
        if self.config['entropy_backup']:
            next_log_probs = jnp.clip(next_log_probs, -100, 100)
            next_q = next_q - temp * next_log_probs
        target_q = jax.lax.stop_gradient(jnp.clip(
            batch['rewards']
            + self.config['discount'] ** batch.get('chunk_length', self.config['horizon_length'])
            * batch['masks'] * next_q, -500., 500.))
        q = self.critic_network.select('critic')(
            batch['observations'], actions=batch['true_actions'], params=grad_params)
        loss = jnp.mean((q - target_q) ** 2)
        info = dict(critic_loss=loss, q_mean=q.mean(), q_max=q.max(), q_min=q.min(),
                    target_q_mean=target_q.mean(), temp=temp,
                    next_actions_mean=next_actions.mean(), next_actions_min=next_actions.min(),
                    next_actions_max=next_actions.max(), next_actions_std=next_actions.std(),
                    rewards_mean=batch['rewards'].mean(), rewards_min=batch['rewards'].min(),
                    rewards_max=batch['rewards'].max())
        if self.config['entropy_backup']:
            info['next_log_probs_mean'] = next_log_probs.mean()
        return loss, info

    # DSRL+EXPO's SAC loss evaluates the distilled critic and clips log-probs.
    actor_loss = DSRLEXPOAgent.actor_loss
    noise_critic_total_loss = DSRLEXPOAgent.noise_critic_total_loss

    def noise_critic_distill_loss(self, batch, grad_params, rng):
        """Regress each Q(s,z) head onto Q(s, decode(V_k z + V_perp eps)).

        As in DSRL+EXPO, use fresh tanh-normal samples instead of replay noise.
        PSS's actor already includes action_magnitude in its output bounds, so
        the same scale is applied to z here. A fresh complement gives an unbiased
        sample of the conditional teacher value for this z.
        """
        noise_rng, complement_rng = jax.random.split(rng)
        z = self.config['action_magnitude'] * jnp.tanh(jax.random.normal(
            noise_rng, (batch['observations'].shape[0], self.config['pss_dim'])))
        actions = self.compute_flow_actions(
            batch['observations'], noises=self.reconstruct_noise(z, complement_rng))
        target_qs = jax.lax.stop_gradient(self.critic_network.select('critic')(
            batch['observations'], actions=actions))
        pred_qs = self.noise_critic_network.select('noise_critic')(
            batch['observations'], actions=z, params=grad_params)
        loss = jnp.mean((pred_qs - target_qs) ** 2)
        return loss, dict(noise_critic_distill_loss=loss,
                          noise_critic_pred_q_mean=pred_qs.mean(),
                          noise_critic_target_q_mean=target_qs.mean())

    @staticmethod
    def _online_update(agent, batch, batch_success=None, success_flag=False):
        new_rng, critic_rng, extra_rng, distill_rng, actor_rng, bc_rng = jax.random.split(agent.rng, 6)
        critic_state, critic_info = agent.critic_network.apply_loss_fn(
            lambda p: agent.critic_total_loss(batch, p, critic_rng))
        agent.target_update(critic_state, 'critic')
        agent = agent.replace(critic_network=critic_state)
        agent = agent._extra_critic_update(agent, batch, batch_success, success_flag, extra_rng)

        # Distill from the updated action critic before updating the SAC actor.
        def distill_step(state, key):
            return state.apply_loss_fn(lambda p: agent.noise_critic_total_loss(batch, p, key))
        keys = jax.random.split(distill_rng, agent.config['noise_critic_grad_steps'])
        noise_state, noise_infos = jax.lax.scan(distill_step, agent.noise_critic_network, keys)
        agent = agent.replace(noise_critic_network=noise_state)

        def actor_loss_fn(p):
            loss, info = agent.actor_total_loss(batch, p, actor_rng)
            if batch_success is not None:
                def add_bc():
                    bc_loss, bc_info = agent.actor_bc_loss(batch_success, p, bc_rng)
                    return loss + bc_loss, {**info, **bc_info}

                def no_bc():
                    bc_info = {key: jnp.asarray(0.) for key in (
                        'actor_bc_loss', 'actor_bc_loss_unweighted', 'actor_noise_mean',
                        'actor_noise_std', 'successful_noise_mean', 'successful_noise_std',
                        'bc_mse_error')}
                    return loss, {**info, **bc_info}
                return jax.lax.cond(success_flag, add_bc, no_bc)
            return loss, info

        actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        noise_info = jax.tree.map(lambda x: x[-1], noise_infos)
        return agent.replace(actor_network=actor_state, rng=new_rng), {
            **actor_info, **critic_info, **noise_info}

    @staticmethod
    def _update(agent, batch):
        return agent._online_update(agent, batch)

    @staticmethod
    def _update_with_bc(agent, batch_tuple):
        return agent._online_update(agent, *batch_tuple)

    def batch_update(self, batch, batch_success=None, success_flag=False):
        self._require_basis()
        for replay_batch in (batch, batch_success):
            if replay_batch is None:
                continue
            if replay_batch['actions'].shape[-1] != self.config['pss_dim']:
                raise ValueError('PSS replay actions must be k-dimensional z coordinates')
            if ('true_actions' not in replay_batch
                    or replay_batch['true_actions'].shape[-1] != self.config['noise_dim']):
                raise ValueError('PSS replay true_actions must contain the full decoded action chunk')
        return super().batch_update(batch, batch_success, success_flag)

    def bc_update(self, batch):
        raise ValueError('PSS uses a frozen offline checkpoint; BC training is disabled')

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        defaults = dict(pss_dim=8, pss_num_probes=64, pss_delta=0.05,
                        pss_probe_batch_size=64, critic_updates_per_actor=1,
                        noise_critic_grad_steps=10, lr_noise_critic=3e-4)
        for key, value in defaults.items():
            if config.get(key) is None:
                config[key] = value
        if config.get('encoder') is not None:
            raise ValueError('PSS currently supports state or pre-encoded observations')
        if config.get('noise_chunk_length', -1) != -1:
            raise ValueError('PSS requires a full action-chunk decoder noise input')
        if config['critic_updates_per_actor'] not in (1, 2):
            raise ValueError('critic_updates_per_actor must be 1 or 2')
        if (not isinstance(config['noise_critic_grad_steps'], int)
                or config['noise_critic_grad_steps'] < 1):
            raise ValueError('noise_critic_grad_steps must be a positive integer')
        action_dim = ex_actions.shape[-1]
        horizon = config['horizon_length'] if config['action_chunking'] else 1
        dim, k = action_dim * horizon, config['pss_dim']
        if not isinstance(k, int) or not 1 <= k <= dim:
            raise ValueError(f'pss_dim must be between 1 and decoder noise dimension {dim}')
        if config['pss_num_probes'] < 1 or config['pss_delta'] <= 0:
            raise ValueError('PSS probe count and finite-difference step must be positive')
        if config['pss_probe_batch_size'] < 1:
            raise ValueError('pss_probe_batch_size must be positive')
        config.update(ob_dims=ex_observations.shape, action_dim=action_dim, noise_dim=dim)
        if config['target_entropy'] is None:
            # Matched DSRL uses -k/2; paper-style config opts into standard -k.
            config['target_entropy'] = -float(k) * config.get('pss_entropy_scale', 0.5)

        two_tier = dict(obs_two_tier=config.get('obs_two_tier', False),
                        two_tier_img_dim=config.get('_two_tier_img_dim', 0),
                        two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
                        two_tier_fused_dim=config.get('two_tier_fused_dim', 0))
        actor = Actor(hidden_dims=config['actor_hidden_dims'], action_dim=k,
                      layer_norm=config['actor_layer_norm'], tanh_squash=True,
                      state_dependent_std=True, low=-config['action_magnitude'],
                      high=config['action_magnitude'], **two_tier)
        critic = Value(hidden_dims=config['value_hidden_dims'],
                       layer_norm=config['layer_norm'], num_ensembles=config['num_qs'],
                       critic_loss_type=config['critic_loss_type'], num_bins=config['num_bins'],
                       q_min=config['q_min'], q_max=config['q_max'], **two_tier)
        time_embedding = (SinusoidalTimeEmbedding(config.get('time_embedding_dim', 32))
                          if config.get('time_embedding') == 'sinusoidal' else None)
        decoder = ActorVectorField(hidden_dims=config['actor_hidden_dims'], action_dim=dim,
                                   layer_norm=config['actor_layer_norm'],
                                   time_embedding=time_embedding, **two_tier)
        actor_def = ModuleDict(dict(actor=actor, temp=LogParam(config['init_temperature'])))
        critic_def = ModuleDict(dict(critic=critic, target_critic=copy.deepcopy(critic)))
        noise_def = ModuleDict(dict(noise_critic=copy.deepcopy(critic)))
        decoder_def = ModuleDict(dict(actor_bc_flow=decoder))
        rng, actor_key, critic_key, decoder_key, noise_key = jax.random.split(jax.random.PRNGKey(seed), 5)
        actor_params = actor_def.init(actor_key, actor=(ex_observations,), temp=())['params']
        critic_params = critic_def.init(
            critic_key, critic=(ex_observations, jnp.zeros(dim)),
            target_critic=(ex_observations, jnp.zeros(dim)))['params']
        critic_params['modules_target_critic'] = critic_params['modules_critic']
        noise_params = noise_def.init(
            noise_key, noise_critic=(ex_observations, jnp.zeros(k)))['params']
        decoder_params = decoder_def.init(
            decoder_key, actor_bc_flow=(ex_observations, jnp.zeros(dim), jnp.zeros(1)))['params']

        def optimizer(lr, kind, weight_decay=1e-4):
            if kind == 'adam':
                tx = optax.adam(float(lr))
            elif kind == 'adamw':
                tx = optax.adamw(float(lr), weight_decay=weight_decay)
            elif kind == 'sgd':
                tx = optax.sgd(float(lr))
            else:
                raise ValueError(f'Unknown optimizer: {kind}')
            return optax.chain(optax.clip_by_global_norm(float(config['clip_grad_norm'])), tx)

        return cls(
            rng=rng,
            actor_network=TrainState.create(actor_def, actor_params,
                optimizer(config['lr_actor'], config['optimizer_actor'])),
            critic_network=TrainState.create(critic_def, critic_params,
                optimizer(config['lr_critic'], config['optimizer_critic'],
                          config.get('critic_weight_decay', 1e-4))),
            noise_critic_network=TrainState.create(noise_def, noise_params,
                optimizer(config['lr_noise_critic'], config['optimizer_critic'])),
            bc_flow_network=TrainState.create(decoder_def, decoder_params, optax.set_to_zero()),
            config=flax.core.FrozenDict(config), steering_basis=jnp.eye(dim),
            response_eigenvalues=jnp.zeros(dim), basis_ready=jnp.asarray(False),
            decoder_loaded=jnp.asarray(False))
