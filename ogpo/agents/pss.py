"""Principal Steering Subspaces (Ni et al., 2026), Sec. III, Eqs. (3)--(5).

Only the latent SAC interface changes: the BC flow architecture and its weights
are identical to the shared OGPO/DSRL decoder. SAC and the matched success-buffer
updates reuse DSRL's losses; the actor/critic action width becomes k instead of D.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ogpo.agents.dsrl import DSRLBestOfNAgent
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

    def batch_update(self, batch, batch_success=None, success_flag=False):
        self._require_basis()
        if batch['actions'].shape[-1] != self.config['pss_dim']:
            raise ValueError('PSS replay actions must be k-dimensional z coordinates')
        return super().batch_update(batch, batch_success, success_flag)

    def bc_update(self, batch):
        raise ValueError('PSS uses a frozen offline checkpoint; BC training is disabled')

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        defaults = dict(pss_dim=8, pss_num_probes=64, pss_delta=0.05,
                        pss_probe_batch_size=64, critic_updates_per_actor=1)
        for key, value in defaults.items():
            config.setdefault(key, value)
        if config.get('encoder') is not None:
            raise ValueError('PSS currently supports state or pre-encoded observations')
        if config.get('noise_chunk_length', -1) != -1:
            raise ValueError('PSS requires a full action-chunk decoder noise input')
        if config['critic_updates_per_actor'] not in (1, 2):
            raise ValueError('critic_updates_per_actor must be 1 or 2')
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
        decoder_def = ModuleDict(dict(actor_bc_flow=decoder))
        rng, actor_key, critic_key, decoder_key = jax.random.split(jax.random.PRNGKey(seed), 4)
        actor_params = actor_def.init(actor_key, actor=(ex_observations,), temp=())['params']
        critic_params = critic_def.init(
            critic_key, critic=(ex_observations, jnp.zeros(k)),
            target_critic=(ex_observations, jnp.zeros(k)))['params']
        critic_params['modules_target_critic'] = critic_params['modules_critic']
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
            bc_flow_network=TrainState.create(decoder_def, decoder_params, optax.set_to_zero()),
            config=flax.core.FrozenDict(config), steering_basis=jnp.eye(dim),
            response_eigenvalues=jnp.zeros(dim), basis_ready=jnp.asarray(False),
            decoder_loaded=jnp.asarray(False))
