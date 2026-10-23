"""AnchorQ (2610.22852v1), Algorithms 1--4, with the shared OGPO decoder.

Only Q(s,z) is learned. Reversed cardinal commands and a base-prior defer
candidate are refined by value-weighted particle barycenters, then ranked by Q.
The matched config retains the baseline trainable Value MLP and physical-step
budget. Frozen randomized priors, REDQ targets and prioritized replay preserve
the method-specific learning rules from Appendix B.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field


def cardinal_reference_actions(action_dim, horizon, magnitude=1., gripper=0.):
    """Six Cartesian delta commands, constant over the action horizon.

    This adapter is for single-arm OSC_POSE Robomimic (xyz, rotation, gripper).
    Rotation is zero; gripper=0 is neutral. The paper does not specify these
    Robomimic command amplitudes, so they are explicit experiment parameters.
    """
    if action_dim != 7:
        raise ValueError('Cardinal anchors require single-arm 7D OSC_POSE actions')
    commands = np.zeros((6, horizon, action_dim), dtype=np.float32)
    for axis in range(3):
        commands[2 * axis, :, axis] = magnitude
        commands[2 * axis + 1, :, axis] = -magnitude
    commands[:, :, -1] = gripper
    return commands.reshape(6, horizon * action_dim)


def value_tilted_barycenters(particles, head_values, beta=1., adaptive=True,
                             uncertainty_floor=0.01):
    """Algorithm 3: head_values=(K,...,C,M+1), particles=(...,C,M+1,D)."""
    means = head_values.mean(axis=0)
    uncertainty = jnp.median(jnp.std(head_values, axis=0), axis=-1)
    intensity = (1. / jnp.maximum(uncertainty, uncertainty_floor) if adaptive
                 else jnp.full_like(uncertainty, beta))
    logits = (means - means.max(axis=-1, keepdims=True)) * intensity[..., None]
    weights = jax.nn.softmax(logits, axis=-1)
    refined = jnp.sum(weights[..., None] * particles, axis=-2)
    return refined, weights, intensity


class AnchorQAgent(flax.struct.PyTreeNode):
    rng: Any
    critic_network: TrainState
    prior_network: TrainState
    bc_flow_network: TrainState
    reference_actions: Any
    decision_step: Any
    decoder_loaded: Any
    config: Any = nonpytree_field()

    # Identical ODE decoder, checkpoint shape checks, and Polyak updates as DSRL.
    compute_flow_actions = DSRLBestOfNAgent.compute_flow_actions
    target_update = DSRLBestOfNAgent.target_update

    def restore_bc_flow_params(self, checkpoint_path):
        agent = DSRLBestOfNAgent.restore_bc_flow_params(self, checkpoint_path)
        return agent.replace(decoder_loaded=jnp.asarray(True))

    def configure_reference_actions(self, action_min=None, action_max=None):
        """Convert raw OSC commands to the decoder's action coordinates if needed."""
        refs = cardinal_reference_actions(
            self.config['action_dim'], self.config['horizon_length'],
            self.config['anchorq_reference_magnitude'], self.config['anchorq_reference_gripper'])
        if (action_min is None) != (action_max is None):
            raise ValueError('Both action normalization bounds must be supplied')
        if action_min is not None:
            low, high = np.asarray(action_min), np.asarray(action_max)
            if (low.shape != (self.config['action_dim'],) or high.shape != low.shape
                    or not np.all(high > low)):
                raise ValueError('Invalid action normalization bounds')
            refs = refs.reshape(6, self.config['horizon_length'], -1)
            refs = np.clip(2. * (refs - low) / (high - low) - 1., -1., 1.)
            refs = refs.reshape(6, -1)
        return self.replace(reference_actions=jnp.asarray(refs))

    @jax.jit
    def reverse_flow_actions(self, observations, actions):
        """Reverse Euler from action time 1 to noise time 0, without clipping.

        The shared decoder flows noise -> action (the reverse of pi0's time
        convention). Negating dt and evaluating t=1,...,1/N implements inversion
        of its continuous vector field, not a purported exact discrete inverse.
        """
        count = self.config['flow_steps']

        def step(i, x):
            times = jnp.full((*x.shape[:-1], 1), 1. - i / count)
            velocity = self.bc_flow_network.select('actor_bc_flow')(observations, x, times)
            return x - velocity / count

        return jax.lax.fori_loop(0, count, step, actions)

    @jax.jit
    def build_anchor_bank(self, observations, rng):
        """Construct one bank per observation; caller caches it in replay."""
        batch_shape = observations.shape[:-1]
        refs = jnp.broadcast_to(self.reference_actions, (*batch_shape, *self.reference_actions.shape))
        obs = jnp.broadcast_to(observations[..., None, :], (*batch_shape, 6, observations.shape[-1]))
        anchors = self.reverse_flow_actions(obs, refs)
        if self.config['anchorq_include_defer']:
            defer = jax.random.normal(rng, (*batch_shape, 1, self.config['noise_dim']))
            anchors = jnp.concatenate((anchors, defer), axis=-2)
        return anchors

    def critic_values(self, observations, actions, target=False, params=None):
        name = 'target_critic' if target else 'critic'
        values = self.critic_network.select(name)(observations, actions=actions, params=params)
        if self.config['anchorq_prior_scale']:
            prior = self.prior_network.select('critic')(observations, actions=actions)
            values = values + self.config['anchorq_prior_scale'] * jax.lax.stop_gradient(prior)
        return values

    def _candidate_qs(self, observations, latents, target=False):
        """Evaluate candidates with bounded activation memory (including TD batches)."""
        # observations: (...,obs_dim), latents: (...,C[,P],D).
        extra = latents.ndim - observations.ndim
        obs = jnp.reshape(observations, (*observations.shape[:-1], *((1,) * extra), observations.shape[-1]))
        obs = jnp.broadcast_to(obs, (*latents.shape[:-1], observations.shape[-1]))
        shape = latents.shape[:-1]
        noises = latents.reshape(-1, latents.shape[-1])
        obs = obs.reshape(-1, observations.shape[-1])
        size = noises.shape[0]
        width = min(size, self.config['anchorq_query_batch_size'])
        padding = (-size) % width
        noises = jnp.pad(noises, ((0, padding), (0, 0))).reshape(-1, width, noises.shape[-1])
        obs = jnp.pad(obs, ((0, padding), (0, 0))).reshape(-1, width, obs.shape[-1])
        def evaluate(inputs):
            o, z = inputs
            return self.critic_values(o, z, target=target)

        # (query_blocks,K,width) -> (K,...,C[,P]).
        values = jax.lax.map(evaluate, (obs, noises))
        values = jnp.moveaxis(values, 1, 0).reshape(self.config['num_qs'], -1)[:, :size]
        return values.reshape(self.config['num_qs'], *shape)

    def _particle_cloud(self, anchors, rng, sigma):
        count = self.config['anchorq_particle_count']
        perturbation = jax.random.normal(rng, (*anchors.shape[:-1], count, anchors.shape[-1]))
        perturbation = jnp.concatenate((jnp.zeros_like(anchors[..., None, :]), perturbation), axis=-2)
        return anchors[..., None, :] + sigma[..., None, None, None] * perturbation

    @jax.jit
    def refine_anchors(self, observations, anchors, rng):
        """Algorithms 3--4, with independent probe and refinement particle draws."""
        batch_shape = observations.shape[:-1]
        count = self.config['anchorq_particle_count']
        if count == 0:
            return anchors
        probe_rng, refine_rng = jax.random.split(rng)

        def refine(_):
            if self.config['anchorq_adaptive_sigma']:
                sigma_max = jnp.full(batch_shape, self.config['anchorq_sigma_max'])
                probes = self._particle_cloud(anchors, probe_rng, sigma_max)
                probe_values = self._candidate_qs(observations, probes)
                uncertainty = jnp.median(jnp.std(probe_values, axis=0), axis=(-2, -1))
                sigma = (self.config['anchorq_sigma_min']
                         + (self.config['anchorq_sigma_max'] - self.config['anchorq_sigma_min'])
                         * jnp.clip(uncertainty, 0., 1.))
            else:
                ramp = self.config['anchorq_sigma_ramp_decisions']
                scale = (jnp.clip((self.decision_step - self.config['anchorq_particle_warmup_decisions'])
                                  / max(1, ramp), 0., 1.) if ramp else 1.)
                sigma = jnp.full(batch_shape, self.config['anchorq_particle_sigma'] * scale)
            particles = self._particle_cloud(anchors, refine_rng, sigma)
            values = self._candidate_qs(observations, particles)
            refined, _, _ = value_tilted_barycenters(
                particles, values, self.config['anchorq_tilt_beta'],
                self.config['anchorq_adaptive_beta'], self.config['anchorq_uncertainty_floor'])
            return refined

        return jax.lax.cond(
            self.decision_step >= self.config['anchorq_particle_warmup_decisions'],
            refine, lambda _: anchors, operand=None)

    @jax.jit
    def select_latent(self, observations, anchors, rng, epsilon=0.):
        refine_rng, explore_rng, choice_rng = jax.random.split(rng, 3)
        refined = self.refine_anchors(observations, anchors, refine_rng)
        head_values = self._candidate_qs(observations, refined)
        scores = head_values.mean(axis=0)
        greedy = jnp.argmax(scores, axis=-1)
        random = jax.random.randint(choice_rng, greedy.shape, 0, anchors.shape[-2])
        index = jnp.where(jax.random.uniform(explore_rng, greedy.shape) < epsilon, random, greedy)
        selected = jnp.take_along_axis(refined, index[..., None, None], axis=-2)[..., 0, :]
        return selected, dict(anchor_index=index, selected_q=jnp.take_along_axis(
            scores, index[..., None], axis=-1)[..., 0],
            refinement_distance=jnp.linalg.norm(refined - anchors, axis=-1).mean(),
            candidate_q_spread=(scores.max(axis=-1) - scores.min(axis=-1)).mean(),
            ensemble_std=jnp.std(head_values, axis=0).mean())

    @jax.jit
    def exploration_epsilon(self):
        fraction = jnp.clip(self.decision_step / max(1, self.config['anchorq_epsilon_decay_decisions']), 0., 1.)
        return (self.config['anchorq_epsilon_start']
                + fraction * (self.config['anchorq_epsilon_end'] - self.config['anchorq_epsilon_start']))

    @jax.jit
    def sample_actions(self, observations, rng=None, deterministic=False):
        """Evaluation always uses greedy anchor choice, with fresh local particles.

        deterministic suppresses epsilon exploration, not the paper's particles
        or base-prior defer draw. Collection uses select_latent with epsilon.
        """
        bank_rng, selection_rng = jax.random.split(rng)
        anchors = self.build_anchor_bank(observations, bank_rng)
        latent, _ = self.select_latent(observations, anchors, selection_rng, epsilon=0.)
        return self.compute_flow_actions(observations, noises=latent)

    def critic_loss(self, batch, grad_params, rng):
        selection_rng, subset_rng = jax.random.split(rng)
        latents, _ = self.select_latent(
            batch['next_observations'], batch['next_anchors'], selection_rng, epsilon=0.)
        targets = self.critic_values(batch['next_observations'], latents, target=True)
        if self.config['anchorq_target_aggregation'] == 'min':
            subset = jax.random.choice(subset_rng, self.config['num_qs'],
                                       (self.config['anchorq_target_heads'],), replace=False)
            next_q = targets[subset].min(axis=0)
        else:
            next_q = targets.mean(axis=0)
        exponent = batch.get('chunk_length', 1)
        target = jax.lax.stop_gradient(jnp.clip(
            batch['rewards'] + self.config['discount'] ** exponent * batch['masks'] * next_q,
            -500., 500.))
        values = self.critic_values(batch['observations'], batch['actions'], params=grad_params)
        td = values - target
        loss = jnp.mean(batch.get('weights', jnp.ones_like(target))[None] * td ** 2)
        return loss, dict(critic_loss=loss, q_mean=values.mean(),
                          target_q_mean=target.mean(), td_abs_mean=jnp.abs(td).mean(),
                          td_errors=jax.lax.stop_gradient(jnp.abs(td).mean(axis=0)))

    @jax.jit
    def critic_update(self, batch):
        """One optimizer step, exposing per-row TD errors for immediate PER refresh."""
        rng, key = jax.random.split(self.rng)
        state, info = self.critic_network.apply_loss_fn(
            lambda params: self.critic_loss(batch, params, key))
        self.target_update(state, 'critic')
        errors = info.pop('td_errors')
        return self.replace(rng=rng, critic_network=state), info, errors

    @staticmethod
    def _update(agent, batch_tuple):
        batch, success_batch, success_flag = batch_tuple
        rng, first_rng, extra_rng = jax.random.split(agent.rng, 3)

        def update(current, data, key):
            state, info = current.critic_network.apply_loss_fn(
                lambda params: current.critic_loss(data, params, key))
            current.target_update(state, 'critic')
            return current.replace(critic_network=state), info

        agent, info = update(agent, batch, first_rng)
        extra = jnp.logical_or(agent.config['critic_updates_per_actor'] == 2,
                              jnp.logical_and(success_flag, agent.config['use_success_buffer_q']))
        chosen = jax.lax.cond(success_flag, lambda: success_batch, lambda: batch)
        agent, _ = jax.lax.cond(extra, lambda a: update(a, chosen, extra_rng),
                                lambda a: (a, info), agent)
        return agent.replace(rng=rng), info

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        if batch_success is None:
            batch_success = batch
        def step(agent, inputs):
            return self._update(agent, (*inputs, success_flag))
        agent, infos = jax.lax.scan(step, self, (batch, batch_success))
        return agent, jax.tree.map(lambda value: value.mean(), infos)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        defaults = dict(
            anchorq_reference_magnitude=1., anchorq_reference_gripper=0.,
            anchorq_include_defer=True, anchorq_particle_count=32,
            anchorq_particle_sigma=0.5, anchorq_particle_warmup_decisions=10000,
            anchorq_sigma_ramp_decisions=0, anchorq_adaptive_sigma=True,
            anchorq_sigma_min=0.05, anchorq_sigma_max=0.3,
            anchorq_tilt_beta=1., anchorq_adaptive_beta=True,
            anchorq_uncertainty_floor=0.01, anchorq_epsilon_start=1.,
            anchorq_epsilon_end=0.05, anchorq_epsilon_decay_decisions=10000,
            anchorq_target_aggregation='min', anchorq_target_heads=2,
            anchorq_prior_scale=2., anchorq_prioritized_replay=True,
            anchorq_per_alpha=0.6, anchorq_per_beta_start=0.4, anchorq_per_epsilon=1e-6,
            anchorq_query_batch_size=256)
        for key, value in defaults.items():
            if config.get(key) is None:
                config[key] = value
        if config.get('encoder') is not None or not config['action_chunking']:
            raise ValueError('AnchorQ requires state/pre-encoded observations and action chunking')
        if config.get('noise_chunk_length', -1) != -1:
            raise ValueError('AnchorQ requires full-dimensional decoder latents')
        for key in ('anchorq_particle_count', 'anchorq_particle_warmup_decisions',
                    'anchorq_sigma_ramp_decisions', 'anchorq_epsilon_decay_decisions'):
            if not isinstance(config[key], int) or config[key] < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if config['flow_steps'] < 1 or config['anchorq_query_batch_size'] < 1:
            raise ValueError('Flow steps and query batch size must be positive')
        for key in ('anchorq_reference_magnitude', 'anchorq_reference_gripper',
                    'anchorq_particle_sigma', 'anchorq_sigma_min', 'anchorq_sigma_max',
                    'anchorq_tilt_beta', 'anchorq_uncertainty_floor'):
            if not np.isfinite(config[key]):
                raise ValueError(f'{key} must be finite')
        if not 0 < config['anchorq_reference_magnitude'] <= 1 or abs(config['anchorq_reference_gripper']) > 1:
            raise ValueError('Reference commands must lie within the OSC action bounds')
        if (not 0 <= config['anchorq_sigma_min'] <= config['anchorq_sigma_max']
                or config['anchorq_particle_sigma'] < 0 or config['anchorq_tilt_beta'] < 0
                or config['anchorq_uncertainty_floor'] <= 0):
            raise ValueError('Invalid particle scale, tilt, or uncertainty floor')
        if not 0 <= config['anchorq_epsilon_end'] <= config['anchorq_epsilon_start'] <= 1:
            raise ValueError('Epsilon schedule must decrease within [0,1]')
        if config['anchorq_target_aggregation'] not in ('mean', 'min'):
            raise ValueError('AnchorQ target aggregation must be mean or min')
        if not 1 <= config['anchorq_target_heads'] <= config['num_qs']:
            raise ValueError('Target subset size must fit within the critic ensemble')
        if not np.isfinite(config['anchorq_prior_scale']) or config['anchorq_prior_scale'] < 0:
            raise ValueError('Prior scale must be finite and nonnegative')
        refs = cardinal_reference_actions(ex_actions.shape[-1], config['horizon_length'],
                                         config['anchorq_reference_magnitude'], config['anchorq_reference_gripper'])
        # Reuse baseline initialization exactly, then discard its unused SAC actor.
        # AnchorQ has no actor parameters, temperature, or actor optimizer in its state.
        base = DSRLBestOfNAgent.create(seed, ex_observations, ex_actions, config)
        prior_def = ModuleDict({'critic': base.critic_network.model_def.modules['critic']})
        prior_params = prior_def.init(jax.random.fold_in(base.rng, 9137),
            critic=(ex_observations, jnp.zeros(config['horizon_length'] * ex_actions.shape[-1])))['params']
        prior = TrainState.create(prior_def, prior_params)
        return cls(rng=base.rng, critic_network=base.critic_network, prior_network=prior,
                   bc_flow_network=base.bc_flow_network, reference_actions=jnp.asarray(refs),
                   decision_step=jnp.asarray(0, jnp.int32), decoder_loaded=jnp.asarray(False),
                   config=base.config)
