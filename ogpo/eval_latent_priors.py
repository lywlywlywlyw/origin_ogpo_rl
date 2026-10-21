"""Evaluate Gaussian, tanh Gaussian and the untrained online actor.

Run via scripts/grace/eval_square_latent_priors.sh. --check-only checks the
real decoder checkpoint and all samplers without creating any simulator.
All evaluation metrics are printed to the terminal; no result files are saved.
Each policy runs exactly 200 episodes (20 environments, 10 rounds).
"""

import argparse
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(
        '/root/OGPO_public/exp/ogpo/OGPO/square/square-mh-low_dim/'
        'ogpo_squareseed1/params_90000.pkl'))
    parser.add_argument('--seed', type=int, default=1, help='Actor initialization seed.')
    parser.add_argument('--eval-seed', type=int, default=100001,
                        help='Reset/sampling seed, matching the first online evaluation.')
    parser.add_argument('--use-layer-norm', action='store_true',
                        help='Use the new actor LayerNorm; default matches uumviw91.')
    parser.add_argument('--check-only', action='store_true',
                        help='Check checkpoint and sampler numerics; no environment or rollouts.')
    return parser.parse_args()


def make_sampler(agent, policy):
    """Return the exact latent sampler; no deterministic mean or Q selection."""
    import jax
    import jax.numpy as jnp

    if policy not in ('A', 'B', 'C'):
        raise ValueError(f'Unknown policy: {policy}')

    @jax.jit
    def sample(observations, rng):
        if policy == 'C':
            # Includes state-dependent mu, sigma and the actor's tanh bijector.
            return agent.sample_noise(observations, rng=rng, deterministic=False)
        epsilon = jax.random.normal(
            rng, observations.shape[:-1] + (agent.config['noise_dim'],))
        return epsilon if policy == 'A' else jnp.tanh(epsilon)

    return sample


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # All heavy imports are deferred so --help cannot start JAX or MuJoCo.
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np

    from envs.robomimic_utils import TASK_CONFIG, _check_dataset_exists
    from ogpo.agents.grace import GRACEAgent
    from ogpo.utils.config_loader import load_config, config_to_agent_dict
    from ogpo.utils.main_helper import get_q_bounds

    config = load_config('grace_ogpo_square_shared')
    config['experiment']['seed'] = args.seed
    config['checkpoint']['restore_bc_flow_path'] = str(args.checkpoint.resolve())
    config['agent'].update(use_layer_norm=args.use_layer_norm,
                           encoder_sphere_limit=False, decoder_add_noise=False)
    agent_config = config_to_agent_dict(config)
    env_name = config['env']['name']
    if env_name != 'square-mh-low_dim':
        raise ValueError(f'This matched evaluation requires square-mh-low_dim: {env_name}')
    if agent_config['action_magnitude'] != 1.0 or agent_config['noise_chunk_length'] != -1:
        raise ValueError('The A/B/C comparison requires magnitude=1 and full-chunk latent.')
    if not agent_config['action_chunking'] or agent_config['encoder'] is not None:
        raise ValueError('This script requires chunked, low-dimensional state policies.')

    # Read one transition only. Do not load the entire offline replay dataset.
    dataset_path = Path(_check_dataset_exists(env_name))
    with h5py.File(dataset_path, 'r') as dataset:
        demo = dataset['data'][next(iter(dataset['data']))]
        example_obs = np.concatenate([
            demo['obs'][key][0] for key in TASK_CONFIG['square']['low_dim_keys']
        ]).astype(np.float32)
        example_action = np.asarray(demo['actions'][0], dtype=np.float32)

    agent_config['q_min'], agent_config['q_max'] = get_q_bounds(
        env_name, agent_config['discount'])
    # Same create(seed, single observation, single action) path as ogpo.main.
    agent = GRACEAgent.create(args.seed, jnp.asarray(example_obs),
                              jnp.asarray(example_action), agent_config)
    agent = agent.restore_bc_flow_params(str(args.checkpoint))
    samplers = {policy: make_sampler(agent, policy) for policy in ('A', 'B', 'C')}

    observations = jnp.asarray(np.stack([example_obs, example_obs]))
    key = jax.random.PRNGKey(args.eval_seed)
    epsilon = jax.random.normal(key, (2, agent.config['noise_dim']))
    expected = {'A': epsilon, 'B': jnp.tanh(epsilon),
                'C': agent.constrain_actor_output(
                    agent.actor_network.select('actor')(observations).sample(seed=key))}
    for policy, sample in samplers.items():
        latent = sample(observations, key)
        np.testing.assert_allclose(latent, expected[policy], rtol=1e-5, atol=1e-6)
        actions = np.asarray(agent.noise_to_actions_public(
            observations, latent, noises_constrained=True))
        assert actions.shape == (2, example_action.size * agent_config['horizon_length'])
        assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.000001
        if policy == 'A':
            np.testing.assert_allclose(actions, agent.sample_actions_normal(observations, key),
                                       rtol=1e-5, atol=1e-6)
        elif policy == 'C':
            np.testing.assert_allclose(actions, agent.sample_actions(observations, key),
                                       rtol=1e-5, atol=1e-6)
    print('Checkpoint / A,B,C sampler / decoder checks passed.', flush=True)
    if args.check_only:
        print('check-only: no simulator created; no episodes evaluated.', flush=True)
        return

    from types import SimpleNamespace
    from ogpo.utils.evaluation import evaluate_parallel
    from ogpo.utils.main_helper import make_vectorized_env

    # Independent evaluation budget; leave the online training YAML unchanged.
    # 200 is not divisible by the training setup's 32 parallel environments.
    n_envs = 20
    episodes = 10
    results = {}
    labels = {'A': 'Gaussian N(0,I)', 'B': 'tanh(Gaussian)',
              'C': 'Initialized actor (stochastic)'}
    print(f'Checkpoint: {args.checkpoint.resolve()}\n'
          f'Dataset: {dataset_path}\n'
          f'Actor seed: {args.seed}; evaluation seed: {args.eval_seed}\n'
          f'Actor LayerNorm: {agent_config["actor_layer_norm"] or args.use_layer_norm}\n'
          f'Encoder clip: {agent.config["use_encoder_clip"]}\n'
          f'Latent dimension: {agent.config["noise_dim"]}\n'
          f'Each policy: {n_envs} environments x {episodes} = {n_envs * episodes} episodes.',
          flush=True)
    for policy, sample in samplers.items():
        # Recreate with identical construction/reset seeds for each policy.
        # evaluate_parallel also resets its policy RNG to PRNGKey(eval_seed).
        random.seed(args.seed)
        np.random.seed(args.seed)
        envs = make_vectorized_env(
            env_name, num_envs=n_envs, seed=args.seed + 1000,
            rew_fn=config['env']['rew_fn'],
            post_success_steps=config['env'].get('post_success_steps', 0))
        try:
            if envs.single_observation_space.shape != example_obs.shape:
                raise ValueError('Dataset/environment observation dimensions differ.')

            @jax.jit
            def actor_fn(observations, rng):
                latent = sample(observations, rng)
                return agent.noise_to_actions_public(
                    observations, latent, noises_constrained=True)

            flags = SimpleNamespace(n_eval_envs=n_envs, eval_episodes=episodes,
                                    discount=agent.config['discount'], env_name=env_name,
                                    run_name=f'latent_prior_{policy}')
            print(f'Evaluating policy {policy}...', flush=True)
            stats = evaluate_parallel(envs, agent, flags, example_action.size,
                                      args.eval_seed, actor_fn=actor_fn, plot=False)
            results[policy] = {k: float(v) for k, v in stats.items()}
            print(f'\n[{policy}] {labels[policy]}', flush=True)
            for metric, value in sorted(results[policy].items()):
                print(f'  {metric}: {value:.6f}', flush=True)
            print(f'  Success rate: {results[policy].get("success", float("nan")):.2%}',
                  flush=True)
        finally:
            envs.close()
    print('\nEvaluation summary', flush=True)
    print(f'{"Policy":<42} {"Episodes":>8} {"Success rate":>14}', flush=True)
    for policy, stats in results.items():
        rate = stats.get('success', float('nan'))
        print(f'{policy + ": " + labels[policy]:<42} {n_envs * episodes:>8} {rate:>13.2%}',
              flush=True)


if __name__ == '__main__':
    main()
