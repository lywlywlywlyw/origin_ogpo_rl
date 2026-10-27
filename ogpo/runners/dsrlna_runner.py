"""DSRL-NA with the shared environment and physical-step update budget."""
import jax
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.runners.bc_runner import _build_eval_flags, _run_evaluation
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.sharding import shard_batch


def _reshape_batch(batch, count, batch_size):
    return jax.tree.map(lambda x: x.reshape((count, batch_size) + x.shape[1:]), batch)


def run_dsrlna_online_rl(
    agent, env, eval_env, eval_envs, train_dataset, config, logger, save_dir,
    algo_name, online_steps, start_training, batch_size, horizon_length, discount,
    utd_online=1, offline_ratio=0., use_success_buffer=False, best_of_n=1,
    eval_interval=10000, n_eval_envs=32, eval_episodes=2, video_episodes=0,
    video_frame_skip=3, run_name='dsrlna', env_name='', log_interval=5000,
    save_interval=500000, log_enabled=True, buffer_size=2000000, seed=0,
    start_step=0, force_jax_sync=True, mesh=None, encode_fn=None,
    is_resumed=False, **common,
):
    """One environment; each eligible physical step earns 1 actor, 1 Qa, 0.5 Qz.

    Rewards and termination come directly from the common environment. Chunks
    store discounted rewards and actual lengths; truncation keeps bootstrap.
    SAC updates precede a separate student block, as in reference DSRL-NA.
    """
    if algo_name != 'dsrlna':
        raise ValueError('Expected the dsrlna agent')
    if use_success_buffer or offline_ratio != 0. or best_of_n != 1:
        raise ValueError('DSRLNA uses ordinary online replay and single-candidate sampling')
    if is_resumed:
        raise ValueError('DSRLNA currently requires a fresh online run')
    if encode_fn is not None:
        raise ValueError('DSRLNA currently supports the low-dimensional shared checkpoint')
    if config['num_train_envs'] != 1:
        raise ValueError('Matched DSRLNA requires exactly one training environment')
    if any(common.get(key, 0) for key in ('bc_refine_steps', 'q_warmup_steps', 'mc_regression')):
        raise ValueError('DSRLNA requires zero generic pretraining/refinement steps')
    if isinstance(utd_online, bool) or not isinstance(utd_online, int) or utd_online < 1:
        raise ValueError('utd_online must be a positive integer')
    for key in ('utd_q', 'utd_pi'):
        if common.get(key) not in (None, utd_online):
            raise ValueError('DSRLNA uses utd_online SAC groups, each with one action-Q step')
    if horizon_length != config['horizon_length'] or discount != config['discount']:
        raise ValueError('Runner horizon/discount must match the DSRLNA agent')

    action_dim = agent.config['action_dim']
    noise_dim = agent.config['noise_dim']
    example = train_dataset.sample(())['observations']
    transition = dict(observations=np.zeros_like(example), actions=np.zeros(noise_dim, np.float32),
                      rewards=np.float32(0), masks=np.float32(1), terminals=np.float32(0),
                      chunk_length=np.int32(1), next_observations=np.zeros_like(example))
    replay = ReplayBuffer.create(transition, buffer_size,
                                 numpy_rng=np.random.default_rng(seed + 30000))
    observation = env.reset(seed=seed + 10000)[0]
    episodes = 0
    rng = jax.random.PRNGKey(seed + 40000)
    total_steps = actor_updates = critic_updates = student_updates = 0
    update_info = {}
    flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)

    def evaluate(step):
        result = _run_evaluation(
            agent, eval_env, eval_envs, flags, action_dim, agent.sample_actions,
            step, eval_episodes, video_episodes, video_frame_skip, run_name, False)
        logger.log(result, 'eval', step=step)
        return result

    def budget():
        return dict(env_steps=total_steps, gradient_updates=actor_updates,
                    critic_gradient_updates=critic_updates, noise_critic_updates=student_updates,
                    decoder_updates=0)

    eval_info = evaluate(start_step)
    with tqdm.tqdm(total=online_steps, desc='DSRLNA Online RL', dynamic_ncols=True) as bar:
        while total_steps < online_steps:
            before = total_steps
            rng, key = jax.random.split(rng)
            # Match GRACE: collect with the actor even during update-free warmup.
            actions = np.asarray(agent.sample_actions(observation, key))
            previous = observation
            reward_sum = 0.
            length = 0
            terminated = truncated = False
            for index, action in enumerate(actions.reshape(horizon_length, action_dim)):
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += discount ** index * reward
                length += 1
                total_steps += 1
                bar.update(1)
                if terminated or truncated or total_steps >= online_steps:
                    break
            done = bool(terminated or truncated)
            replay.add_transition(dict(
                observations=previous, actions=actions, rewards=np.float32(reward_sum),
                masks=np.float32(not terminated), terminals=np.float32(done),
                chunk_length=np.int32(length), next_observations=observation))
            if done:
                episodes += 1
                observation, _ = env.reset(seed=seed + 10000 + episodes)

            # A chunk crossing warmup only earns updates for its eligible steps.
            eligible = max(0, total_steps - max(before, start_training))
            count = eligible * utd_online
            if count:
                batch = _reshape_batch(replay.sample(batch_size * count), count, batch_size)
                # Carry fractional student credit across short chunks/warmup boundaries.
                student_count = int((actor_updates + count) *
                                    config['noise_critic_updates_per_actor']) - student_updates
                student = (_reshape_batch(replay.sample(batch_size * student_count),
                                          student_count, batch_size) if student_count else None)
                if mesh is not None:
                    batch = shard_batch(batch, mesh)
                    if student is not None:
                        student = shard_batch(student, mesh)
                agent, update_info = agent.batch_update(batch, student)
                actor_updates += count
                critic_updates += count
                student_updates += student_count
                if force_jax_sync:
                    jax.block_until_ready(agent)
            log_step = start_step + total_steps
            if log_interval > 0 and total_steps // log_interval > before // log_interval:
                if update_info:
                    logger.log(update_info, 'online_agent', step=log_step)
                logger.log(budget(), 'online_budget', step=log_step)
            if total_steps == online_steps or (
                    eval_interval > 0 and total_steps // eval_interval > before // eval_interval):
                eval_info = evaluate(log_step)
            if log_enabled and save_interval > 0 and total_steps // save_interval > before // save_interval:
                save_agent(agent, save_dir, log_step)
    log_step = start_step + total_steps
    logger.log(budget(), 'online_budget', step=log_step)
    save_agent(agent, save_dir, log_step)
    return agent, log_step, eval_info, replay
