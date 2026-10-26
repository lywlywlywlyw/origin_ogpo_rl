# JGCW Square diagnostics

Run `bash scripts/jgcw/square_shared_ogpo_bc.sh`. This uses
`ogpo/configs/algos/jgcw_ogpo_square_shared.yaml`, which inherits the matched
`dsrl_ogpo_square_shared.yaml` experiment just like GRACE. General JGCW defaults
are in `ogpo/configs/algos/jgcw.yaml`. All diagnostic options live under `agent`
in the Square config. For example:

```bash
bash scripts/jgcw/square_shared_ogpo_bc.sh \
  --agent.train_q_sa=true --agent.cal_q_accuracy=true \
  --agent.accuracy_s_num=16 --agent.accuracy_a_num=8
```

For a background run from the repository root:

```bash
mkdir -p run_logs
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/jgcw/square_shared_ogpo_bc.sh \
  > run_logs/jgcw_square_shared_ogpo_bc.log 2>&1 &
```

The Square config enables latent augmentation consistency by default; action-Q
training, Q accuracy evaluation, and action coverage default to disabled. It restores the shared OGPO BC checkpoint and starts online
training directly.

Q(s,z) learns its own TD target and controls the actor. With `use_aug_z=false`,
its updates match DSRL exactly. Square defaults to `use_aug_z=true`,
`aug_z_num=1`, and `aug_z_sigma=0.02`. For every latent critic update (including
extra success-buffer updates), add independent N(0, sigma²) noise to each replay
action chunk, invert the augmented actions with GRACE's reverse Euler method,
and add `mean((Q(s,z_aug) - stop_gradient(Q(s,z)))²)` to the TD loss with unit
weight. The loss averages over samples, augmentations and ensemble members.
Actions and recovered latents are not clipped. The frozen decoder and actor do
not receive gradients from this loss. Inversion is approximate because Euler
integration and the forward decoder's final action clipping are not exactly
invertible. One augmentation keeps the extra reverse-flow batch at replay batch
size; larger `aug_z_num` increases inversion and critic evaluation cost linearly.
This requires `noise_chunk_length=-1`. Log `critic/td_loss`, `critic/aug_z_loss`
and their sum `critic/critic_loss` under `online_agent`. `train_q_sa` additionally trains a separate action-space TD critic with
the same architecture, optimizer and replay minibatches. It has its own target
network and optimizer state. It does not supervise Q(s,z) or affect the actor.
It receives the same extra success-buffer critic updates. Both critics' update
counts are logged separately in `online_budget`.

## Fixed Monte Carlo test bank

`cal_q_accuracy` requires `train_q_sa=true`, low-dimensional Robomimic
observations, discounted-sum chunk rewards, and `entropy_backup=false`.
Before online training, a spawned CPU worker creates its own simulator and
selects 16 distinct states across four temporal quarters of successful
HDF5 demonstrations, preferring different demos within each quarter. This
covers early, middle and late task progress; temporal quarters are not
hand-labeled semantic stages. Model XML, simulator states and episode metadata
are restored, rather than attempting to reconstruct a simulator from a policy
observation vector.

For each state, sample one fixed latent from the initial stochastic latent
actor and decode its fixed action chunk. This produces paired (s,z,a=f(s,z));
it does not require an approximate inverse of a demonstration action. Save the
bank and its demo/progress metadata under `<save_dir>/q_accuracy/`.

At online step 0, then approximately every 100,000 physical steps, send an
immutable policy/critic snapshot to the worker. There is only one outstanding
job; busy intervals do not accumulate a queue. Each of the 16 states is
restored 8 times. Every rollout **executes the same fixed first action chunk**,
then continues with the snapshot's stochastic current policy. Discount every
physical reward by gamma=.99, including the existing success reward shaping.
A policy-only rollout from s would estimate V(s), so is not used as the target
for Q(s,z) or Q(s,a). Fixed seeds across snapshots reduce comparison noise;
repetitions use separate RNG streams.

Training bootstraps through time-limit truncations. Therefore the diagnostic
continues across artificial time limits, stops on a true terminal, and caps
at `accuracy_rollout_steps=1000`. With |reward|<=5, the unobserved discounted
tail is bounded by `5*.99**1000/(1-.99)`, approximately .0216. Log cutoff and
time-limit fractions explicitly. Changing gamma or the cap changes this bound.

Defaults are a starting computational budget, not a guarantee of small MC
error: 16 states x 8 continuations = 128 trajectories per snapshot, at most
128,000 simulator steps. Check `mc_se_mean`, `mc_se_max`, and per-state MC
standard errors; increase repeats when these are too large for the Q error
being measured. Initialization prepares the bank before online training; during
training only snapshot transfer runs in the parent. End of training waits for
outstanding results and evaluates the final snapshot. `accuracy_timeout=3600`
seconds is the startup/evaluation watchdog. The worker uses 2 CPU cores and
cannot allocate the training GPU.

A production-network timing check with the existing 90k frozen BC checkpoint,
4 phase-balanced states and 8 continuations per state took 252 seconds for
12,566 simulator steps. Mean/max Monte Carlo SE were 1.82/3.37 return units,
and state restoration error was 0. Extrapolating by state count gives roughly
17 minutes for the default 16x8 bank on this CPU; actual runtime depends strongly
on policy success and rollout lengths. This is why the default uses 16 rather
than 24 states. There are no claims of full-training or multi-seed validation.

Relevant W&B/CSV metrics:

- `q_accuracy/q_sz/{mae,rmse,bias}` and `q_accuracy/q_sa/{mae,rmse,bias}`;
  smaller MAE/RMSE is better, positive bias means overestimation.
- `q_accuracy/mc_se_mean`, `mc_se_max`, `state_N/mc_se`, and per-stage errors.
- `q_accuracy/snapshot_step` identifies the evaluated **online** step. The
  W&B event is written at the current training log step when the result arrives;
  W&B automatically uses `q_accuracy/snapshot_step` as the default x-axis for
  `q_accuracy/*` charts. Existing customized panels may require selecting it
  manually.
- `q_accuracy/*/mc_noise_corrected_mse` subtracts estimated MC sampling noise
  from squared error and can be negative. It is an estimate, not an accuracy
  percentage.

All individual rollout returns and predictions are saved as
`q_accuracy/snapshot_<online_step>.npz`, alongside JSON metrics. State restoration
must reproduce the fixed observation within 1e-4 or evaluation raises an error.

## Action coverage (JGCW and OGPO)

`cal_action_coverage=true` enables the same implementation in both runners.
OGPO Square's YAML is `ogpo/configs/algos/ogpo.yaml`, used by
`scripts/ogpo/square.sh`. Enable it with:

```bash
bash scripts/ogpo/square.sh --cal_action_coverage=true
```

Use `--agent.cal_action_coverage=true` when invoking `ogpo.main` directly.
Coverage uses physical single-step actions in their executed [-1,1] coordinates,
not z and not flattened action chunks. Each query finds exact K=32 nearest
states among **all strictly earlier online physical transitions**, with fixed
state-coordinate scaling from offline observation standard deviations (floor
1e-3). The current transition is inserted only after the query. Rebuilding a
KD-tree every 5,000 steps is an optimization: all newer points are merged into
each exact nearest-neighbor result. There is no reservoir or history cap.

By default, evaluate novelty at every physical step (`coverage_sample_interval=1`),
retaining every step in history. M counts every eligible transition after at
least K historical points exist, matching the requested duplicate-rate formula.
For cheaper sampled statistics, explicitly set the interval to 100; M then
counts evaluated queries rather than all transitions. The epsilon=.1 threshold is L2 in
normalized physical-action coordinates, without division by action dimension.
It is configurable and should be held fixed between algorithms.

Metrics in `action_coverage/` include `novelty_mean`, novelty quantiles,
cumulative `duplicate_rate`, `window_duplicate_rate`, historical action min/max
and per-coordinate spans, local action spans, and averaged local covariance
entries/trace/largest eigenvalue. `state_knn_radius_mean` shows how local the
selected neighbors actually are; novel states can have distant neighbors.
Covariance is the sample covariance of the K historical actions in that local
state neighborhood. Window statistics reset after logging; the cumulative
rate and action range retain all history. Online coverage excludes offline
training data and prior process history on a resumed run.

Tests: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_jgcw.py`.
