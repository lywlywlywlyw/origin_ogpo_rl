# AnchorQ with the shared square BC checkpoint

Implementation source: `ref_papers/2610.22852v1.pdf`, Sections 3.1--3.3 and
Appendix B, Algorithms 1--4. Launch:

```bash
cd /root/OGPO_public
mkdir -p run_logs
nohup env \
ROBOMIMIC_DATASET_ROOT="/root/datasets" \
MUJOCO_GL=egl \
EGL_DEVICE_ID=0 \
MUJOCO_EGL_DEVICE_ID=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/anchorq/square_shared_ogpo_bc.sh \
  > run_logs/anchorq-square-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

The launcher accepts dotted CLI overrides such as `--experiment.seed=2`.
The matched YAML inherits `dsrl_ogpo_square_shared.yaml` directly, including
`params_90000.pkl` from `exp/ogpo/OGPO/square/square-mh-low_dim/ogpo_squareseed1`.
No offline training or automatic process resume occurs.

## Selection and learning

1. Construct six constant Cartesian delta chunks (+/- x, y, z) in the single-arm
   7D OSC_POSE command space. Rotation and gripper commands are zero. The motion
   magnitude (default 1.0) and gripper command are explicit parameters because
   the paper does not give a Robomimic-specific reference map. If the action
   wrapper uses normalization, commands are converted to the decoder coordinates.
2. Reverse the frozen vector field using 10 backward Euler-time steps from
   action time 1 to noise time 0. This is reverse-time explicit Euler integration,
   not the implicit backward-Euler solver. It is a numerical inversion of the
   continuous ODE, not an exact inverse of the discretized forward solver.
3. Append an independent Gaussian base-prior defer latent. Cache this bank for
   the current observation and in the replay transition's `next_anchors` for
   bootstrap states. True terminals do not need a next bank; time-limit
   truncations retain the final observation/bank and bootstrap normally.
4. From the first latent decision, construct M=32 additional Gaussian particles
   around each candidate, retaining the anchor itself. Adaptive reach probes a
   separate cloud at sigma_max=0.3, computes the median head standard deviation
   over all probe particles/candidates, and sets sigma=0.05+0.25*clip(u,0,1).
5. On each fresh refinement cloud, beta=1/max(median_particle_std,0.01) is
   computed separately per anchor. The softmax of beta*mean(Q) weights the latent
   barycenter. There is no latent clipping, best-particle substitution, or
   gradient optimization through the decoder. `anchorq_uncertainty_floor=0.01`
   is an explicit implementation choice; the paper specifies a positive floor
   without a value in its benchmark table.
6. Select among barycenters using epsilon-greedy with epsilon decaying linearly
   from 1 to 0.05 over 10,000 decisions. Evaluation uses epsilon=0 while retaining
   fresh particles/defer draws, as in the paper. Q-learning uses fresh refinements
   of cached next-state banks, online-Q selection, and target-Q evaluation.

The exploration schedule counts decisions; the inherited training
and evaluation budgets count physical environment steps.

## Comparison settings

The shared experiment keeps the square baseline's 4x512 decoder, 4x512 Value
MLP with layer normalization and 10 independent ensemble heads, horizon 4,
discount 0.99, batch 256, learning rate 4.5e-5, AdamW weight decay 1e-5,
tau 0.05, 10,000-step warmup, 2,000,000 physical training steps, and 64 evaluation
episodes (32 environments x 2). Only Q(s,z) is trained. The decoder is frozen,
and there is no latent actor, temperature optimizer, or latent BC objective.
This retains the critic architecture of DSRL; PSS/GRACE's additional action or
distillation critics are method-specific and are not added to AnchorQ.

Two critic updates run per eligible physical step, including the same extra
successful-trajectory Q update when a successful batch is available; otherwise
the extra update uses ordinary replay. `online_budget` records actual optimizer
steps and zero actor/decoder updates. The expected critic budget is 3,980,000
steps. Matching critic steps does not match total compute: AnchorQ evaluates many
candidates, while SAC baselines also update an actor. Candidate Q queries are
processed in bounded batches to limit activation memory.

The corrected v2 configuration restores Appendix B's method-specific components:
REDQ min over two randomly sampled target heads, frozen independent randomized
prior functions (scale 2), and prioritized replay with importance weights. Q
means and uncertainty both include the prior. Only the original Value residual
network is optimized; each prior has the same shape and no optimizer. This adds
frozen parameters and inference cost, not trainable capacity or critic steps.

PER priorities are the mean absolute TD error over heads, refreshed after every
optimizer step before the next sample. Importance weights use the global minimum
sampling probability. Alpha=0.6, beta=0.4 annealed to 1 across the online budget,
and epsilon=1e-6 are explicit choices because the paper does not tabulate these
values. The baseline's extra successful-trajectory update remains uniform within
successful chunks, with unit weights; its rows also refresh their PER priorities.
This success-buffer rule and physical-step reward/discount/update budgets remain
benchmark adaptations. This is not a claim of an exact LIBERO experiment.

Particle warmup and epsilon decay remain 10,000 **decisions**, about 40,000
physical steps with horizon 4. Do not interpret the first 40k as a test of active
particle refinement. To reproduce the previous method ablation, override
`--agent.anchorq_prior_scale=0 --agent.anchorq_prioritized_replay=false
--agent.anchorq_target_aggregation=mean` and use a separate experiment name.

Parameters/CSV logs are stored in
`exp/ogpo/OGPO/square/square-mh-low_dim/anchorq_square_shared_ogpo_bc_v2seed1/`.
`anchorq_references.npz` saves the actual reference actions and
`anchorq_method.json` records the comparison choices. Parameter snapshots include
the critic/target, frozen priors, frozen decoder, reference actions and decision counter; exact
process resume is rejected because simulator, pending episodes and collector RNG
state are not saved together.
