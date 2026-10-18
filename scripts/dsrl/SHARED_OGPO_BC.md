# DSRL matched to the completed OGPO square run

Reference: `exp/ogpo/OGPO/square/square-mh-low_dim/ogpo_squareseed1`.
Use the BC snapshot `params_90000.pkl`, not the later online snapshots. The
recorded upper limit was 500,000 BC steps, but the run stopped BC at 90,000.

```bash
cd /root/OGPO_public
CUDA_VISIBLE_DEVICES=0 bash scripts/dsrl/square_shared_ogpo_bc.sh
```

The launcher uses `ogpo/configs/algos/dsrl_ogpo_square_shared.yaml`. Its algorithm
is still `dsrl`; the separate config name selects this specific comparison.
No existing experiment is resumed. The frozen flow decoder receives the OGPO
actor's exact weights; latent actor, temperature and noise-space critic start
fresh. No further offline optimization runs. Log steps start at 90,000 and end
at 2,090,000 so they line up with the saved OGPO run.

| Controlled quantity | Completed OGPO | New DSRL |
| --- | --- | --- |
| Offline flow weights | 90,000-step BC snapshot | Exact same snapshot, frozen online |
| Environment/data settings | `square-mh-low_dim`, all trajectories, proportion 1 | Same recorded settings; offline ratio 0 online |
| Physical training environment interactions | 2,000,000 | Exactly 2,000,000 |
| Start learning | Interaction 10,000 inclusive | Same |
| Online actor optimizer updates | 1,990,001 | Exactly 1,990,001 |
| Online critic optimizer updates | 3,980,002 | Exactly 3,980,002 |
| Batch size / action horizon / discount | 256 / 4 / .99 | Same |
| Critic ensemble / aggregation / EMA tau | 10 / mean / .05 | Same |
| Online critic optimizer | AdamW, constant 4.5e-5, weight decay 1e-5 | Same |
| Evaluation | Every 10,000 interactions; 32 envs x 2 episodes | Same episode budget and interval; a partial chunk can delay an evaluation by up to 3 interactions |

Counts above were checked against optimizer `step` fields: the final OGPO
actor/critic states have steps 1,990,002 / 3,980,003, starting from step 1 after
online optimizer reset. This OGPO run performed two critic updates for every
actor update (including its successful-trajectory update).

Implementation details affecting the interpretation:

- DSRL's legacy UTD is per action chunk. This config counts UTD per physical
  environment step and applies the updates together at each chunk end.
- The extra critic update uses successful replay when available, otherwise
  ordinary replay. This ensures equal optimizer counts even if DSRL collects
  successful trajectories later than OGPO. It is a matched-budget DSRL variant.
- Rewards use discounted sums within each chunk, matching OGPO sequence reward
  aggregation only when the sampled sequence stays within an episode. Partial chunks store their actual length for bootstrapping;
  collection stops immediately on termination or when the interaction budget
  is exhausted.
- The config matches OGPO's success-buffer toggle, batch size and availability
  threshold measured in physical transitions. It also matches gradient clipping.
- `reset_optimizers_with_lr()` in the current OGPO code actually constructs the
  online critic schedule with `ppo_lr` (4.5e-5), despite printing `critic_lr`
  (3e-4). The comparison uses the effective value.
- The latent SAC actor retains DSRL's optimizer/entropy objective. OGPO uses
  policy optimization of its flow and best-of-8 selection; DSRL uses its latent
  actor. Those are algorithm differences. Equal optimizer counts do not imply
  equal FLOPs, inference cost or wall-clock time.
- Compare DSRL `eval.csv` with OGPO `eval_sde.csv`, which evaluates the OGPO policy
  used during collection (best-of-8). OGPO `eval.csv` instead evaluates its ODE
  flow. This experiment compares shared initialization and equal training
  budgets; a single seed cannot establish statistical superiority.

New output directory:
`exp/ogpo/OGPO/square/square-mh-low_dim/dsrl_square_shared_ogpo_bcseed1`.
`online_budget.csv` records physical interactions and actor/critic update counts.
The matched runner fails if the replay buffer would force it to skip updates,
and verifies final counts instead of silently accepting a different budget.

Validation (CPU; no long training run):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  .venv/bin/python -m unittest discover -s tests -p test_dsrl_shared_checkpoint.py -v
```

## Follow-up audit

The audit corrected three implementation issues: registering `online_budget`
with the real CSV logger; sampling successful chunks with replacement so the
physical-step threshold is effective; and respecting `checkpoint.resume=false`
when looking up old W&B run IDs, as well as when loading agent state.

The configuration supports a shared-initialization, equal-interaction and
equal-optimizer-budget comparison. Algorithm-specific differences remain:
OGPO uses best-of-8 selection and flow policy optimization; DSRL uses a latent
SAC actor. Actor objectives/schedules, replay representation and update timing
are not identical, and equal optimizer counts do not imply equal FLOPs.

Terminal handling also differs. OGPO samples fixed-length sequences and masks
ones that cross a termination boundary out of critic loss using `valid[..., -1]`.
Although its intermediate cumulative reward tensor can include rewards from
the next episode, these invalid sequences do **not** contribute to TD loss.
DSRL stores executed chunks and trains on partial terminal chunks with their
actual duration. This is a difference in sample selection, not evidence of
cross-episode reward contamination in OGPO critic training. The audit did not
change the completed OGPO baseline or start a new training run.
