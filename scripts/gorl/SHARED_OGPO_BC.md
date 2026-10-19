# GoRL-FM with the shared Square offline decoder

Reference: `ref_codes/GoRL/configs/methods/gorl_fm.toml` (DMControl profile),
`src/flow_policy/{ppo,rollouts,networks}.py` and
`src/gorl/algorithms/decoder_training.py`. This implements the alternating
GoRL-FM method with pretrained initialization and the existing Robomimic protocol;
it is not a reproduction of the original from-scratch benchmark scores.

## Run

```bash
cd /root/OGPO_public
mkdir -p run_logs
nohup env \
  ROBOMIMIC_DATASET_ROOT=/root/datasets \
  MUJOCO_GL=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  bash scripts/gorl/square_shared_ogpo_bc.sh \
  > run_logs/gorl-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Configuration: `ogpo/configs/algos/gorl_ogpo_square_shared.yaml`, inheriting
`dsrl_ogpo_square_shared.yaml`. W&B project: `OGPO`; group: `square`;
run: `gorl_square_shared_ogpo_bc`. Output:
`exp/ogpo/OGPO/square/square-mh-low_dim/gorl_square_shared_ogpo_bcseed1`.

## Shared initialization and comparison protocol

Initial decoder weights come from
`exp/ogpo/OGPO/square/square-mh-low_dim/ogpo_squareseed1/params_90000.pkl`.
Only decoder weights are imported, with tree/shape/finite-value checks.
Initial decoder observation preprocessing is identity, preserving the original
BC input convention. Encoder PPO normalization never changes decoder inputs.

Matched to the existing runs: task, dataset preprocessing, reward wrapper,
512 x 4 encoder/decoder backbone, full 28-dimensional latent/action chunk,
action horizon 4, sinusoidal time embedding, ten forward Euler steps, action
clipping to [-1,1], discount .99 per physical step, 2M total physical training
steps, and stochastic evaluation with 32 environments x 2 episodes every 10k
steps. Compare GoRL `eval.csv` with DSRL `eval.csv` and OGPO `eval_sde.csv`.
No offline data enters online PPO or decoder refinement losses.

This is equal interaction budget, not equal computation or wall time.

## Restored reference algorithm settings

* Each stage starts with fresh random Gaussian encoder parameters, a fresh
  single V(s) (256 x 5, SiLU, no layer normalization), fresh Adam states and PPO
  observation statistics. Reference `fresh` initialization is not an exact
  state-independent N(0,I) encoder. Gaussian scale is softplus(raw)+.001, capped
  at 10. Encoder backbone/head shapes and initialization remain DSRL-compatible.
* PPO: joint actor/value gradient clipping at .5, Adam .001, 16 epochs,
  sequence length 30, requested minibatch size 1024 sequences and 32 minibatches.
  Clip epsilon .15 in stage zero and .3 later; entropy .01; value coefficient
  .25; reward scaling 10; latent regularization .0005 then .001.
* GAE targets are recomputed from the current value function at each update;
  lambda=.95 once per chunk. Physical discount is gamma**executed_chunk_length.
  Truncated transitions mask the TD residual, policy advantage before
  normalization and value loss, following the reference. Cumulative observation
  statistics are installed after all PPO epochs on that rollout.
* Decoder remains frozen during PPO. At each non-final boundary, use the best
  encoder by stage evaluation return to collect fresh stochastic actions. This
  reuses shared scheduled evaluations; no extra evaluation episodes are added.
* Decoder refinement starts from a fresh zero-output velocity network; Adam
  .0003, no gradient clipping, 50,000 updates, batch 8192, eight independent
  Gaussian/time samples per action. Decoder normalization uses train-split
  observations only. This fresh initialization applies to later refinements,
  never to the initial offline checkpoint.
* Split by episode with 10% validation (transition fallback only if collection
  contains one episode). Validate every 100 updates on eight fixed sampled
  batches with fresh Gaussian/time draws. Initial validation at step zero is
  eligible for best-checkpoint selection. No warm-refinement variant is enabled.
* Four stages have weights 2:2:1:1; the last stage finishes after encoder training.
  There is no SAC replay warmup, Q ensemble, target Q or success replay buffer.

## Necessary adaptations that remain

1. **Checkpoint and action interface:** the reference starts from identity,
   uses a reverse flow and pre-tanh action targets. Here the initial decoder is
   the shared BC model; forward CFM and bounded chunk targets preserve its ODE
   and environment interface. Unexecuted chunk coordinates are masked. Forward
   and reverse CFM have equivalent time-reversed objectives, but the complete
   bounded-action parameterization differs from the reference pre-tanh setup.
2. **Network backbone:** encoder/decoder use the existing DSRL networks;
   the reference's Gaussian policy semantics and single value network remain.
3. **Sampling budget:** original MJX uses 2048 PPO environments, 32 x 1024 x 30
   chunk queries per rollout, and 4096 collection episodes in 64 environments.
   Here Robomimic collection is serial. PPO rollouts stop at 10k physical steps,
   the next evaluation boundary, or the stage boundary. Consequently actual
   sequence counts/minibatches are smaller than requested; a smaller final
   minibatch and masked sequence padding preserve contiguous trajectories.
   Each of the three decoder collections stops at 10k physical steps or 4096
   complete episodes, whichever comes first. All collection interactions count
   toward 2M. These effective sampling sizes are explicit adaptations, not claims
   that the original batch sizes were used unchanged.
4. **Evaluation and discount:** use the existing baselines' stochastic evaluation,
   cadence and physical discount rather than reference deterministic 128-episode
   evaluation and gamma=.995. Final policy is reported; best-stage selection is
   used only for the reference's decoder data collection.

Stage boundaries are 666667, 1333333, 1666667 and 2000000 physical steps.
Default decoder collection totals 30,000 steps within that budget. Each PPO
rollout logs its actual queries, sequences and minibatches per epoch. Evaluation
at the final step is performed once after the final PPO update.

## Artifacts and limitations

* `gorl_protocol.json`: initial decoder hash, requested sampling sizes, caps,
  actual physical/gradient counts and stage collection selection details.
* `gorl_decoder_rollout_stageN.npz`: fresh states, sampled latents, bounded
  actions, masks, rewards, executed lengths and episode IDs. Latents are stored
  for provenance; CFM samples independent noise and does not fit rollout pairs.
* `gorl_decoder_split_stageN.npz`: training/validation indices.
* `online_budget.csv`, `online_agent.csv`, `gorl_decoder.csv` and W&B metrics;
  decoder metrics use a monotonic optimizer-update axis while environment steps
  remain fixed during decoder fitting.
* Standard checkpoints and refined stage checkpoints paired with the next
  stage's fresh encoder. Exact interrupted-process resume is not implemented.

This shared checkpoint is a flow model, so the implemented variant is GoRL-FM.
Single-device training with flat states/frozen embeddings is supported. Original
GoRL decoder batch 8192 x eight noise samples is retained and may use substantial
GPU memory with the larger shared backbone; runtime/memory have not been profiled
for this revision.
