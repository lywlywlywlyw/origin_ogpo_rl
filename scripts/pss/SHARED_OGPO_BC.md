# PSS with the shared Square BC flow

Implementation: `ogpo/agents/pss.py` and `ogpo/runners/pss_rl_runner.py`.
Reference: Ni et al., *Principal Steering Subspaces for Online Adaptation of
Frozen Generative Robot Policies* (2026), local PDF in `ref_papers`, Sec. III.

```bash
cd /root/OGPO_public
mkdir -p run_logs
nohup env ROBOMIMIC_DATASET_ROOT=/root/datasets MUJOCO_GL=egl \
  EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  bash scripts/pss/square_shared_ogpo_bc.sh \
  > run_logs/pss-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

## Algorithm

1. Restore only the BC decoder from
   `exp/ogpo/OGPO/square/square-mh-low_dim/ogpo_squareseed1/params_90000.pkl`.
   Its architecture and parameters are unchanged; actor/critic/temperature are
   initialized fresh. The loader also accepts a DSRL checkpoint's frozen decoder.
2. Sample 64 offline observations before online collection to fit the fixed basis.
   No extra environment interactions or optimizer updates are used.
   Independently sample one Gaussian base noise per observation.
3. Compute forward differences with delta=0.05 of normalized, output-clipped
   chunks. Form the **uncentered** mean R R^T (Eq. 4), then its descending
   eigenbasis. This uses 64*(28+1)=1,856 decoder queries, with no extra simulation
   steps. Probe outputs, basis, eigenvalues, energy and decoder hash are saved in
   `pss_subspace.npz` and `pss_subspace.json`.
4. SAC controls top-8 coordinates z. Every policy query independently samples
   the remaining 20 Gaussian coordinates and decodes w=V_k z+V_perp epsilon.
   Replay and critic inputs use z. Early warmup entries are projected into the
   fixed basis before any learning. The basis and decoder remain frozen.
5. Evaluation is stochastic: both z and the complement are sampled. The optional
   deterministic actor mode fixes z only; the complement still gets resampled.

The decoder is a flow model because the existing shared checkpoint is a flow
model. This does not reimplement the paper's separate diffusion backbone or its
hardware deployment system.

## Comparison with the completed local runs

The default launcher selects `pss_ogpo_square_shared.yaml`, which inherits the
exact existing matched DSRL config and changes only the experiment name and PSS
interface. In particular:

| Quantity | Matched PSS / existing matched DSRL |
| --- | --- |
| Offline decoder | Same OGPO 90,000-step snapshot |
| New physical interactions | 2,000,000, including actor-policy collection warmup |
| First update | After the first 10,000 interactions (at the next chunk boundary) |
| Actor / critic updates | 1,990,000 / 3,980,000 |
| Batch / horizon / physical-step discount | 256 / 4 / 0.99 |
| Critic ensemble / aggregation / tau | 10 / mean / 0.05 |
| Critic optimizer | AdamW, 4.5e-5, weight decay 1e-5 |
| Actor optimizer / bounds | Adam, 3e-4 / [-1, 1] per controlled coordinate |
| Hidden layers / flow steps | 512 x 4 / 10 |
| Evaluation | Every 10,000 interactions; 32 environments x 2 episodes |
| Success replay | Same actor imitation, extra critic step and threshold |
| Target entropy | Same rule, -controlled_dimension/2 (PSS -4, DSRL -14) |

As with matched DSRL, each executed chunk uses discounted rewards and
gamma**actual_length, and UTD is counted per physical interaction then applied at
chunk boundaries. Actual optimizer step counters are checked at completion;
missing replay capacity raises an error rather than silently reducing updates.
Logs start at 90,000 and finish at 2,090,000. Compare PSS `eval.csv` to DSRL
`eval.csv` and OGPO `eval_sde.csv`, which measures OGPO's collection policy.

These settings intentionally differ from the paper's twin/min critics,
entropy-regularized targets, tau=0.005, 20 updates per query, gamma=0.999 per
Square query, reward sums, eight decoder steps, and 200-episode evaluation.
`pss.yaml` exposes the paper-style online settings with the repository's existing
backbone; it requires an explicit compatible offline checkpoint. The matched
launcher uses the existing DSRL objectives, including its success-buffer additions
and entropy_backup=false. It is a matched-budget PSS variant, not an exact
reproduction of the paper's experimental numbers. The paper does not specify its
target-entropy value; `pss.yaml` uses the standard -k convention.

PSS's smaller actor/critic input/output dimensions are intrinsic to its method;
the hidden layers and decoder are matched. The extra forward probe queries are
reported separately: equal environment/update budgets do not mean equal FLOPs.
The response spectrum should be measured on this checkpoint; the paper's reported
retained-energy values cannot be assumed to carry over. Multiple seeds are needed
for a statistical comparison.

Output: `exp/ogpo/OGPO/square/square-mh-low_dim/pss_square_shared_ogpo_bcseed1`.
Regular snapshots include the fixed basis. Process resume is explicitly rejected
because exact simulator/pending-trajectory/PRNG restoration is not implemented.
Existing OGPO/DSRL experiment outputs are never used as output directories.
