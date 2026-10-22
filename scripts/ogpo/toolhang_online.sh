#!/usr/bin/env bash
set -euo pipefail
cd /root/OGPO_public
export ROBOMIMIC_DATASET_ROOT="${ROBOMIMIC_DATASET_ROOT:-/root/datasets}"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Skip BC and initialize from the task's offline checkpoint; accept CLI overrides.
exec .venv/bin/python -m ogpo.main --algo=ogpo_toolhang_online "$@"
