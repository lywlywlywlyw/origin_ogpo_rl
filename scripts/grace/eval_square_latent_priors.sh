#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export ROBOMIMIC_DATASET_ROOT="${ROBOMIMIC_DATASET_ROOT:-/root/datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec .venv/bin/python -u -m ogpo.eval_latent_priors "$@"
