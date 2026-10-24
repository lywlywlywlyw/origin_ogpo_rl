#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBOMIMIC_DATASET_ROOT="${ROBOMIMIC_DATASET_ROOT:-/root/datasets}"
# Shared OGPO BC checkpoint; skip offline training and disable auto-resume.
# Additional CLI overrides use the same --section.key=value format as GRACE.
exec .venv/bin/python -m ogpo.main --algo=jgcw_ogpo_square_shared "$@"
