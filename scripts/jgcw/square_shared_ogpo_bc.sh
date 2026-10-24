#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBOMIMIC_DATASET_ROOT="${ROBOMIMIC_DATASET_ROOT:-/root/datasets}"
# Shared OGPO BC checkpoint; skip offline training and disable auto-resume.
# Additional CLI overrides use the same --section.key=value format as GRACE.
exec "$PROJECT_ROOT/.venv/bin/python" -m ogpo.main --algo=jgcw_ogpo_square_shared "$@"
