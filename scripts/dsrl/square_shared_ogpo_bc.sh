#!/usr/bin/env bash
set -euo pipefail
cd /root/OGPO_public
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# This independent config disables auto-resume and skips BC. Extra CLI overrides
# are accepted, but changing budgets/environment invalidates the matched setup.
exec .venv/bin/python -m ogpo.main --algo=dsrl_ogpo_square_shared "$@"
