#!/usr/bin/env bash
set -euo pipefail
cd /root/OGPO_public
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Restore the shared frozen decoder and skip offline training/auto-resume.
# Extra CLI overrides are forwarded; defaults use DSRL-NA's 1:1:0.5 update ratio.
exec .venv/bin/python -m ogpo.main --algo=dsrlna_ogpo_square_shared "$@"
