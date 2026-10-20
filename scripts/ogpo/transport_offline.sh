#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
# Same offline settings as the full task config; no online training.
# Keep 64 evaluation episodes using eight workers per task.
exec .venv/bin/python -u -m ogpo.main \
    --algo=ogpo_transport \
    --eval.n_eval_envs=8 \
    --eval.eval_episodes=8 \
    "$@" \
    --training.bc_only=true \
    --training.online_steps=0
