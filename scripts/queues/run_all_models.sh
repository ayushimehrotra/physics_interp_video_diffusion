#!/usr/bin/env bash
# Run the PhysLaw benchmark (stages 04-09 + 11) for every model on every rendered family, over all GPUs.
# Run from anywhere after scripts/00_setup_envs.sh and scripts/queues/render_all.sh (data/<family>/ rendered):
#
#   bash scripts/queues/run_all_models.sh --dry-run          # the task list, skipped models and why; no GPU work
#   bash scripts/queues/run_all_models.sh --smoke 2          # FIRST: 2 groups per family -> smoke/models/ (~1-2 h)
#   bash scripts/queues/run_all_models.sh                    # everything -> cache/, results/, results/leaderboard.md
#   bash scripts/queues/run_all_models.sh --models physrvg,vjepa2_vitl --families f1,f7 --gpus 0,1
#
# Resumable: rerun the same command; finished tasks (logs/models/_done/<task>) are skipped, failed ones retried.
# Progress: tail -f logs/models/_status.tsv (time, task, exit code, minutes, GPU); per-task logs in logs/models/.
# Each GPU runs one model at a time; GPUs smaller than a model's MIN_GPU_GB (scripts/queues/run_models.py) are
# never given it. Skipped automatically: Cosmos-Predict2.5 until its gated weights + cosmos_venv exist, PC1 / PC2
# until stage 10 has trained them. Everything else: scripts/queues/run_models.py --help.
#
# Env: PHYSLAW_MODELS (default /workspace/models), P (physics venv python), P_COSMOS (cosmos venv python).
set -u
cd "$(dirname "$0")/../.."
export PHYSLAW_MODELS=${PHYSLAW_MODELS:-/workspace/models} HF_HUB_OFFLINE=1
P=${P:-/root/envs/physics_venv/bin/python}
mkdir -p logs/models
"$P" scripts/queues/run_models.py "$@" 2>&1 | tee -a logs/models/run_all_models.log
exit "${PIPESTATUS[0]}"
