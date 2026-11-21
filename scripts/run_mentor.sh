#!/usr/bin/env bash
# PhysLaw: every model on every law (rendered family), on this machine's GPUs. One command, resumable.
#
# Prerequisites: NVIDIA GPU(s) (the Wan2.2-I2V-A14B model needs an 80 GB GPU; everything else fits in 48 GB),
# ~1.5 TB free for weights at $PHYSLAW_MODELS, the rendered data in data/<family>/ (e.g. data/f1, data/f2, ...),
# and for Cosmos only: the licences of nvidia/Cosmos-Predict2.5-2B and nvidia/Cosmos-1.0-Guardrail accepted on
# huggingface.co and `hf auth login` (or HF_TOKEN set). Without them Cosmos is skipped; everything else runs.
#
#   git clone -b physlaw-v2-merged https://github.com/ayushimehrotra/physics_interp_video_diffusion.git physlaw
#   cd physlaw                                                     # put / link the rendered data in data/<family>/
#   PHYSLAW_MODELS=/path/with/space/models bash scripts/run_mentor.sh            # 1) SMOKE: 2 groups per family
#   FULL=1 PHYSLAW_MODELS=/path/with/space/models bash scripts/run_mentor.sh     # 2) FULL run
#
# SMOKE (default) writes ONLY to smoke/models/ (never cache/ or results/): a pipeline check, not results.
# FULL=1 writes cache/<model>/<family> and results/<model>/<family>, then results/leaderboard.md.
# Rerun the same command after any interruption: finished tasks are skipped (logs/models/_done/), failed ones retried.
# Progress: tail -f logs/models/_status.tsv     Per-task logs: logs/models/<task>.log
# Steps: (1) venvs (scripts/00_setup_envs.sh; the kubric renderer is not needed), (2) weights at the pinned
# revisions (scripts/download_weights.py), (3) the run (scripts/queues/run_models.py: dry-run plan, then all tasks).
# Extra args go to run_models.py, e.g.  --models physrvg,vjepa2_vitl  --families f1,f2  --gpus 0,1
set -euo pipefail
cd "$(dirname "$0")/.."
export PHYSLAW_MODELS="${PHYSLAW_MODELS:-/workspace/models}"
export ENVS="${ENVS:-/root/envs}"
export P="$ENVS/physics_venv/bin/python" P_COSMOS="$ENVS/cosmos_venv/bin/python"
mkdir -p logs/models "$PHYSLAW_MODELS"

echo "== 1/3 venvs ($ENVS)"
SKIP_KUBRIC=1 PHYSLAW_COSMOS=1 bash scripts/00_setup_envs.sh 2>&1 | tee -a logs/models/setup.log

echo "== 2/3 weights ($PHYSLAW_MODELS; resumable, a gated model that fails is skipped later)"
"$P" scripts/download_weights.py 2>&1 | tee -a logs/models/download.log || true

echo "== 3/3 run"
ls -d data/*/ >/dev/null 2>&1 || { echo "no rendered data under data/<family>/"; exit 1; }
if [ -n "${FULL:-}" ]; then MODE=(); else MODE=(--smoke 2); fi
bash scripts/queues/run_all_models.sh --dry-run "${MODE[@]}" "$@"
bash scripts/queues/run_all_models.sh "${MODE[@]}" "$@"
echo "== done: $( [ -n "${FULL:-}" ] && echo 'results/ and results/leaderboard.md' || echo 'smoke/models/ (pipeline check only)' )"
