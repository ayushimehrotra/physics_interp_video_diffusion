#!/usr/bin/env bash
# Run an experiment config across every visible GPU.
#
#     ./scripts/pipeline/run_experiment.sh configs/red_ball_slope.yaml
#     ./scripts/pipeline/run_experiment.sh configs/foo.yaml --limit 2 --keep-going
#     NUM_GPUS=2 ./scripts/pipeline/run_experiment.sh configs/foo.yaml
#     CUDA_VISIBLE_DEVICES=0,3 ./scripts/pipeline/run_experiment.sh configs/foo.yaml
#
# One full model replica per GPU, each taking its own slice of the run grid.
# There is no communication between ranks, so a crashed rank costs you its runs
# and nothing else — re-running the same command picks up whatever is missing.
#
# Any extra arguments are forwarded to scripts/run_experiment.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG.yaml [extra args for run_experiment.py]" >&2
  exit 2
fi

VPY=".venv/bin/python"
if [[ ! -x "$VPY" ]]; then
  echo "no .venv found -- run ./scripts/pipeline/setup.sh first" >&2
  exit 1
fi

# GPU count: explicit NUM_GPUS wins, else count what CUDA can actually see
# (which already respects CUDA_VISIBLE_DEVICES).
if [[ -n "${NUM_GPUS:-}" ]]; then
  N="$NUM_GPUS"
else
  N="$("$VPY" -c 'import torch; print(torch.cuda.device_count())')"
fi

if [[ "$N" -lt 1 ]]; then
  echo "no GPUs visible; running single-process on CPU (this will be very slow)" >&2
  N=1
fi

# Downloads are already warm after setup.sh; keep this on for any straggler files.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Each rank owns one GPU, so intra-op thread pools just fight each other.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Cut fragmentation from the VAE decode's large transient allocations.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==> launching on $N GPU(s): $*"

if [[ "$N" -eq 1 ]]; then
  # No torchrun for the single-GPU case: a plain process gives cleaner
  # tracebacks and exit codes.
  exec "$VPY" scripts/pipeline/run_experiment.py "$@"
fi

exec .venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$N" \
  scripts/pipeline/run_experiment.py "$@"
