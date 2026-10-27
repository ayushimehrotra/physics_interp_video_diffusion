#!/usr/bin/env bash
# One-time environment setup. Run once per machine:
#
#     ./scripts/pipeline/setup.sh
#
# Creates .venv, installs pinned requirements, and pre-downloads the model
# weights so the first experiment does not spend 20 minutes downloading (and so
# a later multi-GPU launch does not have N processes racing for the same cache).
#
# Options:
#   --python PATH     interpreter to build the venv with (default: python3)
#   --no-download     skip weight prefetch
#   --model REPO      HF repo to prefetch (default: zai-org/CogVideoX-2b)
#   --cache-dir DIR   HF cache location (default: HF_HOME, else ~/.cache/huggingface)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="python3"
DOWNLOAD=1
MODEL_REPO="zai-org/CogVideoX-2b"
CACHE_DIR="${HF_HOME:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)     PYTHON_BIN="$2"; shift 2 ;;
    --no-download) DOWNLOAD=0; shift ;;
    --model)      MODEL_REPO="$2"; shift 2 ;;
    --cache-dir)  CACHE_DIR="$2"; shift 2 ;;
    # Print the header comment block, stopping at the first non-comment line so
    # this stays correct as the header grows.
    -h|--help)    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

echo "==> repo:   $REPO_ROOT"
echo "==> python: $($PYTHON_BIN --version 2>&1)"

# Python 3.10+ required: the codebase uses PEP 604 unions at runtime in places.
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"need Python >= 3.10, got {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  echo "==> creating .venv"
  "$PYTHON_BIN" -m venv .venv
else
  echo "==> reusing existing .venv"
fi

VPY=".venv/bin/python"
echo "==> upgrading pip"
"$VPY" -m pip install -q --upgrade pip setuptools wheel

echo "==> installing requirements (this pulls ~10GB of CUDA wheels; be patient)"
"$VPY" -m pip install -r requirements.txt

echo "==> verifying"
"$VPY" - <<'PY'
import torch, diffusers, transformers
print(f"    torch        {torch.__version__}")
print(f"    diffusers    {diffusers.__version__}")
print(f"    transformers {transformers.__version__}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"    cuda         {torch.version.cuda}, {n} GPU(s)")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"      [{i}] {p.name}, {p.total_memory / 2**30:.0f} GiB")
else:
    print("    cuda         NOT AVAILABLE -- generation will not work")
PY

# Import the package itself so a syntax error surfaces now, not 20 minutes into
# a download on a machine you don't have a shell on.
"$VPY" -c "import physics_interp; print(f'    physics_interp {physics_interp.__version__}')"

if [[ "$DOWNLOAD" -eq 1 ]]; then
  echo "==> prefetching weights: $MODEL_REPO"
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  ${CACHE_DIR:+HF_HOME="$CACHE_DIR"} \
  "$VPY" - "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
# Skip the duplicate .bin copies; diffusers loads the safetensors.
path = snapshot_download(repo, ignore_patterns=["*.bin", "*.pth", "*.msgpack"])
print(f"    cached at {path}")
PY
fi

cat <<EOF

==> setup complete.

    smoke test (1 video, ~2 min on an A100):
      .venv/bin/python scripts/pipeline/run_experiment.py configs/red_ball_slope.yaml --limit 1

    full config on every GPU:
      ./scripts/pipeline/run_experiment.sh configs/red_ball_slope.yaml

EOF
