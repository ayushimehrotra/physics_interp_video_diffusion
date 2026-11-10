#!/usr/bin/env bash
# Stage 0 — build both venvs after a fresh machine / pod restart (~12 min).
#   bash scripts/00_setup_envs.sh
# Venvs live in /root/envs. The Kubric source (pinned commit 61f2422) is copied to /root/src/kubric
# before installing, so the install never writes egg-info into the source tree.
# Overrides: KUBRIC_SRC (default /root/kubric), PHYSLAW_MODELS (default /root/models).
set -euo pipefail
export UV_CACHE_DIR=/root/.cache/uv
export PATH="$HOME/.local/bin:$PATH"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENVS=/root/envs
KUBRIC_SRC="${KUBRIC_SRC:-/root/kubric}"
mkdir -p "$ENVS" /root/src

# uv + a uv-managed Python 3.12 (the system may only have 3.11).
command -v uv >/dev/null || python3.11 -m pip install --user -q uv
uv python install 3.12

# Kubric venv (py3.11, bpy 4.3, numpy 1.26).
if [ ! -x "$ENVS/kubric_venv/bin/python" ]; then
  rm -rf /root/src/kubric && cp -r "$KUBRIC_SRC" /root/src/kubric && rm -rf /root/src/kubric/.git /root/src/kubric/*.egg-info
  grep -v '^-e git+' "$HERE/envs/kubric_requirements_locked.txt" > "$ENVS/kubric_reqs.txt"
  uv venv -q --python /usr/bin/python3.11 "$ENVS/kubric_venv"
  uv pip install --python "$ENVS/kubric_venv/bin/python" -r "$ENVS/kubric_reqs.txt"
  uv pip install --python "$ENVS/kubric_venv/bin/python" --no-deps /root/src/kubric
  uv pip install --python "$ENVS/kubric_venv/bin/python" pytest jsonschema pyyaml
fi
"$ENVS/kubric_venv/bin/python" -c "import bpy, kubric, pybullet; print('kubric venv OK, bpy', bpy.app.version_string)"

# Model venv (torch 2.8 / cu128, diffusers, transformers) + this repo's extras.
if [ ! -x "$ENVS/physics_venv/bin/python" ]; then
  uv venv -q --python 3.12 "$ENVS/physics_venv"
  # exact versions frozen on the H100 pod (2026-11-05): torch 2.8.0+cu128, diffusers 0.35.1, transformers 4.56.1
  uv pip install --python "$ENVS/physics_venv/bin/python" --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu128 -r "$HERE/envs/physics_venv_freeze.txt"
fi
"$ENVS/physics_venv/bin/python" -c "import torch; print('physics venv OK, torch', torch.__version__, torch.cuda.is_available())"
# Cosmos-Predict2.5 (output track A) only: its own venv, with the dependencies NVIDIA's cosmos-predict2.5
# scripts/diffusers_inference.py pins (torch 2.10, transformers 4.57.1, cosmos-guardrail, diffusers with
# Cosmos2_5_PredictBasePipeline = 0.39.0). physics_venv keeps diffusers 0.35.1, so no other model's outputs change.
#   PHYSLAW_COSMOS=1 bash scripts/00_setup_envs.sh ; then run stage 07 with $ENVS/cosmos_venv/bin/python
if [ -n "${PHYSLAW_COSMOS:-}" ] && [ ! -x "$ENVS/cosmos_venv/bin/python" ]; then
  uv venv -q --python 3.12 "$ENVS/cosmos_venv"
  uv pip install --python "$ENVS/cosmos_venv/bin/python" --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision transformers==4.57.1 \
      diffusers==0.39.0 "cosmos-guardrail @ git+https://github.com/codeJRV/cosmos-guardrail" accelerate \
      imageio==2.37.0 imageio-ffmpeg==0.6.0 pyyaml==6.0.2 pillow scipy opencv-python-headless pytest
  "$ENVS/cosmos_venv/bin/python" -c "import diffusers; from diffusers import Cosmos2_5_PredictBasePipeline; print('cosmos venv OK, diffusers', diffusers.__version__)"
fi
# Model weights: plain files under ${PHYSLAW_MODELS:-/root/models} (one root for all models).
echo "models: ${PHYSLAW_MODELS:-/root/models}; set HF_HUB_OFFLINE=1"
