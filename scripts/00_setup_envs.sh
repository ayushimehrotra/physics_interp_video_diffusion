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
# Model weights: plain files under ${PHYSLAW_MODELS:-/root/models} (one root for all models).
echo "models: ${PHYSLAW_MODELS:-/root/models}; set HF_HUB_OFFLINE=1"
