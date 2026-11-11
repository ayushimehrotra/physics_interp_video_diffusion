# physlaw — notes for Claude

Benchmark: physical laws inside video models. Spec: `docs/SPEC.md`. State, results, plan: `docs/RESUME.md`
(the only status doc; update it instead of adding new docs). Smoke-test outputs go in `smoke/`.
One branch: `physlaw-v2-merged`.

## Environment

* Repo, data, cache, results and weights live on `/workspace` (geesefs: no symlinks/chmod/`sed -i`/rsync; keep uv
  caches off it). `/root` is not persistent and has 50 GB.
* Two venvs: `/root/envs/kubric_venv` (py3.11, bpy, pybullet, NO torch) runs `physlaw/data/*`;
  `/root/envs/physics_venv` (py3.12, uv-managed, torch) runs models/eval. Keep `physlaw/__init__.py`,
  `physlaw/config.py` and `physlaw/data/*` torch-free. Rebuild both with
  `KUBRIC_SRC=/workspace/kubric bash scripts/00_setup_envs.sh`.
* Model weights: `/workspace/models`. Run models with `PHYSLAW_MODELS=/workspace/models HF_HUB_OFFLINE=1`.
* F1 stays at the 64-group subset in `data/f1`.
* Rendering needs `KUBRIC_USE_GPU=1`; check `render_device` in labels (should be `OPTIX: ...`).
* Never `pkill -f` a pattern that also appears in your own shell command line: it kills that shell.
* Diffusion models: use ONLY the settings pinned in `configs/models/*.yaml` (the model cards'
  official resolution / frames / fps / steps / guidance / scheduler). Getting these wrong is
  what invalidated the predecessor's diffusion results.

## Conventions

* Every stage: separate CLI in `scripts/`, reads/writes disk, idempotent (skip existing unless `--force`).
* Configs are YAML with `defaults:` inheritance and `--set key.path=value` overrides;
  `config_hash` is written into every output.
* Labels: twin velocities live in `physlaw.post_velocity`. The legacy `collision` block
  describes the honest simulation, not the twin.
