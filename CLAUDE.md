# physlaw — notes for Claude

Benchmark: physical laws inside video models. Spec: `docs/SPEC.md`. State: `docs/STATUS.md`.
Successor of `../physics_interp_video_diffusion`. Its CLAUDE.md is copied to
`docs/prior_work/` for METHODOLOGY only. Its numbers are not PhysLaw results (diffusion models were
run off their official configs; see `docs/prior_work/README.md`).

## Environment (current machine, since 2026-11-09)

* Everything is on local disk under `/root`: repo `/root/meta/physlaw`, weights `/root/models`,
  Kubric source `/root/kubric` (pinned commit 61f2422), venvs `/root/envs`. No `/workspace`.
* GPU: one RTX 6000 Ada (48 GB, sm_89). Host: 8 CPUs, **15 GB RAM**. Load large models straight
  to the GPU (or with offload); a full CPU-side load of a 5B+ pipeline can run out of host RAM.
* Two venvs: `/root/envs/kubric_venv` (py3.11, bpy, pybullet, NO torch) runs `physlaw/data/*`;
  `/root/envs/physics_venv` (py3.12, uv-managed, torch) runs models/eval. Keep `physlaw/__init__.py`,
  `physlaw/config.py` and `physlaw/data/*` torch-free. Rebuild both with `scripts/00_setup_envs.sh`.
* Model weights: plain files in `/root/models` (configs: `${env:PHYSLAW_MODELS,/root/models}`).
  Run models with `HF_HUB_OFFLINE=1`.
* F1 stays at the 64-group subset in `data/f1` (decision 2026-11-09: the other 336 groups are not rendered).
* Earlier pods used a geesefs `/workspace` (no symlinks/chmod/`sed -i`/rsync, torch import hangs);
  `docs/RESUME.md` and `docs/claude_memory/` describe that setup.
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
