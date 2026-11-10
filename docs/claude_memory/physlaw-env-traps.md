---
name: physlaw-env-traps
description: Pod environment traps for the physlaw repo - geesefs, two venvs under /root/envs, Kubric GC bug
metadata:
  type: project
---
/workspace is geesefs (rsync, sed -i, symlinks, chmod fail; torch import from it hangs). Venvs are rebuilt after each pod restart by `bash /workspace/meta/physlaw/scripts/00_setup_envs.sh` into /root/envs/kubric_venv (py3.11, bpy, no torch) and /root/envs/physics_venv (py3.12, torch). Rendering needs KUBRIC_USE_GPU=1; OptiX works on H100 (~2.3 s/frame, per-frame overhead dominated, so parallelise workers).

**Why:** these cost real debugging time on 2026-11-04 (rsync failures, a Kubric PyBullet __del__ client-id reuse bug, a pkill pattern killing its own shell).

**How to apply:** use cp/Python for file ops on /workspace, check /root/envs exists before running anything, and see physlaw `CLAUDE.md` for the full list. Related: [[physlaw-successor-project]].
