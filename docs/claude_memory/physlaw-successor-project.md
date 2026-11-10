---
name: physlaw-successor-project
description: /workspace/meta/physlaw is the new benchmark repo (from physlaw_benchmark_spec.md); old physics_interp_video_diffusion results are considered outdated
metadata:
  node_type: memory
  type: project
  originSessionId: ec457d78-1f1c-47d5-bfaa-c7a1cbe32a7c
  modified: 2026-11-04T22:43:41.047Z
---

On 2026-11-04 the user had `/workspace/meta/physlaw` created as the successor to `/workspace/meta/physics_interp_video_diffusion`, following `/workspace/physlaw_benchmark_spec.md` (copied to `docs/SPEC.md`). The user considers many results in the old CLAUDE.md outdated because the diffusion models were not run at their correct (official) configurations.

**Why:** off-spec settings (e.g. CogVideoX1.5 at 45 frames, V-JEPA2 frames repeated) invalidate comparisons; the new benchmark pins official settings per model in `configs/models/*.yaml`.

**How to apply:** treat old numbers as methodology/reference only, never as results; check `docs/STATUS.md` in physlaw for current milestone state and the deliberate spec deviations (48 fps x 289-frame master render, KE-reducing split sign). See [[physlaw-env-traps]] for pod setup.
