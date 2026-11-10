# Resuming on a new machine

## Current machine (2026-11-09): local disk, RTX 6000 Ada 48 GB, 8 CPUs, 15 GB RAM

No `/workspace`; everything is under `/root` (repo `/root/meta/physlaw`, weights `/root/models`,
Kubric source `/root/kubric`). Setup:

```bash
cd /root/meta/physlaw
bash scripts/00_setup_envs.sh          # installs uv + Python 3.12 if missing, builds /root/envs/{kubric_venv,physics_venv}
export HF_HUB_OFFLINE=1
```
Model configs default to `/root/models`, so `PHYSLAW_MODELS` need not be set. F1 stays at the 64-group
subset (decision 2026-11-09). The sections below are the 2026-11-05 notes for the geesefs pods.

---

# Resuming on a new pod (written 2026-11-05, moving H100 -> A100)

Everything needed is on `/workspace` (persistent). The container disk (`/root`: venvs, HF cache,
Claude's memory) is lost on pod stop and must be rebuilt as below.

## 1. Rebuild the environments (~12 min)

```bash
cd /workspace/meta/physlaw
bash scripts/00_setup_envs.sh          # /root/envs/kubric_venv + /root/envs/physics_venv, exact frozen versions
export HF_HUB_OFFLINE=1                # all weights are local; nothing needs the Hub
```
* Pinned versions: `envs/physics_venv_freeze.txt` (torch 2.8.0+cu128, diffusers 0.35.1, transformers
  4.56.1, opencv-python-headless 4.12), `envs/kubric_venv_freeze.txt`. cu128 wheels support A100 (sm_80).
* Weights: plain files under `/workspace/models` (configs read `${env:PHYSLAW_MODELS,/workspace/models}`):
  | dir | model | identity |
  |---|---|---|
  | `vjepa2-vitl-fpc64-256/` | V-JEPA 2 ViT-L (+ untrained control's config) | HF facebook/vjepa2-vitl-fpc64-256 @ b3c1679b |
  | `videomae2/vit_g_hybrid_pt_1200e.pth` | VideoMAE v2-giant pre-training ckpt (with decoder) | sha256 5773c881… (sidecar `.sha256`, checked at load) |
  | `CogVideoX-2b/` | CogVideoX-2b | HF zai-org/CogVideoX-2b @ 1137dacf, every file hash-verified |
  | `CogVideoX1.5-5B-I2V/` | CogVideoX1.5-5B-I2V | HF @ 46c90528 |
  | `Wan2.2-TI2V-5B-Diffusers/` | Wan 2.2 TI2V-5B (not used yet) | |
* geesefs reads are slow (CogVideoX1.5-I2V took ~35 min to load). If the A100 pod has local disk, copy the
  models you will use to e.g. `/root/models` and `export PHYSLAW_MODELS=/root/models` (one root for all).
  Do NOT copy them into an HF cache layout on `/workspace` (geesefs drops symlinks -> 0-byte files).
* Config hashes exclude file locations, so moving weights does not change them (VideoMAE caches carry the
  pre-change hash 2c78b257e18c; same weights; now 4531024ff411).

## 2. State of every stage (F1 subset: 64 groups x 5 twins = 320 clips in `data/f1`)

| stage | V-JEPA 2 | untrained | VideoMAE v2-g | CogVideoX-2b | CogVideoX1.5-I2V |
|---|---|---|---|---|---|
| 04 cache (`cache/<model>/f1`) | done | done | done | done (+ `gen_seed0.mp4` per group) | not run |
| 05 pixel baselines | done | done (copied) | done | done | — |
| L0 | done | done | done | done (table repaired 2026-11-05) | — |
| L2 (incl. motion-adjusted) | done | done | done | done | — |
| L1 | done | n/a | done | done | — |
| 07 gate / 09 outputs | n/a | n/a | n/a | done (gate FAILED 0/20; no extra seeds) | — |

Experiments (`scripts/experiments/`, results under `results/<model>/f1/`):
* `naturalness_cogvideox.py` — done (4 scenes x 4 variants): CogVideoX-2b never continues the context.
* `native_i2v_cogvideox15.py` — done (2 scenes x 2 variants): balls persist with native I2V.

## 3. Resume commands

All CogVideoX-2b stages finished before the pod stopped; nothing is half-done. Every stage is
resumable (stage 4 skips clips with an existing .npz; 07 skips existing seeds; 06/09 just rerun).
Queue scripts used on the H100 are in `scripts/queues/` (they log to /root/envs; adjust paths); logs in `logs/`.

### Next (not started)
* CogVideoX1.5-5B-I2V through stage 4 + L0-L2 (config fixed to the official repo's DPM + dynamic CFG,
  VAE tiling). Cost on the H100: ~45k tokens/pass; expect ~5 h cache (+ generation if the gate passes);
  A100 is ~2x slower. Decide first how its forecast should be conditioned (see STATUS: native I2V vs
  latent replacement).
* Wan 2.2 TI2V-5B adapter (configs/models/wan22_ti2v_5b.yaml).
* Step 5 positive control (deferred by decision).

## 4. A100 notes
* Rendering (stage 01, mentor's 336 groups): Cycles OptiX runs on A100 but without RT cores, so expect it
  to be several times slower than on the H100; `KUBRIC_USE_GPU=1` still applies. The render loop's worker count
  (default 6) was tuned on the H100; re-tune with `--workers`.
* Memory: all stages so far peaked < 35 GB (CogVideoX-2b caching 25 GB); fine on A100-80GB. On A100-40GB
  run one GPU job at a time.
* bf16 / fp16 / sdpa all supported on sm_80.
