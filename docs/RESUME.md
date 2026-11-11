# Resuming on a new machine

Everything lives in S3 bucket `s3://ayushi-runpod-datasets` (region us-west-2), mirroring `/root`:

| S3 prefix | Local path | What |
|---|---|---|
| `meta/physlaw/` | `/root/meta/physlaw` | repo incl. `.git` (all branches: `physlaw-v2`, `wp-*`), `data/`, `cache/`, `results/`, `logs/` |
| `models/<name>/` | `/root/models/<name>` | model weights (configs default to `/root/models`, or set `PHYSLAW_MODELS`) |
| `kubric/` | `/root/kubric` | Kubric source, pinned commit 61f2422 |
| `meta/physlaw_testruns/2026-11-10/` | (reference only) | test-run outputs of the 2026-11-10 build (renders, caches, PC1/PC2 checkpoints) |

Code is also on GitHub: branch `physlaw-v2` of `ayushimehrotra/physics_interp_video_diffusion`.

## Restore (H100 or any CUDA 12.8 machine)

```bash
B=s3://ayushi-runpod-datasets
aws s3 sync $B/meta/physlaw /root/meta/physlaw
aws s3 sync $B/kubric /root/kubric
aws s3 sync $B/models /root/models --exclude "Wan2.2-I2V-A14B-bf16/*" --exclude "*.download.log"
cd /root/meta/physlaw
git worktree prune                      # the wp-* worktrees were on the old machine; branches stay in .git
bash scripts/00_setup_envs.sh           # uv + Python 3.12, builds /root/envs/{kubric_venv,physics_venv} (~12 min)
export HF_HUB_OFFLINE=1
/root/envs/physics_venv/bin/python -m pytest -q tests     # expect all pass (12-14 skips need PHYSLAW_DATA)
```

Notes:
* Rendering needs `KUBRIC_USE_GPU=1`; check `render_device` in a clip's labels (`OPTIX: ...`).
* Models not in the bucket: Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P (disk), Cosmos-Predict2.5-2B (gated:
  accept the licence on Hugging Face, `hf auth login`, then download). Their configs pin `hf_id` and `revision`.
* Never `pkill -f` a pattern that also appears in your own command line: it kills that shell.
* State, results and the per-stage commands for full runs: `docs/STATUS.md`.
