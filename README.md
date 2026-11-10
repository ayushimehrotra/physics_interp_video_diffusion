# PhysLaw-Internal

Do video models obey physical laws **inside** their representations, and does that match
whether their **generated outputs** obey them? Full specification: [`docs/SPEC.md`](docs/SPEC.md).
What is built and what deviates from the spec: [`docs/STATUS.md`](docs/STATUS.md).

Successor to `../physics_interp_video_diffusion`. Reusable code was copied in (see
STATUS); its results are **not** carried over — see [`docs/prior_work/README.md`](docs/prior_work/README.md).

## Layout (spec §1)

```
configs/{data,models,eval}/   YAML; data/_master.yaml holds the render format
physlaw/data/                 generate.py (twin groups), twins.py (edit algebra), kubric_scene.py
                              (vendored Kubric scene), collision_physics.py, splits.py, schema/
physlaw/models/               base.py (ModelAdapter, TokenGridSpec, frame sampling) + adapters
physlaw/tokens/               project.py (masks -> token ROIs), pool.py
physlaw/probes/               ridge.py (grouped nested CV, grid-max nulls), observer.py
physlaw/eval/ track/ analysis/   levels L0-L4, output-level eval, tracking, aggregation
physlaw/_legacy/              old physics_interp package + scripts, kept as porting reference
scripts/00..11_*              one CLI per stage, each reads/writes disk, resumable
tests/                        twin identities, probes + tokens (pure), data acceptance (needs data)
```

## Running

```bash
bash scripts/00_setup_envs.sh                         # /root/envs/{kubric_venv,physics_venv}

# Stage 1: render F1 (spec milestone 1). ~950 frames/group at ~2.3 s/frame/worker on an H100.
KUBRIC_USE_GPU=1 /root/envs/kubric_venv/bin/python scripts/01_generate_data.py \
    --config configs/data/f1_collisions.yaml --workers 8            # -> data/f1/
#   pilot: --n-groups 8 --out data/f1_pilot ; physics only: --no-render

# Stage 2: spec §2.3 acceptance tests + report + contact sheet
/root/envs/kubric_venv/bin/python scripts/02_validate_data.py --data data/f1

# Unit tests (no GPU, no data)
/root/envs/kubric_venv/bin/python -m pytest -q -p no:cacheprovider tests/
```

Data layout: `data/<family>/<family>_<seed>/<twin>/{video.mp4, masks.npz, labels.json, _COMPLETE}`,
plus `_GROUP_COMPLETE` per group and `_runlog.jsonl` per family. Labels: old generator's full
per-frame physics plus a `physlaw` block (family, twin, group ID, split, contact frame, camera
K and extrinsics, masses/radii/densities, pre/post velocities, edit record, raw-frame hashes).

## Disk estimate

Per F1 group (5 twins, 289 frames, 832x480): mp4 + masks ~ measured in the pilot report
(`data/f1_pilot/_validation/report.json`). Activation-cache estimates per model: TODO at milestone 2.
