# F1 data: handoff for rendering the full set

## What exists

`data/f1/` holds the **first 64 of 400** F1 twin groups (seeds 100000–100063, 320 videos), rendered
on an H100 on 2026-11-05 and validated with `scripts/02_validate_data.py` (report in
`data/f1/_validation/`). The subset is a strict prefix of the full set: same config, same seeds,
same generator.

Each group has 5 twins that are byte-identical before contact and differ only afterwards:

| twin | after contact | physically possible |
|---|---|---|
| `conserve_e1` | elastic | yes |
| `conserve_e015` | e = 0.15 | yes |
| `mom_violate_e1` | elastic; impulse split moved off m_A/m_B | no (momentum) |
| `mom_violate_e015` | same split change at e = 0.15 | no (momentum) |
| `energy_create` | e = 1.5, momentum exact | no (energy) |

Clip: 832x480, 48 fps, 289 frames (6.0 s), contact at frames 112–128. Per twin:
`video.mp4`, `masks.npz` (instance ids 0 floor, 1 A, 2 B), `labels.json` (schema in
`physlaw/data/schema/labels.schema.json`; PhysLaw fields under the `physlaw` key).

## Rendering the remaining 336 groups

```bash
bash scripts/00_setup_envs.sh          # once per pod: /root/envs/{kubric_venv,physics_venv}
cd /root/meta/physlaw
KUBRIC_USE_GPU=1 /root/envs/kubric_venv/bin/python scripts/01_generate_data.py \
    --config configs/data/f1_collisions.yaml --workers 6 --out data/f1
/root/envs/kubric_venv/bin/python scripts/02_validate_data.py --data data/f1
```

The driver skips any group that already has `_GROUP_COMPLETE`, so it renders only seeds
100064–100399, and it can be interrupted and re-run safely. To split across machines, give each a
disjoint range by running into separate directories with
`--set seed_offset=<start> n_groups=<count>` and merging the group folders afterwards (group
names encode the seed, so they never collide).

**Do not change** `configs/data/_master.yaml` or `configs/data/f1_collisions.yaml` between parts:
every label records `config_hash`, and the set must be one configuration. (Throughput-only keys
`persistent_data`, `render_threads` and `decode_workers` do not change the pixels beyond denoiser
noise, but leave them alone too.)

## Cost (H100 80GB)

~950 rendered frames per group; GPU saturates at 4+ workers at ~5.4 frames/s → ~3 min/group,
336 groups ≈ 17 h. On a GPU without RT cores or with less memory, measure first with
`--n-groups` on a scratch `--out`. The first frame on a fresh pod can take several minutes
(Cycles kernel load). Check `render_device` in any `labels.json`: it should start with `OPTIX`.

## Known caveats

* `mom_violate_e1` also creates kinetic energy in about half the groups. This can't be avoided at
  e = 1. Use `physlaw.energy_physical` to stratify. `mom_violate_e015` never creates energy (enforced).
* Splits are by hash of the group ID, so roughly 70/15/15, and all twins of a scene share a split.
* Environment traps (geesefs, venvs under /root, pkill self-match): see `CLAUDE.md`.
