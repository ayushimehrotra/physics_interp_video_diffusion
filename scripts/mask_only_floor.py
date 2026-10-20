"""§20's mask-only leak control, for whatever pool spec a probe run used.

Object-local pooling selects tokens with the ground-truth segmentation, so the
*choice of region* is an information channel independent of the tokens'
contents. The control is to fit the same probe on the pooling weights ALONE,
with no activations anywhere: whatever that scores is the floor an object-local
number has to be read against.

This matters more for a concatenated pool spec than it ever did for a single
pool. §20 found `log_mass_ratio` clean (floors −2.470 / −0.024) *because* the
ratio needs both objects across both phases and one pool's mask holds neither —
an argument that dies the moment the probe is handed `A.pre+A.post+B.pre+B.post`,
whose mask holds exactly those ingredients.

No model and no GPU: the pooling geometry is rebuilt from the capture's own
metadata (`frame_indices`, `input_size`, the token layout and the phase split),
so the masks are the same ones the capture pooled with, by construction.

    .venv/bin/python scripts/mask_only_floor.py \
        --capture-dir outputs_v3_videomae-base \
        --dataset-dir momentum_dataset/probe_set_v3_latent \
        --pools "A.pre+A.post+B.pre+B.post,A.pre" \
        --targets log_mass_ratio,delta_v_A --where momentum_conserved=true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_interp.interp import extract_label, fit_probe, select_scenes  # noqa: E402
from physics_interp.models.base import TokenLayout  # noqa: E402
from physics_interp.runner import build_pool_weights  # noqa: E402
from physics_interp.video import load_segmentation, masks_to_latent_grid  # noqa: E402

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def parse_where(spec: str | None) -> dict:
    out = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k.strip()] = v.lower() == "true"
        else:
            out[k.strip()] = float(v) if v.replace(".", "", 1).isdigit() else v
    return out


def scene_mask_rows(scene_dir: Path, meta: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The capture's own pooling weights for one scene, rebuilt from metadata.

    Returns ``(normalised, occupancy, names)``, both ``[n_pools, n_tokens]``.

    *normalised* is the exact row :func:`build_pool_weights` handed the recorder
    — the mask as the probe saw it, summing to 1. *occupancy* is the same mask
    before that division. The difference is not cosmetic: normalising fixes the
    row sum at 1 and so removes the object's **area**, which is the channel
    `radius_A` leaks through (0.25 raw vs −0.02 normalised on VideoMAE A.pre).
    Report both — normalised is the faithful description of what the pooled
    activation actually is, occupancy is the strictly stronger control, since it
    is the normalised row times a per-scene scalar the probe can then use.
    """
    lay = meta["metadata"]["layout"]
    layout = TokenLayout(kind=lay["kind"], frames=lay["frames"], height=lay["height"],
                         width=lay["width"], text_len=lay["text_len"])
    idx = meta["frame_indices"]
    size = meta["metadata"]["model"]["input_size"]
    tubelet = len(idx) // layout.frames
    if tubelet * layout.frames != len(idx):
        raise ValueError(
            f"{len(idx)} input frames do not group evenly into {layout.frames} "
            "temporal cells; the capture's geometry is not what this assumes."
        )

    seg = load_segmentation(scene_dir)[idx]
    # Nearest, never bilinear: these are instance ids, and an interpolated 1.5
    # belongs to neither sphere and is dropped by the `== seg_id` test.
    m = torch.as_tensor(np.asarray(seg)).float()[:, None]
    m = torch.nn.functional.interpolate(m, size=(size, size), mode="nearest")
    seg = m[:, 0].round().to(torch.uint8).numpy()

    occ = masks_to_latent_grid(seg, layout, ids=(1, 2),
                               frame_to_cell=lambda t: t // tubelet)
    regions = {"A": occ[0], "B": occ[1]}
    regions["all"] = np.ones((layout.frames, layout.height, layout.width), np.float32)
    phases = {k: tuple(v) for k, v in meta["phases"].items()}
    weights, names = build_pool_weights(layout, regions, phases)

    raw = []
    for occupancy in regions.values():
        for start, end in phases.values():
            masked = np.zeros_like(occupancy)
            masked[start:end] = occupancy[start:end]
            raw.append(masked.reshape(-1))
    return weights.numpy(), np.stack(raw).astype(np.float32), names


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-dir", required=True,
                   help="an outputs_* directory whose captures define the geometry")
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v3_latent")
    p.add_argument("--experiment", default="momentum_probe")
    p.add_argument("--pools", default="A.pre+A.post+B.pre+B.post",
                   help="comma-separated pool specs; '+' concatenates on the feature axis")
    p.add_argument("--targets", default="log_mass_ratio")
    p.add_argument("--where", default=None)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the table as JSON here")
    args = p.parse_args(argv)

    act_dir = Path(args.capture_dir) / args.experiment / "activations"
    paths = sorted(act_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no captures under {act_dir}")

    rows, raws, labels, pool_names = [], [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
        w, raw, names = scene_mask_rows(Path(args.dataset_dir) / path.stem, meta)
        if pool_names is None:
            pool_names = names
        elif names != pool_names:
            raise ValueError(f"{path.stem} pools {names} != {pool_names}")
        rows.append(w)
        raws.append(raw)
        labels.append(meta["labels"])

    M = np.stack(rows)                                   # [scenes, pools, tokens]
    R = np.stack(raws)
    keep = select_scenes(labels, parse_where(args.where) or None)
    M, R, labels = M[keep], R[keep], [l for l, k in zip(labels, keep) if k]
    idx = {n: i for i, n in enumerate(pool_names)}
    print(f"{len(labels)} scene(s), {M.shape[1]} pools x {M.shape[2]} tokens "
          f"from {args.capture_dir}", flush=True)

    # The all-ones `all.*` masks are identical in every scene and must score ~0.
    # If they ever do not, the measurement itself is broken.
    checks = [p for p in ("all.pre", "all.post") if p in idx]

    out = []
    print(f"\n{'target':<20} {'pool spec':<30} {'normalised':>11} {'occupancy':>11} "
          f"{'shuf':>8}")
    print("-" * 84)
    for target in [t.strip() for t in args.targets.split(",")]:
        Y = np.stack([extract_label(l, target) for l in labels])
        if np.ptp(Y, axis=0).max() == 0:
            print(f"{target:<20} constant on this subset — skipped")
            continue
        for spec in [s.strip() for s in args.pools.split(",")] + checks:
            fits = {}
            for kind, src in (("normalised", M), ("occupancy", R)):
                X = np.concatenate([src[:, idx[p], :] for p in spec.split("+")],
                                   axis=-1).astype(np.float64)
                fits[kind] = fit_probe(
                    X, Y, alphas=ALPHAS, train_fraction=args.train_fraction,
                    split_seed=args.split_seed, standardize=True)
            tag = spec + ("  <- sanity, ~0" if spec in checks else "")
            print(f"{target:<20} {tag:<30} {fits['normalised']['r2']:>11.3f} "
                  f"{fits['occupancy']['r2']:>11.3f} "
                  f"{fits['normalised']['r2_shuffled']:>8.3f}")
            out.append({"target": target, "pool": spec,
                        "r2_normalised": fits["normalised"]["r2"],
                        "r2_occupancy": fits["occupancy"]["r2"],
                        "r2_shuffled": fits["normalised"]["r2_shuffled"],
                        "dim": fits["normalised"]["dim"],
                        "n_train": fits["normalised"]["n_train"],
                        "n_test": fits["normalised"]["n_test"],
                        "sanity_check": spec in checks})

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"capture_dir": args.capture_dir, "dataset_dir": args.dataset_dir,
             "where": args.where, "num_scenes": len(labels), "results": out}, indent=2))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
