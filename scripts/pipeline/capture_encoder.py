#!/usr/bin/env python3
"""Capture pooled residual-stream activations of a frozen video encoder for every clip of a set.

    python scripts/pipeline/capture_encoder.py --model videomae-base \
        --dataset datasets/conservation_set_v1 \
        --out outputs/conservation_set_v1/videomae-base/activations

    # truncation rungs for §47 (frames the encoder sees; the rest hold the last frame)
    for k in 10 14 20 30 38 48; do
      python scripts/pipeline/capture_encoder.py --model videomae-base --dataset datasets/conservation_set_v1 \
          --truncate $k --out outputs/conservation_trunc/videomae-base/k$k/activations; done

    # §59: V-JEPA2's own predictor surprise, one JSON for the whole set
    python scripts/pipeline/capture_encoder.py --model vjepa2-vitl --dataset datasets/scene_ladder_v1 \
        --surprise --out outputs/scene_ladder_v1/vjepa2-vitl/surprise.json

Resumable: clips with an existing .npz are skipped. Every capture records the
frame spec, tubelet layout, phase split, truncation, device and the labels'
render_device, because encodings must never be compared across render batches
or GPUs (§58 trap).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from physics_interp.momentum.captures import write_capture
from physics_interp.momentum.encoders import ENCODERS, Encoder, parse_frames
from physics_interp.video import load_frames, load_segmentation


def clip_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "labels.json").exists()
                  and (d / "_COMPLETE").exists())


def even_indices(k: int, n: int) -> list[int]:
    """n frame indices spread evenly over [0, k], as §43's truncation_curve.py: a
    temporally denser but normal-looking clip, rather than one that stops dead."""
    return [int(round(x)) for x in np.linspace(0, k, n)]


def _dest(out: Path, k, d: Path) -> Path:
    return (out / f"k{k}" if k is not None else out) / f"{d.name}.npz"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(ENCODERS))
    p.add_argument("--hf-id", default=None, help="override the Hugging Face checkpoint id")
    p.add_argument("--dataset", required=True, type=Path, help="a set root with one clip dir per clip")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", default=None, help="frame spec start:stop:step[xrepeat]; default per model")
    p.add_argument("--truncate", type=int, default=None, help="frames the encoder sees (hold the last one after)")
    p.add_argument("--resample-to", default=None,
                   help="comma list of k: per k, sample the encoder's N frames evenly over [0, k] (§43's "
                        "truncation, not the hold-last-frame one) and write to OUT/k<k>/. Only the *.all "
                        "pools are meaningful: pre/post splits are not exact on resampled frames")
    p.add_argument("--allow-straddle", action="store_true")
    p.add_argument("--surprise", action="store_true", help="V-JEPA2 predictor surprise instead of activations")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keep-going", action="store_true")
    args = p.parse_args()

    spec = ENCODERS[args.model]
    frames = parse_frames(args.frames or spec.default_frames, spec.n_frames)
    clips = clip_dirs(args.dataset)
    if not clips:
        sys.exit(f"no complete clips under {args.dataset}")
    if args.limit:
        clips = clips[: args.limit]
    print(f"{args.model} ({spec.hf_id}): {len(clips)} clips, frames {frames[0]}..{frames[-1]} "
          f"({len(frames)}), truncate={args.truncate}", flush=True)

    rungs = [int(k) for k in args.resample_to.split(",")] if args.resample_to else [None]
    enc = Encoder(spec, device=args.device, hf_id=args.hf_id)
    print(f"loaded: {len(enc.blocks)} blocks; sites {enc.sites[0]} .. {enc.sites[-1]}", flush=True)
    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    failures = []
    surprises = []
    t0 = time.time()
    for i, d in enumerate(clips, 1):
        todo = [k for k in rungs if args.surprise or not _dest(args.out, k, d).exists()]
        if not todo:
            continue
        try:
            labels = json.loads((d / "labels.json").read_text())
            rgb = load_frames(d, num_frames=49)
            masks = load_segmentation(d)
            if args.surprise:
                s = enc.surprise(rgb, masks, frames, allow_straddle=args.allow_straddle)
                surprises.append({"clip_id": d.name, "labels": labels, "surprise": s})
            else:
              for k in todo:
                fr = frames if k is None else even_indices(k, spec.n_frames)
                pooled, names, layout = enc.capture(rgb, masks, fr, truncate=args.truncate,
                                                    allow_straddle=args.allow_straddle or k is not None)
                meta = {
                    "labels": labels, "pools": names, "sites": enc.sites, "model": args.model,
                    "hf_id": args.hf_id or spec.hf_id, "frames": fr, "tubelet": layout.tubelet,
                    "grid": [layout.n_tubelets, layout.grid_h, layout.grid_w],
                    "phase_split_frame": 20, "truncate": k if k is not None else args.truncate,
                    "truncate_mode": "resample" if k is not None else ("hold" if args.truncate else None),
                    "capture_device": device_name,
                    "render_device": labels.get("scene_constants", {}).get("render_device"),
                    "clip_id": d.name,
                }
                write_capture(_dest(args.out, k, d), pooled, meta)
            if i % 10 == 0 or i == len(clips):
                rate = (time.time() - t0) / i
                print(f"  [{i}/{len(clips)}] {d.name}  {rate:.1f}s/clip", flush=True)
        except Exception:
            failures.append(d.name)
            print(f"  FAILED {d.name}:\n{traceback.format_exc()}", flush=True)
            if not args.keep_going:
                break
    enc.close()
    if args.surprise and surprises:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": args.model, "frames": frames, "capture_device": device_name,
                                        "clips": surprises}, default=str))
        print(f"surprise table -> {args.out}")
    if failures:
        print(f"{len(failures)} clip(s) failed: {failures[:10]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
