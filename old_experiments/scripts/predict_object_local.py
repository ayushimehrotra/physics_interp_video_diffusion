#!/usr/bin/env python
"""Pre-contact-only prediction, read through OBJECT-LOCAL pools.

The mean-pooled version of this test (and §15's) throws away most of the signal:
measured on the v1 capture, `radius_A` reads 0.881 from an object-local pool at
the last block and 0.070 from a frame-wide one — a factor of 12 on a *static*
property that no amount of withheld future information could affect. Every
physics target here is object-local, so the frame-wide readout was the wrong
instrument.

This rebuilds the test with the pooling the capture path already uses:
segmentation masks mapped onto the token grid, A and B pooled separately.

  encoder    pre-only encoding, pooled over A's PRE-contact tokens
  predictor  predictor output, pooled over A's POST-contact tokens
  delta      does the predictor add anything over the encoding it came from?

**The control that decides whether any of it means anything.** Object-local
pooling selects *which* tokens to average, and the selection is made from the
ground-truth segmentation. Since transformers carry positional information, the
pooled vector encodes roughly *where the object is* — and for a target like
"post-contact velocity", where the object is over time nearly is the answer. So
this script also fits a probe on the pooling mask ALONE, with no activations in
it. Whatever that scores is leakage, and any object-local number has to be read
against it rather than against zero. This applies to every object-local result
in the project, not just this test.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402

from steer_velocity import fit_readout, split_indices                # noqa: E402
from physics_interp.interp import extract_label                      # noqa: E402
from physics_interp.video import (                                   # noqa: E402
    load_frames, load_segmentation, masks_to_latent_grid, tubelet_frame_to_cell,
)

TARGETS = ["velocity_A_post", "velocity_B_post", "velocity_A_pre", "radius_A"]


def score(X, Y, tr, te):
    X = np.asarray(X, dtype=np.float64)
    r = fit_readout(X, Y, tr)
    P = (X[te] - r["mu"]) / r["sd"] @ r["W"] + r["b"]
    return float(1 - ((Y[te] - P) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum())


def pool(H, w):
    """Weighted mean of [tokens, dim] by [tokens]; None if the region is empty."""
    s = w.sum()
    if s <= 0:
        return None
    return (H * (w / s)[:, None]).sum(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="datasets/probe_set")
    p.add_argument("--repo", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--n-pre-frames", type=int, default=20)
    p.add_argument("--all-scenes", dest="conserved_only", action="store_false", default=True)
    p.add_argument("-o", "--out", default="outputs/predict_vjepa2-vitl")
    args = p.parse_args()

    from transformers import AutoModel
    from physics_interp.models.encoders import VJEPA2Adapter

    dev = "cuda:0"
    scenes = sorted(Path(x).parent for x in
                    glob.glob(f"{args.dataset_dir}/scene_*/labels.json"))
    print(f"{len(scenes)} scenes in {args.dataset_dir}", flush=True)

    m = AutoModel.from_pretrained(args.repo, dtype=torch.float32).eval().to(dev)
    ad = VJEPA2Adapter(pipe=m, repo=args.repo, device=torch.device(dev),
                       dtype=torch.float32)
    layout = ad.latent_layout(49, 480, 720)
    n_pre_cell = args.n_pre_frames // ad.tubelet          # 10 tubelets
    per_cell = layout.height * layout.width               # 256
    n_pre_tok = n_pre_cell * per_cell                     # 2560
    n_tok = layout.frames * per_cell                      # 6144
    print(f"layout {layout.frames}x{layout.height}x{layout.width}; "
          f"context 0-{n_pre_tok}, target {n_pre_tok}-{n_tok}", flush=True)
    ctx = [torch.arange(n_pre_tok, device=dev)[None]]
    tgt = [torch.arange(n_pre_tok, n_tok, device=dev)[None]]

    rows, labels, t0 = {k: [] for k in
                        ("enc_A", "enc_mean", "pred_A", "pred_B", "pred_mean",
                         "mask_pre", "mask_post")}, [], time.perf_counter()
    for i, d in enumerate(scenes, 1):
        lab = json.loads((d / "labels.json").read_text())
        fr = load_frames(d, num_frames=49)
        idx = ad.frame_indices(len(fr))
        seg = ad.preprocess_masks(load_segmentation(d)[idx])
        occ = masks_to_latent_grid(
            seg, layout, ids=(1, 2),
            frame_to_cell=lambda t: tubelet_frame_to_cell(t, ad.tubelet))
        # [2, cells, H, W] -> flat token masks over the two token ranges
        A_pre = occ[0, :n_pre_cell].reshape(-1)
        A_post = occ[0, n_pre_cell:].reshape(-1)
        B_post = occ[1, n_pre_cell:].reshape(-1)
        if A_pre.sum() <= 0 or A_post.sum() <= 0 or B_post.sum() <= 0:
            print(f"  {d.name}: SKIP (empty region)")
            continue

        with torch.no_grad():
            x = ad.preprocess(fr[list(range(1, args.n_pre_frames + 1))]).to(dev)
            hp = m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
            pr = m.predictor(encoder_hidden_states=hp, context_mask=ctx,
                             target_mask=tgt).last_hidden_state
        He = hp[0].float().cpu().numpy()
        Hp = pr[0].float().cpu().numpy()
        if He.shape[0] != A_pre.shape[0] or Hp.shape[0] != A_post.shape[0]:
            raise SystemExit(f"token mismatch: encoder {He.shape}, predictor "
                             f"{Hp.shape}, masks {A_pre.shape}/{A_post.shape}")

        rows["enc_A"].append(pool(He, A_pre))
        rows["enc_mean"].append(He.mean(0))
        rows["pred_A"].append(pool(Hp, A_post))
        rows["pred_B"].append(pool(Hp, B_post))
        rows["pred_mean"].append(Hp.mean(0))
        # The mask itself, as a feature vector -- no activations at all.
        rows["mask_pre"].append(A_pre.astype(np.float32))
        rows["mask_post"].append(A_post.astype(np.float32))
        labels.append(lab)
        if i % 25 == 0 or i == len(scenes):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(scenes)}] {el:6.1f}s, ~{el/(i/len(scenes))-el:5.1f}s left",
                  flush=True)

    R = {k: np.stack(v) for k, v in rows.items()}
    keep = np.arange(len(labels))
    if args.conserved_only:
        keep = np.array([i for i, l in enumerate(labels) if l["momentum_conserved"]])
    L = [labels[i] for i in keep]
    R = {k: v[keep] for k, v in R.items()}
    tr, te = split_indices(len(L))
    print(f"\n{len(L)} scenes ({len(tr)} train / {len(te)} test)"
          + (" — conserved only" if args.conserved_only else ""))

    print(f"\n{'target':<17} {'enc A.pre':>10} {'pred A.post':>12} {'delta':>8} "
          f"{'enc mean':>9} {'pred mean':>10} {'pred B.post':>12} {'MASK ONLY':>10}")
    out = {}
    for t in TARGETS:
        Y = np.stack([extract_label(l, t) for l in L])
        ea = score(R["enc_A"], Y, tr, te)
        pa = score(R["pred_A"], Y, tr, te)
        em = score(R["enc_mean"], Y, tr, te)
        pm = score(R["pred_mean"], Y, tr, te)
        pb = score(R["pred_B"], Y, tr, te)
        mk = score(R["mask_post"], Y, tr, te)
        out[t] = dict(enc_A=ea, pred_A=pa, delta=pa - ea, enc_mean=em,
                      pred_mean=pm, pred_B=pb, mask_only=mk)
        print(f"{t:<17} {ea:>10.3f} {pa:>12.3f} {pa-ea:>+8.3f} {em:>9.3f} "
              f"{pm:>10.3f} {pb:>12.3f} {mk:>10.3f}")

    # Proper variance decomposition: subtracting R2 values is not valid, so
    # residualise the target against the mask-only prediction and ask what the
    # activations explain of what is LEFT. That is the predictor's unique
    # contribution, with the pooling-region leak already removed.
    print(f"\n{'target':<17} {'mask only':>10} {'acts only':>10} {'mask+acts':>10} "
          f"{'acts | mask':>12}   <- unique contribution of the activations")
    for t in TARGETS:
        Y = np.stack([extract_label(l, t) for l in L])
        for nm, A in (("encoder A.pre", R["enc_A"]), ("predictor A.post", R["pred_A"])):
            mk = R["mask_pre"] if "encoder" in nm else R["mask_post"]
            r = fit_readout(np.asarray(mk, float), Y, tr)
            resid = Y - ((np.asarray(mk, float) - r["mu"]) / r["sd"] @ r["W"] + r["b"])
            print(f"  {t:<15} {nm:<17} {score(mk, Y, tr, te):>10.3f} "
                  f"{score(A, Y, tr, te):>10.3f} "
                  f"{score(np.c_[mk, A], Y, tr, te):>10.3f} "
                  f"{score(A, resid, tr, te):>12.3f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    f = Path(args.out) / "predict_object_local.json"
    f.write_text(json.dumps({"args": vars(args), "n": len(L), "results": out}, indent=2))
    print(f"\nwrote {f}")
    print("\nRead every object-local column against MASK ONLY, not against zero:"
          "\nthat column is what the choice of pooling region alone predicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
