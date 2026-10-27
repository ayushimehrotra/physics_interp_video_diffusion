#!/usr/bin/env python
"""Can the model predict the collision aftermath from pre-contact frames alone?

This is the honest version of the prediction question, and three design points
decide whether it means anything.

**Encode pre-contact frames ONLY.** HF's `VJEPA2Model` runs the encoder on the
full clip and only then splits context/target masks, so its "context" tokens
have already attended to the future. Measured previously: encoding 20 frames
gives tokens only cos = 0.74 similar to the first 2560 tokens of a 48-frame
encode. This script encodes frames 1-20 and calls `m.predictor(...)` directly.

**Use v1, not v3.** Conservation-based prediction requires the mass ratio, and
v3's `latent_mass` arm makes mass invisible by construction — no observer could
do it. On v1 masses are visible through material and size.

**Do not score total momentum.** On conserved scenes `p_post = p_pre` to machine
precision (measured: median relative difference 1.3e-16), so "predicting" total
post-collision momentum is copying the pre-collision total. It is an identity,
and a model scoring well on it has learned to copy, not to apply conservation.
The informative targets are the *individual* post-collision velocities — the
split — which require mass ratio, restitution and contact geometry together.

**Fit on conserved scenes only.** In a violating scene the aftermath was edited
after the fact, so it is not determined by the pre-contact state and behaves as
noise for any observer.

Three readouts are compared, all with the same split protocol:
  encoder  linear probe on the pre-only encoding      <- the control that matters
  predictor  linear probe on the predictor's output   <- does the predictor add?
  oracle   linear/quadratic readout of the TRUE pre-contact labels
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
from physics_interp.video import load_frames                         # noqa: E402

TARGETS = ["velocity_A_post", "velocity_B_post", "velocity_A_pre",
           "momentum_post", "momentum_pre", "radius_A"]


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def score(X, Y, tr, te):
    X = np.asarray(X, dtype=np.float64)
    r = fit_readout(X, Y, tr)
    return r2(Y[te], (X[te] - r["mu"]) / r["sd"] @ r["W"] + r["b"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="datasets/probe_set")
    p.add_argument("--repo", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--n-pre-frames", type=int, default=20)
    p.add_argument("--n-frames", type=int, default=48)
    p.add_argument("--conserved-only", action="store_true", default=True)
    p.add_argument("--all-scenes", dest="conserved_only", action="store_false")
    p.add_argument("-o", "--out", default="outputs/predict_vjepa2-vitl")
    args = p.parse_args()

    from transformers import AutoModel
    from physics_interp.models.encoders import VJEPA2Adapter

    dev = "cuda:0"
    scenes = sorted(Path(x).parent for x in
                    glob.glob(f"{args.dataset_dir}/scene_*/labels.json"))
    print(f"{len(scenes)} scenes in {args.dataset_dir}")

    print("loading V-JEPA2 ...", flush=True)
    m = AutoModel.from_pretrained(args.repo, dtype=torch.float32).eval().to(dev)
    ad = VJEPA2Adapter(pipe=m, repo=args.repo, device=torch.device(dev),
                       dtype=torch.float32)

    # Token counts follow from the tubelet grid: 2 frames per tubelet, 16x16
    # patches. 20 pre-contact frames -> 10 tubelets -> 2560 tokens.
    per_tubelet = 256
    n_pre_tok = (args.n_pre_frames // 2) * per_tubelet
    n_tok = (args.n_frames // 2) * per_tubelet
    ctx = [torch.arange(n_pre_tok, device=dev)[None]]
    tgt = [torch.arange(n_pre_tok, n_tok, device=dev)[None]]
    print(f"context tokens 0-{n_pre_tok}, target tokens {n_pre_tok}-{n_tok}")

    H_enc, H_pred, labels, t0 = [], [], [], time.perf_counter()
    for i, d in enumerate(scenes, 1):
        lab = json.loads((d / "labels.json").read_text())
        contact = float(lab["collision"]["actual_frame"])
        if contact < args.n_pre_frames:
            print(f"  {d.name}: SKIP (contact at frame {contact})")
            continue
        fr = load_frames(d, num_frames=49)
        with torch.no_grad():
            x = ad.preprocess(fr[list(range(1, args.n_pre_frames + 1))]).to(dev)
            hp = m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
            pr = m.predictor(encoder_hidden_states=hp, context_mask=ctx,
                             target_mask=tgt).last_hidden_state
        H_enc.append(hp.mean(1)[0].float().cpu().numpy())
        H_pred.append(pr.mean(1)[0].float().cpu().numpy())
        labels.append(lab)
        if i % 25 == 0 or i == len(scenes):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(scenes)}] {el:6.1f}s, ~{el/(i/len(scenes))-el:5.1f}s left",
                  flush=True)
    H_enc, H_pred = np.stack(H_enc), np.stack(H_pred)
    print(f"encoded {len(labels)} scenes; encoder {H_enc.shape}, predictor {H_pred.shape}")

    keep = np.arange(len(labels))
    if args.conserved_only:
        keep = np.array([i for i, l in enumerate(labels) if l["momentum_conserved"]])
        print(f"restricting to {len(keep)} conserved scenes "
              "(a violating aftermath is not determined by the pre-contact state)")
    L = [labels[i] for i in keep]
    He, Hp = H_enc[keep], H_pred[keep]
    tr, te = split_indices(len(L))

    g = lambda t: np.stack([extract_label(l, t) for l in L])          # noqa: E731
    vA, vB = g("velocity_A_pre"), g("velocity_B_pre")
    mA = np.array([[l["objects"]["A"]["mass"]] for l in L])
    mB = np.array([[l["objects"]["B"]["mass"]] for l in L])
    rA = np.array([[l["objects"]["A"]["radius"]] for l in L])
    rB = np.array([[l["objects"]["B"]["radius"]] for l in L])
    ip = np.array([[l["derived"]["impact_parameter"]] for l in L])
    base = np.c_[vA, vB, mA, mB, rA, rB, ip]

    def poly(X):
        cols = [X]
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append((X[:, i] * X[:, j])[:, None])
        return np.c_[tuple(cols)]

    print(f"\n{'target':<18} {'encoder':>9} {'predictor':>10} {'delta':>8} "
          f"{'oracle-lin':>11} {'oracle-quad':>12}")
    rows = {}
    for t in TARGETS:
        Y = g(t)
        e, q = score(He, Y, tr, te), score(Hp, Y, tr, te)
        ol, oq = score(base, Y, tr, te), score(poly(base), Y, tr, te)
        rows[t] = dict(encoder=e, predictor=q, delta=q - e, oracle_lin=ol, oracle_quad=oq)
        print(f"{t:<18} {e:>9.3f} {q:>10.3f} {q-e:>+8.3f} {ol:>11.3f} {oq:>12.3f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / ("predict_conserved.json" if args.conserved_only
                            else "predict_all.json")
    out.write_text(json.dumps({"args": vars(args), "n_scenes": len(L),
                               "n_train": len(tr), "n_test": len(te),
                               "results": rows}, indent=2))
    print(f"\nwrote {out}   (n={len(L)}, {len(tr)} train / {len(te)} test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
