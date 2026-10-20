#!/usr/bin/env python
"""§40b-A1 / R1 -- when does the model commit to the collision outcome?

Re-encode each clip TRUNCATED at frame k and probe the POST-collision velocity
from what the model has seen so far. Contact is at frame 20.03 in every scene.

  decodable BEFORE k=20  ->  the model ANTICIPATES the outcome
  only after             ->  it READS the aftermath, and §4's "the probes were
                             reading, not predicting" is settled properly

WHY THIS IS THE ONLY PREDICTION TEST LEFT
-----------------------------------------
§7 closed generation (CogVideoX cannot inpaint a two-ball continuation), §15 and
§21 closed the V-JEPA2 predictor (a pass-through: everything decodable from its
output was already in the pre-contact encoding). Truncation needs neither. It
also removes §4's confound BY CONSTRUCTION: §4 could not tell prediction from
hindsight because attention is bidirectional, so a "pre-contact" token has
already attended to the aftermath. **A frame that does not exist cannot be
attended to.**

It is also the experiment §41 demands. §41 found that patching A's incoming
velocity does not move B's outgoing velocity (error ratio 1.002 / 0.990). But
the aftermath is VISIBLE there, so the model never needs to infer it. This
removes the visible answer.

TWO DESIGN CHOICES, both forced
-------------------------------
**Resample, do not freeze.** `frame_indices` hard-codes a 49-frame clip, so a
truncated clip needs its own subsample. Sampling N frames evenly from [0, k]
keeps the input in-distribution (a normal-looking, temporally denser clip);
freezing the tail instead would show the model a video that stops dead, which is
far more out-of-distribution. The cost is that apparent inter-frame displacement
shrinks with k -- **absorbed by refitting the probe at every k**, since a
systematic rescaling is a linear change and only information LOSS survives it.
`velocity_A_pre` is the internal check: it lives entirely in frames 0-20, so if
it holds up across k the resampling is not destroying velocity information.

**Read whole-clip pools, not pre/post.** Below k=20 there are no post-contact
cells at all, so `B.post` does not exist and the pre/post split is undefined.
Pooling over the whole truncated clip (`A.all`, `B.all`, `all.all`) is defined at
every k, is the natural question ("given everything seen so far, what is B's
outgoing velocity?"), and sidesteps `tubelet_phase_split`'s straddle error.

CONTROLS, all required
----------------------
  radius_A       static: it cannot change with k, so any decline is the encoding
                 degrading rather than physics disappearing. The scale bar.
  velocity_A_pre fully contained in frames 0-20: must be flat for k >= 20 and
                 available well before. Catches the resampling confound.
  mask floor     recomputed at every k (§20) -- the pooling mask shortens too.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import extract_label, ridge_fit          # noqa: E402
from physics_interp.runner import (                                 # noqa: E402
    ActivationRecorder, VideoDiffusionRunner, build_pool_weights,
)
from physics_interp.video import (                                  # noqa: E402
    load_frames, load_segmentation, masks_to_latent_grid,
)

LADDER = [10, 14, 17, 19, 20, 22, 25, 30, 38, 48]
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]


def even_indices(k: int, n: int) -> list[int]:
    """n frame indices spread evenly over [0, k]. Duplicates allowed when k < n:
    the model needs a fixed count, and a repeated frame is honest (it says the
    clip is short) where an interpolated one would invent motion."""
    return [int(round(x)) for x in np.linspace(0, k, n)]


def capture(args) -> dict:
    cfg = load_config(args.config); cfg.model.id = args.model
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    adapter = runner.adapter
    n_in = adapter.num_input_frames
    layout = runner.layout
    sites = ["embed.out"] + [f"block.{b}.resid_post"
                             for b in range(adapter.num_blocks)]

    scenes = sorted(Path(args.dataset_dir).glob("scene_*"))
    scenes = [s for s in scenes if (s / "labels.json").exists()]
    # ALL scenes, conserving and violating. The violation edits POST-contact
    # velocities only, so frames 0-20 are honest physics in every scene; the
    # corrupted label is repaired by reconstruction below (§43a). This doubles n,
    # and n is what the direct probe is starved of.
    keep = [(s, json.loads((s / "labels.json").read_text())) for s in scenes]
    keep = keep[: args.n_scenes]
    print(f"{args.model}: {len(keep)} conserving scenes, {n_in} input frames, "
          f"{len(sites)} sites, ladder {LADDER}\n")

    out = {k: {"acts": [], "masks": []} for k in LADDER}
    labels = []
    t0 = time.perf_counter()
    for si, (sdir, lab) in enumerate(keep, 1):
        frames_all = load_frames(sdir, num_frames=cfg.generation.num_frames)
        seg_all = load_segmentation(sdir)
        labels.append(lab)
        for k in LADDER:
            idx = even_indices(k, n_in)
            seg = adapter.preprocess_masks(seg_all[idx])
            occ = masks_to_latent_grid(
                seg, layout, ids=(1, 2),
                frame_to_cell=lambda t: min(t // adapter.tubelet, layout.frames - 1))
            regions = {"A": occ[0], "B": occ[1],
                       "all": np.ones((layout.frames, layout.height, layout.width),
                                      np.float32)}
            w, names = build_pool_weights(layout, regions, {"all": (0, layout.frames)})
            rec = ActivationRecorder(w, names)
            runner.hooks.reset(); runner.hooks.guidance_scale = 1.0
            runner.hooks.step_override = 0
            with runner.hooks.hook(sites, rec, steps=[0], name="rec"):
                with torch.no_grad():
                    adapter.encode(frames_all[idx])
            runner.hooks.step_override = None
            got = rec.collect([0])
            out[k]["acts"].append(np.stack([got[s][0] for s in sites]))   # [sites,pools,dim]
            # occupancy (un-normalised) mask rows, for the §20 floor at this k
            out[k]["masks"].append(np.stack([occ[0].reshape(-1), occ[1].reshape(-1)]))
        if si % 10 == 0 or si == len(keep):
            el = time.perf_counter() - t0
            print(f"  [{si}/{len(keep)}] {el/si:.1f}s/scene  eta {(len(keep)-si)*el/si/60:.1f} min")

    return {"sites": sites, "pools": names, "labels": labels,
            "acts": {k: np.stack(v["acts"]) for k, v in out.items()},
            "masks": {k: np.stack(v["masks"]) for k, v in out.items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set")
    p.add_argument("--n-scenes", type=int, default=150)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    D = capture(args)
    sites, pools, labels = D["sites"], D["pools"], D["labels"]
    pi = {q: i for i, q in enumerate(pools)}
    np.savez_compressed(f"outputs_trunc_{args.model}_acts.npz",
                        **{f"k{k}": v.astype(np.float16) for k, v in D["acts"].items()},
                        sites=np.array(sites), pools=np.array(pools),
                        labels=np.array([json.dumps(l) for l in labels]))

    def reconstructed(r, which):
        """The aftermath the SIMULATION would produce, from the pre-contact state.
        Valid in violating scenes, where the recorded `post` block was overwritten
        after the fact. Exact to 1.6e-16 on the conserving half (§43a)."""
        c = r["collision"]
        vA = np.asarray(c["pre"]["velocity_A"][:2], float)
        vB = np.asarray(c["pre"]["velocity_B"][:2], float)
        n = np.asarray(c["contact_normal"][:2], float); n /= np.linalg.norm(n)
        mA = r["objects"]["A"]["mass"]; mB = r["objects"]["B"]["mass"]
        mu = mA * mB / (mA + mB); J = (1.0 + r["restitution"]) * mu * ((vA - vB) @ n)
        return vA - (J / mA) * n if which == "A" else vB + (J / mB) * n

    def folds(n, k=5, seed=0):
        o = np.random.default_rng(seed).permutation(n)
        return [(np.setdiff1d(o, o[i::k]), o[i::k]) for i in range(k)]

    def cv_r2(X, Y):
        """5-fold CV with alpha chosen on an inner split of each training fold, so
        the test fold is never used for selection. Every scene is tested once."""
        n = len(Y); P = np.zeros_like(Y)
        for tr, te in folds(n):
            cut = int(0.8 * len(tr)); itr, iva = tr[:cut], tr[cut:]
            best = None
            for a in ALPHAS:
                mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6
                W, b = ridge_fit((X[itr] - mu) / sd, Y[itr], a)
                p = (X[iva] - mu) / sd @ W + b
                r = 1 - ((Y[iva] - p) ** 2).sum() / max(((Y[iva] - Y[iva].mean(0)) ** 2).sum(), 1e-12)
                if best is None or r > best[0]: best = (r, a)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            W, b = ridge_fit((X[tr] - mu) / sd, Y[tr], best[1]); P[te] = (X[te] - mu) / sd @ W + b
        return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

    TARGETS = [("velocity_B_post", lambda l: reconstructed(l, "B")),
               ("velocity_A_post", lambda l: reconstructed(l, "A")),
               ("velocity_A_pre",  lambda l: extract_label(l, "velocity_A_pre")),
               ("velocity_B_pre",  lambda l: extract_label(l, "velocity_B_pre")),
               ("radius_A",        lambda l: extract_label(l, "radius_A"))]
    SPECS = ["A.all", "B.all", "all.all", "A.all+B.all"]
    print(f"\n{'='*96}\nR2 vs TRUNCATION -- n={len(labels)}, 5-fold CV, best over {len(sites)} sites")
    print(f"contact at frame 20.03, so k<=20 is STRICTLY pre-contact\n{'='*96}")
    res = {}
    for tgt, fn in TARGETS:
        Y = np.stack([np.atleast_1d(fn(l)) for l in labels])
        print(f"\n{tgt}")
        print(f"  {'pool':<14}" + "".join(f"{('k=' + str(k)):>8}" for k in LADDER))
        for spec in SPECS:
            row = []
            for k in LADDER:
                A = D["acts"][k].astype(np.float64)
                X = np.concatenate([A[:, :, pi[p], :] for p in spec.split("+")], axis=2)
                row.append(max(cv_r2(X[:, s], Y) for s in range(len(sites))))
            res[f"{tgt}|{spec}"] = row
            star = "  <--" if spec == max(SPECS, key=lambda sp: 0) and False else ""
            print(f"  {spec:<14}" + "".join(f"{v:8.3f}" for v in row) + star)
        best = max(SPECS, key=lambda sp: res[f"{tgt}|{sp}"][LADDER.index(20)])
        print(f"  best pool at k=20: {best}  ({res[f'{tgt}|{best}'][LADDER.index(20)]:.3f})")
    for tgt in ("velocity_B_post",):
        Y = np.stack([reconstructed(l, "B") for l in labels])
        row = [cv_r2(D["masks"][k].reshape(len(labels), -1).astype(np.float64), Y) for k in LADDER]
        res[f"MASKFLOOR {tgt}"] = row
        print(f"\n  {'MASK-ONLY FLOOR':<14}" + "".join(f"{v:8.3f}" for v in row))

    dest = Path(args.out or f"outputs_trunc_{args.model}.json")
    dest.write_text(json.dumps({"model": args.model, "ladder": LADDER,
                                "sites": sites, "results": res}, indent=1))
    print(f"\ncontact is at frame 20.03: columns k=10..19 are STRICTLY pre-contact.")
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
