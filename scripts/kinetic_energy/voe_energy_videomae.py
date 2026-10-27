#!/usr/bin/env python
"""VoE on `energy_voe_pilot` using VideoMAE's OWN decoder reconstruction loss.

The second, independent surprise channel. `voe_energy_twins.py` asks V-JEPA2's predictor
which continuation it forecast; this asks VideoMAE's pretraining head how well it can
reconstruct each continuation from the frames before contact. Both are the model's own
training objective evaluated on clips it has never seen, and neither fits anything.

    surprise(clip) = the model's masked-reconstruction loss on that clip

THE MASK IS IDENTICAL ACROSS THE THREE TWINS OF A SEED. The loss is stochastic in the
mask; drawing independently per clip would put that variance straight into the per-seed
difference. Holding it fixed is the same discipline `voe_cogvideox.py` applies to its
noise tensor and timestep grid, and it is what makes a 20-seed paired test readable.

MASKING IS RESTRICTED TO POST-CONTACT TUBELETS. The three clips are pixel-identical
before contact, so a mask covering pre-contact patches contributes an identical term to
all three and dilutes the comparison. `VideoMAEAdapter` subsamples frames 3,6,...,48
precisely so the collision lands on a tubelet boundary: tubelet 2 = {15,18} is the last
pre-contact and tubelet 3 = {21,24} the first post-contact, so tubelets 3-7 are the
aftermath.

Two mask modes:
  all     every post-contact patch masked. Deterministic, one forward per clip -- the
          model must rebuild the entire aftermath from the pre-contact frames, which is
          the closest pixel-space analogue of the predictor experiment.
  random  `--draws` random subsets of the post-contact patches, averaged. Closer to the
          90% tube masking VideoMAE was trained under, at the cost of sampling.

    .venv/bin/python scripts/kinetic_energy/voe_energy_videomae.py --mask-mode all
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from math import comb
from pathlib import Path
import numpy as np
import torch
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from physics_interp.video import load_frames                       # noqa: E402
from physics_interp.models.encoders import VideoMAEAdapter          # noqa: E402

N_TUBELETS, PATCHES_PER_TUBELET = 8, 196      # 16 frames / 2 ; (224/16)^2
FIRST_POST_TUBELET = 3                        # tubelet 2 = {15,18} pre, 3 = {21,24} post


def sign_p(k: int, n: int) -> float:
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="datasets/energy_voe_pilot")
    ap.add_argument("--repo", default="MCG-NJU/videomae-base")
    ap.add_argument("--mask-mode", choices=("all", "random"), default="all")
    ap.add_argument("--draws", type=int, default=24, help="random mode only")
    ap.add_argument("--mask-ratio", type=float, default=0.75,
                    help="random mode: fraction of POST-contact patches masked")
    ap.add_argument("--n-seeds", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/voe_energy_videomae.json")
    args = ap.parse_args()

    from transformers import VideoMAEForPreTraining
    m = VideoMAEForPreTraining.from_pretrained(args.repo).eval()
    ad = VideoMAEAdapter(pipe=m.videomae, repo=args.repo,
                         device=torch.device("cpu"), dtype=torch.float32)

    tri, ks = defaultdict(dict), {}
    for p in sorted(Path(args.dataset_dir).glob("*_seed*/labels.json")):
        L = json.load(open(p)); seed = re.search(r"seed(\d+)$", p.parent.name).group(1)
        e = L["energy_edit"]["e_target"]
        role = "anchor" if abs(e - 1) < 1e-9 else ("impossible" if e > 1 else "legal")
        tri[seed][role] = p.parent
        if role == "impossible":
            ks[seed] = e
    seeds = [s for s in sorted(tri) if len(tri[s]) == 3]
    if args.n_seeds:
        seeds = seeds[: args.n_seeds]
    print(f"{len(seeds)} triples   mask-mode={args.mask_mode}"
          + (f"  draws={args.draws} ratio={args.mask_ratio}" if args.mask_mode == "random" else ""),
          flush=True)

    post = np.zeros((N_TUBELETS, PATCHES_PER_TUBELET), bool)
    post[FIRST_POST_TUBELET:] = True
    post_idx = np.flatnonzero(post.reshape(-1))
    print(f"  post-contact patches: {len(post_idx)}/{N_TUBELETS * PATCHES_PER_TUBELET}",
          flush=True)

    @torch.no_grad()
    def losses(sdir, masks):
        x = ad.preprocess(load_frames(sdir, num_frames=49)[ad.frame_indices(49)])
        out = []
        for mk in masks:
            out.append(float(m(pixel_values=x,
                               bool_masked_pos=torch.from_numpy(mk)[None]).loss))
        return out

    rows = []
    for i, seed in enumerate(seeds, 1):
        rng = np.random.default_rng(int(seed))
        if args.mask_mode == "all":
            masks = [post.reshape(-1).copy()]
        else:
            masks = []
            for _ in range(args.draws):
                mk = np.zeros(N_TUBELETS * PATCHES_PER_TUBELET, bool)
                mk[rng.choice(post_idx, int(round(args.mask_ratio * len(post_idx))),
                              replace=False)] = True
                masks.append(mk)
        r = {"seed": seed, "k": ks[seed]}
        for role in ("anchor", "legal", "impossible"):
            v = losses(tri[seed][role], masks)
            r[role] = float(np.mean(v))
        rows.append(r)
        print(f"  [{i}/{len(seeds)}] seed {seed}  legal {r['legal']:.4f}  "
              f"anchor {r['anchor']:.4f}  impossible {r['impossible']:.4f}", flush=True)

    D = np.array([[r["legal"], r["anchor"], r["impossible"]] for r in rows])
    n = len(D)
    k = int((D[:, 2] > D[:, 0]).sum())
    anchor_lowest = int(((D[:, 1] < D[:, 0]) & (D[:, 1] < D[:, 2])).sum())
    mono = int(((D[:, 0] < D[:, 1]) & (D[:, 1] < D[:, 2])).sum())
    C = D - D.mean(axis=1, keepdims=True)
    print(f"\n{'=' * 74}\nVideoMAE decoder reconstruction loss, {n} seeds")
    print(f"  centred means   legal {C[:,0].mean():+.5f}   anchor {C[:,1].mean():+.5f}"
          f"   impossible {C[:,2].mean():+.5f}")
    print(f"  impossible > legal in {k}/{n}   p = {sign_p(k, n):.4f}")
    print(f"  anchor least surprising in {anchor_lowest}/{n} "
          f"(a departure-magnitude account requires ALL)")
    print(f"  legal < anchor < impossible in {mono}/{n}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"mask_mode": args.mask_mode, "n_seeds": n, "rows": rows,
               "impossible_gt_legal": k, "sign_p": sign_p(k, n),
               "anchor_lowest": anchor_lowest, "monotone": mono,
               "centred_means": {r: float(C[:, j].mean())
                                 for j, r in enumerate(("legal", "anchor", "impossible"))}},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
