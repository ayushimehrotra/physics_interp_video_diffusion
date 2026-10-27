#!/usr/bin/env python
"""VoE on matched twins with CogVideoX-2b's DENOISING LOSS -- the generative version.

The V-JEPA2 version asked a JEPA predictor whether the violating future matched
its latent forecast, and the answer was no effect (and the per-tubelet trend ran
BACKWARDS, killing the residual lean). But a JEPA predictor's L2 is a weak
likelihood proxy. CogVideoX-2b is an actual generative model, so its denoising
loss is the quantity that means "how probable is this video" -- the closest thing
to a likelihood available here.

    z_t = alpha_t * z_0 + sigma_t * eps
    loss(t) = || eps_pred(z_t, t) - eps ||^2

Lower loss = the clip sits where the model expects video to be. If CogVideoX has
any expectation that momentum is conserved, the violating twin should cost more.

THREE DESIGN POINTS, each of which decides whether this measures anything
------------------------------------------------------------------------
**Identical noise for both twins.** The loss is stochastic in eps and in t. Draw
them independently per clip and the per-pair difference drowns in sampling noise.
Here the SAME eps tensor and the SAME timestep grid are used for both twins of a
pair, so everything except the video itself cancels. This is the whole reason a
150-pair paired test can see anything at all.

**Score the post-contact latent frames, not the clip.** CogVideoX's causal VAE
puts pixel frames 17-20 in latent frame 5 and 21-24 in latent 6, so latent frames
0-5 are purely pre-contact and 6-12 purely post-contact, with no straddling cell.
The twins are pixel-identical before contact, so including latents 0-5 adds terms
that are equal on both sides -- the same dilution that hid the effect in the
V-JEPA2 run until it was masked. Reported at three regions: whole clip,
post-contact latents, and ball-occupied cells within them.

**v_prediction.** CogVideoX-2b's scheduler is v_prediction, not epsilon, so the
raw denoiser output is converted through `split_prediction` before the residual
is taken. Comparing raw model outputs across parameterisations would be wrong.

Reported per timestep as well as pooled: at very high noise nothing is
recoverable and at very low noise the task is trivial, so any real effect should
live in the middle of the ladder rather than being flat across it.
"""
from __future__ import annotations
import argparse, glob, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import sys
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from physics_interp.config import load_config                                  # noqa: E402
from physics_interp.runner import VideoDiffusionRunner                         # noqa: E402
from physics_interp.video import load_frames, load_segmentation, pixel_frame_to_latent  # noqa: E402


def latent_ball_mask(sdir, n_lat, h, w, n_frames=49):
    """[n_lat, h, w] bool: does either sphere occupy this latent cell?"""
    seg = load_segmentation(sdir)[:n_frames]                  # [T, H, W] instance ids
    occ = (seg == 1) | (seg == 2)
    H, W = occ.shape[1], occ.shape[2]
    sy, sx = H // h, W // w
    out = np.zeros((n_lat, h, w), bool)
    for t in range(n_frames):
        f = pixel_frame_to_latent(t)
        if f >= n_lat:
            continue
        blk = occ[t, : sy * h, : sx * w].reshape(h, sy, w, sx).any((1, 3))
        out[f] |= blk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    ap.add_argument("--dataset-dir", default="datasets/conservation_set_v1")
    ap.add_argument("--n-pairs", type=int, default=0)
    ap.add_argument("--n-timesteps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="docs/artifacts/voe_cogvideox.json")
    args = ap.parse_args()

    pairs = defaultdict(dict)
    for p in sorted(glob.glob(f"{args.dataset_dir}/*_seed*/labels.json")):
        d = Path(p).parent
        seed = re.search(r"seed(\d+)$", d.name).group(1)
        cell = d.name.split("_seed")[0]
        pairs[(cell.replace("_violating", "").replace("_honest", ""), seed)][
            "viol" if "violating" in cell else "honest"] = d
    pairs = {k: v for k, v in sorted(pairs.items()) if len(v) == 2}
    if args.n_pairs:
        pairs = dict(list(pairs.items())[: args.n_pairs])
    print(f"{len(pairs)} matched pairs")

    cfg = load_config(args.config)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    ad = runner.adapter
    sched = ad.scheduler
    n_train = int(getattr(sched.config, "num_train_timesteps", 1000))
    ts = np.linspace(0.08, 0.92, args.n_timesteps) * (n_train - 1)
    ts = [int(round(t)) for t in ts]
    print(f"prediction_type = {getattr(sched.config,'prediction_type','epsilon')}, "
          f"timesteps {ts}")
    embeds = ad.embed_prompt(cfg.inversion.prompt)

    contact_lat = pixel_frame_to_latent(20)          # 5: last purely pre-contact latent
    rows, t0 = [], time.perf_counter()
    for i, ((cell, seed), tw) in enumerate(pairs.items(), 1):
        z = {}
        for tag in ("honest", "viol"):
            fr = load_frames(tw[tag], num_frames=cfg.generation.num_frames)
            z[tag] = ad.encode_video(torch.from_numpy(fr).float() / 255.0
                                     if fr.dtype == np.uint8 else torch.from_numpy(fr))
        F, C, h, w = z["honest"].shape[1:]
        mask = torch.from_numpy(
            latent_ball_mask(tw["honest"], F, h, w) |
            latent_ball_mask(tw["viol"], F, h, w)).to(z["honest"].device)
        post = torch.zeros(F, dtype=torch.bool, device=mask.device)
        post[contact_lat + 1:] = True
        regions = {
            "all": torch.ones(F, 1, h, w, dtype=torch.bool, device=mask.device),
            "post": post[:, None, None, None].expand(F, 1, h, w),
            "post_balls": (post[:, None, None] & mask)[:, None, :, :],
        }
        # SANITY: the twins must differ only AFTER contact. Latents 0-5 are purely
        # pre-contact (causal VAE puts pixel frames 17-20 in latent 5), so their
        # difference should be at VAE-noise level while 6-12 carry the violation.
        dz = (z["honest"] - z["viol"])[0].abs().float().mean((1, 2, 3)).cpu().numpy()

        gen = torch.Generator(device="cpu").manual_seed(args.seed + int(seed))
        rec = {r: {"honest": [], "viol": []} for r in regions}
        for t in ts:
            eps = torch.randn(z["honest"].shape, generator=gen).to(
                z["honest"].device, z["honest"].dtype)
            alpha, sigma = ad.alpha_sigma(t)
            tt = torch.tensor([t], device=z["honest"].device)
            for tag in ("honest", "viol"):
                zt = alpha * z[tag] + sigma * eps
                with torch.no_grad():
                    out = ad.denoise(zt, tt, embeds)
                _, eps_pred = ad.split_prediction(out, zt, alpha, sigma)
                err = (eps_pred.float() - eps.float()) ** 2          # [1,F,C,h,w]
                for rname, rm in regions.items():
                    m = rm.expand(F, C, h, w)
                    rec[rname][tag].append(float(err[0][m].mean()))
        rows.append(dict(cell=cell, seed=seed, dz_by_latent_frame=[float(x) for x in dz],
                         ball_frac=float(regions["post_balls"].expand(F, C, h, w).float().mean()
                                         / max(regions["post"].expand(F, C, h, w).float().mean(), 1e-9)),
                         **{r: {k: rec[r][k] for k in ("honest", "viol")} for r in regions}))
        if i % 5 == 0 or i == len(pairs):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(pairs)}] {el:6.1f}s  eta {el/i*(len(pairs)-i)/60:5.1f} min",
                  flush=True)

    from math import comb
    rep = {"timesteps": ts, "n_pairs": len(rows), "rows": rows, "summary": {}}
    bf = np.array([r["ball_frac"] for r in rows])
    print(f"\nball cells: {bf.mean():.1%} of the post-contact latent cells")
    DZ = np.array([r["dz_by_latent_frame"] for r in rows])
    rep["dz_by_latent_frame"] = DZ.mean(0).tolist()
    print("SANITY  mean |z_honest - z_viol| by latent frame (0-5 pre-contact, 6-12 post):")
    print("  latent  " + "".join(f"{j:>8}" for j in range(DZ.shape[1])))
    print("  |dz|    " + "".join(f"{v:>8.4f}" for v in DZ.mean(0)))
    for rname in ("all", "post", "post_balls"):
        H = np.array([r[rname]["honest"] for r in rows])       # [pairs, timesteps]
        V = np.array([r[rname]["viol"] for r in rows])
        hm, vm = H.mean(1), V.mean(1)
        k, n = int((vm > hm).sum()), len(hm)
        p = sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n
        dif = vm - hm
        mde = 2.8 * dif.std(ddof=1) / np.sqrt(n)
        print(f"\n{'='*88}\nREGION = {rname.upper()}")
        print(f"  pooled over timesteps: honest {hm.mean():.6f}  violating {vm.mean():.6f}  "
              f"diff {dif.mean():+.6f}")
        print(f"  violating costs more in {k}/{n} pairs   sign-test p = {p:.4f}   "
              f"(detectable {mde:.6f})")
        print(f"  {'timestep':<10}" + "".join(f"{t:>8}" for t in ts))
        print(f"  {'frac worse':<10}" + "".join(f"{(V[:,j]>H[:,j]).mean():>8.2f}"
                                                for j in range(len(ts))))
        rep["summary"][rname] = {"honest": float(hm.mean()), "viol": float(vm.mean()),
                                 "diff": float(dif.mean()), "n_viol_worse": k, "n": n,
                                 "p": float(p), "mde": float(mde),
                                 "frac_worse_by_t": [float((V[:, j] > H[:, j]).mean())
                                                     for j in range(len(ts))]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
