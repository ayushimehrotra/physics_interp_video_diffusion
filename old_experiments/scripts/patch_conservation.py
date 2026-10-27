#!/usr/bin/env python
"""Does the model ENFORCE conservation? — intervene before the collision, read after.

Probing cannot answer this and never will (§12, §26): a probe is a readout, and a
readout can always do the arithmetic itself, so a probe only ever says whether the
information is present. This is the causal version of the question.

THE EXPERIMENT
--------------
Patch object A's PRE-contact tokens toward a donor scene's, which changes what the
model believes about the momentum going in. Then read the model's POST-contact
momentum, out of tokens that were never touched.

    if the model enforces conservation   the change must carry across the collision
                                         -- what goes in has to come out
    if it just decodes each phase        the post-contact readout does not move

The headline is the **carry ratio**: the component of the induced post-contact
momentum change along the induced pre-contact one, over the pre-contact change.

    carry ~ 1    the model propagated the whole change. Conservation is enforced.
    carry ~ 0    pre and post are read independently. No law in the loop.

WHY THE PASSIVE CHANNEL IS ZERO HERE, NOT ARGUED AWAY
-----------------------------------------------------
The patched tokens are A-local AND pre-contact; the read pool is post-contact. They
are disjoint in *both* space and time, so the read pool's pooled vector cannot move
by arithmetic -- only by the network actually propagating something. The overlap is
computed per scene and printed (`w_overlap`); it should be exactly 0.000. Contrast
§18a, where edit and readout shared tokens and a purely passive effect scored
observed/passive = 1.02 and looked like a success.

CONTROLS
--------
  variants     full / velocity / radius / random, as in `patch_velocity.py`. The
               radius arm is the specificity control (a different real factor,
               same construction) and random is the "any k-dim slice of a real
               activation difference does this" null.
  landed       transfer fraction on `velocity_A_pre`. If the patch did not land,
               a carry ratio of 0 means nothing. Check this column first.
  KE PLACEBO   the same carry ratio computed on kinetic energy, which is NOT
               conserved in these collisions. If KE carries as strongly as
               momentum, what is being measured is generic propagation through
               the network, not a conservation constraint. This is the control
               the result stands on -- the same one that decided `lawful_errors`.

DATASET
-------
Default is v1 (`probe_set`), where masses are visible so momentum is decodable at
both phases (`momentum_pre` reads 0.825 frame-wide). The question here is not
whether the model can see momentum -- it is whether the two phases are linked.

USAGE
-----
    .venv/bin/python old_experiments/scripts/patch_conservation.py --model videomae-base \
        --capture-dir outputs/videomae-base --dataset-dir datasets/probe_set \
        --block 6 --n-scenes 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402

from steer_velocity import (                                        # noqa: E402
    SceneRunner, apply_readout, fit_readout, load_captures, r2_of, split_indices,
)
from patch_velocity import inlp_subspace                            # noqa: E402
from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import ActivationSteer, extract_label    # noqa: E402
from physics_interp.runner import VideoDiffusionRunner              # noqa: E402

#: (target, pool) pairs the experiment reads. Momentum at both phases is the
#: measurement; KE at both phases is the placebo; the rest are controls.
READS = (
    ("momentum_pre", "all.pre"),
    ("momentum_post", "all.post"),
    ("kinetic_energy_pre", "all.pre"),
    ("kinetic_energy_post", "all.post"),
    ("velocity_A_pre", "A.pre"),
    ("velocity_B_post", "B.post"),
    ("radius_A", "A.pre"),
)


def raw(r, x):
    """Readout applied and converted back to label units (apply_readout divides
    by the training label sd, which would make pre and post incomparable)."""
    return apply_readout(r, x)[0] * r["y_sd"]


def carry(d_post, d_pre):
    """Component of the post-contact change along the pre-contact change.

    A projection rather than a ratio of norms: the sign matters. A model that
    moves its post estimate the *wrong way* should score negative, not positive.
    """
    denom = float(d_pre @ d_pre) + 1e-12
    return float(d_post @ d_pre / denom)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs/videomae-base")
    p.add_argument("--dataset-dir", default="datasets/probe_set")
    p.add_argument("--block", type=int, default=6, help="where the patch is written")
    p.add_argument("--read-block", default=None,
                   help="default: the block where momentum reads best at BOTH phases")
    p.add_argument("--rounds", type=int, default=8, help="INLP rounds -> k = 2*rounds")
    p.add_argument("--n-scenes", type=int, default=40)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.id = args.model
    out_dir = Path(args.out) if args.out else Path(f"outputs/steer_{args.model}")

    acts, labels, pools, scene_ids = load_captures(Path(args.capture_dir), cfg.name)
    n = len(labels)
    pi = {q: i for i, q in enumerate(pools)}
    blocks = sorted(int(s.split(".")[1]) for s in acts if s.startswith("block."))
    site = f"block.{args.block}.resid_post"
    tr, te = split_indices(n)

    Y = {t: np.stack([extract_label(l, t) for l in labels]) for t, _ in READS}

    # Read where momentum decodes best at BOTH phases. Picking on one phase would
    # bias the carry ratio: a post readout that barely works cannot move.
    if args.read_block:
        read_site = args.read_block
    else:
        best, read_site = -np.inf, None
        for b in blocks:
            s = f"block.{b}.resid_post"
            q = min(r2_of(fit_readout(acts[s][:, pi[pl], :].astype(np.float64),
                                      Y[t], tr),
                          acts[s][:, pi[pl], :].astype(np.float64), Y[t], te)
                    for t, pl in (("momentum_pre", "all.pre"),
                                  ("momentum_post", "all.post")))
            if q > best:
                best, read_site = q, s
        print(f"read block chosen on min(R2 pre, R2 post) = {best:.3f}")

    print(f"patch at {site}, read at {read_site}, {len(tr)} train / {len(te)} test")

    X = acts[site][:, pi["A.pre"], :].astype(np.float64)
    vA = Y["velocity_A_pre"]
    k = 2 * args.rounds
    V = inlp_subspace(X, vA, tr, args.rounds)
    U = inlp_subspace(X, Y["radius_A"], tr, k)
    rng = np.random.default_rng(0)
    R, _ = np.linalg.qr(rng.normal(size=(X.shape[1], k)))

    readouts, r2s = {}, {}
    for target, pool in READS:
        Xr = acts[read_site][:, pi[pool], :].astype(np.float64)
        r = fit_readout(Xr, Y[target], tr)
        readouts[(target, pool)] = r
        r2s[f"{target}|{pool}"] = r2_of(r, Xr, Y[target], te)
    print("readout R2:")
    for a, b in r2s.items():
        print(f"    {a:<28} {b:+.3f}")
    if min(r2s["momentum_pre|all.pre"], r2s["momentum_post|all.post"]) < 0.2:
        print("\nWARNING: momentum does not read at both phases here. The carry "
              "ratio will not be interpretable.")

    te_list = list(te)[: args.n_scenes]
    donors = {}
    for i in te_list:
        j = i
        while j == i:
            j = int(rng.choice(te))
        donors[i] = j

    projections = {"full": None, "velocity": V, "radius": U, "random": R}

    print(f"\nloading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    rec_sites = [f"block.{b}.resid_post" for b in blocks]

    rows, t0 = [], time.perf_counter()
    for si, i in enumerate(te_list, 1):
        sid, jid = scene_ids[i], scene_ids[donors[i]]
        try:
            sc = SceneRunner(runner, Path(args.dataset_dir) / sid,
                             cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})")
            continue
        if sc.n_steer_tokens == 0:
            continue

        # The passive channel, computed rather than assumed: how much of each read
        # pool's pooling weight sits on tokens the patch touches. Disjoint in space
        # AND time for the post pools, so this must be exactly 0.
        spi = {q: idx for idx, q in enumerate(sc.pool_names)}
        overlap = {pl: float(sc.weights[spi[pl]] @ sc.steer_mask)
                   for _, pl in READS}

        base = sc.run(rec_sites)
        bp = {kk: raw(r, base[read_site][pi[kk[1]]]) for kk, r in readouts.items()}
        diff = X[donors[i]] - X[i]
        dv_target = vA[donors[i]] - vA[i]

        for pname, P in projections.items():
            delta = diff if P is None else (P @ (P.T @ diff))
            nrm = float(np.linalg.norm(delta))
            if nrm < 1e-9:
                continue
            steer = ActivationSteer(delta / nrm, alpha=nrm, scale="absolute",
                                    tokens=sc.steer_mask)
            out = sc.run(rec_sites, steer_site=site, steer=steer)

            d = {kk: raw(r, out[read_site][pi[kk[1]]]) - bp[kk]
                 for kk, r in readouts.items()}

            kA = ("velocity_A_pre", "A.pre")
            tgt = dv_target
            row = {
                "scene": sid, "donor": jid, "variant": pname, "delta_norm": nrm,
                "w_overlap_post": overlap["all.post"],
                "w_overlap_Bpost": overlap["B.post"],
                # Did the patch land? Everything below is void if this is ~0.
                "landed": float(d[kA] @ tgt / (float(tgt @ tgt) + 1e-12)),
                "d_p_pre": float(np.linalg.norm(d[("momentum_pre", "all.pre")])),
                "d_p_post": float(np.linalg.norm(d[("momentum_post", "all.post")])),
                "carry_momentum": carry(d[("momentum_post", "all.post")],
                                        d[("momentum_pre", "all.pre")]),
                "carry_ke": carry(d[("kinetic_energy_post", "all.post")],
                                  d[("kinetic_energy_pre", "all.pre")]),
                "d_vB_post": float(np.linalg.norm(d[("velocity_B_post", "B.post")])),
                "d_radius_A": float(d[("radius_A", "A.pre")][0]),
            }
            rows.append(row)
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(te_list)}] {sid}<-{jid}  {el:6.1f}s, "
              f"~{el / (si / len(te_list)) - el:5.1f}s left", flush=True)
    runner.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"patch_conservation_block{args.block}_k{k}.json"
    path.write_text(json.dumps(
        {"args": vars(args), "read_site": read_site, "readout_r2": r2s,
         "k": k, "rows": rows}, indent=2))
    print(f"\nwrote {path}")
    summarise(rows)
    return 0


def boot_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarise(rows):
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["variant"], []).append(r)

    ov = max(abs(r["w_overlap_post"]) for r in rows) if rows else float("nan")
    print(f"\npassive channel: max pooling-weight overlap of the patch with "
          f"`all.post` = {ov:.6f}   (must be 0)")

    print(f"\n{'variant':<10} {'|delta|':>8} {'landed':>8} {'d_p_pre':>9} "
          f"{'d_p_post':>9} {'CARRY_p':>9} {'95% CI':>18} {'carry_KE':>9} "
          f"{'d_vB_post':>10}")
    for v, rs in sorted(groups.items()):
        cm = [r["carry_momentum"] for r in rs]
        lo, hi = boot_ci(cm)
        m = lambda kk: float(np.median([r[kk] for r in rs]))         # noqa: E731
        print(f"{v:<10} {m('delta_norm'):>8.2f} {m('landed'):>8.3f} "
              f"{m('d_p_pre'):>9.4f} {m('d_p_post'):>9.4f} "
              f"{float(np.mean(cm)):>9.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"{float(np.mean([r['carry_ke'] for r in rs])):>9.3f} "
              f"{m('d_vB_post'):>10.4f}")

    full = groups.get("full", [])
    if full:
        cm, ck = np.mean([r["carry_momentum"] for r in full]), \
                 np.mean([r["carry_ke"] for r in full])
        print(f"\nfull variant: carry(momentum) {cm:+.3f} vs carry(KE) {ck:+.3f}")
        print("  carry ~ 1 and above KE  -> the model propagates momentum across "
              "the collision")
        print("  carry ~ 0               -> the two phases are read independently")
        print("  carry ~ KE              -> generic propagation, not a conservation "
              "constraint")


if __name__ == "__main__":
    raise SystemExit(main())
