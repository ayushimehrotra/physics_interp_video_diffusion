#!/usr/bin/env python3
"""§54b — the three-point chord test, as a live script (it was a one-off before).

    python scripts/chord_test.py --captures outputs/energy_voe_450/*/activations

Three clips of one seed differ only in bounciness, so physics puts their
outcomes on a straight line with the middle one at a fixed position along the
chord joining the outer two: t = (e_mid - e_lo) / (e_hi - e_lo), read from
restitution_realised (never labels["restitution"], the label trap). The
encoder's three vectors need not be collinear — a nonlinear map bends a line —
but a bend keeps the order and the position along the chord. So, per seed and
site, the middle clip is projected onto the encoder chord and compared with the
physics value; the spread of that position across seeds and the perpendicular
residual (curvature) are reported by depth, together with the correlation of
encoder step lengths with physical step lengths.

Works on any set whose seeds have exactly three non-violating arms; with more
arms pass --arms lo mid hi (arm tags).
"""
import _momentum_path  # noqa: F401

import numpy as np

from physics_interp.momentum import cli, physics
from physics_interp.momentum.ladders import ladders
from physics_interp.momentum.stats import bootstrap_ci


def main():
    p = cli.base_parser(__doc__, site=False)
    p.set_defaults(pools=["all.post"])
    p.add_argument("--arms", nargs=3, default=None, metavar=("LO", "MID", "HI"),
                   help="arm tags to use when a seed has more than three arms")
    args = p.parse_args()
    sets = cli.load_sets(args)
    out = {"protocol": "§54b", "models": {}}
    for name, cs in sets.items():
        lads = ladders(cs, min_arms=3)
        triples = []
        for L in lads:
            if args.arms:
                tags = list(cs.arm[L.idx])
                if not all(a in tags for a in args.arms):
                    continue
                sel = np.array([L.idx[tags.index(a)] for a in args.arms])
            elif len(L.idx) == 3:
                sel = L.idx
            else:
                continue
            e = cs.restitution[sel]
            if not (e[0] < e[1] < e[2]):
                continue
            o = [physics.outcome_from_labels(cs.labels[i]) for i in sel]
            t_phys = (e[1] - e[0]) / (e[2] - e[0])
            triples.append((sel, e, t_phys, o))
        if not triples:
            print(f"[{name}] no three-arm seeds found")
            continue
        t_phys_all = np.array([t for _, _, t, _ in triples])
        d_phys_hi = np.array([np.linalg.norm(o[2] - o[0]) for _, _, _, o in triples])
        d_phys_mid = np.array([np.linalg.norm(o[1] - o[0]) for _, _, _, o in triples])
        rec = {"n_seeds": len(triples), "t_physics_median": float(np.median(t_phys_all)),
               "t_physics_all_equal": bool(np.ptp(t_phys_all) < 1e-9), "pools": {}}
        print(f"\n[{name}] {len(triples)} seeds, physics chord position {np.median(t_phys_all):.3f}"
              + ("" if rec["t_physics_all_equal"] else " (varies by seed; deviations below are t - t_phys)"))
        depth = cs.depth()
        for pool in args.pools:
            rows = []
            print(f"  pool {pool}")
            print(f"  {'site':24s} {'t median':>9s} {'t mean':>7s} {'sd':>6s} {'dev CI95':>18s} {'perp/chord':>10s} "
                  f"{'corr|d| hi':>10s} {'corr|d| mid':>11s}")
            for site in cs.sites:
                X, valid = cs.features(site, pool)
                ts, perps, d_hi, d_mid, keep = [], [], [], [], []
                for k, (sel, e, t_phys, o) in enumerate(triples):
                    if not valid[sel].all():
                        continue
                    lo, mid, hi = X[sel[0]], X[sel[1]], X[sel[2]]
                    chord = hi - lo
                    cc = float(chord @ chord)
                    if cc == 0:
                        continue
                    t = float((mid - lo) @ chord / cc)
                    ts.append(t)
                    perps.append(float(np.linalg.norm(mid - lo - t * chord) / np.sqrt(cc)))
                    d_hi.append(np.sqrt(cc))
                    d_mid.append(float(np.linalg.norm(mid - lo)))
                    keep.append(k)
                ts, perps = np.array(ts), np.array(perps)
                keep = np.array(keep)
                dev = ts - t_phys_all[keep]
                _, lo_ci, hi_ci = bootstrap_ci(dev, np.mean, seed=args.seed)
                row = {
                    "site": site, "depth": depth[site], "n": int(len(ts)),
                    "t_median": float(np.median(ts)), "t_mean": float(ts.mean()), "t_sd": float(ts.std()),
                    "t_iqr": float(np.subtract(*np.quantile(ts, [0.75, 0.25]))),
                    "deviation_mean": float(dev.mean()), "deviation_ci95": [lo_ci, hi_ci],
                    "perp_over_chord_median": float(np.median(perps)),
                    "corr_step_hi": float(np.corrcoef(d_hi, d_phys_hi[keep])[0, 1]),
                    "corr_step_mid": float(np.corrcoef(d_mid, d_phys_mid[keep])[0, 1]),
                    "order_correct_frac": float(np.mean((ts > 0) & (ts < 1))),
                }
                rows.append(row)
                print(f"  {site:24s} {row['t_median']:9.3f} {row['t_mean']:7.3f} {row['t_sd']:6.3f} "
                      f"[{lo_ci:+.3f}, {hi_ci:+.3f}] {row['perp_over_chord_median']:10.3f} "
                      f"{row['corr_step_hi']:10.2f} {row['corr_step_mid']:11.2f}")
            if len(rows) >= 2:
                rec["spread_tightening_embed_over_deep"] = rows[0]["t_sd"] / max(rows[-1]["t_sd"], 1e-12)
                print(f"  spread tightening embed -> deepest: {rec['spread_tightening_embed_over_deep']:.2f}x")
            rec["pools"][pool] = rows
        out["models"][name] = rec
    cli.write_artifact(out, args.out, "chord_test.json")


if __name__ == "__main__":
    main()
