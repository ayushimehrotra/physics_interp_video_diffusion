"""Build the generation plan for `energy_voe_v1` -- the cell `conservation_set_v1` is missing.

THE EMPTY BOX. Cross what is actually conserved in `conservation_set_v1`'s 300 clips:

                              KE created   KE lost   KE kept
    momentum CONSERVED             0          73        77
    momentum BROKEN               57          90         3

Zero clips conserve momentum while creating kinetic energy. Every one of the 57
energy-creating clips also breaks momentum, because that is how they were made --
the violation edit targets momentum and moves energy only as a side effect. Surprise
measured on those cannot be attributed to the invented energy rather than to the
missing momentum.

AND THE HONEST HALF DOES NOT FILL IT. Losing kinetic energy is LEGAL. An inelastic
collision converts it to heat and deformation and violates nothing, so the 73
momentum-conserving energy-losing clips are ordinary physics. They can tell you
whether a model NOTICES energy changed -- §44 answered that, at AUC 0.946/0.983 and
mass-free -- but not whether it holds conservation as a LAW, because no law is being
broken for it to react to.

WHAT THIS SET ADDS. Clips that are impossible: kinetic energy created from nothing,
`e > 1`, with the momentum books balanced to 1e-15. No physics engine will produce
that, so it is edited in by `apply_energy_edit` -- a COMMON rescale of both
deflections, `lambda = (1+e_target)/(1+e_sim)`, which holds the split `b/a = m_A/m_B`
that conservation fixes and moves only the sum that restitution fixes.

THE CONTROL, and it is the reason the set is shaped this way. Each seed is rendered
at `e = k` AND at `e = 1/k`. Both sit the same distance from elastic in log space;
one separates the balls k times too fast, the other k times too slow; both are
equally unusual as kinematics. **Exactly one is impossible.** A model more surprised
by `k` than by `1/k` holds an energy prior. A model equally surprised by both is
detecting novelty, not physics. The anchor `e = 1.0` runs through the same edit path
(verified a bit-exact no-op) so all three twins are structurally identical.

READ THIS BEFORE PROBING IT: **this set is for violation-of-expectation, not for a
probe.** `e` is mass-free and visible, so a probe separates the classes trivially by
reading separation speed -- that is §44's whole finding and it has not gone away. The
point here is different: the violation MUST be visible or the model could not react
to it at all. What is being asked is whether the model treats *impossible* differently
from merely *unusual*, which is what the `k` / `1/k` pairing isolates.

    python momentum_dataset/plan_energy_voe_set.py --seeds 60 --k 1.25 1.5 2.0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gs", HERE / "generate_scene.py")
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

# Elastic base. The edit rescales from whatever the simulator realised, so the base
# only needs to be a clean collision -- but pinning it at 1.0 keeps `lambda` close to
# 1 for every target and keeps the anchor clip a true no-op.
E_BASE = 1.0


def probe(seed: int, e_target: float, pair: str, scratch: Path):
    """Physics only, WITH the edit applied. Returns labels, or None if unusable.

    The feasibility check runs through the generator's own machinery rather than a
    reimplementation: `generate(render=False)` already writes `per_frame[*].in_frame`
    and `identifiability`, both computed after the edit, so a ball the edit flings out
    of shot or one whose post-contact travel is too small to see is caught here by the
    same code that would catch it in the real render.
    """
    try:
        L = gs.generate(seed, scratch / f"{seed}_{e_target}", f"s{seed}", render=False,
                        violate=False, arm="legacy", restitution=E_BASE,
                        energy_restitution=e_target, material_pair=pair)
    except Exception as exc:                      # noqa: BLE001 - reason is recorded
        return None, f"generate_failed: {type(exc).__name__}: {exc}"

    for row in L["per_frame"]:
        for tag in ("A", "B"):
            if not row[tag]["in_frame"]:
                return None, f"{tag}_leaves_frame_at_{row['frame']}"

    ident = L["identifiability"]
    if ident["min_delta_displacement_px"] < 8.0:
        return None, f"post_contact_travel_too_small_{ident['min_delta_displacement_px']:.1f}px"

    c = L["collision"]
    if c["momentum_rel_error"] > 1e-9:
        return None, f"momentum_not_conserved_{c['momentum_rel_error']:.2e}"
    if abs(c["restitution_measured"] - e_target) > 1e-6:
        return None, (f"restitution_missed_target_{c['restitution_measured']:.6f}"
                      f"_vs_{e_target:.6f}")
    return L, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=60,
                    help="seeds to ACCEPT; scanning continues past rejections")
    ap.add_argument("--k", type=float, nargs="+", default=[1.25, 1.5, 2.0],
                    help="restitution ladder. Each k contributes an IMPOSSIBLE clip at "
                         "e=k and its matched LEGAL clip at e=1/k")
    ap.add_argument("--seed-start", type=int, default=9000,
                    help="offset from conservation_set_v1's seeds so the two sets never "
                         "share a scene and can be pooled without a hidden duplicate")
    ap.add_argument("--material-pair", default="mixed", choices=("mixed", "same", "random"),
                    help="`mixed` matches conservation_set_v1's main arm, so appearance "
                         "statistics stay comparable across the two sets")
    ap.add_argument("--max-scan", type=int, default=400)
    ap.add_argument("-o", "--out", default="momentum_dataset/energy_voe_v1/plan.json")
    args = ap.parse_args()

    logging.basicConfig(level="WARNING")
    for k in args.k:
        if k <= 1.0:
            raise SystemExit(f"--k must be > 1 (its partner 1/k supplies the legal side); got {k}")

    targets = [(1.0, "anchor")]
    for k in args.k:
        targets.append((k, f"super{k:g}"))
        targets.append((1.0 / k, f"sub{k:g}"))

    clips, rejected, seed = [], [], args.seed_start - 1
    accepted = 0
    with tempfile.TemporaryDirectory(prefix="energy_voe_plan_") as tmp:
        scratch = Path(tmp)
        while accepted < args.seeds and seed - args.seed_start < args.max_scan:
            seed += 1
            got, why = {}, None
            for e_target, tag in targets:
                L, reason = probe(seed, e_target, args.material_pair, scratch)
                if L is None:
                    why = f"{tag}: {reason}"
                    break
                got[tag] = L
            if why is not None:
                rejected.append({"seed": seed, "reason": why})
                continue

            # every twin of a seed must share its pre-contact segment exactly, which is
            # what makes the k / 1/k comparison a matched one rather than a between-scene one
            ref = got["anchor"]["per_frame"]
            for tag, L in got.items():
                for row_a, row_b in zip(ref, L["per_frame"]):
                    if row_a["frame"] > gs.PRE_FRAME:
                        break
                    for t in ("A", "B"):
                        if not np.allclose(row_a[t]["position"], row_b[t]["position"], atol=1e-12):
                            why = f"{tag}: pre_contact_diverges_at_frame_{row_a['frame']}"
                if why:
                    break
            if why is not None:
                rejected.append({"seed": seed, "reason": why})
                continue

            for e_target, tag in targets:
                L = got[tag]
                c = L["collision"]
                clips.append({
                    "name": f"{tag}_seed{seed}",
                    "seed": seed,
                    "energy_restitution": float(e_target),
                    "material_pair": args.material_pair,
                    "arm": "legacy",
                    "restitution": E_BASE,
                    "cell": ("impossible" if e_target > 1.0
                             else "anchor" if e_target == 1.0 else "legal"),
                    "k": (float(e_target) if e_target > 1.0
                          else 1.0 if e_target == 1.0 else float(1.0 / e_target)),
                    "log_e": float(np.log(e_target)),
                    "ke_ratio": c["kinetic_energy_ratio"],
                    "momentum_rel_error": c["momentum_rel_error"],
                    "restitution_measured": c["restitution_measured"],
                })
            accepted += 1
            if accepted % 10 == 0:
                print(f"  {accepted}/{args.seeds} seeds accepted (scanned to {seed})", flush=True)

    if accepted < args.seeds:
        print(f"WARNING: only {accepted}/{args.seeds} seeds accepted within --max-scan")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_cell = {}
    for c in clips:
        by_cell.setdefault(c["cell"], []).append(c)
    summary = {cell: {
        "n": len(v),
        "ke_ratio_range": [min(x["ke_ratio"] for x in v), max(x["ke_ratio"] for x in v)],
        "max_momentum_rel_error": max(x["momentum_rel_error"] for x in v),
    } for cell, v in by_cell.items()}

    json.dump({"clips": clips, "summary": summary, "n_clips": len(clips),
               "n_seeds": accepted, "targets": [t[0] for t in targets],
               "seeds_rejected": rejected, "config": vars(args)},
              open(out, "w"), indent=1)

    print(f"\nwrote {out}  ({len(clips)} clips, {accepted} seeds, "
          f"{len(rejected)} seeds rejected)")
    for cell, v in sorted(summary.items()):
        print(f"  {cell:11s} n={v['n']:4d}  KE ratio "
              f"[{v['ke_ratio_range'][0]:.3f}, {v['ke_ratio_range'][1]:.3f}]  "
              f"max momentum rel err {v['max_momentum_rel_error']:.2e}")
    print("\n  every clip conserves momentum; only the `impossible` cell creates energy.")


if __name__ == "__main__":
    main()
