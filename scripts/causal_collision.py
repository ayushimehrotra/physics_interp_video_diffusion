#!/usr/bin/env python
"""§40b-A2 -- does the model PROPAGATE the collision, or just track two balls?

Every probe in this project asks whether a quantity is PRESENT. This asks
whether the network performs the interaction: patch ball A's **pre-contact**
representation with a donor scene's, then read ball B's **POST-contact**
velocity. Physics says B's outgoing velocity depends on A's incoming velocity by
a computable amount. Does the readout move by that amount?

WHY THIS DESIGN AND NOT A PROBE
-------------------------------
Decoding cannot answer it. `velocity_B_post` decodes at 0.95 whether the model
computed the collision or merely watched the aftermath (§4: post-contact
quantities read as well from pre-contact pools, because attention is
bidirectional). An intervention on the *cause* that moves the *effect* by the
right amount is a different claim, and it is the one a physics engine would pass.

THE COUNTERFACTUAL, stated exactly
----------------------------------
For a frictionless two-sphere impact with restitution e, the impulse is along
the contact normal n and

    J = (1 + e) * mu_red * ((v_A - v_B) . n),        v_B_post = v_B_pre + (J/m_B) n

Verified on all 75 conserving scenes: reconstructing v_B_post this way from the
labels matches the simulator to a **median 1.95e-16**. Holding the contact
geometry (n, e, masses, v_B_pre) fixed and replacing A's incoming velocity with
a donor's gives a pure linear response,

    required dv_B_post = k * ((v_A_donor - v_A_i) . n) * n,   k = (1+e) mu_red / m_B

Median 0.86 label-sd over random donor pairs, so it is measurable.

**The honest caveat:** a real re-simulation with a different v_A would also move
the contact point, hence n. Holding n fixed is the local linearisation, and it
is the only version of the question that is well posed without re-rendering. It
is stated, not hidden.

WHY THE PASSIVE CHANNEL IS EXACTLY ZERO HERE
--------------------------------------------
§18a needed a passive-propagation control because the edit and the readout
shared tokens, so a residual stream carries the added vector into the readout
with no computation at all -- and a purely arithmetic effect scored obs/passive
1.02 and looked like success. Here the patched tokens (A-local AND pre-contact)
and the read tokens (B-local AND post-contact) are **disjoint in both region and
phase**. The pooled B.post vector therefore cannot move by arithmetic: its
pooling weight over the patched tokens is exactly 0, which the script asserts
per scene rather than assuming. Anything that arrives has been *computed*.
This is §31's floor/cube trick applied to the collision.

CONTROLS, all reported
----------------------
  B.pre       required change is EXACTLY ZERO -- B's incoming velocity cannot
              depend on A's. The before/after dissociation.
  A.pre       did the patch land at all? (§18d's transfer, ~0.86 expected)
  radius_B    an appearance property the patch must not touch.
  random      a random direction of matched norm.
  mismatched  observed scored against a DIFFERENT pair's required change --
              catches a readout that drifts in a fixed direction regardless.
  r(obs, req) the scene-dependent test: the required change FLIPS SIGN with the
              sign of (v_A_donor - v_A_i).n, so a constant response cannot fake
              a correlation. This is the strongest single number here.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steer_velocity import (                                        # noqa: E402
    SceneRunner, apply_readout, fit_readout, load_captures, r2_of, split_indices,
)
from physics_interp.config import load_config                       # noqa: E402
from physics_interp.interp import ActivationSteer, extract_label    # noqa: E402
from physics_interp.runner import VideoDiffusionRunner              # noqa: E402


def collision_geometry(labels):
    """(n, k) -- contact normal A->B, and the linear-response gain (1+e)mu/m_B."""
    c = labels["collision"]
    vB = np.asarray(c["pre"]["velocity_B"][:2], dtype=np.float64)
    vBp = np.asarray(c["post"]["velocity_B"][:2], dtype=np.float64)
    dvB = vBp - vB
    n = dvB / (np.linalg.norm(dvB) + 1e-12)
    mA = float(labels["objects"]["A"]["mass"]); mB = float(labels["objects"]["B"]["mass"])
    e = float(labels["restitution"])
    return n, (1.0 + e) * (mA * mB / (mA + mB)) / mB


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default=None)
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set")
    p.add_argument("--block", type=int, default=None, help="steer block")
    p.add_argument("--read-block", default=None)
    p.add_argument("--n-scenes", type=int, default=30)
    p.add_argument("--donors", type=int, default=3, help="donor draws per recipient")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config); cfg.model.id = args.model
    cap = args.capture_dir or f"outputs_{args.model}"
    blk = args.block if args.block is not None else (6 if "videomae" in args.model else 7)
    read_site = args.read_block or (
        "block.11.resid_post" if "videomae" in args.model else "block.23.resid_post")
    site = f"block.{blk}.resid_post"

    acts, labels, pools, scene_ids = load_captures(Path(cap), cfg.name)
    keep = [i for i, l in enumerate(labels) if l.get("momentum_conserved", True)]
    acts = {s: v[keep].astype(np.float64) for s, v in acts.items()}
    labels = [labels[i] for i in keep]; scene_ids = [scene_ids[i] for i in keep]
    n = len(labels); pi = {q: i for i, q in enumerate(pools)}
    tr, te = split_indices(n, split_seed=0)

    TARGETS = {("velocity_B_post", "B.post"): None, ("velocity_B_pre", "B.pre"): None,
               ("velocity_A_pre", "A.pre"): None, ("radius_B", "B.all"): None}
    readouts = {}
    print(f"{args.model}   steer {site} -> read {read_site}   "
          f"n={n} conserving, {len(tr)} train / {len(te)} test\n")
    print("readout quality on held-out scenes (fit on train only):")
    for (tgt, pool) in TARGETS:
        Y = np.stack([extract_label(l, tgt) for l in labels])
        X = acts[read_site][:, pi[pool], :]
        r = fit_readout(X, Y, tr); readouts[(tgt, pool)] = r
        print(f"   {tgt:<18} {pool:<7} R2 {r2_of(r, X, Y, te):.3f}  alpha {r['alpha']:g}")

    vA = np.stack([extract_label(l, "velocity_A_pre") for l in labels])
    geo = [collision_geometry(l) for l in labels]

    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    rng = np.random.default_rng(0)
    te_list = list(te)[: args.n_scenes]
    rows, t0 = [], time.perf_counter()

    for si, i in enumerate(te_list, 1):
        sid = scene_ids[i]
        try:
            sc = SceneRunner(runner, Path(args.dataset_dir) / sid, cfg.generation.num_frames)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [{si}] {sid}: SKIP ({exc})"); continue
        if sc.n_steer_tokens == 0:
            continue

        # ---- the passive channel, COMPUTED not argued -------------------- #
        wB_post = sc.weights.numpy()[sc.pool_names.index("B.post")]
        leak = float((wB_post * sc.steer_mask).sum())
        if leak > 1e-9:
            print(f"  [{si}] {sid}: patched and read tokens OVERLAP "
                  f"(weight {leak:.3e}) -- this scene is not clean"); continue

        base = sc.run([read_site])
        bp = {k: apply_readout(r, base[read_site][pi[k[1]]])[0] for k, r in readouts.items()}
        n_i, k_i = geo[i]

        for d in range(args.donors):
            j = int(rng.choice([x for x in te if x != i]))
            for variant in ("patch", "random"):
                delta = (acts[site][j, pi["A.pre"], :] - acts[site][i, pi["A.pre"], :])
                nrm = float(np.linalg.norm(delta))
                if variant == "random":
                    g = rng.normal(size=delta.shape)
                    delta = g / np.linalg.norm(g) * nrm
                steer = ActivationSteer(delta / nrm, alpha=nrm, scale="absolute",
                                        tokens=sc.steer_mask)
                out = sc.run([read_site], steer_site=site, steer=steer)

                obs = {k: apply_readout(r, out[read_site][pi[k[1]]])[0] - bp[k]
                       for k, r in readouts.items()}
                # required change, in label-sd units so obs and req are commensurate
                du_n = float((vA[j] - vA[i]) @ n_i)
                req_raw = k_i * du_n * n_i
                req = req_raw / readouts[("velocity_B_post", "B.post")]["y_sd"]
                o = obs[("velocity_B_post", "B.post")]
                rows.append({
                    "scene": sid, "donor": scene_ids[j], "variant": variant,
                    "delta_norm": nrm, "du_n": du_n,
                    "req_norm": float(np.linalg.norm(req)),
                    "transfer": float(o @ req / (req @ req + 1e-12)),
                    "obs_along_n": float(o @ (n_i / readouts[("velocity_B_post","B.post")]["y_sd"])
                                         / (np.linalg.norm(n_i / readouts[("velocity_B_post","B.post")]["y_sd"]) + 1e-12)),
                    "req_along_n": float(np.sign(du_n) * np.linalg.norm(req)),
                    "obs_vBpost_rms": float(np.sqrt((o ** 2).mean())),
                    "obs_vBpre_rms": float(np.sqrt((obs[("velocity_B_pre","B.pre")] ** 2).mean())),
                    "obs_vApre_rms": float(np.sqrt((obs[("velocity_A_pre","A.pre")] ** 2).mean())),
                    "obs_radiusB": float(obs[("radius_B", "B.all")][0]),
                    "vA_transfer": float(
                        obs[("velocity_A_pre","A.pre")] @
                        ((vA[j]-vA[i]) / readouts[("velocity_A_pre","A.pre")]["y_sd"]) /
                        (np.linalg.norm((vA[j]-vA[i]) / readouts[("velocity_A_pre","A.pre")]["y_sd"])**2 + 1e-12)),
                })
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(te_list)}] {sid}  ({el/si:.1f}s/scene)")

    out_dir = Path(args.out) if args.out else Path(f"outputs_steer_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "causal_collision.json"
    dest.write_text(json.dumps({"model": args.model, "steer_site": site,
                                "read_site": read_site, "rows": rows}, indent=1))
    summarise(rows)
    print(f"\n-> {dest}")
    return 0


def summarise(rows):
    import collections
    if not rows:
        print("no rows"); return
    by = collections.defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    print(f"\n{'='*76}\nRESULT  ({len(by['patch'])} patch pairs)\n{'='*76}")
    print(f"{'variant':<9}{'transfer':>10}{'+-':>8}{'dvB_post':>10}{'dvB_pre':>10}"
          f"{'dvA_pre':>10}{'d radB':>9}{'r(obs,req)':>12}")
    for v in ("patch", "random"):
        R = by[v]
        if not R: continue
        t = np.array([x["transfer"] for x in R])
        o = np.array([x["obs_along_n"] for x in R]); q = np.array([x["req_along_n"] for x in R])
        r_sc = float(np.corrcoef(o, q)[0, 1]) if len(o) > 3 else np.nan
        print(f"{v:<9}{t.mean():10.3f}{t.std()/np.sqrt(len(t)):8.3f}"
              f"{np.mean([x['obs_vBpost_rms'] for x in R]):10.3f}"
              f"{np.mean([x['obs_vBpre_rms'] for x in R]):10.3f}"
              f"{np.mean([x['obs_vApre_rms'] for x in R]):10.3f}"
              f"{np.mean([abs(x['obs_radiusB']) for x in R]):9.3f}{r_sc:12.3f}")
    P = by["patch"]
    tA = float(np.mean([x["vA_transfer"] for x in P]))
    tB = float(np.mean([x["transfer"] for x in P]))
    print(f"\n  patch landed on A:  vA transfer {tA:.3f}   (§18d measured 0.856 on v3)")
    print(f"  PROPAGATION EFFICIENCY  transfer_B / transfer_A = {tB/tA if tA else float('nan'):+.3f}")
    print("    (the model only received {:.0%} of the donor's velocity, so a network that".format(max(tA,0)))
    print("     propagates the collision perfectly would score 1.0 on this ratio, not on transfer)")

    # mismatched-required control, averaged over many shuffles + a permutation p
    o = np.array([x["obs_along_n"] for x in P]); q = np.array([x["req_along_n"] for x in P])
    rng = np.random.default_rng(0)
    obs_r = float(np.corrcoef(o, q)[0, 1])
    null = np.array([np.corrcoef(o, rng.permutation(q))[0, 1] for _ in range(2000)])
    print(f"\n  r(obs, req) = {obs_r:+.3f}   shuffled null {null.mean():+.3f} +- {null.std():.3f}"
          f"   p = {float((np.abs(null) >= abs(obs_r)).mean()):.3f}  (2-sided, 2000 perms)")
    print(f"  n = {len(P)} patch pairs")
    print(f"\n  B.pre is the DISSOCIATION: its required change is EXACTLY zero, so"
          f"\n  {np.mean([x['obs_vBpre_rms'] for x in P]):.3f} label-sd of movement there is pure leak,"
          f"\n  against {np.mean([x['obs_vBpost_rms'] for x in P]):.3f} on B.post where physics DOES require movement.")


if __name__ == "__main__":
    raise SystemExit(main())
