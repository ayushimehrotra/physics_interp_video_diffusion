#!/usr/bin/env python
"""Per-object velocity steering in a video encoder — the §16 pre-registered test.

The question is *not* whether velocity is decodable (it is, at 0.98) but whether
it is **addressable per object**: if a direction is written into the tokens
sphere A occupies, does A's velocity readout move, and does B's move with it?

§16 measured that A-local tokens already carry B's velocity at 0.75-0.83 and
predicted from that alone that per-object control should be poor. This script
runs the causal version of that prediction.

Why the leakage number is the trustworthy one. The on-target effect is partly
passive: a residual stream carries an added vector forward, so a direction
written at block N will show up in a readout at block M > N whether or not the
model *used* it. The leak measurement has no such loophole — the edit is confined
to A's tokens and the leak readout reads B's tokens, so information can only get
there by attention actually moving it. Read the on-target column as a
manipulation check and the leak column as the result.

Three controls, all at matched alpha, because §5 is emphatic that a fitted
direction has to beat its own shuffled twin and usually does not:

  shuffled  same pipeline on permuted labels
  random    isotropic unit vector
  radius_A  the appearance readout — if it moves as much as velocity, the steer
            damaged the representation rather than edited a variable

Stage 1 (`--mode pilot`) sweeps alpha on one block to find the window between
"nothing happens" and "representation destroyed"; stage 2 (`--mode sweep`) runs
the block x direction grid inside that window.

    .venv/bin/python scripts/steer_velocity.py --mode pilot
    .venv/bin/python scripts/steer_velocity.py --mode sweep --alphas 2,4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_interp.config import load_config                      # noqa: E402
from physics_interp.interp import (                                # noqa: E402
    ActivationSteer, build_directions, extract_label, ridge_fit,
)
from physics_interp.runner import VideoDiffusionRunner, build_pool_weights  # noqa: E402
from physics_interp.video import (                                 # noqa: E402
    load_frames, load_segmentation, masks_to_latent_grid, tubelet_frame_to_cell,
)

ALPHAS_RIDGE = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4)


# --------------------------------------------------------------------------- #
# fitting (no GPU — reads the existing captures)
# --------------------------------------------------------------------------- #

def load_captures(capture_dir: Path, name: str):
    """-> acts {site: [n, pools, dim]}, labels [n], pools, scene_ids."""
    files = sorted((capture_dir / name / "activations").glob("scene_*.npz"))
    if not files:
        raise SystemExit(f"no captures under {capture_dir / name / 'activations'}")
    acts, labels, ids, pools = {}, [], [], None
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            if pools is None:
                pools = meta["pools"]
                acts = {k[4:]: [] for k in z.files if k.startswith("act.")}
            for s in acts:
                # [steps, pools, dim] with exactly one step for an encoder.
                acts[s].append(z["act." + s][0])
            labels.append(meta["labels"])
            ids.append(meta["scene_id"])
    return {s: np.stack(v) for s, v in acts.items()}, labels, pools, ids


def split_indices(n: int, train_fraction: float = 0.6, split_seed: int = 0):
    """The same scene split fit_probe uses, so numbers stay comparable to §16."""
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    return order[:n_train], order[n_train:]


def fit_readout(X, Y, tr, alphas=ALPHAS_RIDGE):
    """Ridge readout that keeps its coefficients, so it can be applied to steered
    activations. fit_probe() returns only scores; the protocol here is identical
    (standardize on train, alpha chosen on an inner split of train)."""
    n_inner = max(4, int(round(0.75 * len(tr))))
    inner_tr, inner_val = tr[:n_inner], tr[n_inner:]
    if len(inner_val) < 2:
        inner_tr, inner_val = tr, tr

    def prep(idx):
        return X[idx].mean(0), X[idx].std(0) + 1e-6

    mu, sd = prep(inner_tr)
    scores = []
    for a in alphas:
        W, b = ridge_fit((X[inner_tr] - mu) / sd, Y[inner_tr], a)
        pred = (X[inner_val] - mu) / sd @ W + b
        ss_res = ((Y[inner_val] - pred) ** 2).sum()
        ss_tot = ((Y[inner_val] - Y[inner_val].mean(0)) ** 2).sum()
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    alpha = float(alphas[int(np.nanargmax(scores))])

    mu, sd = prep(tr)
    W, b = ridge_fit((X[tr] - mu) / sd, Y[tr], alpha)
    return {"W": W, "b": b, "mu": mu, "sd": sd, "alpha": alpha,
            "y_sd": Y[tr].std(0) + 1e-12}


def apply_readout(r, x):
    """x is [dim] or [n, dim] -> prediction in units of the training label sd."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return ((x - r["mu"]) / r["sd"] @ r["W"] + r["b"]) / r["y_sd"]


def r2_of(r, X, Y, te):
    pred = (X[te] - r["mu"]) / r["sd"] @ r["W"] + r["b"]
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# one scene, many conditions (frames loaded once — I/O dominates model time)
# --------------------------------------------------------------------------- #

class SceneRunner:
    """Holds one scene's pixels and geometry so N steered passes cost N forward
    passes and one read. Mirrors runner.encode_scene()'s setup exactly; the
    duplication buys a ~3x speedup on a network filesystem, where loading 49 PNGs
    costs more than the forward pass it feeds."""

    def __init__(self, runner, scene_dir: Path, num_frames: int):
        from physics_interp.models.encoders import tubelet_phase_split

        self.runner = runner
        adapter = runner.adapter
        self.labels = json.loads((scene_dir / "labels.json").read_text())
        frames = load_frames(scene_dir, num_frames=num_frames)
        self.idx = adapter.frame_indices(len(frames))
        self.frames = frames[self.idx]

        contact = float(self.labels["collision"]["actual_frame"])
        n_pre = tubelet_phase_split(self.idx, contact, adapter.tubelet)

        layout = runner.layout
        to_cell = lambda t: tubelet_frame_to_cell(t, adapter.tubelet)  # noqa: E731
        seg = adapter.preprocess_masks(load_segmentation(scene_dir)[self.idx])
        occ = masks_to_latent_grid(seg, layout, ids=(1, 2), frame_to_cell=to_cell)
        regions = {"A": occ[0], "B": occ[1],
                   "all": np.ones((layout.frames, layout.height, layout.width), np.float32)}
        phases = {"pre": (0, n_pre), "post": (n_pre, layout.frames),
                  "all": (0, layout.frames)}
        self.weights, self.pool_names = build_pool_weights(layout, regions, phases)

        # Binary A-local pre-contact token mask for the steer. NOT the pooling
        # row: that is normalised to sum to 1, so using it would scale the edit
        # by 1/n_tokens and silently make alpha mean something different in every
        # scene.
        m = np.zeros_like(occ[0])
        m[:n_pre] = occ[0][:n_pre]
        self.steer_mask = (m.reshape(-1) > 0).astype(np.float32)
        self.n_steer_tokens = int(self.steer_mask.sum())
        # Raw occupancy kept so a caller can build its own region masks — §31's
        # floor/cube patching needs region B as well, and needs them disjoint.
        self.occ = occ
        self.n_pre = n_pre

    def run(self, sites, steer_site=None, steer=None):
        from physics_interp.runner import ActivationRecorder

        runner = self.runner
        recorder = ActivationRecorder(self.weights, self.pool_names)
        prev = runner.hooks.guidance_scale
        try:
            runner.hooks.reset()
            runner.hooks.guidance_scale = 1.0
            runner.hooks.step_override = 0
            if steer is not None:
                # Registered before the recorder, so a recorder on the same site
                # sees the steered value rather than the pre-steer one.
                with runner.hooks.hook(steer_site, steer, steps=[0], name="steer"):
                    with runner.hooks.hook(sites, recorder, steps=[0], name="rec"):
                        runner.adapter.encode(self.frames)
            else:
                with runner.hooks.hook(sites, recorder, steps=[0], name="rec"):
                    runner.adapter.encode(self.frames)
        finally:
            runner.hooks.step_override = None
            runner.hooks.guidance_scale = prev
        # {site: [1, pools, dim]} -> {site: [pools, dim]}
        return {k: v[0] for k, v in recorder.collect([0]).items()}


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="configs/momentum_probe.yaml")
    p.add_argument("--model", default="videomae-base")
    p.add_argument("--capture-dir", default="outputs_v3_videomae-base")
    p.add_argument("--dataset-dir", default="momentum_dataset/probe_set_v3_latent")
    p.add_argument("--mode", choices=("pilot", "sweep"), default="pilot")
    p.add_argument("--component", type=int, default=0,
                   help="velocity component to steer: 0=x, 1=y")
    p.add_argument("--blocks", default=None, help="comma list; default picks from the depth curve")
    p.add_argument("--alphas", default=None, help="comma list; default depends on --mode")
    p.add_argument("--directions", default="probe,diff_means,residual,temporal,shuffled,random")
    p.add_argument("--read-block", default=None, help="site to read out at; default = last block")
    p.add_argument("--n-scenes", type=int, default=None)
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.model.id = args.model
    name = cfg.name
    capture_dir = Path(args.capture_dir)
    out_dir = Path(args.out) if args.out else Path(f"outputs_steer_{args.model}")
    comp_name = "xy"[args.component]

    # ---- phase A: everything that does not need a GPU --------------------- #
    print(f"loading captures from {capture_dir} ...", flush=True)
    acts, labels, pools, scene_ids = load_captures(capture_dir, name)
    n = len(labels)
    pi = {q: i for i, q in enumerate(pools)}
    blocks = sorted(int(s.split(".")[1]) for s in acts if s.startswith("block."))
    read_site = args.read_block or f"block.{blocks[-1]}.resid_post"
    tr, te = split_indices(n)
    print(f"{n} scenes, blocks {blocks[0]}-{blocks[-1]}, {len(tr)} train / {len(te)} test")

    Y = {t: np.stack([extract_label(l, t) for l in labels])
         for t in ("velocity_A_pre", "velocity_B_pre", "radius_A")}
    y_steer = Y["velocity_A_pre"][:, args.component]

    # Depth curve for the steered quantity, so the block choice is evidence-based.
    print(f"\nvelocity_A_pre[{comp_name}] from A.pre, by block (test R2):")
    curve = {}
    for b in blocks:
        X = acts[f"block.{b}.resid_post"][:, pi["A.pre"], :].astype(np.float64)
        r = fit_readout(X, y_steer[:, None], tr)
        curve[b] = r2_of(r, X, y_steer[:, None], te)
        print(f"  block {b:2d}  {curve[b]: .3f}")
    if args.blocks:
        steer_blocks = [int(x) for x in args.blocks.split(",")]
    elif args.mode == "pilot":
        steer_blocks = [max(curve, key=curve.get)]
    else:
        steer_blocks = sorted({blocks[1], blocks[len(blocks) // 3],
                               max(curve, key=curve.get), blocks[-2]})
    print(f"steering at blocks {steer_blocks}, reading at {read_site}")

    if args.alphas:
        alphas = [float(x) for x in args.alphas.split(",")]
    else:
        alphas = [1.0, 2.0, 4.0, 8.0, 16.0] if args.mode == "pilot" else [2.0, 4.0]
    want = args.directions.split(",")

    # Directions, fitted on train scenes only at the site they are written to.
    print("\nfitting directions ...", flush=True)
    directions = {}
    for b in steer_blocks:
        site = f"block.{b}.resid_post"
        X = acts[site][:, pi["A.pre"], :].astype(np.float64)
        # Appearance controls to project out for the `residual` estimator.
        ctrl = {t: build_directions(X[tr], Y[t][tr, 0])["probe"]
                for t in ("radius_A",)}
        tmean = acts[site][:, [pi["A.pre"], pi["A.post"]], :].mean(1).astype(np.float64)
        d = build_directions(X[tr], y_steer[tr], controls=ctrl,
                             temporal_mean=tmean[tr])
        directions[b] = {k: v for k, v in d.items()
                         if k in want and np.all(np.isfinite(v))}
        print(f"  block {b}: {sorted(directions[b])}")

    # Readouts, fit on train at the read site.
    readouts = {}
    for target, pool in (("velocity_A_pre", "A.pre"), ("velocity_B_pre", "B.pre"),
                         ("velocity_B_pre", "A.pre"), ("radius_A", "A.pre")):
        X = acts[read_site][:, pi[pool], :].astype(np.float64)
        r = fit_readout(X, Y[target], tr)
        r["r2"] = r2_of(r, X, Y[target], te)
        readouts[(target, pool)] = r
        print(f"  readout {target:16s} from {pool:6s} R2 {r['r2']: .3f}")

    # ---- phase B: GPU ----------------------------------------------------- #
    test_ids = [scene_ids[i] for i in te][: args.n_scenes]
    dataset = Path(args.dataset_dir)
    print(f"\nloading {args.model} ...", flush=True)
    runner = VideoDiffusionRunner.from_config(cfg, device="cuda")
    sites = [f"block.{b}.resid_post" for b in blocks]

    conditions = [("baseline", None, 0.0)]
    for b in steer_blocks:
        for dname, vec in directions[b].items():
            for a in alphas:
                conditions.append((dname, (b, vec), a))
    print(f"{len(test_ids)} scenes x {len(conditions)} conditions = "
          f"{len(test_ids) * len(conditions)} forward passes", flush=True)

    rows = []
    t0 = time.perf_counter()
    for si, sid in enumerate(test_ids, 1):
        scene_dir = dataset / sid
        try:
            sc = SceneRunner(runner, scene_dir, cfg.generation.num_frames)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [{si}/{len(test_ids)}] {sid}: SKIP ({exc})")
            continue
        if sc.n_steer_tokens == 0:
            print(f"  [{si}/{len(test_ids)}] {sid}: SKIP (no A-local pre tokens)")
            continue

        base = sc.run(sites)
        base_pred = {k: apply_readout(r, base[read_site][pi[k[1]]])[0]
                     for k, r in readouts.items()}
        for dname, spec, alpha in conditions:
            if spec is None:
                continue
            b, vec = spec
            steer = ActivationSteer(vec, alpha=alpha, tokens=sc.steer_mask)
            out = sc.run(sites, steer_site=f"block.{b}.resid_post", steer=steer)
            if steer.calls == 0:
                raise SystemExit(f"steer never fired at block {b} — site not hooked")

            # Realised perturbation on the tokens actually edited.
            # float64 is not optional: activations are stored fp16 and V-JEPA2's
            # reach |x| ~ 264, so a sum of squares overflows fp16's 65504 ceiling
            # and np.linalg.norm silently returns inf -- which turns every ratio
            # below into a clean, plausible-looking 0.000.
            h0 = base[f"block.{b}.resid_post"][pi["A.pre"]].astype(np.float64)
            h1 = out[f"block.{b}.resid_post"][pi["A.pre"]].astype(np.float64)
            pert = float(np.linalg.norm(h1 - h0) / (np.linalg.norm(h0) + 1e-12))

            row = {"scene": sid, "block": b, "direction": dname, "alpha": alpha,
                   "perturbation": pert, "n_steer_tokens": sc.n_steer_tokens}
            for k, r in readouts.items():
                d = apply_readout(r, out[read_site][pi[k[1]]])[0] - base_pred[k]
                # RMS over components, in training-label sd units.
                row[f"d_{k[0]}_from_{k[1]}"] = float(np.sqrt((d ** 2).mean()))

            # The control that decides whether any of this means anything.
            # A residual stream carries an added vector forward, so a readout at
            # block M > N moves whether or not the model *read* the direction.
            # Predict the readout change on the assumption that the pooled delta
            # measured at the steered block arrives at the read site untouched;
            # observed / passive ~ 1 means the effect is pure propagation and the
            # on-target column is arithmetic, not evidence about the model.
            kA = ("velocity_A_pre", "A.pre")
            dp = apply_readout(readouts[kA], base[read_site][pi["A.pre"]] + (h1 - h0))[0] \
                - base_pred[kA]
            row["passive_velocity_A_pre_from_A.pre"] = float(np.sqrt((dp ** 2).mean()))

            # Readout-free version of the same question: did the network do
            # anything to the injected vector, or does it ride the residual
            # stream untouched? cos ~ 1 with norm ratio ~ 1 means fully inert,
            # in which case no readout-based number can be evidence about
            # mechanism. leak_B is the cross-object channel measured in
            # representation space, with no probe in the loop at all.
            f64 = lambda k, p: out[k][pi[p]].astype(np.float64)    # noqa: E731
            b64 = lambda k, p: base[k][pi[p]].astype(np.float64)   # noqa: E731
            dA = f64(read_site, "A.pre") - b64(read_site, "A.pre")
            dB = f64(read_site, "B.pre") - b64(read_site, "B.pre")
            inj = h1 - h0
            nrm = lambda v: float(np.linalg.norm(v)) + 1e-12          # noqa: E731
            row["prop_norm_ratio"] = nrm(dA) / nrm(inj)
            row["prop_cos"] = float(inj @ dA / (nrm(inj) * nrm(dA)))
            row["leak_rel_A"] = nrm(dA) / nrm(b64(read_site, "A.pre"))
            row["leak_rel_B"] = nrm(dB) / nrm(b64(read_site, "B.pre"))
            rows.append(row)
        done = si / len(test_ids)
        el = time.perf_counter() - t0
        print(f"  [{si}/{len(test_ids)}] {sid}  {el:6.1f}s elapsed, "
              f"~{el / done - el:5.1f}s left", flush=True)
    runner.close()

    # ---- phase C ---------------------------------------------------------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_v{comp_name}_read{read_site.split('.')[1]}"
    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "read_site": read_site, "steer_blocks": steer_blocks, "alphas": alphas,
        "depth_curve": curve,
        "readout_r2": {f"{t}|{p}": readouts[(t, p)]["r2"] for (t, p) in readouts},
        "n_scenes": len(test_ids),
        "rows": rows,
    }
    path = out_dir / f"steer_{tag}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")
    summarise(rows)
    return 0


def summarise(rows):
    if not rows:
        print("no rows")
        return
    keys = [k for k in rows[0] if k.startswith(("d_", "passive_", "prop_", "leak_"))]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["block"], r["alpha"], r["direction"]), []).append(r)
    short = {
        "d_velocity_A_pre_from_A.pre": "dvA|A",
        "d_velocity_B_pre_from_B.pre": "dvB|B",
        "d_velocity_B_pre_from_A.pre": "dvB|A",
        "d_radius_A_from_A.pre": "drA|A",
        "passive_velocity_A_pre_from_A.pre": "passive",
        "prop_norm_ratio": "|dh11|/|dh6|",
        "prop_cos": "cos",
        "leak_rel_A": "relA",
        "leak_rel_B": "relB",
    }
    hdr = [short.get(k, k) for k in keys]
    print(f"\n{'block':>5} {'alpha':>6} {'direction':<10} {'|dh|/|h|':>9} "
          + " ".join(f"{h:>13}" for h in hdr))
    for (b, a, d), rs in sorted(groups.items()):
        pert = np.median([r["perturbation"] for r in rs])
        cells = " ".join(f"{np.median([r[k] for r in rs]):>13.3f}" for k in keys)
        print(f"{b:>5} {a:>6.1f} {d:<10} {pert:>9.3f} {cells}")


if __name__ == "__main__":
    raise SystemExit(main())
