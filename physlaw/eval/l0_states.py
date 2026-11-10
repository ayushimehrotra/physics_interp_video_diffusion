"""L0 states (spec §5): can per-object states be decoded from pooled activations?

Targets, per object o in {A, B} and phase in {pre, post}:
  velocity (2-D world xy, one block), speed, direction (cos, sin), log_mass, radius.
Grid per target: layers x {roi_<o>.<phase>, roi_pair.<phase>, frame.<phase>}.
Score: out-of-fold R^2, grouped 5-fold by twin group, 12 fold seeds, nested alpha
(physlaw.probes.grid, identical to ridge.fit_probe).

Baselines, through the same pipeline:
  * observers   obs_vel / obs_speed / obs_quad / obs_normal (true pre-contact kinematics)
  * mask floor  the pooling weights themselves (flattened [T, gh, gw]) as features, for the best
                cell's pool (old project §20: a ROI pool leaks state through where it sits). If
                token features carried position only, every probe on the pooled vector would be
                linear in these weights, so this is the ceiling of a content-free model.
                Frame pools have uniform weights (floor 0).
  * tracker     both balls' token-space centre, radius, presence per token-frame of the window
                (an ideal tracker on the masks; a reference for what is visible, not a floor)
  * untrained   the same grid on the untrained control (separate run; merged in the report)
  * null        500 group-block permutations, max over the target's grid -> corrected p.

Note (F1, legacy appearance arm): density is 1.5 or 6.0 and tied to the material, so mass is
visible through size AND material; the tracker (size only) bounds the size part.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from physlaw.probes.grid import grid_max_p, grid_probe, group_block_perms
from physlaw.probes.observer import OBSERVERS, observer_matrix

TARGETS = ("velocity", "speed", "direction", "log_mass", "radius")
OBJECTS = ("A", "B")
PHASES = ("pre", "post")
POOL_KINDS = ("roi_own", "roi_pair", "frame")

_G: dict = {}          # fork-shared state for workers


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def load_index(cache_dir: Path) -> list[dict]:
    recs = {}
    for line in open(cache_dir / "_index.jsonl"):
        r = json.loads(line)
        recs[r["clip"]] = r                      # last write wins (re-runs append)
    return [recs[k] for k in sorted(recs)]


def complete_groups(recs: list[dict]) -> list[dict]:
    """Keep only groups that hold the full twin set (needed for group-block permutations)."""
    twins_of: dict[str, set] = {}
    for r in recs:
        twins_of.setdefault(r["group"], set()).add(r["twin"])
    full = max(len(s) for s in twins_of.values())
    return [r for r in recs if len(twins_of[r["group"]]) == full]


def load_pooled(cache_dir: Path, recs: list[dict]):
    first = np.load(cache_dir / recs[0]["file"])
    pools = [str(p) for p in first["pools"]]
    layers = [int(l) if str(l).lstrip("-").isdigit() else str(l) for l in first["layers"]]
    A = np.stack([np.load(cache_dir / r["file"])["pooled"] for r in recs]).astype(np.float32)  # [n, P, L, D]
    return A, pools, layers


def targets(labels: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for o in OBJECTS:
        for ph in PHASES:
            v = np.array([l["physlaw"][f"{ph}_velocity"][o] for l in labels], float)[:, :2]
            s = np.linalg.norm(v, axis=1)
            out[f"{o}.{ph}.velocity"] = v
            out[f"{o}.{ph}.speed"] = s
            out[f"{o}.{ph}.direction"] = v / np.maximum(s, 1e-9)[:, None]
            out[f"{o}.{ph}.log_mass"] = np.log([l["physlaw"]["objects"][o]["mass"] for l in labels])
            out[f"{o}.{ph}.radius"] = np.array([l["physlaw"]["objects"][o]["radius"] for l in labels])
    return out


def _mask_feat_one(args):
    clip_dir, frames, spec, fpc = args
    from physlaw.tokens.pool import pool_weights
    from physlaw.tokens.project import project_masks, token_frame_phase
    inst = np.load(Path(clip_dir) / "masks.npz")["instance"]
    obj = project_masks(inst, list(frames), spec)
    phases = token_frame_phase(list(frames), spec, fpc)
    ph = np.array(phases)
    out = {}
    for w in PHASES:     # tracker: both balls' token-space centre, radius, presence per token-frame
        cols = []
        for o in OBJECTS:
            c = obj[o].center[ph == w]
            present = ~np.isnan(c[:, 0])
            cols += [np.nan_to_num(c[:, 0]), np.nan_to_num(c[:, 1]), obj[o].radius[ph == w], present.astype(float)]
        out[f"geom.{w}"] = np.concatenate(cols)
    W = pool_weights(obj, phases)     # mask floor: the pooling weights of each ROI pool, flattened
    for name, wt in W.items():
        if name.startswith("roi_") and not name.endswith(".all"):
            out[f"w.{name}"] = wt.reshape(-1).astype(np.float32)
    return out


def mask_features(cache_dir: Path, data_dir: Path, recs: list[dict], labels: list[dict], spec,
                  workers: int = 32) -> dict[str, np.ndarray]:
    """{'geom.pre'|'geom.post': [n, F], 'w.<pool>': [n, T*gh*gw]}, cached in <cache_dir>/_maskfeat_v2.npz."""
    f = cache_dir / "_maskfeat_v2.npz"
    clips = [r["clip"] for r in recs]
    if f.exists():
        z = np.load(f)
        if list(z["clips"]) == clips:
            return {k: z[k] for k in z.files if k != "clips"}
    jobs = [(str(data_dir / r["clip"]), np.load(cache_dir / r["file"])["frames"], spec,
             l["physlaw"]["first_post_contact_frame"]) for r, l in zip(recs, labels)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_mask_feat_one, jobs, chunksize=4))
    out = {}
    for k in res[0]:   # token-frame counts can differ by one if contact jitters; pad to max
        L = max(len(r[k]) for r in res)
        out[k] = np.stack([np.pad(r[k], (0, L - len(r[k]))) for r in res])
    np.savez(f, clips=np.array(clips), **out)
    return out


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def _lname(l) -> str:
    """Cell label for a layer key: int layers -> "L8"; diffusion keys are already "snr0/L15"."""
    return f"L{l}" if isinstance(l, int) else str(l)


def _pool_name(kind: str, o: str, ph: str) -> str:
    return {"roi_own": f"roi_{o}", "roi_pair": "roi_pair", "frame": "frame"}[kind] + f".{ph}"


def _run_cell(job):
    key, xfun, block_names = job
    X = _G["X"](xfun)
    blocks = {b: _G["Y"][b] for b in block_names}
    return key, grid_probe(X, blocks, _G["groups"], alphas=_G["alphas"], k=_G["k"],
                           fold_seeds=_G["fold_seeds"], perms=_G["perms"] if xfun[0] == "act" else None)


def _features(xfun):
    kind = xfun[0]
    if kind == "act":
        _, pi, li = xfun
        return _G["acts"][:, pi, li]
    if kind == "obs":
        return _G["obs"][xfun[1]]
    if kind == "mask":
        return _G["mask"][xfun[1]]
    raise KeyError(kind)


def run(cache_dir: Path, data_dir: Path, cfg, model_cfg, workers: int = 48, log=print) -> dict:
    """Returns the L0 result dict (also the per-layer curves)."""
    from physlaw.models.base import TokenGridSpec
    probe = cfg["probe"]
    recs = complete_groups(load_index(cache_dir))
    labels = [json.loads((data_dir / r["clip"] / "labels.json").read_text()) for r in recs]
    acts, pools, layers = load_pooled(cache_dir, recs)
    groups = np.array([r["group"] for r in recs])
    twins = np.array([r["twin"] for r in recs])
    log(f"[L0] {len(recs)} clips / {len(set(groups))} groups; acts {acts.shape}; layers {layers}")

    from physlaw.models.base import token_spec
    spec = token_spec(model_cfg)
    mask = mask_features(cache_dir, data_dir, recs, labels, spec, workers=min(workers, 32))
    Y = targets(labels)
    want = [t for t in cfg.get("targets", TARGETS)]
    Y = {k: v for k, v in Y.items() if k.split(".")[2] in want}

    _G.update(X=_features, Y=Y, groups=groups, acts=acts, mask=mask,
              obs={o: observer_matrix(labels, o) for o in OBSERVERS},
              alphas=tuple(float(x) for x in probe["alphas"]), k=int(probe["k_folds"]), fold_seeds=int(probe["fold_seeds"]),
              perms=group_block_perms(groups, twins, int(probe["n_permutations"]), seed=0))

    # one job per distinct feature matrix, carrying every target block that searches it
    jobs: dict[tuple, list[str]] = {}
    for b in Y:
        o, ph, _ = b.split(".")
        for kind in POOL_KINDS:
            pi = pools.index(_pool_name(kind, o, ph))
            for li in range(len(layers)):
                jobs.setdefault(("act", pi, li), []).append(b)
        for obs in OBSERVERS:
            jobs.setdefault(("obs", obs), []).append(b)
        jobs.setdefault(("mask", f"geom.{ph}"), []).append(b)
        for kind in ("roi_own", "roi_pair"):
            jobs.setdefault(("mask", f"w.{_pool_name(kind, o, ph)}"), []).append(b)
    log(f"[L0] {len(jobs)} feature cells x their targets; {len(_G['perms'])} permutations; {workers} workers")

    res: dict[tuple, dict] = {}
    with ProcessPoolExecutor(workers) as ex:   # fork: workers inherit _G
        for i, (key, r) in enumerate(ex.map(_run_cell, [(k, k, v) for k, v in jobs.items()], chunksize=1)):
            res[key] = r
            if (i + 1) % 25 == 0:
                log(f"[L0]   {i + 1}/{len(jobs)} cells")

    out = {"n_clips": len(recs), "n_groups": int(len(set(groups))), "layers": layers, "targets": {}}
    for b in Y:
        o, ph, _ = b.split(".")
        cells = {}
        for kind in POOL_KINDS:
            pname = _pool_name(kind, o, ph)
            pi = pools.index(pname)
            for li, l in enumerate(layers):
                cells[f"{_lname(l)}/{pname}"] = res[("act", pi, li)][b]
        best_cv = max(cells, key=lambda c: cells[c]["score_mean"])
        best_q, p, null_max = grid_max_p({c: v["quick"] for c, v in cells.items()},
                                         {c: v["null"] for c, v in cells.items()})
        curve = {kind: [cells[f"{_lname(l)}/{_pool_name(kind, o, ph)}"]["score_mean"] for l in layers] for kind in POOL_KINDS}
        obs = {k: res[("obs", k)][b] for k in OBSERVERS}
        best_obs = max(obs, key=lambda k: obs[k]["score_mean"])
        geom = res[("mask", f"geom.{ph}")][b]
        bpool = best_cv.rsplit("/", 1)[1]          # cell = "<layer key>/<pool>"; diffusion keys contain "/"
        floor = res[("mask", f"w.{bpool}")][b] if ("mask", f"w.{bpool}") in res else None
        out["targets"][b] = {
            "best_cell": best_cv, "r2": cells[best_cv]["score_mean"], "r2_sd": cells[best_cv]["score_sd"],
            "p_corrected": p, "null_max_q95": float(np.quantile(null_max, 0.95)),
            "best_cell_quick": best_q, "curve": curve,
            "observer": {"best": best_obs, "r2": obs[best_obs]["score_mean"],
                         "all": {k: v["score_mean"] for k, v in obs.items()}},
            # floor for the best cell's own pool (frame pools have uniform weights: floor 0)
            "mask_floor": {"pool": bpool, "r2": floor["score_mean"] if floor else 0.0,
                           "r2_sd": floor["score_sd"] if floor else 0.0},
            "mask_floor_by_pool": {k[1][2:]: res[k][b]["score_mean"] for k in res
                                   if k[0] == "mask" and k[1].startswith("w.") and b in res[k]},
            "tracker": {"r2": geom["score_mean"], "r2_sd": geom["score_sd"]},
        }
    return out


def table(res: dict, untrained: dict | None = None) -> str:
    rows = ["| target | best cell | R² (sd) | untrained R² | mask floor (same pool) | tracker | best observer | null q95 | p_corr |",
            "|---|---|---|---|---|---|---|---|---|"]
    for b, t in res["targets"].items():
        u = untrained["targets"].get(b, {}).get("r2") if untrained else None
        rows.append(f"| {b} | {t['best_cell']} | {t['r2']:.3f} ({t['r2_sd']:.3f}) | "
                    f"{'' if u is None else f'{u:.3f}'} | {t['mask_floor']['r2']:.3f} | {t['tracker']['r2']:.3f} | "
                    f"{t['observer']['r2']:.3f} ({t['observer']['best']}) | {t['null_max_q95']:.3f} | {t['p_corrected']:.3g} |")
    return "\n".join(rows)
