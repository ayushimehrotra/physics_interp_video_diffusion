"""L3 invariance (spec §5.5): Galilean (F4), mass independence (F7 drop pairs, F6 slides), transfer (F6, nuisance).

Which analyses run on a family: ``analyses.<family>`` in configs/eval/l3_invariance.yaml. All work from stage 4's
cache (pooled encode() activations, forecast()), on the CPU.

Probes: ridge, grouped by twin group, with the §5 held-out selection rule: the (pool, layer) cell and the penalty
are chosen by inner CV on the training groups only, then the probe predicts groups it never saw
(:func:`crossfit`). ``n_cells`` (variants searched) is reported with every number.

* galilean            probes for restitution e and the CoM-frame scattering angle (angle between the relative velocity
                      before and after the event; both Galilean invariant) trained on u = 0 clips. Per boost level:
                      invariance error = mean |decoded(boosted) - decoded(paired u = 0 clip)| / sd(target), against the
                      within-condition error mean |decoded - true| / sd on held-out u = 0 clips. A boosted clip is
                      predicted by the fold whose test groups hold its u = 0 partner (folds are grouped by the pair key,
                      so the probe never saw the scene). L2 is repeated per boost level on the boosted twins.
* mass_independence   per-object probes on REAL post-event frames (encode() FINAL, own ROI pool) for the target
                      (``accel_z`` fall acceleration, ``decel`` slide deceleration, ``dv`` velocity change), applied to
                      forecast() pooled the same way (and moment-matched, ``_mm``, as L1). Slope of the decoded target
                      against log mass within each group (law: 0), relative to the mean |target|: forecast vs probe
                      floor (the same probes on the real legal clips) vs the violating twin's real clips (sensitivity).
* transfer            train on condition a, test on b (and back): R² against b's own cross-fitted R²; size is the
                      control target (no physics, should transfer).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from physlaw.eval.l0_states import load_index
from physlaw.probes.ridge import DEFAULT_ALPHAS, _SVDRidge, grouped_kfold, r2_score

# --------------------------------------------------------------------------- #
# labels -> targets
# --------------------------------------------------------------------------- #


def get(labels: dict, path: str, default=None):
    """Dotted-path lookup in a labels dict ("physlaw.boost.level")."""
    x = labels
    for k in path.split("."):
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


def event_frame(labels: dict) -> int:
    pl = labels["physlaw"]
    for k in ("event_frame", "branch_frame", "first_post_contact_frame"):
        if k in pl:
            return int(pl[k])
    raise KeyError("labels have no event frame")


def _rel(labels):
    v0 = np.array([labels["physlaw"]["pre_velocity"][o][:2] for o in "AB"], float)
    v1 = np.array([labels["physlaw"]["post_velocity"][o][:2] for o in "AB"], float)
    return v0[0] - v0[1], v1[0] - v1[1], np.asarray(labels["physlaw"]["contact_normal"][:2], float)


def restitution(labels) -> float:
    r0, r1, n = _rel(labels)
    n = n / np.linalg.norm(n)
    return float(-(r1 @ n) / (r0 @ n))


def scatter_angle(labels) -> float:
    """CoM-frame scattering angle: rotation of the relative velocity through the event (rad, [0, pi])."""
    r0, r1, _ = _rel(labels)
    return float(np.arccos(np.clip(r0 @ r1 / (np.linalg.norm(r0) * np.linalg.norm(r1) + 1e-12), -1, 1)))


def post_accel(labels, o: str, window_s: float = 1.0) -> np.ndarray:
    """Mean acceleration vector (3-D) of object o over ``window_s`` after the event, from per-frame velocities."""
    pf, fps = labels["per_frame"], labels["scene_constants"]["frame_rate"]
    e = event_frame(labels)
    fr = [f for f in range(e + 1, min(len(pf), e + 1 + int(window_s * fps))) if pf[f][o].get("in_frame", True)]
    t = np.array(fr, float) / fps
    v = np.array([pf[f][o]["velocity"] for f in fr], float)
    return np.polyfit(t, v, 1)[0] if len(fr) >= 3 else np.full(3, np.nan)


def plane_axes(labels) -> tuple[int, int]:
    """The two world axes spanning the image plane: all but the one most aligned with the camera's viewing direction
    (labels-1 F1 clips carry 2-D velocities: (0, 1))."""
    v = labels["physlaw"]["pre_velocity"]["A"]
    cam = labels["physlaw"].get("camera")
    if len(v) < 3 or not cam:
        return (0, 1)
    depth = int(np.argmax(np.abs(np.asarray(cam["world_to_camera"], float)[2, :3])))
    return tuple(a for a in range(3) if a != depth)


def object_target(labels, o: str, name: str, window_s: float = 1.0) -> float:
    if name == "accel_z":
        return float(post_accel(labels, o, window_s)[2])
    if name == "decel":
        a = post_accel(labels, o, window_s)
        v = np.asarray(labels["per_frame"][event_frame(labels)][o]["velocity"], float)
        return float(-(a @ v) / (np.linalg.norm(v) + 1e-12))
    if name == "dv":
        pl = labels["physlaw"]
        return float(np.linalg.norm(np.subtract(pl["post_velocity"][o][:2], pl["pre_velocity"][o][:2])))
    raise KeyError(name)


def clip_target(labels, name: str) -> np.ndarray:
    """Clip-level targets (object A where per-object), as a 1-D array."""
    pl = labels["physlaw"]
    if name == "restitution":
        return np.array([restitution(labels)])
    if name == "com_angle":
        return np.array([scatter_angle(labels)])
    if name in ("size", "radius"):
        return np.array([pl["objects"]["A"]["radius"]])
    if name == "log_mass":
        return np.array([np.log(pl["objects"]["A"]["mass"])])
    if name == "velocity":                           # the two in-image-plane components (F1: x, y; side views: x, z)
        return np.asarray(pl["pre_velocity"]["A"], float)[list(plane_axes(labels))]
    if name == "accel":
        return np.array([np.linalg.norm(post_accel(labels, "A"))])
    for path in (f"physlaw.{name}", f"physlaw.constants.{name}", f"physlaw.edit.{name}", f"physlaw.scene.{name}",
                 f"sampled_parameters.{name}"):
        v = get(labels, path)
        if v is not None:
            return np.atleast_1d(np.asarray(v, float))
    raise KeyError(f"no target {name!r} in labels")


# --------------------------------------------------------------------------- #
# cache -> features
# --------------------------------------------------------------------------- #


def load_family(cache_dir: Path, data_dir: Path):
    """(records, labels, pooled [n, P, L, D] float32, pool names, layer keys) for every cached clip. Not L0's
    complete_groups: F6 / F7 sub-families have different twin sets, and stage 4 caches complete groups only."""
    recs = load_index(cache_dir)
    labels = [json.loads((data_dir / r["clip"] / "labels.json").read_text()) for r in recs]
    first = np.load(cache_dir / recs[0]["file"])
    pools = [str(p) for p in first["pools"]]
    layers = [str(l) for l in first["layers"]]
    A = np.stack([np.load(cache_dir / r["file"])["pooled"] for r in recs]).astype(np.float32)
    return recs, labels, A, pools, layers


def cells(A, pools, layers, want_pools, want_layers="all") -> dict[str, np.ndarray]:
    """{"<layer>/<pool>": [n, D]} for the searched grid."""
    ls = layers if want_layers == "all" else [str(l) for l in want_layers]
    return {f"{l}/{p}": A[:, pools.index(p), layers.index(l)] for p in want_pools if p in pools for l in ls}


def _mask_weights(cache_dir: Path, data_dir: Path, recs, mcfg) -> dict:
    """Per-clip ROI pooling weights (L0's _maskfeat_v2.npz, built and cached here if L0 has not run)."""
    from physlaw.eval.l0_states import mask_features
    from physlaw.models.base import token_spec
    labels = [json.loads((data_dir / r["clip"] / "labels.json").read_text()) for r in recs]
    return mask_features(cache_dir, data_dir, recs, labels, token_spec(mcfg), workers=4)


def forecast_spaces(cache_dir: Path, data_dir: Path, recs, mcfg, pool_names) -> dict[str, tuple[dict, dict]]:
    """{space: (real {pool: [n, D]}, forecast {pool: [n, D]})}, as L1: encoders' FINAL space (forecast tokens pooled
    with each clip's own weights from L0's _maskfeat_v2.npz), diffusion models' per-(SNR, layer) fc_pooled."""
    zs = [np.load(cache_dir / r["file"]) for r in recs]
    pools = [str(p) for p in zs[0]["pools"]]
    if "fc_pooled" in zs[0].files:
        lk, fk = [str(l) for l in zs[0]["layers"]], [str(x) for x in zs[0]["fc_keys"]]
        return {q: ({p: np.stack([z["pooled"][pools.index(p), lk.index(q)] for z in zs]).astype(np.float64) for p in pool_names},
                    {p: np.stack([z["fc_pooled"][pools.index(p), fk.index(q)] for z in zs]).astype(np.float64) for p in pool_names})
                for q in fk}
    if "pooled_final" in zs[0].files:
        get_ = lambda z, p: z["pooled_final"][pools.index(p)]
    else:
        li = [str(l) for l in zs[0]["layers"]].index("-1")
        get_ = lambda z, p: z["pooled"][pools.index(p), li]
    real = {p: np.stack([get_(z, p) for z in zs]).astype(np.float64) for p in pool_names}
    mf = _mask_weights(cache_dir, data_dir, recs, mcfg)
    tub = int(mcfg["token_grid"]["tubelet"])
    fc = {p: np.zeros_like(real[p]) for p in pool_names}
    memo = {}
    for i, (r, z) in enumerate(zip(recs, zs)):
        if r["group"] not in memo:
            f = np.load(cache_dir / r["group"] / "_forecast.npz")
            memo = {r["group"]: (f["final"].astype(np.float32), int(f["context_frames"]) // tub)}
        F, t0 = memo[r["group"]]
        ph = np.array([str(x) for x in z["phases"]])
        post = ph[t0:] == "post"
        for p in pool_names:
            if p.startswith("frame."):
                fc[p][i] = F[post].reshape(-1, F.shape[-1]).mean(0)
            else:
                w = mf[f"w.{p}"][i][: len(ph) * F.shape[1] * F.shape[2]].reshape(len(ph), *F.shape[1:3])[t0:] * post[:, None, None]
                fc[p][i] = (w.reshape(-1) / max(w.sum(), 1e-12)) @ F.reshape(-1, F.shape[-1])
    return {"final": (real, fc)}


# --------------------------------------------------------------------------- #
# probes with held-out selection
# --------------------------------------------------------------------------- #


def fit_select(Xc: dict, Y: np.ndarray, groups: np.ndarray, rows: np.ndarray, alphas=DEFAULT_ALPHAS, k: int = 5,
               seed: int = 0):
    """Choose (cell, alpha) by grouped inner CV on ``rows`` only; refit there. Returns (cell, predict(X) fn, inner R²)."""
    Y = Y.reshape(len(Y), -1)
    g = groups[rows]
    if len(set(g)) < 2:                      # no CV possible (tiny test sets): first cell, middle penalty
        c, a = next(iter(Xc)), alphas[len(alphas) // 2]
        m = _SVDRidge(Xc[c][rows], Y[rows], True)
        return c, (lambda X, m=m, a=a: m.predict(X, a)), float("nan")
    folds = grouped_kfold(g, min(k, len(set(g))), seed + 10_000)
    best = (-np.inf, None, None)
    for c, X in Xc.items():
        P = {a: np.zeros((len(rows), Y.shape[1])) for a in alphas}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(rows)), te)
            m = _SVDRidge(X[rows[tr]], Y[rows[tr]], True)
            for a in alphas:
                P[a][te] = m.predict(X[rows[te]], a)
        for a in alphas:
            s = r2_score(Y[rows], P[a])
            if np.isfinite(s) and s > best[0]:
                best = (s, c, a)
    s, c, a = best
    m = _SVDRidge(Xc[c][rows], Y[rows], True)
    return c, (lambda X, m=m, a=a: m.predict(X, a)), float(s)


def crossfit(Xc: dict, Y: np.ndarray, groups: np.ndarray, fit_rows: np.ndarray, k: int = 5, seeds: int = 2,
             alphas=DEFAULT_ALPHAS):
    """Held-out predictions for every row: a row whose group appears among ``fit_rows``' groups is predicted by the
    probe of the outer fold that held that group out (selection inside the fold), averaged over ``seeds``; a row of
    any other group by one probe fitted on all ``fit_rows``. Returns (pred [n, k], chosen cells)."""
    Y = Y.reshape(len(Y), -1)
    pred = np.zeros((len(Y), Y.shape[1]))
    hit = np.zeros(len(Y))
    chosen = []
    fg = groups[fit_rows]
    for s in range(seeds):
        for te in grouped_kfold(fg, min(k, len(set(fg))), s):
            test_groups = set(fg[te])
            tr = fit_rows[~np.isin(fg, list(test_groups))]
            c, f, _ = fit_select(Xc, Y, groups, tr, alphas, k, s)
            chosen.append(c)
            rows = np.where(np.isin(groups, list(test_groups)))[0]
            pred[rows] += f(Xc[c][rows])
            hit[rows] += 1
    pred = np.where(hit[:, None] > 0, pred / np.maximum(hit, 1)[:, None], np.nan)
    unseen = np.where(hit == 0)[0]
    if len(unseen):                      # groups never fitted on: one probe on every fit row (pure transfer)
        c, f, _ = fit_select(Xc, Y, groups, fit_rows, alphas, k, 0)
        pred[unseen] = f(Xc[c][unseen])
    return pred, chosen


def _boot(vals_by_group: dict, n: int = 2000, seed: int = 0):
    gs = sorted(vals_by_group)
    if len(gs) < 2:
        return None
    rng = np.random.default_rng(seed)
    b = [np.mean(np.concatenate([vals_by_group[g] for g in rng.choice(gs, len(gs))])) for _ in range(n)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def _cell_count(c: list[str]) -> dict:
    from collections import Counter
    return dict(Counter(c).most_common(3))


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #


def galilean(recs, labels, A, pools, layers, cfg, cache_dir=None, data_dir=None, log=print) -> dict:
    gc = cfg["galilean"]
    base = str(gc["base_level"])
    level = np.array([str(get(l, gc["boost_key"], base)) for l in labels])
    pair = np.array(["/".join(str(get(l, k)) for k in gc["pair_keys"]) for l in labels])   # base <-> boosted partners
    pgroup = np.array([str(get(l, gc["cv_key"], r["group"])) for l, r in zip(labels, recs)])  # CV unit: the scene
    speed = np.array([float(np.linalg.norm(get(l, gc.get("boost_vec_key", ""), [0.0]) or [0.0])) for l in labels])
    Xc = cells(A, pools, layers, gc["pools"], gc.get("layers", "all"))
    fit_rows = np.where(level == base)[0]
    if not len(fit_rows):
        raise RuntimeError(f"galilean: no clips at base level {base!r} of {gc['boost_key']}")
    out = {"n_clips": len(recs), "n_cells": len(Xc), "levels": {}, "targets": {}}
    for tname in gc["targets"]:
        Y = np.stack([clip_target(l, tname) for l in labels])[:, 0]
        pred, chosen = crossfit(Xc, Y, pgroup, fit_rows, int(cfg["probe"]["k_folds"]), int(cfg["fold_seeds"]),
                                tuple(float(a) for a in cfg["probe"]["alphas"]))
        pred = pred[:, 0]
        sd = float(Y[fit_rows].std())
        within = np.abs(pred[fit_rows] - Y[fit_rows]) / sd
        d = {"sd_target": sd, "cells": _cell_count(chosen),
             "within_error": float(within.mean()), "within_r2": r2_score(Y[fit_rows], pred[fit_rows]),
             "within_error_ci95": _boot({g: within[pgroup[fit_rows] == g] for g in set(pgroup[fit_rows])}),
             "by_level": {}}
        base_pred = {pair[i]: pred[i] for i in fit_rows}
        for lv in sorted(set(level) - {base}):
            rows = np.where((level == lv) & np.isfinite(pred))[0]
            paired = [i for i in rows if pair[i] in base_pred]
            inv = np.array([abs(pred[i] - base_pred[pair[i]]) / sd for i in paired])
            err = np.abs(pred[rows] - Y[rows]) / sd
            d["by_level"][str(lv)] = {
                "n": len(rows), "n_paired": len(paired), "boost_speed_mean": float(speed[rows].mean()) if len(rows) else None,
                "invariance_error": float(inv.mean()) if len(inv) else None,
                "invariance_error_ci95": _boot({g: inv[[pgroup[i] == g for i in paired]] for g in set(pgroup[paired])}) if len(inv) else None,
                "transfer_error": float(err.mean()) if len(rows) else None,
                "transfer_r2": r2_score(Y[rows], pred[rows]) if len(rows) > 2 else None,
                "invariance_over_within": float(inv.mean() / within.mean()) if len(inv) else None}
        out["targets"][tname] = d
        log(f"[L3 galilean] {tname}: within {d['within_error']:.3f} (R² {d['within_r2']:.3f}); " + "; ".join(
            f"{lv}: inv {v['invariance_error']} transfer {v['transfer_error']}" for lv, v in d["by_level"].items()))
    for lv in sorted(set(level)):
        out["levels"][str(lv)] = int((level == lv).sum())
    if cache_dir is not None and gc.get("l2_contrasts"):
        out["l2_by_level"] = l2_by_level(cache_dir, data_dir, recs, level, gc["l2_contrasts"], log)
    return out


def l2_by_level(cache_dir: Path, data_dir: Path, recs, level, contrasts, log=print) -> dict:
    """L2 (raw and motion-adjusted) separately on each boost level's twins. ``contrasts``: an L2 family block
    ({calibration: [legal, legal], list: [{viol, legal, ...}]}) whose twin names may contain ``{level}``."""
    from physlaw.eval import l2_surprise
    lv_of = {r["clip"]: lv for r, lv in zip(recs, level)}
    srcs = {"model": cache_dir / "_index.jsonl", "pixel_baselines": cache_dir / "_pixel_baselines.jsonl"}
    fam = data_dir.name
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lv in sorted(set(level)):
            sources = {}
            for name, f in srcs.items():
                if not f.exists():
                    continue
                keep = [l for l in f.read_text().splitlines() if l.strip() and lv_of.get(json.loads(l)["clip"]) == lv]
                (Path(tmp) / f"{name}_{lv}.jsonl").write_text("\n".join(keep) + "\n")
                sources[name] = Path(tmp) / f"{name}_{lv}.jsonl"
            fill = lambda t: t.format(level=lv)
            cfg = {"contrasts": {fam: {"calibration": [fill(t) for t in contrasts["calibration"]],
                                       "list": [{**c, "viol": fill(c["viol"]), "legal": fill(c["legal"])}
                                                for c in contrasts.get("list", [])]}}}
            res = l2_surprise.run(sources, data_dir, cfg, log=lambda *a: None)
            out[str(lv)] = {"contrasts": {c: {src: {m: {k: v[k] for k in ("n", "win_rate", "win_rate_ci95") if k in v}
                                                    for m, v in d.items()} for src, d in by.items()}
                                          for c, by in res["contrasts"].items()},
                            "motion_adjusted": res.get("motion_adjusted")}
            log(f"[L3 galilean] L2 at boost level {lv}: {len(res['contrasts'])} contrasts")
    return out


def _within_slope(y: np.ndarray, x: np.ndarray, g: np.ndarray) -> float:
    """OLS slope of y on x after removing each group's mean (within-group slope)."""
    yc, xc = y.copy(), x.copy()
    for k in set(g):
        s = g == k
        yc[s] -= y[s].mean()
        xc[s] -= x[s].mean()
    return float((xc @ yc) / max(xc @ xc, 1e-12))


def mass_independence(cache_dir, data_dir, recs, labels, mcfg, cfg, family: str, log=print) -> dict:
    mc = cfg["mass_independence"]
    fam = mc["families"][family]
    sel = [i for i, l in enumerate(labels) if fam.get("subfamily") is None or get(l, "physlaw.subfamily") == fam["subfamily"]]
    if len({recs[i]["group"] for i in sel}) < 2:
        return {"skipped": f"{len({recs[i]['group'] for i in sel})} {fam.get('subfamily') or ''} groups"}
    objs = list(fam.get("objects", ["A", "B"]))
    tname = fam["target"]
    pools = {o: f"roi_{o}.post" for o in objs}
    # pooling weights are cached for the whole family (the L0 file); keep the sub-family's rows afterwards
    spaces = {sp: tuple({p: v[sel] for p, v in d.items()} for d in rf)
              for sp, rf in forecast_spaces(cache_dir, data_dir, recs, mcfg, sorted(set(pools.values()))).items()}
    recs, labels = [recs[i] for i in sel], [labels[i] for i in sel]
    groups = np.array([r["group"] for r in recs])
    twins = np.array([r["twin"] for r in recs])
    legal = np.isin(twins, fam["legal"])
    k, seeds = min(int(cfg["probe"]["k_folds"]), len(set(groups))), int(cfg["fold_seeds"])
    alphas = tuple(float(a) for a in cfg["probe"]["alphas"])
    Y = {o: np.array([object_target(l, o, tname, float(fam.get("window_s", 1.0))) for l in labels]) for o in objs}
    logm = {o: np.log([l["physlaw"]["objects"][o]["mass"] for l in labels]) for o in objs}
    scale = float(np.mean(np.abs(np.concatenate([Y[o][legal] for o in objs]))))
    out = {"target": tname, "subfamily": fam.get("subfamily"), "objects": objs, "target_mean_abs": scale,
           "n_groups": int(len(set(groups))), "spaces": {}}
    for space, (real, fc) in spaces.items():
        for mm in (False, True):
            dec_real, dec_fc = {}, {}
            for o in objs:
                X, Xf = real[pools[o]], fc[pools[o]]
                pr, pf = np.zeros(len(recs)), np.zeros(len(recs))
                for s in range(seeds):
                    for te in grouped_kfold(groups, k, s):
                        tr = np.setdiff1d(np.arange(len(recs)), te)
                        c, f, _ = fit_select({"x": X}, Y[o], groups, tr, alphas, k, s)
                        pr[te] += f(X[te])[:, 0] / seeds
                        xf = Xf[te]
                        if mm:
                            xf = (xf - Xf[tr].mean(0)) / (Xf[tr].std(0) + 1e-8) * (X[tr].std(0) + 1e-8) + X[tr].mean(0)
                        pf[te] += f(xf)[:, 0] / seeds
                dec_real[o], dec_fc[o] = pr, pf
            stack = lambda d, rows: np.concatenate([d[o][rows] for o in objs])
            res = {}
            for name, dec, rows in (("forecast", dec_fc, legal), ("floor", dec_real, legal),
                                    *((f"real_{t}", dec_real, twins == t) for t in fam.get("violating", []))):
                if not rows.any():
                    continue
                gg = np.concatenate([groups[rows]] * len(objs))
                x = stack(logm, rows)
                y = stack(dec, rows)
                slope = _within_slope(y, x, gg) / scale
                ug = sorted(set(gg))
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(int(cfg["bootstrap"]["n"])):
                    pick = rng.choice(ug, len(ug))
                    idx = np.concatenate([np.where(gg == p)[0] for p in pick])
                    gi = np.concatenate([np.full((gg == p).sum(), j) for j, p in enumerate(pick)])
                    bs.append(_within_slope(y[idx], x[idx], gi) / scale)
                res[name] = {"slope_rel": slope, "slope_rel_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                             "true_slope_rel": _within_slope(stack(Y, rows), x, gg) / scale, "n_clips": int(rows.sum())}
            vname = space + ("_mm" if mm else "")
            out["spaces"][vname] = res
            log(f"[L3 mass] {vname}: " + "; ".join(f"{n}: {v['slope_rel']:+.3f} (true {v['true_slope_rel']:+.3f})" for n, v in res.items()))
    return out


def transfer(recs, labels, A, pools, layers, cfg, family: str, log=print) -> dict:
    tc = cfg["transfer"][family]
    cond = np.array([str(get(l, tc["split_key"])) for l in labels])
    a, b = (str(x) for x in tc["conditions"])
    groups = np.array([r["group"] for r in recs])
    if tc.get("twins"):
        keep = np.isin([r["twin"] for r in recs], tc["twins"])
    else:
        keep = np.ones(len(recs), bool)
    Xc = cells(A, pools, layers, tc["pools"], tc.get("layers", "all"))
    k, seeds = int(cfg["probe"]["k_folds"]), int(cfg["fold_seeds"])
    alphas = tuple(float(x) for x in cfg["probe"]["alphas"])
    out = {"split_key": tc["split_key"], "conditions": [a, b], "n": {a: int((keep & (cond == a)).sum()), b: int((keep & (cond == b)).sum())},
           "n_cells": len(Xc), "targets": {}}
    if min(len(set(groups[keep & (cond == c)])) for c in (a, b)) < 2:
        return {**out, "skipped": f"fewer than 2 groups in a condition {out['n']}"}
    for tname in tc["targets"]:
        Y = np.stack([clip_target(l, tname) for l in labels])
        d = {}
        for src, dst in ((a, b), (b, a)):
            rs, rd = np.where(keep & (cond == src))[0], np.where(keep & (cond == dst))[0]
            c, f, _ = fit_select(Xc, Y, groups, rs, alphas, k, 0)
            within, ch = crossfit(Xc, Y, groups, rd, k, seeds, alphas)
            d[f"{src}->{dst}"] = {"r2_transfer": r2_score(Y[rd], f(Xc[c][rd])), "cell": c,
                                  "r2_within_dst": r2_score(Y[rd], within[rd]), "cells_within": _cell_count(ch)}
            d[f"{src}->{dst}"]["gap"] = d[f"{src}->{dst}"]["r2_within_dst"] - d[f"{src}->{dst}"]["r2_transfer"]
        out["targets"][tname] = d
        log(f"[L3 transfer] {tname}: " + "; ".join(f"{k_}: {v['r2_transfer']:.3f} vs within {v['r2_within_dst']:.3f}"
                                                   for k_, v in d.items()))
    return out


MIN_GROUPS = 4


def run(cache_dir: Path, data_dir: Path, cfg, mcfg, log=print) -> dict:
    family = data_dir.name.split("_")[0]
    todo = list(cfg["analyses"].get(family, []))
    if not todo:
        raise RuntimeError(f"L3: no analyses configured for family {family} (configs/eval/l3_invariance.yaml analyses)")
    recs, labels, A, pools, layers = load_family(cache_dir, data_dir)
    log(f"[L3] {family}: {len(recs)} clips / {len(set(r['group'] for r in recs))} groups; analyses {todo}")
    out = {"family": family, "n_clips": len(recs), "fold_seeds": int(cfg["fold_seeds"]), "k_folds": int(cfg["probe"]["k_folds"])}
    if cfg.get("stand_in"):
        out["stand_in"] = cfg["stand_in"]
    n_groups = len({r["group"] for r in recs})
    for an in todo:
        if n_groups < int(cfg.get("min_groups", MIN_GROUPS)):   # grouped CV with inner selection needs a few groups
            out[an] = {"skipped": f"{n_groups} groups < {cfg.get('min_groups', MIN_GROUPS)}"}
            log(f"[L3] {an}: skipped ({out[an]['skipped']})")
        elif an == "galilean":
            out["galilean"] = galilean(recs, labels, A, pools, layers, cfg, cache_dir, data_dir, log)
        elif an == "mass_independence":
            out["mass_independence"] = mass_independence(cache_dir, data_dir, recs, labels, mcfg, cfg, family, log)
        elif an == "transfer":
            out["transfer"] = transfer(recs, labels, A, pools, layers, cfg, family, log)
        else:
            raise KeyError(an)
    return out


def table(res: dict) -> str:
    rows = []
    if res.get("stand_in"):
        rows += [f"**Stand-in run:** {res['stand_in']}", ""]
    for an in ("galilean", "mass_independence", "transfer"):
        if "skipped" in res.get(an, {}):
            rows += [f"{an}: skipped ({res[an]['skipped']})", ""]
    g = res.get("galilean")
    if g and "skipped" not in g:
        rows += ["## Galilean invariance", "", f"Clips per boost level: {g['levels']}; cells searched: {g['n_cells']}", "",
                 "| target | within error [95% CI] | within R² | level | n (paired) | invariance error [95% CI] | transfer error | inv / within |",
                 "|---|---|---|---|---|---|---|---|"]
        for t, d in g["targets"].items():
            ci = d["within_error_ci95"]
            for lv, v in d["by_level"].items():
                ic = v["invariance_error_ci95"]
                f = lambda x: "-" if x is None else f"{x:.3f}"
                rows.append(f"| {t} | {d['within_error']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") +
                            f" | {d['within_r2']:.3f} | {lv} | {v['n']} ({v['n_paired']}) | {f(v['invariance_error'])}" +
                            (f" [{ic[0]:.3f}, {ic[1]:.3f}]" if ic else "") + f" | {f(v['transfer_error'])} | {f(v['invariance_over_within'])} |")
        if g.get("l2_by_level"):
            rows += ["", "L2 per boost level (adjusted win rate, first metric per contrast): see l3.json `galilean.l2_by_level`."]
        rows.append("")
    m = res.get("mass_independence")
    if m and "skipped" not in m:
        rows += ["## Mass independence", "", f"Target `{m['target']}` ({m['subfamily'] or 'all'}), slope vs log mass / mean |target| "
                 f"(law: 0); {m['n_groups']} groups", "", "| space | clips | slope [95% CI] | true slope |", "|---|---|---|---|"]
        for sp, d in m["spaces"].items():
            for n, v in d.items():
                ci = v["slope_rel_ci95"]
                rows.append(f"| {sp} | {n} | {v['slope_rel']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {v['true_slope_rel']:+.3f} |")
        rows.append("")
    t = res.get("transfer")
    if t and "skipped" not in t:
        rows += ["## Transfer", "", f"Split `{t['split_key']}` {t['conditions']}, clips {t['n']}, cells searched {t['n_cells']}", "",
                 "| target | direction | R² transfer | R² within target condition | gap |", "|---|---|---|---|---|"]
        for tn, d in t["targets"].items():
            for k, v in d.items():
                rows.append(f"| {tn} | {k} | {v['r2_transfer']:.3f} | {v['r2_within_dst']:.3f} | {v['gap']:+.3f} |")
    return "\n".join(rows)
