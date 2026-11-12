"""L3 invariance (spec §5.0.4, §5.5): Galilean (F4), mass independence (F7 drop pairs, F6 pairs, F9 pendulum pairs),
transfer (clean -> nuisance: F1, F7).

Which analyses run on a family: ``analyses.<family>`` in configs/eval/l3_invariance.yaml. All work from stage 4's
cache (pooled encode() activations, forecast()), on the CPU.

Probes: ridge, grouped by twin group, with the §5 held-out selection rule: the (pool, layer) cell and the penalty
are chosen by inner CV on the training groups only, then the probe predicts groups it never saw
(:func:`crossfit`). ``n_cells`` (variants searched) is reported with every number.

The R1 instrument (physlaw.probes.ridge.fit_instrument, shared with L0, L1 and L4): ONE linear map per (object,
quantity), trained on the real pre- AND post-event windows of the training groups jointly (rows stacked, the clip's
pre and post rows in the same fold), cell and penalty by inner grouped CV, then a per-output inverse calibration fitted
on the inner out-of-fold predictions (v_hat ~ c + d v, decode (v_hat - c) / d: undoes ridge shrinkage); ridge.instrument_folds
cross-fits it, ridge.apply_instrument decodes any feature matrix with the fold that held a row's group out.

* galilean   F4. Velocity instruments for A and B trained on real clips of ALL boost levels (so the instrument is
             valid on boosted clips; its per-level velocity R² and epsilon error are the floor, and a level whose
             floor epsilon error exceeds ``galilean.max_floor_eps_err`` is "unmeasurable"), cross-fitted by scene
             (a clip is decoded by a probe that never saw its scene at any boost level). Per clip, from the DECODED
             pre and post velocities: epsilon and the signed CoM-frame scattering angle theta_cm
             (laws.collision.invariants, true contact normal and masses), and laws.momentum.stats (eta with true masses, antiparallel). Per
             boost level: means, and the paired difference to the u = 0 partner (same scene, same rung): d_epsilon,
             d_theta (laws.galilean.stats; law: 0), against the base level's decoding error. Never a probe
             trained to output epsilon directly (it would read speed, which a boost changes). L2 is repeated per
             boost level on the boosted twins (``l2_contrasts``). Camera boosts (levels ``cam_<level>``: the
             collision unboosted, the camera moving at -u; their labels' states are camera-frame, ``frame: camera``,
             so the instrument's targets are the motion the clip shows) are levels of their own against u = 0, and
             ``camera_vs_ball`` pairs each with its ball-boost twin (same scene, rung, |u|; identical ball image
             motion, only the background differs): decoded stat difference camera - ball (law and ideal: 0).
             This real-video test is the MEASURABILITY FLOOR only (audit H2: real videos are Galilean invariant, so
             an accurate encoder scores d ~ 0 whatever its physics). The law test (``galilean.forecast``, per
             forecast space, :func:`galilean_forecast_spaces`): one R1 velocity instrument per object and space
             (real pre + post rows of every level, cross-fitted by scene, cells ``galilean.forecast_pools``) decodes
             the real pre-event velocities and BOTH the real post-event pools (floor) and the forecast's post-event
             pools (each clip reads its own (group, pre_segment = boost level) forecast). Per boosted clip vs its
             u = 0 partner (and cam_<level> vs <level>): d_epsilon, d_theta from the forecast and from the real
             frames, the same pairs, the same instrument; paired difference forecast - floor with a scene-bootstrap
             CI. Verdict per level: "not Galilean invariant" when a d's difference CI is above 0 (the boost changes
             the forecast's eps / theta_cm beyond what the instrument's own boost error does on real frames), else
             "invariant within the floor"; "unmeasurable" when this space's floor median |eps_hat - eps| on the level
             or the base exceeds ``max_floor_eps_err``. Headline: the space with the smallest base-level floor error
             (``selected_space``; chosen on real clips only).
* mass_independence  the shared R1 instrument (ridge.instrument_folds, inverse-calibrated, cross-fitted by group:
             ONE map per space reading every object, rows stacked: each object's own post-event ROI pool -> its
             own target, so heavy and light are read by the same instrument) on REAL post-event frames for the
             acceleration along g (``accel_g``, F7 drops), the deceleration along the motion after the branch (``decel``, F6 pairs)
             or the pendulum period after the branch (``period``, F9 pairs: io.pendulum_periods, twice the
             interval between passes through the bottom), applied to forecast() pooled the same way (and
             moment-matched, ``_mm``, as L1). Per clip the law's heavy / light ratio (law: 1): laws.gravity
             mass_ratio (drops), laws.friction mass_ratio (F6 pairs), laws.oscillation mass_period_ratio (pendulum
             pairs; ``mass_independence.families.<fam>.law``). Floor-relative score
             S = laws.ratio_score(t_forecast, t_floor, t_violator): t_floor = the same probes on the real legal clips,
             t_violator = on the real violating twin (heavier falls faster); "unmeasurable" when the probes cannot
             tell the violator from the floor (|log t_violator - log t_floor| < ``min_log_sensitivity``).
             Mask-only control (audit M0): the forecast is read through ROI pools placed by each clip's REAL
             post-event masks, which already move as the law says on legal clips. So the same instrument protocol is
             fitted on the pooling weights alone (L0's _maskfeat_v2.npz ``w.roi_<o>.post``), its heavy / light ratio
             on the legal clips scored against the same floor and violator (``S_mask_only``), and the forecast
             counts only if S - S_mask_only has a group-bootstrap CI above 0 (``beyond_mask``; verdict).
* transfer   train on condition a, test on b (and back): R² against b's own cross-fitted R²; size is the
             control target (no physics, should transfer).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from physlaw.eval.l0_states import _pool_name, event_inputs, load_index
from physlaw.eval.l1_forecast_laws import clip_segment, forecast_path, forecast_segments
from physlaw.probes.ridge import (DEFAULT_ALPHAS, apply_instrument, fit_select, grouped_kfold, instrument_folds,
                                  moment_match, r2_score)

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
    """``accel_g``: acceleration along g (> 0 falling); ``decel``: deceleration along the motion; ``period``: the
    pendulum period after the event (io.pendulum_periods; aux rods skipped)."""
    if name == "period":
        from physlaw.data.io import pendulum_periods
        names = [n for n, ob in labels["physlaw"]["objects"].items() if not ob.get("aux")]
        return float(pendulum_periods(labels, names)[1][names.index(o)])
    a = post_accel(labels, o, window_s)
    if name == "accel_g":
        return float(-a[2])
    if name == "decel":
        v = np.asarray(labels["per_frame"][event_frame(labels)][o]["velocity"], float)
        return float(-(a @ v) / (np.linalg.norm(v) + 1e-12))
    raise KeyError(name)


def clip_target(labels, name: str) -> np.ndarray:
    """Clip-level transfer targets (object A where per-object), as a 1-D array."""
    pl = labels["physlaw"]
    if name in ("size", "radius"):
        return np.array([pl["objects"]["A"]["radius"]])
    if name == "log_mass":
        return np.array([np.log(pl["objects"]["A"]["mass"])])
    if name == "velocity":                           # the two in-image-plane components (F1: x, y; side views: x, z)
        return np.asarray(pl["pre_velocity"]["A"], float)[list(plane_axes(labels))]
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


def phase_cells(A, pools, layers, o: str, kinds, want_layers="all") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Instrument cells for object o: {"<layer>/<kind>": (X_pre [n, D], X_post [n, D])}, kinds of
    l0_states.POOL_KINDS (roi_own, roi_pair, frame)."""
    ls = layers if want_layers == "all" else [str(l) for l in want_layers]
    out = {}
    for kind in kinds:
        pp, pq = _pool_name(kind, o, "pre"), _pool_name(kind, o, "post")
        if pp in pools and pq in pools:
            for l in ls:
                li = layers.index(l)
                out[f"{l}/{kind}"] = (A[:, pools.index(pp), li], A[:, pools.index(pq), li])
    return out


def _mask_weights(cache_dir: Path, data_dir: Path, recs, mcfg) -> dict:
    """Per-clip ROI pooling weights (L0's _maskfeat_v2.npz, built and cached here if L0 has not run)."""
    from physlaw.eval.l0_states import mask_features
    from physlaw.models.base import token_spec
    labels = [json.loads((data_dir / r["clip"] / "labels.json").read_text()) for r in recs]
    return mask_features(cache_dir, data_dir, recs, labels, token_spec(mcfg), workers=4)


def forecast_spaces(cache_dir: Path, data_dir: Path, recs, mcfg, pool_names) -> dict[str, tuple[dict, dict]]:
    """{space: (real {pool: [n, D]}, forecast {pool: [n, D]})}, as L1: encoders' FINAL space (forecast tokens pooled
    with each clip's own weights from L0's _maskfeat_v2.npz; each clip reads its own (group, pre_segment) forecast,
    l1_forecast_laws.forecast_path), diffusion models' per-(SNR, layer) fc_pooled. Real pre-event pools are
    returned as cached (their forecast entry is meaningless)."""
    zs = [np.load(cache_dir / r["file"]) for r in recs]
    pools = [str(p) for p in zs[0]["pools"]]
    if "fc_pooled" in zs[0].files:
        lk, fk = [str(l) for l in zs[0]["layers"]], [str(x) for x in zs[0]["fc_keys"]]
        if not set(fk) & set(lk):
            raise RuntimeError(f"{cache_dir}: no forecast key {fk} is among the encode layers, so no real rows share "
                               "its space (R1); forecast_layers in configs/models must be a subset of layers")
        return {q: ({p: np.stack([z["pooled"][pools.index(p), lk.index(q)] for z in zs]).astype(np.float64) for p in pool_names},
                    {p: np.stack([z["fc_pooled"][pools.index(p), fk.index(q)] for z in zs]).astype(np.float64) for p in pool_names})
                for q in fk if q in lk}
    if "pooled_final" in zs[0].files:
        get_ = lambda z, p: z["pooled_final"][pools.index(p)]
    else:
        li = [str(l) for l in zs[0]["layers"]].index("-1")
        get_ = lambda z, p: z["pooled"][pools.index(p), li]
    real = {p: np.stack([get_(z, p) for z in zs]).astype(np.float64) for p in pool_names}
    mf = _mask_weights(cache_dir, data_dir, recs, mcfg)
    tub = int(mcfg["token_grid"]["tubelet"])
    fc = {p: np.zeros_like(real[p]) for p in pool_names}
    # one forecast per (group, pre_segment): F4's boost levels each have their own pre-event frames
    seg = forecast_segments({r["clip"]: clip_segment(json.loads((data_dir / r["clip"] / "labels.json").read_text()))
                             for r in recs})
    memo = {}
    for i, (r, z) in enumerate(zip(recs, zs)):
        f_ = forecast_path(cache_dir, r["group"], seg[r["clip"]])
        if f_ not in memo:
            f = np.load(f_)
            memo = {f_: (f["final"].astype(np.float32), int(f["context_frames"]) // tub)}
        F, t0 = memo[f_]
        ph = np.array([str(x) for x in z["phases"]])
        post = ph[t0:] == "post"
        for p in pool_names:
            if p.startswith("frame."):
                fc[p][i] = F[post].reshape(-1, F.shape[-1]).mean(0)
            elif f"w.{p}" in mf:
                w = mf[f"w.{p}"][i][: len(ph) * F.shape[1] * F.shape[2]].reshape(len(ph), *F.shape[1:3])[t0:] * post[:, None, None]
                fc[p][i] = (w.reshape(-1) / max(w.sum(), 1e-12)) @ F.reshape(-1, F.shape[-1])
    return {"final": (real, fc)}


# --------------------------------------------------------------------------- #
# probes with held-out selection
# --------------------------------------------------------------------------- #


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
            c, f, *_ = fit_select(Xc, Y, groups, tr, alphas, k, s)
            chosen.append(c)
            rows = np.where(np.isin(groups, list(test_groups)))[0]
            pred[rows] += f(Xc[c][rows])
            hit[rows] += 1
    pred = np.where(hit[:, None] > 0, pred / np.maximum(hit, 1)[:, None], np.nan)
    unseen = np.where(hit == 0)[0]
    if len(unseen):                      # groups never fitted on: one probe on every fit row (pure transfer)
        c, f, *_ = fit_select(Xc, Y, groups, fit_rows, alphas, k, 0)
        pred[unseen] = f(Xc[c][unseen])
    return pred, chosen


def epsilons(v_pre: np.ndarray, v_post: np.ndarray, ctxs: list) -> np.ndarray:
    """laws.epsilon per clip: v_pre / v_post [n, n_obj, d], ctxs [n] (event kind, normal / g)."""
    from physlaw.eval import laws
    return np.array([laws.epsilon({"v": a}, {"v": b}, c) for a, b, c in zip(v_pre, v_post, ctxs)], float)


def _boot(vals_by_group: dict, n: int = 2000, seed: int = 0):
    gs = sorted(vals_by_group)
    if len(gs) < 2 or n <= 0:
        return None
    rng = np.random.default_rng(seed)
    b = [np.mean(np.concatenate([vals_by_group[g] for g in rng.choice(gs, len(gs))])) for _ in range(n)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def _cell_count(c: list[str]) -> dict:
    from collections import Counter
    return dict(Counter(c).most_common(3))


# --------------------------------------------------------------------------- #
# Galilean invariance (F4)
# --------------------------------------------------------------------------- #


def paired_by_level(s: np.ndarray, level: np.ndarray, pair: np.ndarray, pgroup: np.ndarray, base: str,
                    n_boot: int = 2000) -> dict:
    """Per level: mean of the per-clip statistic s (scene-bootstrap CI) and, for non-base levels, the paired
    difference boosted - its base partner (mean, CI, mean |diff|). Galilean law: difference 0."""
    base_row = {pair[i]: i for i in np.where(level == base)[0]}
    out = {}
    for lv in sorted(set(level)):
        rows = np.where((level == lv) & np.isfinite(s))[0]
        d = {"n": int(len(rows)), "mean": float(s[rows].mean()) if len(rows) else None,
             "ci95": _boot({g: s[rows][pgroup[rows] == g] for g in set(pgroup[rows])}, n_boot) if len(rows) else None}
        if lv != base:
            pr = [(i, base_row[pair[i]]) for i in rows if pair[i] in base_row and np.isfinite(s[base_row[pair[i]]])]
            diff = np.array([s[i] - s[j] for i, j in pr])
            gi = np.array([pgroup[i] for i, _ in pr])
            d.update(n_paired=len(pr), diff=float(diff.mean()) if len(pr) else None,
                     diff_ci95=_boot({g: diff[gi == g] for g in set(gi)}, n_boot) if len(pr) else None,
                     abs_diff=float(np.abs(diff).mean()) if len(pr) else None)
        out[str(lv)] = d
    return out


def galilean_stats(v_pre: np.ndarray, v_post: np.ndarray, ctxs: list) -> dict[str, np.ndarray]:
    """Per-clip statistics of two-body events from velocities [n, 2, d]: epsilon and the signed CoM-frame scattering
    angle theta_cm (laws.collision.invariants, ctx masses), and laws.momentum.stats' eta and antiparallel."""
    from physlaw.eval import laws
    from physlaw.eval.laws.collision import invariants
    inv = np.array([invariants({"v": a}, {"v": b}, c) for a, b, c in zip(v_pre, v_post, ctxs)], float)
    mom = [laws.momentum.stats({"v": a}, {"v": b}, c) for a, b, c in zip(v_pre, v_post, ctxs)]
    return {"epsilon": inv[:, 0], "theta_cm": inv[:, 1],
            "eta": np.array([m["eta"] for m in mom], float), "antiparallel": np.array([m["antiparallel"] for m in mom], float)}


def galilean_pairs(v_pre, v_post, ctxs, level, pair, pgroup, base: str, n_boot: int = 2000) -> dict:
    """laws.galilean.stats of every boosted clip against its u = 0 partner (ctx ref = the partner's states):
    per level mean d_epsilon = |eps - eps_ref| and d_theta = |theta_cm - theta_ref| / pi with scene-bootstrap CIs."""
    from physlaw.eval import laws
    base_row = {pair[i]: i for i in np.where(level == base)[0]}
    out = {}
    for lv in sorted(set(level) - {base}):
        rows = [i for i in np.where(level == lv)[0] if pair[i] in base_row]
        st = [laws.galilean.stats({"v": v_pre[i]}, {"v": v_post[i]},
                                  {**ctxs[i], "ref": ({"v": v_pre[base_row[pair[i]]]}, {"v": v_post[base_row[pair[i]]]})})
              for i in rows]
        d = {"n_paired": len(rows)}
        for k_ in ("d_epsilon", "d_theta"):
            x = np.array([s_[k_] for s_ in st], float)
            ok = np.isfinite(x)
            g = pgroup[np.array(rows, int)][ok] if len(rows) else np.array([])
            d[k_] = float(x[ok].mean()) if ok.any() else None
            d[f"{k_}_ci95"] = _boot({h: x[ok][g == h] for h in set(g)}, n_boot) if ok.any() else None
        out[str(lv)] = d
    return out


def _pair_rows(level: np.ndarray, pair: np.ndarray, ref: str, lv: str) -> list[tuple[int, int]]:
    """(row of level lv, row of its partner at level ref with the same pair key) for every partnered clip."""
    ref_row = {pair[i]: i for i in np.where(level == ref)[0]}
    return [(i, ref_row[pair[i]]) for i in np.where(level == lv)[0] if pair[i] in ref_row]


def pair_d(v_pre: np.ndarray, v_post: np.ndarray, ctxs: list, rows: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """laws.galilean.stats per pair (i, j) (clip i vs its reference j, both read from the same arrays):
    {"d_epsilon": [n], "d_theta": [n]}."""
    from physlaw.eval import laws
    st = [laws.galilean.stats({"v": v_pre[i]}, {"v": v_post[i]}, {**ctxs[i], "ref": ({"v": v_pre[j]}, {"v": v_post[j]})})
          for i, j in rows]
    return {k: np.array([s[k] for s in st], float) for k in ("d_epsilon", "d_theta")}


GALILEAN_D = ("d_epsilon", "d_theta")


def galilean_forecast_compare(v_pre, v_post_fc, v_post_real, ctxs, level, pair, pgroup, base: str, n_boot: int = 2000,
                              measurable: dict | None = None) -> dict:
    """Forecast vs real-video floor, per boosted level against the base (and cam_<level> against <level>): for the
    same pairs, d from the forecast's post-event velocities and from the real ones (pre-event velocities: the real
    decodes in both), their means with scene-bootstrap CIs, and the paired difference forecast − floor. Verdict:
    ``not Galilean invariant`` when any d's difference CI is above 0, else ``invariant within the floor``;
    ``unmeasurable`` when ``measurable[level]`` or ``measurable[ref]`` is False."""
    level, pair, pgroup = np.asarray(level), np.asarray(pair), np.asarray(pgroup)
    lvls = set(level.tolist())
    comps = [(str(lv), str(lv), base) for lv in sorted(lvls - {base})]
    comps += [(f"{lv}_vs_{lv[4:]}", lv, lv[4:]) for lv in sorted(lvls) if lv.startswith("cam_") and lv[4:] in lvls]
    out = {}
    for key, lv, ref in comps:
        rows = _pair_rows(level, pair, ref, lv)
        fc, fl = pair_d(v_pre, v_post_fc, ctxs, rows), pair_d(v_pre, v_post_real, ctxs, rows)
        g = pgroup[np.array([i for i, _ in rows], int)] if rows else np.array([])
        rec = {"vs": ref, "n_paired": len(rows)}
        beyond = False
        for k_ in GALILEAN_D:
            ok = np.isfinite(fc[k_]) & np.isfinite(fl[k_])
            a, b, gg = fc[k_][ok], fl[k_][ok], g[ok]
            by = lambda x: {h: x[gg == h] for h in set(gg)}
            d = {"n": int(ok.sum()), "forecast": float(a.mean()) if ok.any() else None,
                 "forecast_ci95": _boot(by(a), n_boot) if ok.any() else None,
                 "floor": float(b.mean()) if ok.any() else None, "floor_ci95": _boot(by(b), n_boot) if ok.any() else None,
                 "diff": float((a - b).mean()) if ok.any() else None,
                 "diff_ci95": _boot(by(a - b), n_boot) if ok.any() else None}
            d["ratio"] = d["forecast"] / d["floor"] if ok.any() and d["floor"] > 0 else None
            beyond |= bool(d["diff_ci95"] and d["diff_ci95"][0] > 0)
            rec[k_] = d
        if rows:             # signed eps shift (boost - reference): a systematic boost dependence survives forecast noise
            i_, j_ = (np.array(x, int) for x in zip(*rows))
            for nm, vpost in (("forecast", v_post_fc), ("floor", v_post_real)):
                e = galilean_stats(v_pre, vpost, ctxs)["epsilon"]
                s = e[i_] - e[j_]
                ok = np.isfinite(s)
                rec.setdefault("epsilon_shift", {})[nm] = float(s[ok].mean()) if ok.any() else None
                rec["epsilon_shift"][f"{nm}_ci95"] = _boot({h: s[ok][g[ok] == h] for h in set(g[ok])}, n_boot) if ok.any() else None
        meas = measurable is None or (measurable.get(lv, True) and measurable.get(ref, True))
        rec["measurable"] = bool(meas)
        rec["verdict"] = ("unmeasurable (real-video floor)" if not meas else
                          "not Galilean invariant: the boost changes the forecast's eps / theta_cm beyond the floor"
                          if beyond else "invariant within the real-video floor")
        out[key] = rec
    return out


def galilean_forecast_spaces(spaces: dict, Vpre, Vpost, ctxs, level, pair, pgroup, base: str, kinds, k: int,
                             seeds: int, alphas, n_boot: int, max_err: float) -> dict:
    """The forecast-based Galilean test per forecast space. spaces: {space: (real {pool: [n, D]}, forecast {pool:
    [n, D]})} (:func:`forecast_spaces`). One R1 velocity instrument per object and space over the cells ``kinds``
    (l0_states.POOL_KINDS), cross-fitted by scene; decodes real pre, real post (floor) and forecast post."""
    from physlaw.eval.l0_states import _pool_name
    level = np.asarray(level)
    true_eps = galilean_stats(Vpre, Vpost, ctxs)["epsilon"]
    out = {"spaces": {}}
    for sp, (real, fc) in spaces.items():
        dp, dr, df = np.zeros_like(Vpre), np.zeros_like(Vpost), np.zeros_like(Vpost)
        for oi, o in enumerate("AB"):
            ks = [kd for kd in kinds if _pool_name(kd, o, "pre") in real and _pool_name(kd, o, "post") in real]
            Xc = {kd: (real[_pool_name(kd, o, "pre")], real[_pool_name(kd, o, "post")]) for kd in ks}
            folds = instrument_folds(Xc, (Vpre[:, oi], Vpost[:, oi]), pgroup, None, k, seeds, alphas)
            dp[:, oi] = apply_instrument(folds, {c: x[0] for c, x in Xc.items()}, pgroup)
            dr[:, oi] = apply_instrument(folds, {c: x[1] for c, x in Xc.items()}, pgroup)
            df[:, oi] = apply_instrument(folds, {kd: fc[_pool_name(kd, o, "post")] for kd in ks}, pgroup)
        eps_fl = galilean_stats(dp, dr, ctxs)["epsilon"]
        err = {str(lv): float(np.nanmedian(np.abs(eps_fl - true_eps)[level == lv])) for lv in sorted(set(level))}
        meas = {lv: bool(e <= max_err) for lv, e in err.items()}
        out["spaces"][sp] = {"floor_eps_abs_err_median": err, "measurable": meas,
                             "levels": galilean_forecast_compare(dp, df, dr, ctxs, level, pair, pgroup, base, n_boot, meas)}
    if out["spaces"]:
        sel = min(out["spaces"], key=lambda s: out["spaces"][s]["floor_eps_abs_err_median"].get(base, np.inf))
        out["selected_space"] = sel
        out["selection"] = "smallest base-level floor median |eps_hat - eps| (real clips only)"
        out["verdict"] = {lv: r["verdict"] for lv, r in out["spaces"][sel]["levels"].items()}
    return out


def galilean(recs, labels, A, pools, layers, cfg, cache_dir=None, data_dir=None, log=print, mcfg=None) -> dict:
    gc = cfg["galilean"]
    base = str(gc["base_level"])
    level = np.array([str(get(l, gc["boost_key"], base)) for l in labels])
    pair = np.array(["/".join(str(get(l, k)) for k in gc["pair_keys"]) for l in labels])   # base <-> boosted partners
    pgroup = np.array([str(get(l, gc["cv_key"], r["group"])) for l, r in zip(labels, recs)])  # CV unit: the scene
    if not (level == base).any():
        raise RuntimeError(f"galilean: no clips at base level {base!r} of {gc['boost_key']}")
    ev = [event_inputs(l) for l in labels]
    Vpre, Vpost, ctxs = np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev]), [e[2] for e in ev]
    k, seeds = int(cfg["probe"]["k_folds"]), int(cfg["fold_seeds"])
    alphas = tuple(float(a) for a in cfg["probe"]["alphas"])
    n = len(recs)
    dec_pre, dec_post, chosen, n_cells = np.zeros_like(Vpre), np.zeros_like(Vpost), [], 0
    for oi, o in enumerate("AB"):
        Xc = phase_cells(A, pools, layers, o, gc["pools"], gc.get("layers", "all"))
        n_cells = len(Xc)
        folds = instrument_folds(Xc, (Vpre[:, oi], Vpost[:, oi]), pgroup, None, k, seeds, alphas)
        chosen += [c for *_, c, _ in folds]
        dec_pre[:, oi] = apply_instrument(folds, {c: x[0] for c, x in Xc.items()}, pgroup)
        dec_post[:, oi] = apply_instrument(folds, {c: x[1] for c, x in Xc.items()}, pgroup)
    dec, true = galilean_stats(dec_pre, dec_post, ctxs), galilean_stats(Vpre, Vpost, ctxs)
    nb = int(cfg.get("bootstrap", {}).get("n", 2000))
    out = {"n_clips": n, "n_cells": n_cells, "cells": _cell_count(chosen),
           "levels": {str(lv): int((level == lv).sum()) for lv in sorted(set(level))},
           "max_floor_eps_err": float(gc["max_floor_eps_err"]), "floor": {}, "stats": {}}
    for lv in sorted(set(level)):                 # floor: the instrument's own accuracy on this level's real clips
        r = level == lv
        err = np.abs(dec["epsilon"][r] - true["epsilon"][r])
        out["floor"][str(lv)] = {
            "r2_velocity_pre": r2_score(Vpre[r].reshape(r.sum(), -1), dec_pre[r].reshape(r.sum(), -1)),
            "r2_velocity_post": r2_score(Vpost[r].reshape(r.sum(), -1), dec_post[r].reshape(r.sum(), -1)),
            "eps_abs_err_median": float(np.nanmedian(err)),
            "theta_abs_err_median": float(np.nanmedian(np.abs(dec["theta_cm"][r] - true["theta_cm"][r]))),
            "measurable": bool(np.nanmedian(err) <= float(gc["max_floor_eps_err"]))}
    for name in dec:
        d = paired_by_level(dec[name], level, pair, pgroup, base, nb)
        tr = paired_by_level(true[name], level, pair, pgroup, base, 0)
        base_err = float(np.nanmean(np.abs(dec[name] - true[name])[level == base]))
        for lv, v in d.items():
            v["true_mean"] = tr[lv]["mean"]
            if v.get("abs_diff") is not None:
                v["abs_diff_over_base_err"] = v["abs_diff"] / base_err if base_err > 0 else None
        out["stats"][name] = {"base_abs_err": base_err, "by_level": d}
    out["galilean"] = {"decoded": galilean_pairs(dec_pre, dec_post, ctxs, level, pair, pgroup, base, nb),
                       "true": galilean_pairs(Vpre, Vpost, ctxs, level, pair, pgroup, base, 0)}
    for lv, v in out["galilean"]["decoded"].items():
        v["measurable"] = out["floor"][lv]["measurable"] and out["floor"][base]["measurable"]
    log("[L3 galilean] " + "; ".join(f"{lv}: eps err {f['eps_abs_err_median']:.3f} d_eps "
                                      f"{out['galilean']['decoded'].get(lv, {}).get('d_epsilon')}"
                                      for lv, f in out["floor"].items()))
    cam = sorted(lv for lv in set(level) if lv.startswith("cam_") and lv[len("cam_"):] in set(level))
    if cam:     # camera boost vs ball boost: the same ball image motion, the background moves in one only
        out["camera_vs_ball"] = {}
        for lv in cam:
            r = np.isin(level, [lv, lv[len("cam_"):]])
            out["camera_vs_ball"][lv] = {
                name: {**{k_: v for k_, v in paired_by_level(dec[name][r], level[r], pair[r], pgroup[r],
                                                             lv[len("cam_"):], nb)[lv].items() if k_ != "mean"},
                       "true_diff": paired_by_level(true[name][r], level[r], pair[r], pgroup[r], lv[len("cam_"):], 0)[lv]["diff"]}
                for name in dec}
        log("[L3 galilean] camera vs ball boost: " + "; ".join(
            f"{lv}: d_eps {v['epsilon']['diff']}" for lv, v in out["camera_vs_ball"].items()))
    if cache_dir is not None and mcfg is not None:      # the law test: the model's forecast (audit H2)
        kinds = list(gc.get("forecast_pools") or gc["pools"])
        from physlaw.eval.l0_states import _pool_name
        names = sorted({_pool_name(kd, o, ph) for kd in kinds for o in "AB" for ph in ("pre", "post")})
        spaces = forecast_spaces(cache_dir, data_dir, recs, mcfg, names)
        out["forecast"] = galilean_forecast_spaces(spaces, Vpre, Vpost, ctxs, level, pair, pgroup, base, kinds, k, seeds,
                                                   alphas, nb, float(gc["max_floor_eps_err"]))
        log("[L3 galilean] forecast (" + str(out["forecast"].get("selected_space")) + "): " + "; ".join(
            f"{lv}: {v}" for lv, v in out["forecast"].get("verdict", {}).items()))
    if cache_dir is not None and gc.get("l2_contrasts"):
        out["l2_by_level"] = l2_by_level(cache_dir, data_dir, recs, level, gc["l2_contrasts"], log)
    return out


def l2_by_level(cache_dir: Path, data_dir: Path, recs, level, contrasts, log=print) -> dict:
    """L2 win rates (model and pixel baselines) separately on each boost level's twins. ``contrasts``: an L2 family block
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
                                          for c, by in res["contrasts"].items()}}
            log(f"[L3 galilean] L2 at boost level {lv}: {len(res['contrasts'])} contrasts")
    return out


# --------------------------------------------------------------------------- #
# mass independence (F7 drops, F6 pairs, F9 pendulum pairs)
# --------------------------------------------------------------------------- #


def mass_ratios(a: dict, masses: dict, rows: np.ndarray, law: str = "gravity") -> np.ndarray:
    """The law's heavy / light ratio per clip from per-object scalars a {o: [n]} and masses {o: [n]}:
      gravity      mass_ratio of the accelerations along g (a = the component along g, > 0 falling)
      friction     mass_ratio of the decelerations along the motion (a = the deceleration, > 0)
      oscillation  mass_period_ratio of the periods after the branch (a = the period)
    NaN when a ratio is not positive."""
    from physlaw.eval import laws
    objs = list(a)
    z, m = np.zeros((len(objs), 2)), lambda i: {"m": [masses[o][i] for o in objs]}
    if law == "gravity":
        f = lambda i: laws.gravity.stats({"v": z}, {"v": z, "a": [[0.0, -a[o][i]] for o in objs]},
                                         {**m(i), "g": [0.0, -1.0]})["mass_ratio"]
    elif law == "friction":           # sliding along +x, decelerating
        f = lambda i: laws.friction.stats({"v": z + [1.0, 0.0], "a": z}, {"v": z + [1.0, 0.0],
                                          "a": [[-a[o][i], 0.0] for o in objs]}, m(i))["mass_ratio"]
    elif law == "oscillation":
        f = lambda i: laws.oscillation.stats({"T": [np.nan] * len(objs)}, {"T": [a[o][i] for o in objs]},
                                             m(i))["mass_period_ratio"]
    else:
        raise KeyError(f"mass_ratios: law {law!r} (gravity | friction | oscillation)")
    r = np.array([f(i) for i in rows], float)
    return np.where((r > 0) & np.isfinite(r), r, np.nan)


def _log_median(r: np.ndarray, g: np.ndarray, n_boot: int, seed: int = 0):
    """Geometric median ratio (exp median log) with a group-bootstrap CI."""
    ok = np.isfinite(r)
    r, g = np.log(r[ok]), g[ok]
    if not len(r):
        return float("nan"), None
    gs = sorted(set(g))
    rng = np.random.default_rng(seed)
    bs = [np.median(np.concatenate([r[g == x] for x in rng.choice(gs, len(gs))])) for _ in range(n_boot)] if len(gs) > 1 else []
    return float(np.exp(np.median(r))), ([float(np.exp(np.quantile(bs, 0.025))), float(np.exp(np.quantile(bs, 0.975)))]
                                         if bs else None)


def decode_objects(real: dict, fc: dict, pools: dict, Y: dict, groups: np.ndarray, k: int, seeds: int, alphas):
    """The shared R1 instrument for a per-object quantity: ONE inverse-calibrated map (ridge.instrument_folds,
    cross-fitted by group) reads every object, its rows stacked (object o's pool ``pools[o]`` -> its target Y[o]),
    so both terms of a heavy / light ratio are read by the same instrument. Fitted on the clips whose targets are all
    finite. Returns {False: (real decodes, forecast decodes), True: (real decodes, moment-matched forecast decodes)},
    each {o: [n]} (NaN for rows of groups no fold held out)."""
    objs = list(pools)
    fit = np.isfinite(np.stack([Y[o] for o in objs])).all(0)
    dg = np.isin(groups, groups[fit])
    folds = instrument_folds({"x": tuple(real[pools[o]] for o in objs)}, tuple(Y[o] for o in objs), groups,
                             np.where(fit)[0], k, seeds, alphas)

    def dec(X, T=None):                                   # T: moment-match X's rows onto T's (diagnostic, _mm)
        out = np.full(len(groups), np.nan)
        out[dg] = apply_instrument(folds, X[dg], groups[dg],
                                   None if T is None else moment_match(X[dg], T[dg], groups[dg]))[:, 0]
        return out
    dec_real = {o: dec(real[pools[o]]) for o in objs}
    return {False: (dec_real, {o: dec(fc[pools[o]]) for o in objs}),
            True: (dec_real, {o: dec(fc[pools[o]], real[pools[o]]) for o in objs})}


def score_vs_mask(r: dict, n_boot: int, seed: int = 0) -> dict:
    """S(forecast) − S(mask_only), S = laws.ratio_score(t, t_floor, t_violator) on geometric-median ratios, with a
    group-bootstrap CI (the medians recomputed per draw). r: {"forecast", "floor", "violator", "mask": (ratios [m],
    groups [m])}."""
    from physlaw.eval import laws
    lg = {k: (np.log(np.where(v > 0, v, np.nan)), np.asarray(g)) for k, (v, g) in r.items()}

    def diff(sel):
        med = {}
        for k, (v, g) in lg.items():
            x = np.concatenate([v[g == h] for h in sel]) if len(sel) else np.array([])
            x = x[np.isfinite(x)]
            med[k] = np.exp(np.median(x)) if len(x) else np.nan
        return float(laws.ratio_score(med["forecast"], med["floor"], med["violator"])
                     - laws.ratio_score(med["mask"], med["floor"], med["violator"]))
    gs = sorted(set(np.concatenate([g for _, g in lg.values()]).tolist()))
    rng = np.random.default_rng(seed)
    bs = np.array([diff(rng.choice(gs, len(gs))) for _ in range(n_boot if len(gs) > 1 else 0)], float)
    bs = bs[np.isfinite(bs)]
    return {"diff": diff(gs), "diff_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            if len(bs) else None}


def mass_independence(cache_dir, data_dir, recs, labels, mcfg, cfg, family: str, log=print) -> dict:
    from physlaw.eval import laws
    mc = cfg["mass_independence"]
    fam = mc["families"][family]
    sel = [i for i, l in enumerate(labels) if fam.get("subfamily") is None or get(l, "physlaw.subfamily") == fam["subfamily"]]
    if len({recs[i]["group"] for i in sel}) < 2:
        return {"skipped": f"{len({recs[i]['group'] for i in sel})} {fam.get('subfamily') or ''} groups"}
    objs = list(fam.get("objects", ["A", "B"]))
    tname, law = fam["target"], fam.get("law", "gravity")
    pools = {o: f"roi_{o}.post" for o in objs}
    # pooling weights are cached for the whole family (the L0 file); keep the sub-family's rows afterwards
    spaces = {sp: tuple({p: v[sel] for p, v in d.items()} for d in rf)
              for sp, rf in forecast_spaces(cache_dir, data_dir, recs, mcfg, sorted(set(pools.values()))).items()}
    mw = _mask_weights(cache_dir, data_dir, recs, mcfg)          # the pooling weights alone (audit M0)
    mask = {p: mw[f"w.{p}"][sel].astype(np.float64) for p in set(pools.values())}
    recs, labels = [recs[i] for i in sel], [labels[i] for i in sel]
    groups = np.array([r["group"] for r in recs])
    twins = np.array([r["twin"] for r in recs])
    legal = np.isin(twins, fam["legal"])
    k, seeds = min(int(cfg["probe"]["k_folds"]), len(set(groups))), int(cfg["fold_seeds"])
    alphas = tuple(float(a) for a in cfg["probe"]["alphas"])
    nb = int(cfg["bootstrap"]["n"])
    Y = {o: np.array([object_target(l, o, tname, float(fam.get("window_s", 1.0))) for l in labels]) for o in objs}
    M = {o: np.array([l["physlaw"]["objects"][o]["mass"] for l in labels], float) for o in objs}
    min_sens = float(mc.get("min_log_sensitivity", np.log(1.1)))
    out = {"target": tname, "law": law, "subfamily": fam.get("subfamily"), "objects": objs, "n_groups": int(len(set(groups))),
           "min_log_sensitivity": min_sens, "spaces": {}}
    viol = [t for t in fam.get("violating", []) if (twins == t).any()]
    lr = np.where(legal)[0]
    dec_mask, _ = decode_objects(mask, mask, pools, Y, groups, k, seeds, alphas)[False]
    r_mask = mass_ratios(dec_mask, M, lr, law)
    med_mask, ci_mask = _log_median(r_mask, groups[lr], nb)
    out["mask_only"] = {"mass_ratio": med_mask, "mass_ratio_ci95": ci_mask, "n_clips": int(np.isfinite(r_mask).sum()),
                        "note": "the same instrument protocol on the ROI pooling weights alone (real post-event masks)"}
    for space, (real, fc) in spaces.items():
        decoded = decode_objects(real, fc, pools, Y, groups, k, seeds, alphas)
        for mm in (False, True):
            dec_real, dec_fc = decoded[mm]
            res = {}
            for name, dec, rows in (("forecast", dec_fc, legal), ("floor", dec_real, legal), ("true", Y, legal),
                                    *((f"real_{t}", dec_real, twins == t) for t in viol),
                                    *((f"true_{t}", Y, twins == t) for t in viol)):
                rr = np.where(rows)[0]
                r = mass_ratios(dec, M, rr, law)
                med, ci = _log_median(r, groups[rr], nb)
                res[name] = {"mass_ratio": med, "mass_ratio_ci95": ci, "n_clips": int(np.isfinite(r).sum())}
            t_fl, t_fc = res["floor"]["mass_ratio"], res["forecast"]["mass_ratio"]
            t_v = res[f"real_{viol[0]}"]["mass_ratio"] if viol else float("nan")
            sens = abs(np.log(t_v) - np.log(t_fl)) if np.isfinite(t_v) and np.isfinite(t_fl) else float("nan")
            ok = bool(np.isfinite(sens) and sens >= min_sens)
            res["score"] = {"S": float(laws.ratio_score(t_fc, t_fl, t_v)) if ok else None,
                            "S_mask_only": float(laws.ratio_score(med_mask, t_fl, t_v)) if ok else None,
                            "sensitivity_log": float(sens), "measurable": ok}
            if ok:
                vr = np.where(twins == viol[0])[0]
                vm = score_vs_mask({"forecast": (mass_ratios(dec_fc, M, lr, law), groups[lr]),
                                    "floor": (mass_ratios(dec_real, M, lr, law), groups[lr]),
                                    "violator": (mass_ratios(dec_real, M, vr, law), groups[vr]),
                                    "mask": (r_mask, groups[lr])}, nb)
                res["score"].update(S_minus_mask=vm["diff"], S_minus_mask_ci95=vm["diff_ci95"],
                                    beyond_mask=bool(vm["diff_ci95"] and vm["diff_ci95"][0] > 0))
            res["score"]["verdict"] = ("unmeasurable" if not ok else
                                       "beyond the mask-only score" if res["score"]["beyond_mask"] else
                                       "not beyond the mask-only score (the ROI masks alone do as well)")
            vname = space + ("_mm" if mm else "")
            out["spaces"][vname] = res
            log(f"[L3 mass] {vname}: forecast {t_fc:.3f} floor {t_fl:.3f} violator {t_v:.3f} mask-only {med_mask:.3f} "
                f"S {res['score']['S']} S_mask {res['score']['S_mask_only']}: {res['score']['verdict']}")
    return out


# --------------------------------------------------------------------------- #
# transfer
# --------------------------------------------------------------------------- #


def transfer(recs, labels, A, pools, layers, cfg, family: str, log=print) -> dict:
    tc = cfg["transfer"][family]
    cond = np.array([str(get(l, tc["split_key"])) for l in labels])
    a, b = (str(x) for x in tc["conditions"])
    groups = np.array([r["group"] for r in recs])
    keep = np.isin([r["twin"] for r in recs], tc["twins"]) if tc.get("twins") else np.ones(len(recs), bool)
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
            c, f, *_ = fit_select(Xc, Y, groups, rs, alphas, k, 0)
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
            out["galilean"] = galilean(recs, labels, A, pools, layers, cfg, cache_dir, data_dir, log, mcfg=mcfg)
        elif an == "mass_independence":
            out["mass_independence"] = mass_independence(cache_dir, data_dir, recs, labels, mcfg, cfg, family, log)
        elif an == "transfer":
            out["transfer"] = transfer(recs, labels, A, pools, layers, cfg, family, log)
        else:
            raise KeyError(an)
    return out


def table(res: dict) -> str:
    f = lambda x, s=".3f": "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, s)
    c = lambda ci: f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    rows = []
    if res.get("stand_in"):
        rows += [f"**Stand-in run:** {res['stand_in']}", ""]
    for an in ("galilean", "mass_independence", "transfer"):
        if "skipped" in res.get(an, {}):
            rows += [f"{an}: skipped ({res[an]['skipped']})", ""]
    g = res.get("galilean")
    if g and "skipped" not in g:
        rows += ["## Galilean invariance (decoded velocities, R1 instrument trained on every boost level)", "",
                 "The real-video rows are the measurability floor (real videos are invariant by construction); the "
                 "law test is the forecast table below.", "",
                 f"Clips per level {g['levels']}; cells searched {g['n_cells']}; level unmeasurable when the floor's median "
                 f"|eps_hat - eps| > {g['max_floor_eps_err']}.", "",
                 "| level | R² v pre / post | floor median abs eps err | measurable |", "|---|---|---|---|"]
        for lv, fl in g["floor"].items():
            rows.append(f"| {lv} | {f(fl['r2_velocity_pre'])} / {f(fl['r2_velocity_post'])} | {f(fl['eps_abs_err_median'])} | "
                        f"{fl['measurable']} |")
        rows += ["", "laws.galilean.stats against the u = 0 partner (law: 0; d_theta in units of pi): decoded (true).", "",
                 "| level | n paired | d_epsilon [95% CI] | d_theta [95% CI] | measurable |", "|---|---|---|---|---|"]
        for lv, v in g["galilean"]["decoded"].items():
            t_ = g["galilean"]["true"][lv]
            rows.append(f"| {lv} | {v['n_paired']} | {f(v['d_epsilon'])}{c(v['d_epsilon_ci95'])} ({f(t_['d_epsilon'])}) | "
                        f"{f(v['d_theta'])}{c(v['d_theta_ci95'])} ({f(t_['d_theta'])}) | {v['measurable']} |")
        rows += ["", "Paired difference boosted - its u = 0 partner (law: 0), against the base level's decoding error.", "",
                 "| statistic | level | n (paired) | mean [95% CI] | true mean | diff [95% CI] | abs diff / base err |",
                 "|---|---|---|---|---|---|---|"]
        for name, st in g["stats"].items():
            for lv, v in st["by_level"].items():
                rows.append(f"| {name} | {lv} | {v['n']} ({v.get('n_paired', '-')}) | {f(v['mean'])}{c(v['ci95'])} | "
                            f"{f(v['true_mean'])} | {f(v.get('diff'), '+.3f')}{c(v.get('diff_ci95'))} | "
                            f"{f(v.get('abs_diff_over_base_err'), '.2f')} |")
        if g.get("camera_vs_ball"):
            rows += ["", "Camera boost - its ball-boost twin (same scene, rung, u; identical ball image motion): decoded "
                     "(true) paired difference, law 0.", "", "| level | statistic | n paired | diff [95% CI] | true diff |",
                     "|---|---|---|---|---|"]
            for lv, by in g["camera_vs_ball"].items():
                for name, v in by.items():
                    rows.append(f"| {lv} | {name} | {v.get('n_paired')} | {f(v.get('diff'), '+.3f')}{c(v.get('diff_ci95'))} | "
                                f"{f(v.get('true_diff'), '+.3f')} |")
        fcg = g.get("forecast")
        if fcg and fcg.get("spaces"):
            rows += ["", f"**Law test (audit H2): the model's FORECAST of boosted vs unboosted collisions**, same instrument "
                     f"as its real-video floor; space {fcg['selected_space']} ({fcg['selection']}). d = mean over pairs; "
                     "diff = forecast − floor (paired, scene bootstrap).", "",
                     "| space | level (vs) | n | d_eps forecast / floor | d_eps diff [CI] | d_theta forecast / floor | "
                     "d_theta diff [CI] | verdict |", "|---|---|---|---|---|---|---|---|"]
            for sp, sv in fcg["spaces"].items():
                for lv, v in sv["levels"].items():
                    de, dt = v["d_epsilon"], v["d_theta"]
                    rows.append(f"| {sp} | {lv} ({v['vs']}) | {v['n_paired']} | {f(de['forecast'])} / {f(de['floor'])} | "
                                f"{f(de['diff'], '+.3f')}{c(de['diff_ci95'])} | {f(dt['forecast'])} / {f(dt['floor'])} | "
                                f"{f(dt['diff'], '+.3f')}{c(dt['diff_ci95'])} | {v['verdict']} |")
        if g.get("l2_by_level"):
            rows += ["", "L2 per boost level: see l3.json `galilean.l2_by_level`."]
        rows.append("")
    m = res.get("mass_independence")
    if m and "skipped" not in m:
        rows += ["## Mass independence", "", f"Target `{m['target']}` ({m['subfamily'] or 'all'}); law {m.get('law', 'gravity')}: "
                 f"heavy / light ratio (geometric median, law 1); S = ratio_score(forecast, floor, real violator); "
                 f"S_mask = the same for the mask-only instrument (ROI pooling weights alone: "
                 f"{f((m.get('mask_only') or {}).get('mass_ratio'))}); the forecast counts only if S − S_mask has a CI "
                 f"above 0 (audit M0); {m['n_groups']} groups", "",
                 "| space | forecast [95% CI] | floor | true | violator real / true | S | S_mask | S − S_mask [CI] | verdict |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for sp, d in m["spaces"].items():
            vk = [k_ for k_ in d if k_.startswith("real_")]
            vr = " / ".join(f"{f(d[k_]['mass_ratio'])} / {f(d['true_' + k_[5:]]['mass_ratio'])}" for k_ in vk) or "-"
            sc = d["score"]
            rows.append(f"| {sp} | {f(d['forecast']['mass_ratio'])}{c(d['forecast']['mass_ratio_ci95'])} | "
                        f"{f(d['floor']['mass_ratio'])} | {f(d['true']['mass_ratio'])} | {vr} | {f(sc['S'], '+.3f')} | "
                        f"{f(sc.get('S_mask_only'), '+.3f')} | {f(sc.get('S_minus_mask'), '+.3f')}{c(sc.get('S_minus_mask_ci95'))} | "
                        f"{sc.get('verdict', '')} |")
        rows.append("")
    t = res.get("transfer")
    if t and "skipped" not in t:
        rows += ["## Transfer", "", f"Split `{t['split_key']}` {t['conditions']}, clips {t['n']}, cells searched {t['n_cells']}", "",
                 "| target | direction | R² transfer | R² within target condition | gap |", "|---|---|---|---|---|"]
        for tn, d in t["targets"].items():
            for k, v in d.items():
                rows.append(f"| {tn} | {k} | {v['r2_transfer']:.3f} | {v['r2_within_dst']:.3f} | {v['gap']:+.3f} |")
    return "\n".join(rows)
