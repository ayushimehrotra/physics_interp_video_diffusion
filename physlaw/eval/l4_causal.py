"""L4 causal propagation (spec §5.6): push the model's mass-ratio representation of object A before the event and
check that its forecast changes as momentum conservation predicts.

1. Direction, from object A's pre-event ROI pool (stage 4 cache, every clip) at the injection layer.
   ``direction: generative`` (default): u ∝ sum_i (y_i - ybar)(x_i - xbar), the activations regressed on the target
   (per-unit covariance), normalised, with a 1-D projection readout y ~ b0 + gain (x . u). ``ridge``: the
   discriminative ridge weight vector (raw activation units), normalised; the predecessor found it nearly inert
   under steering (velocity survived 12 INLP rounds) while the generative direction moved the readout 1.40x as much.
   CV R² of both types (generative: of the 1-D readout) and |cos| between them are reported. ``layer: auto`` takes
   the layer with the best grouped-CV ridge R² for the target. Source family ``direction_data`` (default ``f1``; ``null`` = the evaluated family, F3,
   cross-fitted by group). F3's twins share their pre-event frames and both balls look identical, so mass is
   invisible before contact there and a direction fitted on F3 is likely noise; F1 shows mass through size and look.
   ``direction_target: residual`` (default) fits log(m_A/m_B) residualised on log(r_A/r_B) (OLS on labels only,
   training clips of the fold), i.e. "mass beyond size"; ``raw`` fits log(m_A/m_B) itself. The CV R² of BOTH the
   raw and the residualised direction is reported per fold (``cv_r2_raw``, ``cv_r2_residual``).
2. Intervention. A torch forward hook on the injection block adds alpha * sd * u (sd = sd of the POOLED projection on
   u over the direction's training clips; alpha in ``alphas_sd``) to every token of object A's binary ROI mask (each
   token gets the full shift, no pooling weights) in the pre-event token-frames,
   then forecast() is rerun. Encoders: the hook acts on the context encoder only. DiTs: on the context tokens at every
   sampling step (both CFG branches); the same run gives the generated video.
3. Readout. Post-event velocities decoded from the forecast with probes trained on REAL post-event frames (as L1:
   frame pool = leak-free, roi = twin's own post ROI; ``_mm`` = moment-matched to the cached unperturbed forecasts),
   cross-fitted by group.
4. Metrics. The injected shift implies dlog q = alpha * sd * |w| (|w| = target units per unit along u; residual
   target: the change of log mass ratio at fixed size).
   Primary, restitution-free (``deflection_slope``): with a = -dv_A'.n and b = dv_B'.n (decoded post minus TRUE pre
   velocity, n = contact normal; both > 0 for a real collision), momentum gives b / a = m_A / m_B for any e. Within
   each (group, twin, draw) the alpha = 0 forecast and the steered ones give points (dlog q, log(b/a)); the pooled
   within-unit slope (group-bootstrap CI) should be 1 for the mass direction and ~0 for the controls. Points with
   a <= 0 or b <= 0 are excluded and counted (``n_excluded``).
   Passive-propagation control: the same slope (``passive_slope``) and the mean |change of v_B'.n| when the decoding
   probe is applied to the unperturbed forecast's post-event features plus the same shift alpha * sd * u (the edit's
   linear arithmetic propagated through no model computation; needs equal widths, else None). Reported
   ``slope_over_passive`` and ``change_over_passive_B_n``: a causal effect must exceed passive (the predecessor's
   probe-direction edit scored 1.02-1.04 by pure arithmetic).
   Secondary (needs e): with q0 = the direction's reading of the clip (residual target: plus the size part from the
   clip's radii) and q' = q0 exp(dlog q), momentum at the twin's restitution e predicts
   d(v_B'.n) = (1 + e) (q'/(1 + q') - q0/(1 + q0)) (v_A - v_B).n and d(v_A'.n) = -(1 + e) (1/(1 + q') - 1/(1 + q0))
   (v_A - v_B).n. Score = Pearson r between predicted and observed changes over (group, twin, alpha != 0) with a
   group-bootstrap CI; gain = least-squares observed / predicted. F3 has e = 1 for every twin, so the e-dependence is
   harmless there. Specificity = mean |change of v_B'.t| / mean |change of v_B'.n| (t = tangent; momentum predicts
   no change).
5. Controls, same alphas, scored against the SAME predicted changes: ``random_dir`` (random unit direction, shift
   norm matched to the mass shift), ``random_subspace`` (rank-matched: a random rank-1 subspace, alpha in its own
   projection sd), ``pc_matched`` (the principal component of the training activations whose variance is closest to
   the mass direction's projection variance, alpha in its sd).
6. Output readout (generators, the steering experiment): each steered video is read out with the track B readout
   (eval/outputs.READOUT_SELF); reported per (group, condition, alpha): measurable, measured d(v_B.n) of the
   generated collision, and whether its change from alpha = 0 has the sign momentum predicts.
Interventions are cached per group in ``work`` (resumable).
"""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

from physlaw.eval.l3_invariance import fit_select, forecast_spaces, load_family, restitution
from physlaw.probes.ridge import DEFAULT_ALPHAS, _SVDRidge, grouped_kfold

POST_POOLS = ("frame.post", "roi_A.post", "roi_B.post")


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #


def blocks(m):
    """The adapter's transformer blocks, in order (layer key l = output of blocks[l])."""
    if hasattr(m, "tr"):                                             # diffusers DiTs
        return getattr(m.tr, "transformer_blocks", None) or m.tr.blocks
    if hasattr(m, "model") and hasattr(m.model, "encoder"):          # transformers V-JEPA 2 / ViT encoders
        return m.model.encoder.layer
    raise NotImplementedError(f"L4: no block list known for {type(m).__name__}")


def block_index(layer: str) -> int:
    """Layer key -> block index: "12" -> 12, "snr+8/L15" -> 15."""
    return int(str(layer).rsplit("/L", 1)[-1])


@contextmanager
def steer(m, layer: str, idx, delta):
    """While active, add ``delta`` [D] to token positions ``idx`` of the output of block ``layer``."""
    import torch
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, idx.to(h.device)] += torch.as_tensor(delta, device=h.device, dtype=h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
    handle = blocks(m)[block_index(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# directions
# --------------------------------------------------------------------------- #


def log_q(labels) -> float:
    o = labels["physlaw"]["objects"]
    return float(np.log(o["A"]["mass"] / o["B"]["mass"]))


def log_r(labels) -> float:
    o = labels["physlaw"]["objects"]
    return float(np.log(o["A"]["radius"] / o["B"]["radius"]))


def residualise(y: np.ndarray, x: np.ndarray, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """y minus its OLS fit on [1, x] (fitted on ``rows``; labels only): (residual for every row, coef [b0, b1]).
    With y = log(m_A/m_B) and x = log(r_A/r_B) the residual is "mass beyond size" (density ratio, up to noise)."""
    rows = np.arange(len(y)) if rows is None else rows
    Xd = np.c_[np.ones(len(x)), x]
    if np.ptp(x[rows]) < 1e-12:                      # constant radius ratio (F3): only the mean comes off
        coef = np.array([y[rows].mean(), 0.0])
    else:
        coef = np.linalg.lstsq(Xd[rows], y[rows], rcond=None)[0]
    return y - Xd @ coef, coef


def fit_direction(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas=DEFAULT_ALPHAS, k: int = 5) -> dict:
    """Ridge X -> y; u = raw-unit weight direction; sd = sd of X @ u; gain = |w| (target units per unit along u)."""
    r2 = {a: _cv_r2(X, y, groups, a, k) for a in alphas}
    best = max(alphas, key=lambda a: r2[a])
    r2 = r2[best]
    m = _SVDRidge(X, y.reshape(-1, 1), True)
    w = m.coef(best)[:, 0] / m.sx
    u = w / np.linalg.norm(w)
    return {"u": u, "sd": float((X @ u).std()), "gain": float(np.linalg.norm(w)), "alpha": float(best),
            "cv_r2": r2, "predict": lambda Z, m=m, a=best: m.predict(Z, a)[:, 0]}


def _generative(X: np.ndarray, y: np.ndarray):
    """u ∝ sum_i (y_i - ybar)(x_i - xbar) (the activations regressed on the label, per unit), normalised; then a
    1-D OLS readout y ~ b0 + gain * (x @ u). Returns (u, b0, gain)."""
    yc = y - y.mean()
    c = (X - X.mean(0)).T @ yc
    u = c / max(np.linalg.norm(c), 1e-300)
    p = X @ u
    pc = p - p.mean()
    gain = float(pc @ yc / max(pc @ pc, 1e-300))
    if gain < 0:                                      # orient u so that +u raises the target
        u, gain = -u, -gain
    return u, float(y.mean() - gain * (X @ u).mean()), gain


def fit_generative(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5) -> dict:
    """Generative direction (:func:`_generative`); cv_r2 = grouped-CV R² of the 1-D projection readout (u and the
    readout refitted on each training split). Same keys as :func:`fit_direction` (gain = target units per unit
    along u). The predecessor found the ridge (discriminative) direction nearly inert under steering, the
    generative one moved the readout 1.40x as much."""
    from physlaw.probes.ridge import r2_score
    p = np.zeros(len(y))
    for te in grouped_kfold(groups, min(k, len(set(groups))), 10_000):
        tr = np.setdiff1d(np.arange(len(y)), te)
        u_, b0_, g_ = _generative(X[tr], y[tr])
        p[te] = b0_ + g_ * (X[te] @ u_)
    u, b0, gain = _generative(X, y)
    return {"u": u, "sd": float((X @ u).std()), "gain": gain, "alpha": float("nan"), "cv_r2": r2_score(y, p),
            "predict": lambda Z, u=u, b0=b0, gain=gain: b0 + gain * (np.asarray(Z) @ u)}


def _cv_r2(X, y, groups, a, k):
    from physlaw.probes.ridge import r2_score
    p = np.zeros(len(y))
    for te in grouped_kfold(groups, min(k, len(set(groups))), 10_000):
        tr = np.setdiff1d(np.arange(len(y)), te)
        p[te] = _SVDRidge(X[tr], y[tr].reshape(-1, 1), True).predict(X[te], a)[:, 0]
    return r2_score(y, p)


def control_directions(X: np.ndarray, d: dict, n: dict, seed: int) -> dict[str, list[tuple[np.ndarray, float]]]:
    """{condition: [(unit vector, shift per alpha unit)]}: the mass direction and its controls."""
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    out = {"mass": [(d["u"], d["sd"])]}
    rand = lambda: (lambda r: r / np.linalg.norm(r))(rng.standard_normal(D))
    if n.get("random_dir"):
        out["random_dir"] = [(rand(), d["sd"]) for _ in range(int(n["random_dir"]))]
    if n.get("random_subspace"):
        out["random_subspace"] = [(r, float((X @ r).std())) for r in (rand() for _ in range(int(n["random_subspace"])))]
    if n.get("pc_matched"):
        Xc = X - X.mean(0)
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        var = s ** 2 / len(X)
        order = np.argsort(np.abs(var - d["sd"] ** 2))[: int(n["pc_matched"])]
        out["pc_matched"] = [(vt[i], float(np.sqrt(var[i]))) for i in order]
    return out


# --------------------------------------------------------------------------- #
# intervention runs (GPU)
# --------------------------------------------------------------------------- #


def _pool_forecast(acts: dict, W: dict, kt: int) -> dict:
    """forecast tokens {key: [T_post, gh, gw, D]} -> {space: {pool: [D]}} with the clip's pooling weights
    W {pool: [T_tok, gh, gw]} restricted to the forecast's token-frames (>= kt)."""
    out = {}
    for key, F in acts.items():
        F = F.float().cpu().numpy() if hasattr(F, "cpu") else np.asarray(F, np.float32)
        sp = "final" if str(key) == "-1" else str(key)
        flat = F.reshape(-1, F.shape[-1])
        out[sp] = {}
        for p in POST_POOLS:
            w = W[p][kt: kt + F.shape[0]].reshape(-1)
            out[sp][p] = (w / max(w.sum(), 1e-12)) @ flat
    return out


def run_interventions(m, mcfg, data_dir: Path, groups: list[str], twins_of: dict, plans: dict, layer: str,
                      alphas: list, work: Path, readout: bool, force: bool = False, log=print):
    """For each group: forecast() under every (condition, draw, alpha) of plans[group] ({cond: [(u, sd)]});
    pooled forecasts per twin -> work/<group>.npz, steered videos' track B readouts -> work/<group>_readout.json."""
    import torch
    from physlaw.data.io import load_clip
    from physlaw.eval import outputs as O
    from physlaw.tokens.pool import pool_weights
    from physlaw.tokens.project import project_masks, token_frame_phase
    work.mkdir(parents=True, exist_ok=True)
    diffusion = mcfg.get("kind") == "diffusion"
    for gi, g in enumerate(groups):
        f = work / f"{g}.npz"
        if f.exists() and not force:
            continue
        clips = {t: load_clip(data_dir / g / t) for t in twins_of[g]}
        c0 = clips[twins_of[g][0]]
        fr, k = m.input_frames(c0.labels), m.context_frames(c0.labels)
        kt = m.n_context_tokens(k) if hasattr(m, "n_context_tokens") else k // m.token_grid.tubelet
        x = m.to_tensor(c0.frames[fr])
        objA = project_masks(c0.masks, fr, m.token_grid)["A"].roi[:kt]
        idx = np.flatnonzero(objA.reshape(-1))
        Ws = {}
        for t, c in clips.items():
            obj = project_masks(c.masks, fr, m.token_grid)
            Ws[t] = pool_weights(obj, token_frame_phase(fr, m.token_grid, c.pl["first_post_contact_frame"]))
        rec, rd = {}, {}
        runs = [("base", 0, 0.0, None)] + [(cond, j, a, (u, s)) for cond, dirs in plans[g].items()
                                          for j, (u, s) in enumerate(dirs) for a in alphas if a != 0]
        for cond, j, a, us in runs:
            ctx = nullcontext() if us is None else steer(m, layer, idx, a * us[1] * us[0])
            with ctx:
                with torch.no_grad():
                    if diffusion:
                        acts, vid = m.forecast(x, k, seed=0, return_video=True)
                    else:
                        acts, vid = m.forecast(x, k), None
            for t in clips:
                pooled = _pool_forecast(acts, Ws[t], kt)
                for sp, d in pooled.items():
                    for p, v in d.items():
                        rec[f"{cond}|{j}|{a:g}|{t}|{sp}|{p}"] = v.astype(np.float32)
            if vid is not None and readout:
                r = O.READOUT_SELF[c0.pl["family"]](vid, c0, float(m.fps), gate="loose")
                rd[f"{cond}|{j}|{a:g}"] = O._row(r)
                import imageio.v3 as iio
                iio.imwrite(work / f"{g}_{cond}{j}_a{a:g}.mp4", vid, fps=m.fps, codec="libx264", quality=9)
        np.savez(f, **rec)
        if rd:
            (work / f"{g}_readout.json").write_text(json.dumps(rd, indent=1, default=float))
        log(f"[L4] {g} ({gi + 1}/{len(groups)}): {len(runs)} forecasts")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def _pred_change(labels, q0: float, dlogq: float, e: float) -> tuple[float, float]:
    pl = labels["physlaw"]
    n = np.asarray(pl["contact_normal"][:2], float)
    n /= np.linalg.norm(n)
    vn = float((np.subtract(pl["pre_velocity"]["A"][:2], pl["pre_velocity"]["B"][:2])) @ n)
    q1 = q0 * np.exp(dlogq)
    return ((1 + e) * (q1 / (1 + q1) - q0 / (1 + q0)) * vn, -(1 + e) * (1 / (1 + q1) - 1 / (1 + q0)) * vn)


def _within_slope(y: np.ndarray, x: np.ndarray, u: np.ndarray) -> float:
    yc, xc = y.astype(float).copy(), x.astype(float).copy()
    for k in set(u):
        s_ = u == k
        yc[s_] -= y[s_].mean()
        xc[s_] -= x[s_].mean()
    return float((xc @ yc) / (xc @ xc)) if xc @ xc > 1e-12 else float("nan")


def deflection_slope(x, a_hat, b_hat, unit, group, n_boot: int = 1000, seed: int = 0) -> dict:
    """The e-free momentum test. Points (injected dlog q = x, decoded deflections a_hat = -dv_A.n, b_hat = dv_B.n);
    momentum gives b / a = m_A / m_B whatever e, so log(b_hat / a_hat) should move one-for-one with x. Pooled
    within-``unit`` slope (each (group, twin[, draw]) demeaned) with a bootstrap CI over ``group``; points with
    a_hat <= 0 or b_hat <= 0 (no real collision) are excluded and counted. Law: 1; unrelated readout: 0."""
    x, a_hat, b_hat, unit, group = map(np.asarray, (x, a_hat, b_hat, unit, group))
    ok = (a_hat > 0) & (b_hat > 0)
    out = {"n": int(len(x)), "n_excluded": int((~ok).sum()), "slope": float("nan"), "slope_ci95": None}
    if ok.sum() < 2:
        return out
    x, y, unit, group = x[ok], np.log(b_hat[ok] / a_hat[ok]), unit[ok], group[ok]
    out["slope"] = _within_slope(y, x, unit)
    gs = sorted(set(group))
    if len(gs) > 1 and n_boot > 0:
        idx_of = {g: np.where(group == g)[0] for g in gs}
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(n_boot):
            pick = rng.choice(len(gs), len(gs))
            ii = np.concatenate([idx_of[gs[j]] for j in pick])
            uu = np.concatenate([np.char.add(f"{k}#", unit[idx_of[gs[j]]].astype(str)) for k, j in enumerate(pick)])
            bs.append(_within_slope(y[ii], x[ii], uu))
        out["slope_ci95"] = [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]
    return out


def _corr(a, b):
    a, b = np.asarray(a) - np.mean(a), np.asarray(b) - np.mean(b)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def _family_dir(name) -> Path:
    """``direction_data`` / ``probe_data``: a data path, or a bare family name ("f1" -> data/f1)."""
    from physlaw.config import resolve_path
    p = resolve_path(name)
    return p if p.exists() or "/" in str(name) else resolve_path(Path("data") / str(name))


def _sibling(cache_dir: Path, family: str, mcfg) -> Path:
    """Another family's stage 4 cache: next to this one, else the repo's cache/<model>/<family>."""
    p = cache_dir.parent / family
    return p if p.exists() else Path(__file__).resolve().parents[2] / "cache" / mcfg.name / family


def run(cache_dir: Path, data_dir: Path, cfg, mcfg, work: Path, force: bool = False, log=print) -> dict:
    alphas = [float(a) for a in cfg["alphas_sd"]]
    ridge_alphas = tuple(float(a) for a in cfg["probe"]["alphas"])
    k = int(cfg["probe"]["k_folds"])
    recs, labels, A, pools, layers = load_family(cache_dir, data_dir)
    lab = {r["clip"]: l for r, l in zip(recs, labels)}
    groups_all = sorted({r["group"] for r in recs})
    twins_of = {g: [r["twin"] for r in recs if r["group"] == g and (not cfg.get("twins") or r["twin"] in cfg["twins"])]
                for g in groups_all}
    first = {g: next(i for i, r in enumerate(recs) if r["group"] == g) for g in groups_all}

    # direction source: every clip, labelled with its own mass ratio, CV grouped by twin group. Default F1 (mass is
    # visible there through size and look); target "residual" = log(m_A/m_B) residualised on log(r_A/r_B) (mass
    # beyond size). F3 (direction_data: null) shares the pre-event frames across a group's twins, so its pre-event
    # activations carry ~no mass information.
    same = not cfg.get("direction_data")
    if same:
        sA, slab, spools, slayers = A, labels, pools, layers
        sgroups = np.array([r["group"] for r in recs])
    else:
        sdata = _family_dir(cfg["direction_data"])
        srecs, slab, sA, spools, slayers = load_family(_sibling(cache_dir, sdata.name, mcfg), sdata)
        sgroups = np.array([r["group"] for r in srecs])
    srows = np.arange(len(sgroups))
    sy_raw = np.array([log_q(slab[i]) for i in srows])
    sr = np.array([log_r(slab[i]) for i in srows])
    tmode = cfg.get("direction_target", "residual")
    dkind = cfg.get("direction", "generative")
    if dkind not in ("generative", "ridge"):
        raise ValueError(f"direction {dkind!r}: generative or ridge")
    if tmode not in ("raw", "residual"):
        raise ValueError(f"direction_target {tmode!r}: raw or residual")
    sy = residualise(sy_raw, sr)[0] if tmode == "residual" else sy_raw
    pa = spools.index("roi_A.pre")
    cand = [l for l in slayers if l != "-1"] if cfg["layer"] == "auto" else [str(cfg["layer"])]
    layer, _, _ = fit_select({l: sA[srows, pa, slayers.index(l)].astype(np.float64) for l in cand}, sy, sgroups,
                             srows, ridge_alphas, k)
    SX = sA[srows, pa, slayers.index(layer)].astype(np.float64)
    EX = A[:, pools.index("roi_A.pre"), layers.index(layer)].astype(np.float64) if layer in layers else None
    log(f"[L4] injection layer {layer} (searched {len(cand)}); direction from "
        f"{'this family' if same else cfg['direction_data']}, target {tmode}")

    # folds over the evaluated groups (directions cross-fitted when fitted on this family)
    groups = groups_all[: cfg.get("max_groups") or None]
    folds = grouped_kfold(np.array(groups), min(k, len(groups)), 0) if len(groups) >= 2 else [np.arange(len(groups))]
    plans, dirs, fold_of = {}, [], {}
    for fi, te in enumerate(folds):
        test = [groups[i] for i in te]
        tr = np.array([g not in test for g in sgroups]) if same else np.ones(len(sgroups), bool)
        res_y, coef = residualise(sy_raw, sr, np.where(tr)[0])
        fits = {(kind, tg): (fit_generative(SX[tr], yy[tr], sgroups[tr], k) if kind == "generative"
                             else fit_direction(SX[tr], yy[tr], sgroups[tr], ridge_alphas, k))
                for kind in ("generative", "ridge") for tg, yy in (("raw", sy_raw), ("residual", res_y))}
        d = fits[dkind, tmode]
        d["cv_r2_raw"], d["cv_r2_residual"], d["target"], d["direction"] = (
            fits[dkind, "raw"]["cv_r2"], fits[dkind, "residual"]["cv_r2"], tmode, dkind)
        d["cv_r2_by"] = {f"{kd}|{tg}": f["cv_r2"] for (kd, tg), f in fits.items()}
        d["cos_generative_ridge"] = float(abs(fits["generative", tmode]["u"] @ fits["ridge", tmode]["u"]))
        d["radius_coef"] = coef.tolist()
        # q0: the direction's reading of the clip (residual target: plus the size part from the clip's radii)
        size = (lambda i: coef[0] + coef[1] * log_r(labels[i])) if tmode == "residual" else (lambda i: 0.0)
        d["q0"] = {g: float(np.exp(d["predict"](EX[[first[g]]])[0] + size(first[g]))) if EX is not None
                   else float(np.exp(log_q(labels[first[g]]))) for g in test}
        dirs.append(d)
        cd = control_directions(SX[tr], d, cfg.get("controls") or {}, seed=fi)
        for g in test:
            plans[g], fold_of[g] = cd, fi
        log(f"[L4] fold {fi}: direction CV R² {d['cv_r2']:.3f} (raw {d['cv_r2_raw']:.3f}, residual "
            f"{d['cv_r2_residual']:.3f}), sd {d['sd']:.3f}, gain {d['gain']:.3f}")

    missing = [g for g in groups if force or not (work / f"{g}.npz").exists()]
    if missing:
        from physlaw.models.registry import load_adapter
        m = load_adapter(mcfg, device=cfg.get("device", "cuda"))
        run_interventions(m, mcfg, data_dir, missing, twins_of, plans, layer, alphas, work,
                          bool(cfg.get("generate_readout", True)), force, log)
        del m

    # decoding probes: real post-event frames of the training groups, per space and pool; mm from cached forecasts.
    # probe_data: train them on another family instead (all its groups), e.g. when the evaluated set is tiny
    if cfg.get("probe_data"):
        pdata = _family_dir(cfg["probe_data"])
        precs, plabels = load_family(_sibling(cache_dir, pdata.name, mcfg), pdata)[:2]
        spaces = forecast_spaces(_sibling(cache_dir, pdata.name, mcfg), pdata, precs, mcfg, list(POST_POOLS))
        grp = np.array(["probe:" + r["group"] for r in precs])               # never a test group
    else:
        plabels = labels
        spaces = forecast_spaces(cache_dir, data_dir, recs, mcfg, list(POST_POOLS))
        grp = np.array([r["group"] for r in recs])
    Yv = {o: np.array([l["physlaw"]["post_velocity"][o][:2] for l in plabels], float) for o in "AB"}
    Z = {g: dict(np.load(work / f"{g}.npz")) for g in groups}
    conds = sorted({key.split("|")[0] for key in Z[groups[0]]} - {"base"})
    out = {"layer": layer, "direction_family": cfg.get("direction_data") or data_dir.name, "direction_target": tmode,
           "probe_family": cfg.get("probe_data") or data_dir.name, "alphas_sd": alphas,
           "n_groups": len(groups), "twins": sorted({t for g in groups for t in twins_of[g]}),
           "direction": dkind,
           "directions": [{k_: d[k_] for k_ in ("sd", "gain", "alpha", "cv_r2", "cv_r2_raw", "cv_r2_residual", "cv_r2_by",
                                                "cos_generative_ridge", "radius_coef")}
                          for d in dirs],
           "variants": {}}
    if cfg.get("stand_in"):
        out["stand_in"] = cfg["stand_in"]
    for sp, (real, fc) in spaces.items():
        for vname, pmap, mm in (("frame", {"A": "frame.post", "B": "frame.post"}, False),
                                ("frame_mm", {"A": "frame.post", "B": "frame.post"}, True),
                                ("roi", {"A": "roi_A.post", "B": "roi_B.post"}, False),
                                ("roi_mm", {"A": "roi_A.post", "B": "roi_B.post"}, True)):
            probes = {}
            for fi, te in enumerate(folds):
                test = {groups[i] for i in te}
                tr = np.where(~np.isin(grp, list(test)))[0]
                for o in "AB":
                    X, Xf = real[pmap[o]], fc[pmap[o]]
                    _, f, _ = fit_select({"x": X}, Yv[o], grp, tr, ridge_alphas, k)
                    tf = ((lambda z, X=X, Xf=Xf, tr=tr: (z - Xf[tr].mean(0)) / (Xf[tr].std(0) + 1e-8)
                           * (X[tr].std(0) + 1e-8) + X[tr].mean(0)) if mm else (lambda z: z))
                    probes[fi, o] = (f, tf)
            def dec(g, key, o, pmap=pmap, probes=probes, sp=sp):
                f, tf = probes[fold_of[g], o]
                return f(tf(Z[g][f"{key}|{sp}|{pmap[o]}"][None].astype(np.float64)))[0]

            def passive(g, t, o, delta, pmap=pmap, probes=probes, sp=sp):
                """Decoded post velocity if the edit only shifted the post-event features by the same vector (no
                model computation): probe(base forecast features + delta). None when the spaces differ in width."""
                z = Z[g][f"base|0|0|{t}|{sp}|{pmap[o]}"].astype(np.float64)
                if z.shape[-1] != len(delta):
                    return None
                f, tf = probes[fold_of[g], o]
                return f(tf((z + delta)[None]))[0]
            res = {}
            for cond in conds:
                P, Ob, PA, ObA, tang, by_g = [], [], [], [], [], {}
                sx, sa, sb, su, sg = [], [], [], [], []           # e-free deflection-ratio points
                pa_, pb_, passB, has_passive = [], [], [], True    # passive-propagation control
                for g in groups:
                    d = dirs[fold_of[g]]
                    ndraw = len(plans[g][cond])
                    for t in twins_of[g]:
                        L = lab[f"{g}/{t}"]
                        n = np.asarray(L["physlaw"]["contact_normal"][:2], float)
                        n /= np.linalg.norm(n)
                        tv = np.array([-n[1], n[0]])
                        vpre = {o: np.asarray(L["physlaw"]["pre_velocity"][o][:2], float) for o in "AB"}
                        e = restitution(L) if cfg.get("e_ref", "twin") == "twin" else float(cfg["e_ref"])
                        b0 = {o: dec(g, f"base|0|0|{t}", o) for o in "AB"}
                        for j in range(ndraw):
                            u_, s_ = plans[g][cond][j]
                            for a in alphas:
                                b1 = b0 if a == 0 else {o: dec(g, f"{cond}|{j}|{a:g}|{t}", o) for o in "AB"}
                                sx.append(a * d["sd"] * d["gain"]); su.append(f"{g}|{t}|{j}"); sg.append(g)
                                sa.append(-float((b1["A"] - vpre["A"]) @ n)); sb.append(float((b1["B"] - vpre["B"]) @ n))
                                if has_passive:
                                    bp = b0 if a == 0 else {o: passive(g, t, o, a * s_ * u_) for o in "AB"}
                                    if bp["A"] is None or bp["B"] is None:
                                        has_passive = False
                                    else:
                                        pa_.append(-float((bp["A"] - vpre["A"]) @ n)); pb_.append(float((bp["B"] - vpre["B"]) @ n))
                                        if a != 0:
                                            passB.append(float((bp["B"] - b0["B"]) @ n))
                                if a == 0:
                                    continue
                                pB, pA = _pred_change(L, d["q0"][g], a * d["sd"] * d["gain"], e)
                                P.append(pB); PA.append(pA)
                                Ob.append(float((b1["B"] - b0["B"]) @ n)); ObA.append(float((b1["A"] - b0["A"]) @ n))
                                tang.append(float(abs((b1["B"] - b0["B"]) @ tv)))
                                by_g.setdefault(g, []).append(len(P) - 1)
                P, Ob, PA, ObA = map(np.array, (P, Ob, PA, ObA))
                rng = np.random.default_rng(0)
                gs = sorted(by_g)
                boot = []
                for _ in range(int(cfg.get("bootstrap", {}).get("n", 1000))):
                    ii = np.concatenate([by_g[x] for x in rng.choice(gs, len(gs))])
                    boot.append(_corr(P[ii], Ob[ii]))
                nb = int(cfg.get("bootstrap", {}).get("n", 1000))
                ds = deflection_slope(sx, sa, sb, su, sg, nb)
                pas = deflection_slope(sx, pa_, pb_, su, sg, nb) if has_passive and pa_ else None
                res[cond] = {"deflection_slope": ds, "passive_slope": pas,
                             "slope_over_passive": (ds["slope"] / pas["slope"]) if pas and abs(pas["slope"]) > 1e-12 else None,
                             "change_over_passive_B_n": (float(np.mean(np.abs(Ob))) / float(np.mean(np.abs(passB))))
                                                        if has_passive and passB and np.mean(np.abs(passB)) > 1e-12 else None,
                             "n": len(P), "r_B": _corr(P, Ob), "r_B_ci95": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))] if len(gs) > 1 else None,
                             "gain_B": float(P @ Ob / max(P @ P, 1e-12)), "r_A": _corr(PA, ObA),
                             "mean_abs_change_B_n": float(np.abs(Ob).mean()), "mean_abs_pred_B_n": float(np.abs(P).mean()),
                             "specificity_tangent_over_normal": float(np.mean(tang) / max(np.abs(Ob).mean(), 1e-12))}
            out["variants"][f"{sp}|{vname}" if len(spaces) > 1 else vname] = res
            log(f"[L4] {sp}|{vname}: " + "; ".join(f"{c}: slope {v['deflection_slope']['slope']:+.3f} r_B {v['r_B']:+.3f} "
                                                    f"gain {v['gain_B']:+.2f}" for c, v in res.items()))
    # steering experiment (generators): track B readout of every steered video
    steer_rows = []
    for g in groups:
        f = work / f"{g}_readout.json"
        if not f.exists():
            continue
        rd = json.loads(f.read_text())
        base = rd.get("base|0|0")
        d = dirs[fold_of[g]]
        L = lab[f"{g}/{twins_of[g][0]}"]
        for key, r in rd.items():
            cond, j, a = key.split("|")
            row = {"group": g, "condition": cond, "draw": int(j), "alpha": float(a), "measurable": r["measurable"],
                   "reason": r.get("reason"), "violations": r.get("violations")}
            if r["measurable"]:
                n = np.asarray(r["contact_normal"])
                row["dvB_n"] = float((np.subtract(r["velocity_post"][1], r["velocity_pre"][1])) @ n)
                row["e"] = r.get("e")
            if r["measurable"] and base and base["measurable"] and key != "base|0|0":
                n = np.asarray(base["contact_normal"])
                b = float((np.subtract(base["velocity_post"][1], base["velocity_pre"][1])) @ n)
                row["change_dvB_n"] = row["dvB_n"] - b
                row["pred_change_dvB_n"] = _pred_change(L, d["q0"][g], float(a) * d["sd"] * d["gain"], 1.0)[0]
                row["sign_agrees"] = bool(np.sign(row["change_dvB_n"]) == np.sign(row["pred_change_dvB_n"]))
            steer_rows.append(row)
    if steer_rows:
        out["steering"] = {"rows": steer_rows, "n_videos": len(steer_rows),
                           "n_measurable": sum(r["measurable"] for r in steer_rows),
                           "n_sign_agrees": sum(bool(r.get("sign_agrees")) for r in steer_rows),
                           "n_scored": sum("sign_agrees" in r for r in steer_rows)}
    return out


def table(res: dict) -> str:
    rows = []
    if res.get("stand_in"):
        rows += [f"**Stand-in run:** {res['stand_in']}", ""]
    rows += [f"Injection layer {res['layer']}; {res.get('direction', 'ridge')} direction fitted on {res['direction_family']} (target "
             f"{res.get('direction_target', 'raw')}); alphas (sd units) {res['alphas_sd']}; {res['n_groups']} groups; "
             f"twins {res['twins']}", "",
             "Directions per fold: " + "; ".join(
                 f"CV R² {d['cv_r2']:.3f} (raw {d.get('cv_r2_raw', float('nan')):.3f}, residual on size "
                 f"{d.get('cv_r2_residual', float('nan')):.3f}), sd {d['sd']:.3g}, gain {d['gain']:.3g}"
                 for d in res["directions"]), "",
             "CV R² by direction type | target (generative: 1-D projection readout): " + "; ".join(
                 ", ".join(f"{k_} {v:.3f}" for k_, v in d.get("cv_r2_by", {}).items()) +
                 (f" (|cos gen, ridge| {d['cos_generative_ridge']:.2f})" if "cos_generative_ridge" in d else "")
                 for d in res["directions"]), "",
             "Primary (e-free): slope of log(b/a) (decoded deflections a = -dv_A'.n, b = dv_B'.n) on the injected "
             "dlog(m_A/m_B), within (group, twin), group-bootstrap CI. Law: 1 for `mass`, 0 for the controls. "
             "Points with a <= 0 or b <= 0 excluded (count shown). Passive = the same metric if the edit only shifted the "
             "post-event features by the same vector (probe arithmetic, no model computation); a causal effect must "
             "exceed it (observed / passive > 1).", "",
             "| forecast readout | condition | slope log(b/a) [95% CI] | excluded / points | passive slope | slope / passive | "
             "abs change d(v_B'.n) / passive |", "|---|---|---|---|---|---|---|"]
    for v, d in res["variants"].items():
        for c, s in d.items():
            ds = s.get("deflection_slope")
            if ds:
                ci = ds["slope_ci95"]
                pf = lambda x: "-" if x is None else f"{x:+.3f}"
                ps = s.get("passive_slope")
                rows.append(f"| {v} | {c} | {ds['slope']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "") +
                            f" | {ds['n_excluded']} / {ds['n']} | {pf(ps['slope'] if ps else None)} | "
                            f"{pf(s.get('slope_over_passive'))} | {pf(s.get('change_over_passive_B_n'))} |")
    rows += ["", "Secondary: r between momentum-predicted and observed d(v_B'.n), prediction at the twin's restitution e "
             "(F3: e = 1 for every twin, so the e-dependence is harmless there).", "",
             "| forecast readout | condition | n | r(pred, obs) d(v_B'.n) [95% CI] | gain | r for A | mean abs change / predicted | tangent / normal |",
             "|---|---|---|---|---|---|---|---|"]
    for v, d in res["variants"].items():
        for c, s in d.items():
            ci = s["r_B_ci95"]
            rows.append(f"| {v} | {c} | {s['n']} | {s['r_B']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "") +
                        f" | {s['gain_B']:+.3f} | {s['r_A']:+.3f} | {s['mean_abs_change_B_n']:.4f} / {s['mean_abs_pred_B_n']:.4f} | "
                        f"{s['specificity_tangent_over_normal']:.2f} |")
    st = res.get("steering")
    if st:
        rows += ["", f"## Steering (generated videos, track B readout, loose gate): {st['n_measurable']}/{st['n_videos']} "
                 f"measurable; sign agrees with momentum in {st['n_sign_agrees']}/{st['n_scored']} scored", "",
                 "| group | condition | alpha | measurable | reason | d(v_B.n) | change vs alpha 0 | predicted |", "|---|---|---|---|---|---|---|---|"]
        f = lambda x: "-" if x is None else f"{x:+.3f}"
        for r in st["rows"]:
            rows.append(f"| {r['group']} | {r['condition']}{r['draw']} | {r['alpha']:g} | {r['measurable']} | {r['reason']} | "
                        f"{f(r.get('dvB_n'))} | {f(r.get('change_dvB_n'))} | {f(r.get('pred_change_dvB_n'))} |")
    return "\n".join(rows)
