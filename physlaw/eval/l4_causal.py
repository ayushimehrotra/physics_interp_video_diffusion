"""L4 causal propagation (spec §5.6): push the model's mass-ratio representation of object A before the event and
check that its forecast changes as momentum conservation predicts.

1. Direction. Ridge from object A's pre-event ROI pool (stage 4 cache, one clip per group) at the injection layer to
   log(m_A / m_B); u = the weight vector in raw activation units, normalised. ``layer: auto`` takes the layer with the
   best grouped-CV R². ``direction_data`` names the family the direction is fitted on (F3 hides mass in the pre-event
   frames, so it is fitted where mass is visible, e.g. F1); on the evaluated family itself it is cross-fitted (the
   direction for a group never saw that group).
2. Intervention. A torch forward hook on the injection block adds alpha * sd * u (sd = sd of the projection on u over
   the direction's training clips; alpha in ``alphas_sd``) to object A's ROI tokens in the pre-event token-frames,
   then forecast() is rerun. Encoders: the hook acts on the context encoder only. DiTs: on the context tokens at every
   sampling step (both CFG branches); the same run gives the generated video.
3. Readout. Post-event velocities decoded from the forecast with probes trained on REAL post-event frames (as L1:
   frame pool = leak-free, roi = twin's own post ROI; ``_mm`` = moment-matched to the cached unperturbed forecasts),
   cross-fitted by group.
4. Metric. The direction implies log q' = log q0 + alpha * sd * |w| (q0 = the direction probe's own reading of the
   clip). Momentum conservation at e (the twin's restitution) then predicts the change of B's velocity along the
   contact normal n: d(v_B'.n) = (1 + e) (q'/(1 + q') - q0/(1 + q0)) (v_A - v_B).n, and for A
   d(v_A'.n) = -(1 + e) (1/(1 + q') - 1/(1 + q0)) (v_A - v_B).n. Score = Pearson r between predicted and observed
   changes over (group, twin, alpha != 0), with a group-bootstrap CI; gain = least-squares observed / predicted.
   Specificity = mean |change of v_B'.t| / mean |change of v_B'.n| (t = tangent; momentum predicts no change).
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


def _corr(a, b):
    a, b = np.asarray(a) - np.mean(a), np.asarray(b) - np.mean(b)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def run(cache_dir: Path, data_dir: Path, cfg, mcfg, work: Path, force: bool = False, log=print) -> dict:
    from physlaw.config import resolve_path
    alphas = [float(a) for a in cfg["alphas_sd"]]
    ridge_alphas = tuple(float(a) for a in cfg["probe"]["alphas"])
    k = int(cfg["probe"]["k_folds"])
    recs, labels, A, pools, layers = load_family(cache_dir, data_dir)
    lab = {r["clip"]: l for r, l in zip(recs, labels)}
    groups_all = sorted({r["group"] for r in recs})
    twins_of = {g: [r["twin"] for r in recs if r["group"] == g and (not cfg.get("twins") or r["twin"] in cfg["twins"])]
                for g in groups_all}
    first = {g: next(i for i, r in enumerate(recs) if r["group"] == g) for g in groups_all}

    # direction source: one row per group (the pre-event context is shared by a group's twins)
    same = not cfg.get("direction_data")
    if same:
        sA, slab, spools, slayers = A, labels, pools, layers
        srows = np.array([first[g] for g in groups_all])
        sgroups = np.array(groups_all)
    else:
        sdata = resolve_path(cfg["direction_data"])
        srecs, slab, sA, spools, slayers = load_family(cache_dir.parent / sdata.name, sdata)
        sg = [r["group"] for r in srecs]
        srows = np.array([sg.index(g) for g in sorted(set(sg))])
        sgroups = np.array(sorted(set(sg)))
    sy = np.array([log_q(slab[i]) for i in srows])
    pa = spools.index("roi_A.pre")
    cand = [l for l in slayers if l != "-1"] if cfg["layer"] == "auto" else [str(cfg["layer"])]
    layer, _, _ = fit_select({l: sA[srows, pa, slayers.index(l)].astype(np.float64) for l in cand}, sy, sgroups,
                             np.arange(len(srows)), ridge_alphas, k)
    SX = sA[srows, pa, slayers.index(layer)].astype(np.float64)
    EX = A[:, pools.index("roi_A.pre"), layers.index(layer)].astype(np.float64) if layer in layers else None
    log(f"[L4] injection layer {layer} (searched {len(cand)}); direction from {'this family' if same else cfg['direction_data']}")

    # folds over the evaluated groups (directions cross-fitted when fitted on this family)
    groups = groups_all[: cfg.get("max_groups") or None]
    folds = grouped_kfold(np.array(groups), min(k, len(groups)), 0) if len(groups) >= 2 else [np.arange(len(groups))]
    plans, dirs, fold_of = {}, [], {}
    for fi, te in enumerate(folds):
        test = [groups[i] for i in te]
        tr = np.array([g not in test for g in sgroups]) if same else np.ones(len(sgroups), bool)
        d = fit_direction(SX[tr], sy[tr], sgroups[tr], ridge_alphas, k)
        d["q0"] = {g: float(np.exp(d["predict"](EX[[first[g]]])[0])) if EX is not None else float(np.exp(log_q(labels[first[g]])))
                   for g in test}
        dirs.append(d)
        cd = control_directions(SX[tr], d, cfg.get("controls") or {}, seed=fi)
        for g in test:
            plans[g], fold_of[g] = cd, fi
        log(f"[L4] fold {fi}: direction CV R² {d['cv_r2']:.3f}, sd {d['sd']:.3f}, gain {d['gain']:.3f}")

    missing = [g for g in groups if force or not (work / f"{g}.npz").exists()]
    if missing:
        from physlaw.models.registry import load_adapter
        m = load_adapter(mcfg, device=cfg.get("device", "cuda"))
        run_interventions(m, mcfg, data_dir, missing, twins_of, plans, layer, alphas, work,
                          bool(cfg.get("generate_readout", True)), force, log)
        del m

    # decoding probes: real post-event frames of the training groups, per space and pool; mm from cached forecasts
    spaces = forecast_spaces(cache_dir, recs, mcfg, list(POST_POOLS))
    grp = np.array([r["group"] for r in recs])
    Yv = {o: np.array([l["physlaw"]["post_velocity"][o][:2] for l in labels], float) for o in "AB"}
    Z = {g: dict(np.load(work / f"{g}.npz")) for g in groups}
    conds = sorted({key.split("|")[0] for key in Z[groups[0]]} - {"base"})
    out = {"layer": layer, "direction_family": cfg.get("direction_data") or data_dir.name, "alphas_sd": alphas,
           "n_groups": len(groups), "twins": sorted({t for g in groups for t in twins_of[g]}),
           "directions": [{k_: d[k_] for k_ in ("sd", "gain", "alpha", "cv_r2")} for d in dirs],
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
            res = {}
            for cond in conds:
                P, Ob, PA, ObA, tang, by_g = [], [], [], [], [], {}
                for g in groups:
                    d = dirs[fold_of[g]]
                    ndraw = len(plans[g][cond])
                    for t in twins_of[g]:
                        L = lab[f"{g}/{t}"]
                        n = np.asarray(L["physlaw"]["contact_normal"][:2], float)
                        n /= np.linalg.norm(n)
                        tv = np.array([-n[1], n[0]])
                        e = restitution(L) if cfg.get("e_ref", "twin") == "twin" else float(cfg["e_ref"])
                        b0 = {o: dec(g, f"base|0|0|{t}", o) for o in "AB"}
                        for j in range(ndraw):
                            for a in alphas:
                                if a == 0:
                                    continue
                                b1 = {o: dec(g, f"{cond}|{j}|{a:g}|{t}", o) for o in "AB"}
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
                res[cond] = {"n": len(P), "r_B": _corr(P, Ob), "r_B_ci95": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))] if len(gs) > 1 else None,
                             "gain_B": float(P @ Ob / max(P @ P, 1e-12)), "r_A": _corr(PA, ObA),
                             "mean_abs_change_B_n": float(np.abs(Ob).mean()), "mean_abs_pred_B_n": float(np.abs(P).mean()),
                             "specificity_tangent_over_normal": float(np.mean(tang) / max(np.abs(Ob).mean(), 1e-12))}
            out["variants"][f"{sp}|{vname}" if len(spaces) > 1 else vname] = res
            log(f"[L4] {sp}|{vname}: " + "; ".join(f"{c}: r_B {v['r_B']:+.3f} gain {v['gain_B']:+.2f}" for c, v in res.items()))
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
    rows += [f"Injection layer {res['layer']}; direction fitted on {res['direction_family']}; alphas (sd units) "
             f"{res['alphas_sd']}; {res['n_groups']} groups; twins {res['twins']}", "",
             "Directions per fold: " + "; ".join(f"CV R² {d['cv_r2']:.3f}, sd {d['sd']:.3g}, gain {d['gain']:.3g}"
                                                for d in res["directions"]), "",
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
