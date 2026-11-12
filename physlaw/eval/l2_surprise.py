"""L2 violation sensitivity (spec §5.0.4 L2 / L2-rep): zero-fit surprise contrasts and the energy kink.

Clean window (spec §5.0.2; ``clean_window`` in l2.json). Every surprise is restricted, per group, to the post-event
frames before min over the group's twins of ``clean_until_s`` (the first secondary event: an object leaves the frame,
a second contact, the apex...). Records restrict when they carry per-frame values (``per_frame``, see
:func:`window_surprise`): scripts/05_surprise.py writes them for the pixel baselines. Stage 4's model surprise is a
scalar per clip until the adapters return per-token-frame surprise; those records are used unrestricted and counted
under ``clean_window.sources.<src>.unrestricted``. Old labels without ``clean_until_s`` (F1, frozen): computed from
``per_frame`` (first frame after the event with an object out of frame or the two balls touching again).

PAIRWISE CONTRASTS (``contrasts.<family>.list``; each entry's ``law``, default momentum: momentum F1-F4 and the F6
glide, gravity F7 projectile / drop, friction F6 slide / pair, oscillation F9 pair, rolling F10, angular_momentum F12). For each contrast (violating twin, legal twin)
and surprise distance, over twin groups: win rate P(surprise(viol) > surprise(legal)) (ties 1/2, group-bootstrap CI),
Wilcoxon signed-rank p, Cliff's delta, Spearman of the paired difference with the dose (``magnitude: eta`` = the
momentum-exchange imbalance η of the violating twin from its labels; another value = a labels ``physlaw.summary``
key, violating − legal). Primary: the contrast flagged ``primary`` (F1: mom_violate_e015 vs conserve_e015; an
observer that knows momentum but assumes the wrong restitution still wins it 0.95-1.00, the e1 contrast only ~0.53).
A contrast's ``viol`` may list several twins (``speed_x0.9+speed_x1.1 vs legal``, the F6 glide's speed-matched pair: one
violator slower, one faster than the legal twin, so an observer of speed alone wins exactly half): every (violator,
legal) pair is a row, the CI bootstraps groups and Wilcoxon runs on per-group mean differences.
Ideal-observer bands (labels only, collision families): perfect / law at assumed e / mass-blind observers in the 4-D
Δv space. Every contrast is computed for the model AND for the pixel baselines of scripts/05_surprise.py
(copy-last-frame, constant-velocity extrapolation) on the same frames. Legal-vs-legal controls: ``calibration`` (one
pair, optional: F9 pairs have one legal twin) and ``controls`` (more pairs); ``primary_by_law`` = the contrast
flagged ``primary`` for each law (exactly one per law and family: ``check_primaries``).

Motion control (spec §5.0.4): the model − pixel-baseline win rate (``vs_pixel`` under each model distance: model
win − copy_last / const_velocity win on the same pairs, paired group-bootstrap CI) and, on F2, speed-matched violators
(both drift signs). No regression adjustment: raw win rates are the scores. The legal-vs-legal ``calibration`` pair
is scored as a reference contrast (two legal outcomes that differ in e; nothing is broken).

F2 speed-matched violators (``contrasts.f2.group_contrasts`` kind ``speed_matched``; l2.json ``group_contrasts``):
each momentum violator (both drift signs, mom_violate_e<e>_<pos|neg>) vs the legal ladder twin closest in post-event
KE (or total speed), so speed cannot separate the pair; raw win rates per source, labels-only observers, pixel gap.

ENERGY KINK (``energy_kink`` entries; l2.json ``energy_kink[<name>]``): surprise along a ladder of twins ordered by the
generalised restitution ε (F2 e0.0…e1.5, F7 bounce, F8 pad and F9 swing eps0.5, eps0.55, …, eps1.5), kink statistic at
the law boundary ε = 1, groups whose ``stop_dose`` is within exclude_radius of 1 excluded; see :func:`energy_kink`.
ACROSS-GROUP KINK (``group_kink`` entries; l2.json ``group_kink[<name>]``; F11 statics): the dose is a scene constant
of each group (``constants.overhang_x``), the curve is the per-group contrast c = log S(violating) − log S(legal) averaged
over the groups at each dose, and the same kink statistic runs on it (no expected dose x̂: there is no outcome ladder);
see :func:`group_kink`.
L2-REP (l2.json ``energy_kink_rep[<name>]``): the kink of the post-event activation trajectory along the same ladder;
see :func:`energy_kink_rep`. Energy has no pairwise surprise contrast (R5: a monotone dose is decodable from any
correlate; only a kink at ε = 1 is evidence).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physlaw.probes.select import holm


def _index(path: Path) -> dict[str, dict]:
    recs = {}
    for line in open(path):
        r = json.loads(line)
        recs[r["clip"]] = r
    return recs


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def _win(d: np.ndarray) -> float:
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def paired_stats(sv: np.ndarray, sl: np.ndarray, mag: np.ndarray | None, n_boot: int = 2000, seed: int = 0,
                 groups: np.ndarray | None = None) -> dict:
    """Win rate (bootstrap over ``groups``: rows of one group are drawn together; default one row per group),
    Wilcoxon on per-group mean differences, Cliff's delta, Spearman of the difference with ``mag``."""
    from scipy.stats import spearmanr, wilcoxon
    d = sv - sl
    rng = np.random.default_rng(seed)
    _, gi = np.unique(np.arange(len(d)) if groups is None else np.asarray(groups), return_inverse=True)
    rows_of = [np.flatnonzero(gi == k) for k in range(gi.max() + 1)] if len(d) else []
    boots = [_win(d[np.concatenate([rows_of[i] for i in rng.integers(0, len(rows_of), len(rows_of))])])
             for _ in range(n_boot)]
    gm = np.array([d[r].mean() for r in rows_of])
    out = {"n": int(len(d)), "n_groups": len(rows_of), "n_zero_diff": int(np.sum(d == 0)), "win_rate": _win(d),
           "win_rate_boot_sd": float(np.std(boots)),
           "win_rate_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
           "wilcoxon_p": float(wilcoxon(gm).pvalue) if np.any(gm != 0) else 1.0,
           "cliffs_delta": cliffs_delta(sv, sl),
           "median_diff": float(np.median(d)), "median_rel_diff": float(np.median(d / np.abs(sl).clip(1e-12)))}
    if mag is not None and np.std(mag) > 0 and np.std(d) > 0:
        r = spearmanr(d, mag)
        out["spearman_vs_magnitude"] = float(r.statistic)
        out["spearman_p"] = float(r.pvalue)
    return out


def _magnitude(labels: dict, key: str | None) -> float:
    return float(labels["physlaw"]["summary"][key]) if key else 0.0


def eta_of(labels: dict) -> float:
    """Momentum-exchange imbalance η = |Σ m_i Δv_i| / Σ |m_i Δv_i| of a clip, from its true labels
    (physlaw.pre_velocity / post_velocity / objects.<o>.mass; in-plane components)."""
    pl = labels["physlaw"]
    q = np.array([pl["objects"][o]["mass"] * (np.asarray(pl["post_velocity"][o], float)[:2]
                                              - np.asarray(pl["pre_velocity"][o], float)[:2]) for o in pl["objects"]])
    den = float(np.linalg.norm(q, axis=1).sum())
    return float(np.linalg.norm(q.sum(0)) / den) if den > 0 else float("nan")


def contrast_magnitude(key: str | None, lab_viol: dict, lab_legal: dict) -> float:
    """Dose of one violating/legal pair: ``eta`` -> η of the violating twin; any other key -> the labels
    ``physlaw.summary`` value, violating − legal."""
    if key == "eta":
        return eta_of(lab_viol)
    return _magnitude(lab_viol, key) - _magnitude(lab_legal, key)


# --------------------------------------------------------------------------- ideal-observer reference bands

IDEAL_E = (1.0, 0.5, 0.0)        # assumed restitutions of the law-based observers
UNEQUAL_MASS_RATIO = 1.25


def _win_dist(pred: np.ndarray, o_viol: np.ndarray, o_legal: np.ndarray) -> float:
    a, b = np.linalg.norm(pred - o_viol), np.linalg.norm(pred - o_legal)
    if np.isclose(a, b, rtol=1e-12, atol=1e-12):
        return 0.5
    return 1.0 if a > b else 0.0


def ideal_observer_bands(items: list[tuple], unequal_ratio: float = UNEQUAL_MASS_RATIO) -> dict:
    """Win rates of reference observers on paired outcomes, from kinematics only.

    items: (PreState, o_viol [4], o_legal [4]) per group, o = (dv_A, dv_B) in the plane. An observer predicts
    an outcome ``pred`` and "wins" a group when |pred − o_viol| > |pred − o_legal| (ties 1/2). Observers:
    ``perfect`` (pred = o_legal), ``law_e{e}`` (the frictionless legal outcome at assumed restitution e), and
    ``mass_blind`` (o_legal's relative-velocity change split equally between the balls,
    laws.mass_blind_outcome). Reported overall and on groups with mass ratio >= ``unequal_ratio``, plus
    ``viol_slower`` = share of groups with |v_post(viol)| / |v_post(legal)| < 1 (Frobenius norm over both balls)."""
    from physlaw.data import collision_physics as cp
    from physlaw.eval import laws
    names = ["perfect", *[f"law_e{e:g}" for e in IDEAL_E], "mass_blind"]
    wins = {k: [] for k in names}
    unequal, slower = [], []
    for st, ov, ol in items:
        ov, ol = np.asarray(ov, float).reshape(4), np.asarray(ol, float).reshape(4)
        v0 = np.stack([st.v_a, st.v_b])
        mb = laws.mass_blind_outcome({"v": v0}, {"v": v0 + ol.reshape(2, 2)}, {})["v"] - v0
        preds = {"perfect": ol, **{f"law_e{e:g}": cp.legal_outcome(st, e) for e in IDEAL_E}, "mass_blind": mb.reshape(4)}
        for k, pr in preds.items():
            wins[k].append(_win_dist(pr, ov, ol))
        unequal.append(max(st.m_a, st.m_b) / min(st.m_a, st.m_b) >= unequal_ratio)
        slower.append(np.linalg.norm(v0 + ov.reshape(2, 2)) / np.linalg.norm(v0 + ol.reshape(2, 2)) < 1)
    unequal = np.array(unequal, bool)
    out = {"n": len(items), "n_unequal_mass": int(unequal.sum()), "unequal_mass_ratio": unequal_ratio,
           "win_rate": {k: float(np.mean(v)) for k, v in wins.items()},
           "win_rate_unequal_mass": {k: (float(np.mean(np.array(v)[unequal])) if unequal.any() else float("nan"))
                                     for k, v in wins.items()},
           "viol_slower": float(np.mean(slower)) if slower else float("nan")}
    return out


def ideal_observer(pairs: list[tuple[dict, dict]], **kw) -> dict:
    """``ideal_observer_bands`` from (labels of the violating twin, labels of the legal twin) per group."""
    from physlaw.data import collision_physics as cp
    from physlaw.eval import laws
    items = []
    for lv, ll in pairs:
        pv, qv, ctx = laws.from_labels(lv)
        pl, ql, _ = laws.from_labels(ll)
        st = cp.PreState(float(ctx["m"][0]), float(ctx["m"][1]), pl["v"][0], pl["v"][1], ctx["n"])
        items.append((st, (qv["v"] - pv["v"]).reshape(4), (ql["v"] - pl["v"]).reshape(4)))
    return ideal_observer_bands(items, **kw)


def ideal_line(io: dict) -> str:
    """One md line for a contrast's ideal-observer bands."""
    w, u = io["win_rate"], io["win_rate_unequal_mass"]
    law = " / ".join(f"{w[f'law_e{e:g}']:.2f}" for e in IDEAL_E)
    return (f"ideal observer (labels only, 4-D Δv distance, n={io['n']}): perfect {w['perfect']:.2f}; law e=1/0.5/0 "
            f"{law}; mass-blind {w['mass_blind']:.2f} ({u['mass_blind']:.2f} on {io['n_unequal_mass']} unequal-mass "
            f"groups, ratio >= {io['unequal_mass_ratio']:g}); violator slower in {io['viol_slower']:.0%} of groups")


def family_contrasts(cfg, family: str) -> tuple[tuple[str, str] | None, list[dict]]:
    """(calibration pair or None, contrast dicts, primary first) for a family; (None, []) when the family has no
    pairwise contrasts (F8 / F11: kinks only)."""
    fam = (cfg["contrasts"] or {}).get(family) or {}
    cons = [dict(x) for x in fam.get("list") or []]
    check_primaries(family, fam)
    cal = tuple(fam["calibration"]) if fam.get("calibration") else None
    return cal, sorted(cons, key=lambda x: not x.get("primary", False))


def check_primaries(family: str, fam: dict) -> None:
    """Exactly one ``primary`` contrast per law in a family's ``list`` (the headline number; F6 has two laws: glide
    momentum and friction), and none among ``group_contrasts``. Raises ValueError otherwise."""
    laws: dict = {}
    for c in fam.get("list") or []:
        laws.setdefault(c.get("law", "momentum"), []).extend([contrast_key(c)] if c.get("primary") else [])
    bad = {law: prim for law, prim in laws.items() if len(prim) != 1}
    flagged = [g.get("name") for g in fam.get("group_contrasts") or [] if "primary" in g]
    if bad or flagged:
        raise ValueError(f"L2 contrasts.{family}: need exactly one primary contrast per law (primaries per law: {bad})"
                         + (f"; group_contrasts take no primary flag: {flagged}" if flagged else ""))


def control_pairs(cfg, family: str) -> list[tuple[str, str]]:
    """The family's legal-vs-legal reference pairs: ``calibration`` first, then ``controls``."""
    fam = (cfg["contrasts"] or {}).get(family) or {}
    cal = [tuple(fam["calibration"])] if fam.get("calibration") else []
    return cal + [tuple(c) for c in fam.get("controls") or []]


def viols_of(con: dict) -> list[str]:
    """The violating twin(s) of a contrast (``viol``: one name or a list)."""
    v = con["viol"]
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


def contrast_key(con: dict) -> str:
    return f"{'+'.join(viols_of(con))} vs {con['legal']}"


def _subsets(con: dict, lab) -> dict:
    """{key suffix: predicate on the violating clip} for a contrast: all, each stratum of ``stratify``, and for a
    contrast with several violators (a speed-matched pair) each violator's half alone (`` [half <twin>]``; audit M3:
    the slower and the faster half are reported separately)."""
    out = {"": lambda clip: True}
    f = con.get("stratify")
    if f:
        out[f" [{f}]"] = lambda clip: bool(lab(clip)["physlaw"][f])
        out[f" [not {f}]"] = lambda clip: not bool(lab(clip)["physlaw"][f])
    vs = viols_of(con)
    if len(vs) > 1:
        for v in vs:
            out[f" [half {v}]"] = lambda clip, v=v: clip.rsplit("/", 1)[-1] == v
    return out


def energy_creating_share(lab, groups: list[str], viols: list[str]) -> dict:
    """{violator: share of the groups whose violating clip creates energy (labels physlaw.energy_physical False)}.
    Audit M3: a violator that also creates energy breaks two laws (F6 glide speed_x > 1, F12 spin_x > 1 at e = 0.9,
    F2 mom_violate_*_pos, F6 slide reverse); a model surprised only by energy creation wins such a contrast."""
    out = {}
    for v in viols:
        flags = [not bool(lab(f"{g}/{v}")["physlaw"].get("energy_physical", True)) for g in groups]
        out[v] = float(np.mean(flags)) if flags else float("nan")
    return out


def run(sources: dict[str, Path], data_dir: Path, cfg, log=print) -> dict:
    """sources: name -> _index.jsonl-style file whose records carry {"clip", "group", "twin", "surprise"} (+ optional
    ``per_frame``; the model's records also "file", its stage-4 .npz relative to the index's directory)."""
    idx = {name: _index(p) for name, p in sources.items()}
    labels_cache: dict[str, dict] = {}

    def lab(clip):
        if clip not in labels_cache:
            labels_cache[clip] = json.loads((data_dir / clip / "labels.json").read_text())
        return labels_cache[clip]

    windows = group_windows(data_dir, lab)
    idx, cw = apply_clean_window(idx, windows)
    log(f"[L2] clean window: {cw['n_groups_with_window']}/{cw['n_groups']} groups; "
        + "; ".join(f"{s}: {v}" for s, v in cw["sources"].items()))
    for src, v in cw["sources"].items():
        if v["unrestricted"]:
            log(f"[L2] WARNING {src}: {v['unrestricted']} records without per-frame surprise (cache older than the "
                "clean window, or no clean_until_s): their whole post segment is used")
    model = next((n for n in idx if n != "pixel_baselines"), None)
    out = {"clean_window": cw, **_pairwise(idx, data_dir, cfg, lab, log)}
    out.update(_kink_block(idx, data_dir, cfg, lab, windows, model,
                           Path(sources[model]).parent if model else None, log))
    return out


def _pairwise(idx: dict, data_dir: Path, cfg, lab, log) -> dict:
    """Pairwise contrasts with the pixel gap, legal-vs-legal controls, F2 speed-matched group contrasts."""
    from physlaw.data.io import COLLISION_FAMILIES
    fam = data_dir.name
    cal, cons = family_contrasts(cfg, fam)
    ctrl = control_pairs(cfg, fam)
    if not cons and not ctrl:     # no pairwise contrasts configured for this family: kinks only
        return {"calibration": None, "primary_contrast": None, "contrast_info": {}, "contrasts": {}}
    controls = [{"viol": v, "legal": l, "magnitude": None, "control": True} for v, l in ctrl]
    prim = [contrast_key(c) for c in cons if c.get("primary")]
    by_law: dict = {}
    for c in cons:
        if c.get("primary"):
            by_law.setdefault(c.get("law", "momentum"), contrast_key(c))
    out = {"calibration": list(cal) if cal else None, "controls": [list(c) for c in ctrl],
           "primary_contrast": prim[0] if prim else None, "primary_by_law": by_law, "contrast_info": {}, "contrasts": {}}
    groups_all = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    for con in cons:          # labels-only references: ideal-observer bands per collision contrast
        key = contrast_key(con)
        info = {"viol": con["viol"], "legal": con["legal"], "law": con.get("law", "momentum"),
                "primary": bool(con.get("primary")), "magnitude": con.get("magnitude")}
        gs = [g for g in groups_all if all((data_dir / g / t / "labels.json").exists()
                                           for t in (*viols_of(con), con["legal"]))]
        if fam in COLLISION_FAMILIES and gs and len(viols_of(con)) == 1:
            info["ideal_observer"] = ideal_observer([(lab(f"{g}/{con['viol']}"), lab(f"{g}/{con['legal']}")) for g in gs])
        if gs:
            info["energy_creating_share"] = energy_creating_share(lab, gs, viols_of(con))
            info["also_breaks_energy"] = sorted(v for v, s in info["energy_creating_share"].items() if s > 0)
        out["contrast_info"][key] = info
    row_viols: dict = {}
    for con in cons + controls:
        viols, legal = viols_of(con), con["legal"]
        for suffix, keep in _subsets(con, lab).items():
            key = contrast_key(con) + suffix
            row_viols[key] = [suffix[len(" [half "):-1]] if suffix.startswith(" [half ") else viols
            res = {}
            for src, recs in idx.items():
                groups = sorted({r["group"] for r in recs.values()})
                full = [(g, f"{g}/{v}", f"{g}/{legal}") for g in groups
                        if all(f"{g}/{t}" in recs for t in (*viols, legal)) for v in viols]
                rows = [r for r in full if keep(r[1])]
                if not rows or (suffix and len(rows) == len(full)):      # empty, or a stratum that is everything
                    continue
                mag = (np.array([contrast_magnitude(con["magnitude"], lab(v), lab(l)) for _, v, l in rows])
                       if con.get("magnitude") else None)
                metrics = sorted(set.intersection(*(set(recs[c]["surprise"]) for _, v, l in rows for c in (v, l))))
                rg = np.array([g for g, _, _ in rows])
                res[src] = {m: paired_stats(np.array([recs[v]["surprise"][m] for _, v, _ in rows]),
                                            np.array([recs[l]["surprise"][m] for _, _, l in rows]), mag, groups=rg)
                            for m in metrics}
            if res:
                out["contrasts"][key] = res
                log(f"[L2] {key}: " + "; ".join(f"{s}: {len(v)} metrics" for s, v in res.items()))
    ctrl_keys = {contrast_key(c) for c in controls}
    for src in idx:          # Holm across the violating contrasts, per source and distance
        ms = {m for c, v in out["contrasts"].items() if src in v for m in v[src]}
        for m in ms:
            ph = holm({c: v[src][m]["wilcoxon_p"] for c, v in out["contrasts"].items()
                       if src in v and m in v[src] and c.split(" [")[0] not in ctrl_keys})
            for c, p in ph.items():
                out["contrasts"][c][src][m]["wilcoxon_p_holm"] = p
    names = list(idx)
    model = next((n for n in names if n != "pixel_baselines"), None)
    if "pixel_baselines" in idx and model is not None:
        n_boot = int(cfg.get("group_n_boot", 1000))
        for key, by_src in out["contrasts"].items():
            legal = key.split(" [")[0].split(" vs ")[1]
            for m, rec in by_src.get(model, {}).items():
                rec["vs_pixel"] = _pair_pixel_gap(idx[model], idx["pixel_baselines"], row_viols[key], legal, m, n_boot)
    spec = group_spec(cfg, fam)
    if spec:                 # F2: speed-matched violators over the whole twin group
        gdirs = [g for g in groups_all if any((data_dir / g / t / "labels.json").exists() for t in spec["ladder"])]
        kin = group_kinematics(lab, gdirs, lambda g: sorted(q.name for q in (data_dir / g).iterdir()
                                                             if (q / "labels.json").exists()))
        out["group_contrasts"] = group_contrasts(
            kin, {src: {c: r["surprise"] for c, r in recs.items()} for src, recs in idx.items()}, spec,
            model=model, n_boot=int(cfg.get("group_n_boot", 1000)))
        log(f"[L2] group contrasts: " + ", ".join(f"{k}: {v['n_rows']} rows" for k, v in out["group_contrasts"]["contrasts"].items()))
    return out


def _kink_block(idx: dict, data_dir: Path, cfg, lab, windows: dict, model: str | None, cache_dir: Path | None,
                log) -> dict:
    """{"energy_kink": {name: record}, "energy_kink_rep": {name: record}, "group_kink": {name: record}} for the
    family's kink entries ({} if none configured)."""
    specs = kink_specs(cfg, data_dir.name)
    if not specs:
        return {}
    groups = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    sur = {c: r["surprise"] for c, r in idx[model].items()} if model else {}
    pix = {c: r["surprise"] for c, r in idx["pixel_baselines"].items()} if "pixel_baselines" in idx else None
    n_boot = int(cfg.get("group_n_boot", 1000))
    out: dict = {}
    for s in specs:
        name = s.get("name", s["kind"])
        if s["kind"] == GROUP_KINK_KIND:
            out.setdefault("group_kink", {})[name] = group_kink(s, groups, lab, sur, model, pix, n_boot=n_boot, log=log)
            continue
        out.setdefault("energy_kink", {})
        out.setdefault("energy_kink_rep", {})
        ek = energy_kink(s, groups, lab, sur, model, pix, n_boot=n_boot, log=log)
        out["energy_kink"][name] = ek
        if model is None or cache_dir is None:
            continue
        xh = [d["model"]["x_hat"] for d in ek.get("distances", {}).values() if d.get("model")]
        xh = [v for v in xh if np.isfinite(v)]
        out["energy_kink_rep"][name] = energy_kink_rep(
            s, ek.get("groups_scored", []), lab, data_dir, cache_dir, idx[model], windows,
            x_hat=float(np.median(xh)) if xh else None, n_boot=n_boot,
            n_splits=int(cfg.get("selection", {}).get("n_splits", 50)), log=log,
            stops={g: (v or []) + list(ek.get("kinematic_minima", {}).get(g, {}).values())
                   for g, v in ek.get("stop_doses", {}).items()})
    return out


# Pixel baselines (scripts/05_surprise.py): name -> surprise key ("{region}" = the model metric's region, obj / all).
PIXEL_BASELINES = {"copy_last": "copy_last_mse_{region}", "const_velocity": "constvel_px"}


def pixel_key(name: str, metric: str) -> str:
    region = metric.split("_")[-1]
    return PIXEL_BASELINES[name].format(region=region if region in ("obj", "all") else "obj")


def _pair_pixel_gap(model_recs: dict, pix_recs: dict, viol, legal: str, m: str, n_boot: int = 1000,
                    seed: int = 0) -> dict:
    """Model win − pixel-baseline win on the groups where both are scored (paired: the same group resample).
    ``viol``: one violating twin or a list (every (violator, legal) pair of a group is a row)."""
    viols = list(viol) if isinstance(viol, (list, tuple)) else [viol]
    groups = sorted({r["group"] for r in model_recs.values()})
    out = {}
    has = lambda recs, g, t, key: f"{g}/{t}" in recs and key in recs[f"{g}/{t}"]["surprise"]
    for bn in PIXEL_BASELINES:
        k = pixel_key(bn, m)
        gs = [g for g in groups if all(has(model_recs, g, t, m) and has(pix_recs, g, t, k) for t in (*viols, legal))]
        if len(gs) < 2:
            continue
        d = lambda recs, key: np.array([[recs[f"{g}/{v}"]["surprise"][key] - recs[f"{g}/{legal}"]["surprise"][key]
                                         for v in viols] for g in gs], float)        # [G, n_viols]
        dm, db = d(model_recs, m), d(pix_recs, k)
        rng = np.random.default_rng(seed)
        boots = [(_win(dm[i].ravel()) - _win(db[i].ravel())) for i in (rng.integers(0, len(gs), len(gs)) for _ in range(n_boot))]
        out[bn] = {"n": len(gs), "baseline_win": _win(db.ravel()), "model_win": _win(dm.ravel()),
                   "model_minus_baseline": _win(dm.ravel()) - _win(db.ravel()), "model_minus_baseline_ci95": _q(boots)}
    return out


# --------------------------------------------------------------------------- F2 group contrasts
# One F2 twin group = the legal restitution ladder (e = 0.0 .. 1.0; the energy-creating rungs e1.1 .. e1.5 are only
# used by the kink) + momentum violators mom_violate_e<e>_<pos|neg> (both drift signs), all sharing the pre-contact
# frames. ``speed_matched``: violator vs the ladder twin whose post-contact KE (or total speed) is closest, so speed
# cannot separate the pair. Matching quality: median |log KE_viol / KE_legal|; pairs above ``max_log_gap`` are left
# out (counted as unmatched), and with ``require_bracketed`` so are violators outside the ladder's range.
# ``balance_faster`` weights rows so violators faster / slower than their partner (in the match scalar) count
# equally: an observer of that scalar alone then sits at 0.5. Wilcoxon p uses unweighted per-group means.
# Rows are violator clips, several per group, so every CI is a bootstrap over GROUPS (the same draw for model and
# pixel baselines: ``vs_pixel`` = model win − baseline win) and the Wilcoxon test runs on per-group means.

GROUP_KINDS = ("speed_matched",)


def group_spec(cfg, family: str) -> dict | None:
    """The ``contrasts.<family>`` group-contrast block (speed_matched entries), or None."""
    fam = (cfg["contrasts"] or {}).get(family) or {}
    cons = [dict(x) for x in fam.get("group_contrasts") or [] if x["kind"] in GROUP_KINDS]
    if not cons:
        return None
    return {"ladder": [str(x) for x in fam["ladder"]], "contrasts": cons}


def _post(st, o) -> np.ndarray:
    return np.stack([st.v_a, st.v_b]) + np.asarray(o, float).reshape(2, 2)


def kin_scalar(st, o, by: str = "ke") -> float:
    """Post-contact KE (``ke``) or total speed |v_A| + |v_B| (``speed``) of outcome o = (dv_A, dv_B)."""
    v = _post(st, o)
    if by == "ke":
        return float(0.5 * st.m_a * v[0] @ v[0] + 0.5 * st.m_b * v[1] @ v[1])
    if by == "speed":
        return float(np.linalg.norm(v, axis=1).sum())
    raise ValueError(by)


def eta_outcome(st, o) -> float:
    q = np.array([st.m_a, st.m_b])[:, None] * np.asarray(o, float).reshape(2, 2)
    den = float(np.linalg.norm(q, axis=1).sum())
    return float(np.linalg.norm(q.sum(0)) / den) if den > 0 else float("nan")


def match_partner(st, o_viol, ladder_out: dict, by: str = "ke") -> tuple[str, float, float]:
    """(ladder twin closest to the violator in log ``by``, |log KE ratio|, |log speed ratio|)."""
    x = np.log(kin_scalar(st, o_viol, by))
    best = min(ladder_out, key=lambda k: abs(np.log(kin_scalar(st, ladder_out[k], by)) - x))
    gk = abs(np.log(kin_scalar(st, o_viol, "ke") / kin_scalar(st, ladder_out[best], "ke")))
    gs = abs(np.log(kin_scalar(st, o_viol, "speed") / kin_scalar(st, ladder_out[best], "speed")))
    return best, float(gk), float(gs)


def _segment_dist(o, d: np.ndarray, k0: float, k1: float) -> float:
    """Distance from o to the segment {k d : k in [k0, k1]} in R^4."""
    o = np.asarray(o, float).reshape(4)
    k = float(np.clip(o @ d / (d @ d), k0, k1))
    return float(np.linalg.norm(o - k * d))


def group_observers(st, e_range=(0.0, 1.0)) -> dict:
    """name -> surprise(o) for the labels-only reference observers (4-D Δv space, as ``ideal_observer_bands``):
    ``perfect`` distance to the legal segment e in e_range (every ladder twin scores 0); ``law_e{e}`` distance to
    the legal outcome at assumed e; ``mass_blind`` distance to the mass-blind segment (the relative-velocity change
    (1+e) u_n n split equally, e in e_range); ``speed_e{e}`` |total post speed − total speed of the legal outcome at
    e| (a speed-only observer: knows how fast things should go, not where); ``ke_e{e}`` the same with post KE."""
    from physlaw.data import collision_physics as cp
    u, n = st.approach_speed, st.n
    mb = np.concatenate([-n / 2, n / 2])
    obs = {"perfect": lambda o: cp.distance_to_legal_line(o, st, segment=e_range)}
    for e in IDEAL_E:
        obs[f"law_e{e:g}"] = (lambda o, ol=cp.legal_outcome(st, e): float(np.linalg.norm(np.asarray(o, float) - ol)))
    obs["mass_blind"] = lambda o: _segment_dist(o, mb, (1 + e_range[0]) * u, (1 + e_range[1]) * u)
    for e in IDEAL_E:
        obs[f"speed_e{e:g}"] = (lambda o, s0=kin_scalar(st, cp.legal_outcome(st, e), "speed"): abs(kin_scalar(st, o, "speed") - s0))
    for e in IDEAL_E:
        obs[f"ke_e{e:g}"] = (lambda o, k0=kin_scalar(st, cp.legal_outcome(st, e), "ke"): abs(kin_scalar(st, o, "ke") - k0))
    return obs


def build_rows(kin: dict, spec: dict) -> dict:
    """Label-only bookkeeping per speed_matched contrast. kin: group -> (PreState, {twin: o4}).
    Returns {contrast name: [row dicts]}; a row is one violator clip with its matched legal partner."""
    out = {}
    for con in spec["contrasts"]:
        name, kind = con.get("name", con["kind"]), con["kind"]
        if kind not in GROUP_KINDS:
            raise ValueError(f"unknown group contrast kind {kind!r} (have {GROUP_KINDS})")
        by, gap = con.get("match", "ke"), con.get("max_log_gap")
        prefix = con.get("viol_prefix", "mom_violate")
        rows = []
        for g, (st, outs) in kin.items():
            lad = {t: outs[t] for t in spec["ladder"] if t in outs}
            if not lad:
                continue
            for v in sorted(t for t in outs if t.startswith(prefix)):
                partner, gk, gs = match_partner(st, outs[v], lad, by)
                xs = [kin_scalar(st, o, by) for o in lad.values()]
                bracketed = min(xs) <= kin_scalar(st, outs[v], by) <= max(xs)
                rows.append({"group": g, "viol": v, "sign": v.rsplit("_", 1)[-1], "eta": eta_outcome(st, outs[v]),
                             "energy_physical": kin_scalar(st, outs[v], "ke") <= kin_scalar(st, np.zeros(4), "ke") + 1e-12,
                             "legal": partner, "log_gap_ke": gk, "log_gap_speed": gs, "bracketed": bool(bracketed),
                             "matched": (gap is None or (gk if by == "ke" else gs) <= float(gap))
                             and (bracketed or not con.get("require_bracketed", False))})
        if con.get("balance_faster", False):
            # faster and slower violators (vs their partner, in the matching scalar) get equal total weight: an
            # observer that only knows the expected speed at a ladder edge then sits at exactly 0.5
            for r in rows:
                st, outs = kin[r["group"]]
                r["faster"] = kin_scalar(st, outs[r["viol"]], by) > kin_scalar(st, outs[r["legal"]], by)
            mr = [r for r in rows if r["matched"]]
            nf = sum(r["faster"] for r in mr)
            for r in mr:
                k = nf if r["faster"] else len(mr) - nf
                r["weight"] = len(mr) / (2.0 * k) if 0 < nf < len(mr) else 1.0
        out[name] = rows
    return out


def _row_subsets(rows: list[dict]) -> dict:
    """{subset name: boolean mask}: all, by drift sign, by energy_physical, by violator twin."""
    m = lambda f: np.array([bool(f(r)) for r in rows], bool)
    out = {"all": m(lambda r: True)}
    for sg in sorted({r["sign"] for r in rows}):
        out[f"sign={sg}"] = m(lambda r, sg=sg: r["sign"] == sg)
    out["energy_physical"] = m(lambda r: r["energy_physical"])
    out["not energy_physical"] = ~out["energy_physical"]
    for v in sorted({r["viol"] for r in rows}):
        out[v] = m(lambda r, v=v: r["viol"] == v)
    return {k: v for k, v in out.items() if v.any()}


def _row_index(rows: list[dict], clip_ix: dict) -> tuple[np.ndarray, np.ndarray]:
    """(iv [n], il [n]): clip indices of each row's violator and legal partner; -1 = missing."""
    iv = np.array([clip_ix.get(f"{r['group']}/{r['viol']}", -1) for r in rows], int)
    il = np.array([clip_ix.get(f"{r['group']}/{r['legal']}", -1) for r in rows], int)
    return iv, il


def score_rows(a: np.ndarray, iv: np.ndarray, il: np.ndarray) -> np.ndarray:
    """Per row win = s_viol > s_partner (ties 1/2) from a per-clip surprise array ``a``; NaN where a clip is
    missing (index -1 or a NaN value)."""
    ax = np.append(np.asarray(a, float), np.nan)        # index -1 -> NaN
    sv, sl = ax[iv], ax[il]
    with np.errstate(invalid="ignore"):
        win = np.where(np.isclose(sv, sl, rtol=1e-12, atol=1e-12), 0.5, (sv > sl).astype(float))
    win[~(np.isfinite(sv) & np.isfinite(sl))] = np.nan
    return win


def _wmean(x: np.ndarray, grp: np.ndarray, wg: np.ndarray, mask: np.ndarray | None = None,
           rw: np.ndarray | None = None) -> float:
    """Mean of row values with per-group bootstrap weights (a group drawn k times counts k times) and
    optional row weights ``rw`` (speed_matched ``balance_faster``)."""
    ok = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    w = wg[grp[ok]] * (1.0 if rw is None else rw[ok])
    s = w.sum()
    return float((w * x[ok]).sum() / s) if s > 0 else float("nan")


def _avg(x: np.ndarray, mask: np.ndarray, rw: np.ndarray) -> float:
    """Row-weighted mean over ``mask`` (NaN rows skipped)."""
    return _wmean(x, np.zeros(len(x), int), np.ones(1), mask, rw)


def _rw(rows: list[dict]) -> np.ndarray:
    return np.array([r.get("weight", 1.0) for r in rows], float)


def _q(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [float("nan")] * 2


def _weights(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n_groups] multiplicities of a bootstrap over groups."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups) for _ in range(n_boot)])


def _wilcoxon_groups(d: np.ndarray, grp: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    ok = np.isfinite(d)
    gm = np.array([d[ok & (grp == g)].mean() for g in np.unique(grp[ok])])
    return float(wilcoxon(gm).pvalue) if len(gm) > 1 and np.any(gm != 0) else 1.0


def ideal_group_bands(kin: dict, rows_by: dict, spec: dict) -> dict:
    """Labels-only win rates of the reference observers (``group_observers``), per group contrast and row subset."""
    obs = {g: group_observers(st) for g, (st, _) in kin.items()}
    clips = [(g, t) for g, (_, outs) in kin.items() for t in outs]
    clip_ix = {f"{g}/{t}": i for i, (g, t) in enumerate(clips)}
    names = list(next(iter(obs.values()))) if obs else []
    a = {ob: np.array([obs[g][ob](kin[g][1][t]) for g, t in clips]) for ob in names}
    out = {}
    for con in spec["contrasts"]:
        name = con.get("name", con["kind"])
        rows = [r for r in rows_by[name] if r["matched"]]
        if not rows:
            continue
        iv, il = _row_index(rows, clip_ix)
        subs, rw = _row_subsets(rows), _rw(rows)
        res = {"observers": names, "n": len(rows), "win_rate": {}}
        for ob in names:
            win = score_rows(a[ob], iv, il)
            res["win_rate"][ob] = {k: _avg(win, m, rw) for k, m in subs.items()}
        sp = np.array([kin_scalar(kin[g][0], kin[g][1][t], "speed") for g, t in clips])
        res["viol_faster_than_partner"] = float(np.mean(sp[iv] > sp[il]))
        res["viol_faster_than_partner_weighted"] = _avg((sp[iv] > sp[il]).astype(float), np.ones(len(rows), bool), rw)
        out[name] = res
    return out


def group_contrasts(kin: dict, surprise: dict, spec: dict, model: str | None = None, n_boot: int = 1000,
                    seed: int = 0) -> dict:
    """F2 speed_matched contrasts.

    kin: group -> (PreState, {twin: o4}) from labels; surprise: source -> clip -> {metric: value} (clip =
    "group/twin"), incl. "pixel_baselines" (the pixel gap of ``model``'s rows, ``vs_pixel``)."""
    from scipy.stats import spearmanr
    rows_by = build_rows(kin, spec)
    out = {"ladder": spec["ladder"], "contrasts": {},
           "ideal_observer": ideal_group_bands(kin, rows_by, spec)}
    for con in spec["contrasts"]:
        name, kind = con.get("name", con["kind"]), con["kind"]
        all_rows = rows_by[name]
        rows = [r for r in all_rows if r["matched"]]
        rec = {"kind": kind, "config": con, "n_rows": len(rows), "n_unmatched": len(all_rows) - len(rows),
               "n_groups": len({r["group"] for r in rows}), "raw": {}}
        if rows:
            gk = np.array([r["log_gap_ke"] for r in rows])
            pc: dict = {}
            for r in rows:
                pc[r["legal"]] = pc.get(r["legal"], 0) + 1
            rec["matching"] = {"by": con.get("match", "ke"), "max_log_gap": con.get("max_log_gap"),
                               "require_bracketed": bool(con.get("require_bracketed", False)),
                               "share_bracketed_all_rows": float(np.mean([r["bracketed"] for r in all_rows])),
                               "balance_faster": bool(con.get("balance_faster", False)),
                               "share_faster_in_match_scalar": float(np.mean([r.get("faster", np.nan) for r in rows])) if con.get("balance_faster") else None,
                               "median_abs_log_ke_ratio": float(np.median(gk)), "p90_abs_log_ke_ratio": float(np.quantile(gk, 0.9)),
                               "median_abs_log_speed_ratio": float(np.median([r["log_gap_speed"] for r in rows])),
                               "median_abs_log_ke_ratio_all_rows": float(np.median([r["log_gap_ke"] for r in all_rows])),
                               "partner_counts": dict(sorted(pc.items()))}
        if rows:             # audit M3: the pos (faster) drift creates energy in many groups (always at e = 1)
            rec["energy_creating_share_by_sign"] = {
                sg: float(np.mean([not r["energy_physical"] for r in rows if r["sign"] == sg]))
                for sg in sorted({r["sign"] for r in rows})}
        out["contrasts"][name] = rec
        if not rows:
            continue
        ug = sorted({r["group"] for r in rows})
        grp = np.array([ug.index(r["group"]) for r in rows])
        W = _weights(len(ug), n_boot, seed)
        subs, rw = _row_subsets(rows), _rw(rows)
        eta = np.array([r["eta"] for r in rows])
        pixr = surprise.get("pixel_baselines", {})
        for src, recs in surprise.items():
            clips = sorted(recs)
            clip_ix = {c: i for i, c in enumerate(clips)}
            iv, il = _row_index(rows, clip_ix)
            for m in sorted({m for c in recs.values() for m in c}):
                a = np.array([recs[c].get(m, np.nan) for c in clips], float)
                win = score_rows(a, iv, il)
                d = np.append(a, np.nan)[iv] - np.append(a, np.nan)[il]
                pwin = {}
                if src == model and pixr:
                    for bn in PIXEL_BASELINES:
                        k = pixel_key(bn, m)
                        pw = score_rows(np.array([pixr.get(c, {}).get(k, np.nan) for c in clips], float), iv, il)
                        if np.isfinite(pw).any():
                            pwin[bn] = pw
                for sub, mask in subs.items():
                    ok = mask & np.isfinite(win)
                    if not ok.any():
                        continue
                    r_ = {"n": int(ok.sum()), "n_groups": int(len(np.unique(grp[ok]))), "win_rate": _avg(win, ok, rw),
                          "win_rate_ci95": _q([_wmean(win, grp, w, mask, rw) for w in W]),
                          "wilcoxon_p_group_means": _wilcoxon_groups(np.where(ok, d, np.nan), grp)}
                    if np.std(eta[ok]) > 0 and np.std(d[ok]) > 0:
                        sp = spearmanr(d[ok], eta[ok])
                        r_["dose_spearman_eta"], r_["dose_spearman_p"] = float(sp.statistic), float(sp.pvalue)
                    vp = {}
                    for bn, pw in pwin.items():
                        mk = ok & np.isfinite(pw)
                        if mk.any():
                            vp[bn] = {"n": int(mk.sum()), "baseline_win": _avg(pw, mk, rw), "model_win": _avg(win, mk, rw),
                                      "model_minus_baseline": _avg(win, mk, rw) - _avg(pw, mk, rw),
                                      "model_minus_baseline_ci95": _q([_wmean(win, grp, w, mk, rw) - _wmean(pw, grp, w, mk, rw)
                                                                       for w in W])}
                    if vp:
                        r_["vs_pixel"] = vp
                    rec["raw"].setdefault(src, {}).setdefault(m, {})[sub] = r_
    return out


def group_table(gc: dict) -> str:
    """md for the F2 speed-matched contrasts: matching, ideal observers, win rates (all sources), pixel gap."""
    rows = ["## F2 speed-matched momentum violators (vs the KE-closest legal ladder twin)", ""]
    for name, c in gc["contrasts"].items():
        rows.append(f"### {name}: {c['n_rows']} violator clips in {c['n_groups']} groups"
                    + (f", {c['n_unmatched']} unmatched (left out)" if c.get("n_unmatched") else ""))
        rows.append("")
        if c.get("energy_creating_share_by_sign"):
            rows += ["Energy-creating violators (energy_physical False) by drift sign: " + ", ".join(
                f"{s} {v:.0%}" for s, v in c["energy_creating_share_by_sign"].items()), ""]
        if "matching" in c:
            mt = c["matching"]
            rows.append(f"Matching by {mt['by']} (max |log gap| {mt['max_log_gap']}): median |log KE_viol/KE_legal| "
                        f"{mt['median_abs_log_ke_ratio']:.3f} (p90 {mt['p90_abs_log_ke_ratio']:.3f}; all rows before the cut "
                        f"{mt['median_abs_log_ke_ratio_all_rows']:.3f}), median |log speed ratio| {mt['median_abs_log_speed_ratio']:.3f}; "
                        f"partners {mt['partner_counts']}")
            rows.append("")
        io = gc.get("ideal_observer", {}).get(name)
        if io:
            subs = list(next(iter(io["win_rate"].values())))
            rows += [f"Ideal observers (labels only, n={io['n']}): win rate; violator faster than its partner in "
                     f"{io['viol_faster_than_partner']:.0%}", "", "| observer | " + " | ".join(subs) + " |",
                     "|---|" + "---|" * len(subs)]
            for ob in io["observers"]:
                rows.append(f"| {ob} | " + " | ".join(f"{io['win_rate'][ob][s]:.2f}" for s in subs) + " |")
            rows.append("")
        rows += ["| source | distance | subset | n (groups) | win [95% CI] | Wilcoxon p (group means) | dose ρ(η) | "
                 "model − copy_last [CI] | model − const_vel [CI] |", "|---|---|---|---|---|---|---|---|---|"]
        for src, by_m in c["raw"].items():
            for m, by_s in by_m.items():
                for sub, s in by_s.items():
                    dose = f"{s['dose_spearman_eta']:+.2f}" if "dose_spearman_eta" in s else ""
                    rows.append(f"| {src} | {m} | {sub} | {s['n']} ({s['n_groups']}) | {s['win_rate']:.2f} {_ci(s['win_rate_ci95'])} "
                                f"| {s['wilcoxon_p_group_means']:.2g} | {dose} | {_gap(s.get('vs_pixel'), 'copy_last')} | "
                                f"{_gap(s.get('vs_pixel'), 'const_velocity')} |")
        rows.append("")
    return "\n".join(rows)


def group_kinematics(lab, groups: list[str], twins_of) -> dict:
    """group -> (PreState, {twin: o4}) from labels (laws.from_labels; the twins share one pre-state)."""
    from physlaw.data import collision_physics as cp
    from physlaw.eval import laws
    kin = {}
    for g in groups:
        outs, st = {}, None
        for t in twins_of(g):
            pv, qv, ctx = laws.from_labels(lab(f"{g}/{t}"))
            if st is None:
                st = cp.PreState(float(ctx["m"][0]), float(ctx["m"][1]), pv["v"][0], pv["v"][1], ctx["n"])
            outs[t] = (qv["v"] - pv["v"]).reshape(4)
        if st is not None:
            kin[g] = (st, outs)
    return kin


def _ci(x: list) -> str:
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]"


def contrast_lines(info: dict, primary: str | None) -> list[str]:
    """Per contrast: a header (primary marked), its dose magnitude and the ideal-observer bands."""
    rows = []
    for c, i in info.items():
        rows.append(f"- **{c}** (law {i.get('law', 'momentum')}){' (PRIMARY contrast)' if c == primary else ''}; dose magnitude: "
                    f"{'η of the violating twin' if i.get('magnitude') == 'eta' else i.get('magnitude') or 'none'}")
        if i.get("also_breaks_energy"):
            sh = i["energy_creating_share"]
            rows.append("  - **also breaks energy** (energy_physical False; a model surprised by energy creation alone "
                        "wins these rows): " + ", ".join(f"{v} in {sh[v]:.0%} of groups" for v in i["also_breaks_energy"])
                        + ("; see the `[half <twin>]` rows" if len(sh) > 1 else ""))
        if "ideal_observer" in i:
            rows.append(f"  - {ideal_line(i['ideal_observer'])}")
    return rows


def _gap(vp: dict | None, bn: str) -> str:
    g = (vp or {}).get(bn)
    return f"{g['model_minus_baseline']:+.2f} {_ci(g['model_minus_baseline_ci95'])}" if g else ""


def table(res: dict) -> str:
    rows = []
    cw = res.get("clean_window")
    if cw:
        rows += [f"Clean window: {cw['n_groups_with_window']}/{cw['n_groups']} groups have one (median "
                 f"{cw['window_s_median']:.2f} s after the event); per source "
                 + "; ".join(f"{s}: {v}" for s, v in cw["sources"].items()) + ". " + cw["note"], ""]
    if res.get("contrast_info"):
        rows += [*contrast_lines(res["contrast_info"], res.get("primary_contrast")), ""]
    rows += ["| contrast | source | distance | n | win rate [95% CI] | Wilcoxon p (Holm) | Cliff's δ | median rel. diff | "
             "ρ vs magnitude | model − copy_last [CI] | model − const_vel [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, by_src in res["contrasts"].items():
        for src, by_m in by_src.items():
            for m, s in by_m.items():
                rho = s.get("spearman_vs_magnitude")
                rows.append(f"| {c}{' (primary)' if c == res.get('primary_contrast') else ''} | {src} | {m} | {s['n']} | {s['win_rate']:.2f} "
                            f"[{s['win_rate_ci95'][0]:.2f}, {s['win_rate_ci95'][1]:.2f}] | {s['wilcoxon_p']:.2g} "
                            f"({s.get('wilcoxon_p_holm', float('nan')):.2g}) | "
                            f"{s['cliffs_delta']:+.2f} | {s['median_rel_diff']:+.1%} | "
                            f"{'' if rho is None else f'{rho:+.2f}'} | {_gap(s.get('vs_pixel'), 'copy_last')} | "
                            f"{_gap(s.get('vs_pixel'), 'const_velocity')} |")
    if res.get("group_contrasts"):
        rows += ["", group_table(res["group_contrasts"])]
    if res.get("energy_kink"):
        rows += ["", kink_table(res["energy_kink"])]
    if res.get("energy_kink_rep"):
        rows += ["", rep_table(res["energy_kink_rep"])]
    if res.get("group_kink"):
        rows += ["", group_kink_table(res["group_kink"])]
    return "\n".join(rows)


# --------------------------------------------------------------------------- energy kink (spec §5.0.2, §5.0.4 L2)
# A ladder of twins sharing the pre-event frames, ordered by the generalised restitution ε of the exchange event on an
# equal-step grid (F2 e0.0..e1.5 collision, F7 bounce eps0.5..eps1.5, F8 pad eps0.5..eps1.5). x0 = 1, the law
# boundary, is interior: energy is created exactly when ε > 1. Inside the clean window every kinematic correlate is
# affine in ε, so smooth through ε = 1; a model that knows the energy law should be surprised in a way that BENDS
# at x0, whatever it expects inside the legal range:
#
#   per group, s(x) on the ladder (log surprise, clean window), standardised within the group
#   (zero mean, unit sd over the ladder, so K is in within-group sd units and comparable across sources);
#   second differences  D2(x) = s(x+h) - 2 s(x) + s(x-h)  at the interior points;
#   K = D2(x0) - p(x0), p = the smooth curvature trend: a linear fit of D2(x) over the reference interior points
#       {x : |x - x0| > exclude_radius, |x - x_hat| > exclude_radius} (quadratic when >= 5 points and it wins
#       leave-one-out; :func:`curvature_trend`). A kink is a spike of curvature above the smooth curvature trend.
#
# x_hat = the source's expected dose = argmin of the group-averaged curve (parabolic refinement between grid points),
# fitted LEAVE-ONE-GROUP-OUT (each group's x_hat comes from the other groups, so the exclusion never looks at the
# group being scored). Excluding D2 near x_hat removes the V / U a "novelty" observer has at its own expectation, and
# subtracting the trend removes constant and smoothly varying curvature (convex motion trends, exp-like growth, the
# |v'| hyperbola of a ball that stops beyond the ladder) and every linear trend.
# Inconclusive: |x_hat - x0| < exclude_radius (a V at x_hat and a kink at x0 are the same shape) or x0 is not an
# interior ladder point (no ladder points on both sides). Aggregate: mean K over groups, group-bootstrap 95% CI,
# share of groups with K > 0.
#
# Baselines through the same statistic: the pixel baselines (copy_last, const_velocity; scripts/05_surprise.py) and,
# in a separate run, the untrained model. The model's kink counts only if K_model exceeds every pixel baseline:
# paired per-group differences with group-bootstrap CIs, all lower bounds > 0 (and the untrained model's K CI holds 0;
# PC pass check). Labels-only observers (novelty at x̂, speed-only, motion trends, the law hinge) are not run: by
# construction (x̂ exclusion, curvature-trend subtraction, stop-dose exclusion, affine kinematics) they sit at K = 0 / > 0,
# which tests/test_l2_kink.py asserts.
# Kinematic-V exclusion (from labels, per group; the post velocities are affine in the dose, so speeds are analytic):
# a correlate with a V or a minimum within exclude_radius of x0 kinks there without any law, so the group is
# excluded from K (and K_rep) and scored apart (``excluded_groups``) when, within exclude_radius of x0, a ball stops
# (|v'| minimal: ``stop``; for nearly equal masses A stops at e = m_A/m_B ~ 1), the total post speed S = Σ|v'| or the
# post KE is minimal (``speed_min``), or S(x) = S(x̂) at some x != x̂ (``speed_match``: the second V of an observer
# that expects the speed of the model's leave-one-group-out x̂; per distance). Doses within exclude_radius of any of
# these kinematic-V doses are also left out of the curvature-trend fit (no reference point left: K = NaN, not scored).

KINK_KIND = "energy_kink"
GROUP_KINK_KIND = "group_kink"          # across groups on a scene constant (F11 statics; :func:`group_kink`)
KINK_KINDS = (KINK_KIND, GROUP_KINK_KIND)
KINK_BASELINES = tuple(PIXEL_BASELINES)


def kink_specs(cfg, family: str) -> list[dict]:
    """The ``energy_kink`` and ``group_kink`` entries of ``contrasts.<family>.group_contrasts`` (empty if none)."""
    c = cfg["contrasts"]
    fam = c.get(family) if isinstance(c, dict) else None
    if not fam:
        return []
    return [dict(x) for x in fam.get("group_contrasts") or [] if x["kind"] in KINK_KINDS]


def ladder_doses(twins: list[str], doses=None) -> np.ndarray:
    """Doses of the ladder twins: ``doses`` if given, else the trailing number of each name (e1.3 -> 1.3,
    k0.7 -> 0.7)."""
    import re
    if doses is not None:
        return np.asarray(doses, float)
    out = []
    for t in twins:
        mt = re.search(r"(\d+(?:\.\d+)?)$", t)
        if not mt:
            raise ValueError(f"no dose in twin name {t!r}; give `doses` in the config")
        out.append(float(mt.group(1)))
    return np.array(out)


def fit_x_hat(curves: np.ndarray, x: np.ndarray) -> float:
    """argmin over x of the group-averaged curve (NaN groups skipped), refined by a parabola through the minimum
    and its two neighbours when the minimum is interior and the parabola is convex. Ties: the smallest x."""
    mc = np.nanmean(np.atleast_2d(curves), 0)
    if not np.isfinite(mc).any():
        return float("nan")
    i = int(np.nanargmin(mc))
    if 0 < i < len(x) - 1 and np.isfinite(mc[i - 1:i + 2]).all():
        a, b, c = mc[i - 1], mc[i], mc[i + 1]
        den = a - 2 * b + c
        if den > 1e-12 * max(1.0, abs(a) + abs(b) + abs(c)):
            return float(x[i] + 0.5 * (a - c) / den * (x[i + 1] - x[i]))
    return float(x[i])


def _standardise(curves: np.ndarray) -> np.ndarray:
    c = np.atleast_2d(np.asarray(curves, float))
    mu = c.mean(1, keepdims=True)
    sd = c.std(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12 * np.maximum(1.0, np.abs(mu)), (c - mu) / sd, np.nan)


def curvature_trend(xr: np.ndarray, yr: np.ndarray, x0: float) -> float:
    """The smooth curvature trend at x0 from the reference points (xr, yr = D² or ‖D²a‖): a linear least-squares fit
    (a constant with one point), quadratic when there are >= 5 points and its leave-one-out error is lower."""
    xr, yr = np.asarray(xr, float), np.asarray(yr, float)
    if len(xr) < 2:
        return float(yr[0]) if len(xr) else float("nan")

    def loo(deg):
        return sum((np.polyval(np.polyfit(np.delete(xr, i), np.delete(yr, i), deg), xr[i]) - yr[i]) ** 2
                   for i in range(len(xr)))
    deg = 2 if len(xr) >= 5 and loo(2) < loo(1) else 1
    return float(np.polyval(np.polyfit(xr, yr, deg), x0))


def kink_stat(curves, x, x0: float = 1.0, exclude_radius: float = 0.2, x_hat=None, standardize: bool = True,
              step_tol: float = 1e-6, stops=None, novelty: bool = True) -> dict:
    """The kink statistic per group. curves [G, L] surprise per group on the ladder doses x [L] (equal steps);
    x_hat: None (leave-one-group-out fit of each group's expected dose; in-sample if G == 1), a scalar, or [G];
    novelty False: no expected dose at all (across-group curves, :func:`group_kink`: nothing excluded around x̂);
    stops: per group, kinematic-V doses (stops, minima, speed matches) whose neighbourhood (exclude_radius) is left
    out of the trend fit (a group left with no reference point gets K = NaN: not scored).
    x0 on the grid: the curvature is D²(x0). x0 midway between two interior doses (F11's grid 0.55 … 1.45 around 1):
    the curvature across x0 is the slope change D²(x0 − h/2) + D²(x0 + h/2) (a step at x0 cancels in it), and every
    midpoint's pair sum is the reference curve.
    Returns {K_g [G], x_hat_g [G], x_hat (median of x_hat_g), inconclusive, reason, d2_mean, x_interior, stencil}."""
    x = np.asarray(x, float)
    C = np.atleast_2d(np.asarray(curves, float))
    if C.shape[1] != len(x):
        raise ValueError(f"curves have {C.shape[1]} doses, ladder has {len(x)}")
    steps = np.diff(x)
    if len(x) < 3 or not np.allclose(steps, steps[0], rtol=0, atol=step_tol) or steps[0] <= 0:
        raise ValueError(f"ladder doses must be increasing with an equal step: {x.tolist()}")
    S = _standardise(C) if standardize else C
    G = len(S)
    xi = x[1:-1]
    D2 = S[:, 2:] - 2 * S[:, 1:-1] + S[:, :-2]
    stencil = "point"
    if not np.isclose(xi, x0, rtol=0, atol=step_tol).any() and len(xi) > 1:
        mid = (xi[:-1] + xi[1:]) / 2
        if np.isclose(mid, x0, rtol=0, atol=step_tol).any():
            xi, D2, stencil = mid, D2[:, :-1] + D2[:, 1:], "midpoint"
    out = {"x_interior": xi.tolist(), "step": float(steps[0]), "x0": float(x0), "exclude_radius": float(exclude_radius),
           "n_groups": int(G), "stencil": stencil}
    hit = np.flatnonzero(np.isclose(xi, x0, rtol=0, atol=step_tol))
    if not novelty:
        xh = np.full(G, np.nan)
        out["x_hat_fit"] = "none (no expected dose)"
    elif x_hat is None:
        if G > 1:
            xh = np.array([fit_x_hat(np.delete(S, g, 0), x) for g in range(G)])
            out["x_hat_fit"] = "leave-one-group-out"
        else:
            xh = np.array([fit_x_hat(S, x)])
            out["x_hat_fit"] = "in-sample (one group)"
    else:
        xh = np.broadcast_to(np.asarray(x_hat, float), (G,)).copy()
        out["x_hat_fit"] = "given"
    x_hat_med = float(np.nanmedian(xh)) if np.isfinite(xh).any() else float("nan")
    out.update(x_hat=x_hat_med, x_hat_g=xh.tolist(), d2_mean=np.nanmean(D2, 0).tolist() if G else [])
    out["x_hat_all_groups"] = fit_x_hat(S, x) if novelty else float("nan")
    if not len(hit):
        out.update(K_g=[float("nan")] * G, inconclusive=True,
                   reason=f"ladder lacks points on both sides of x0 = {x0:g} (interior doses {xi.tolist()})")
        return out
    i0 = int(hit[0])
    K = np.full(G, np.nan)
    n_ref = np.zeros(G, int)
    for g in range(G):
        keep = (np.abs(xi - x0) > exclude_radius + step_tol) & np.isfinite(D2[g])
        if np.isfinite(xh[g]):
            keep &= np.abs(xi - xh[g]) > exclude_radius + step_tol
        for c in (stops[g] if stops is not None and stops[g] else []):
            if np.isfinite(c):
                keep &= np.abs(xi - c) > exclude_radius + step_tol
        n_ref[g] = int(keep.sum())
        if keep.any() and np.isfinite(D2[g, i0]):
            K[g] = D2[g, i0] - curvature_trend(xi[keep], D2[g, keep], x0)
    out["K_g"] = K.tolist()
    out["n_reference_points_median"] = float(np.median(n_ref)) if G else 0.0
    reason = None
    if novelty and not np.isfinite(x_hat_med):
        reason = "expected dose x_hat undefined (flat curves)"
    elif novelty and abs(x_hat_med - x0) < exclude_radius - step_tol:
        reason = (f"|x_hat - x0| = {abs(x_hat_med - x0):.2f} < {exclude_radius:g}: a V at x_hat and a kink at x0 "
                  "cannot be told apart")
    elif not np.isfinite(K).any():
        reason = "no reference interior points left after excluding x_hat's neighbourhood"
    out.update(inconclusive=reason is not None, reason=reason)
    return out


def _kink_summary(st: dict, W: np.ndarray) -> dict:
    """Mean K over groups, group-bootstrap CI (W: [n_boot, G] multiplicities), share of groups with K > 0."""
    K = np.asarray(st["K_g"], float)
    ok = np.isfinite(K)
    boots = [float((w[ok] * K[ok]).sum() / w[ok].sum()) if (w[ok].sum() > 0) else np.nan for w in W]
    keys = ("x_hat", "x_hat_all_groups", "x_hat_fit", "inconclusive", "reason", "n_reference_points_median",
            "d2_mean", "x_interior")
    return {"K": float(K[ok].mean()) if ok.any() else float("nan"), "K_ci95": _q(boots),
            "share_K_pos": float((K[ok] > 0).mean()) if ok.any() else float("nan"), "n_groups": int(ok.sum()),
            "x_hat_g_range": [float(np.nanmin(st["x_hat_g"])), float(np.nanmax(st["x_hat_g"]))]
            if np.isfinite(st["x_hat_g"]).any() else [float("nan")] * 2,
            **{k: st[k] for k in keys if k in st}}


def _kink_diff(st_a: dict, st_b: dict, W: np.ndarray) -> dict:
    """Paired per-group difference K_a - K_b: mean and group-bootstrap CI."""
    d = np.asarray(st_a["K_g"], float) - np.asarray(st_b["K_g"], float)
    ok = np.isfinite(d)
    boots = [float((w[ok] * d[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan for w in W]
    return {"diff": float(d[ok].mean()) if ok.any() else float("nan"), "diff_ci95": _q(boots), "n_groups": int(ok.sum())}


def kink_compare(model_curves: np.ndarray, x, x0: float = 1.0, exclude_radius: float = 0.2,
                 baselines: dict | None = None, n_boot: int = 1000, seed: int = 0, stops=None,
                 x_hat=None, x_hat_baselines: dict | None = None) -> dict:
    """K for the model and the given baseline curves (name -> [G, L], the pixel baselines), the paired differences
    model − baseline, and the verdict: ``kink beyond baselines`` only if the model's K CI is above 0 and every
    baseline is present with its difference CI above 0. ``x_hat`` / ``x_hat_baselines[name]``: per-group expected
    doses [G] to use (None: kink_stat's leave-one-group-out fit over these curves)."""
    x = np.asarray(x, float)
    st_m = kink_stat(model_curves, x, x0, exclude_radius, x_hat=x_hat, stops=stops)
    W = _weights(len(np.atleast_2d(model_curves)), n_boot, seed)
    rec = {"model": _kink_summary(st_m, W), "baselines": {}, "vs_baselines": {}}
    for name, c in (baselines or {}).items():
        c = np.atleast_2d(np.asarray(c, float))
        if c.shape != np.atleast_2d(model_curves).shape or not np.isfinite(c).any():
            continue
        st = kink_stat(c, x, x0, exclude_radius, x_hat=(x_hat_baselines or {}).get(name), stops=stops)
        rec["baselines"][name] = _kink_summary(st, W)
        rec["vs_baselines"][name] = _kink_diff(st_m, st, W)
    lo = rec["model"]["K_ci95"][0]
    beats = (set(rec["vs_baselines"]) == set(KINK_BASELINES)
             and all(v["diff_ci95"][0] > 0 for v in rec["vs_baselines"].values()))
    rec["inconclusive"], rec["reason"] = st_m["inconclusive"], st_m["reason"]
    rec["verdict"] = ("inconclusive: " + st_m["reason"] if st_m["inconclusive"] else
                      "kink beyond baselines" if lo > 0 and beats else
                      "kink, not beyond every baseline" if lo > 0 else "no kink")
    return rec


# --------------------------------------------------------------------------- clean window and stop dose (spec §5.0.2)
# Per-frame surprise (written by scripts/05_surprise.py for the pixel baselines; stage 4 should write the same for
# the model's surprise):  rec["per_frame"] = {"t_s": [T], "surprise": {metric: [T]}, "weight": {metric: [T]}}
# t_s = absolute clip time (s) of the LAST master frame of each post-context (token-)frame, so a frame counts only
# when all of it is inside the window; weight (optional, default 1) = the frame's share of the metric's mean (e.g.
# object-mask pixel count). The windowed metric = weighted mean over frames with t_s <= t_end.

CLEAN_NOTE = ("Surprise restricted per group to post-event frames ending before min over the group's twins of "
              "clean_until_s; 'unrestricted' = records with a scalar surprise only (stage 4 stores no per-frame model "
              "surprise yet) or groups without clean_until_s.")


def event_time_s(labels: dict) -> float:
    """Time (s) of the first post-event master frame (physlaw.first_post_contact_frame), else physlaw.event_time_s."""
    pl = labels["physlaw"]
    if pl.get("first_post_contact_frame") is not None:
        return float(pl["first_post_contact_frame"]) / float(labels["scene_constants"]["frame_rate"])
    return float(pl["event_time_s"])


def clean_until_s(labels: dict) -> float | None:
    """physlaw.clean_until_s; for old labels (F1) from per_frame: the time of the first frame after the event at which
    an object is out of frame or the two objects touch again (centre distance <= sum of radii after having separated),
    else the last frame's time. None without either."""
    pl = labels["physlaw"]
    if pl.get("clean_until_s") is not None:
        return float(pl["clean_until_s"])
    pf, fpc = labels.get("per_frame"), pl.get("first_post_contact_frame")
    if not pf or fpc is None or not isinstance(pl.get("objects"), dict):
        return None
    objs = list(pl["objects"])
    rad = [pl["objects"][o].get("radius") for o in objs]
    pair = len(objs) == 2 and all(r is not None for r in rad)
    apart = False
    for f in pf[int(fpc):]:
        if not all(f[o]["in_frame"] for o in objs):
            return float(f["time_s"])
        if pair:
            d = float(np.linalg.norm(np.subtract(f[objs[0]]["position"][:2], f[objs[1]]["position"][:2])))
            if apart and d <= (rad[0] + rad[1]) * 1.001:
                return float(f["time_s"])
            apart |= d > (rad[0] + rad[1]) * 1.01
    return float(pf[-1]["time_s"])


def group_windows(data_dir: Path, lab) -> dict:
    """group -> {"t_event", "t_end"} (t_end = min over the group's twins of clean_until_s), or None when a twin has
    no clean_until_s and no per_frame to compute it from."""
    out = {}
    for gd in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        clips = [f"{gd.name}/{q.name}" for q in sorted(gd.iterdir()) if (q / "labels.json").exists()]
        if not clips:
            continue
        ends = [clean_until_s(lab(c)) for c in clips]
        out[gd.name] = (None if any(e is None for e in ends) else
                        {"t_event": event_time_s(lab(clips[0])), "t_end": float(min(ends))})
    return out


def window_surprise(rec: dict, t_end: float) -> dict | None:
    """{metric: windowed value} from rec["per_frame"] (frames with t_s <= t_end); None if no frame qualifies."""
    pf = rec["per_frame"]
    keep = np.asarray(pf["t_s"], float) <= t_end + 1e-9
    out = {}
    for m, v in pf["surprise"].items():
        v = np.asarray(v, float)
        w = np.asarray((pf.get("weight") or {}).get(m, np.ones(len(v))), float)
        ok = keep & np.isfinite(v) & (w > 0)
        if ok.any():
            out[m] = float((w[ok] * v[ok]).sum() / w[ok].sum())
    return out or None


def apply_clean_window(idx: dict, windows: dict) -> tuple[dict, dict]:
    """(idx with each record's surprise restricted to its group's clean window, the l2.json ``clean_window`` block).
    Records without per_frame (or in groups without a window) are kept unrestricted; records whose window holds no
    post-event frame are dropped."""
    w_s = [w["t_end"] - w["t_event"] for w in windows.values() if w is not None]
    info = {"note": CLEAN_NOTE, "n_groups": len(windows), "n_groups_with_window": len(w_s),
            "window_s_median": float(np.median(w_s)) if w_s else float("nan"), "sources": {}}
    out = {}
    for src, recs in idx.items():
        cnt = {"restricted": 0, "unrestricted": 0, "dropped_empty_window": 0}
        new = {}
        for c, r in recs.items():
            w = windows.get(r["group"])
            if w is None or "per_frame" not in r:
                cnt["unrestricted"] += 1
                new[c] = r
                continue
            s = window_surprise(r, w["t_end"])
            if s is None:
                cnt["dropped_empty_window"] += 1
                continue
            cnt["restricted"] += 1
            new[c] = {**r, "surprise": s}
        out[src] = new
        info["sources"][src] = cnt
    return out, info


def scored_objects(labels: dict) -> list[str]:
    """The labels' objects without the ``aux`` bodies (F9 rods, F11 prop): never scored (spec §5.0.4)."""
    ob = labels["physlaw"]["objects"]
    return [o for o in ob if not (isinstance(ob, dict) and isinstance(ob[o], dict) and ob[o].get("aux"))]


def stop_doses(labs: list[dict], x: np.ndarray) -> list[float] | None:
    """Per-object stop doses of a group: labels physlaw.stop_dose ({object: ε}, the same in every rung); for old
    labels from the rungs' post_velocity (affine in ε: v'(x) = a + b x, |v'| minimal at x* = −a·b / |b|²).
    None when neither is available."""
    for l in labs:
        sd = l["physlaw"].get("stop_dose")
        if sd is not None:
            return [float(v) for v in (sd.values() if isinstance(sd, dict) else sd) if v is not None]
    try:
        objs = scored_objects(labs[0])
        V = np.array([[np.asarray(l["physlaw"]["post_velocity"][o], float)[:2] for o in objs] for l in labs])
    except (KeyError, TypeError, ValueError):
        return None
    A = np.stack([np.ones_like(x), x], 1)
    out = []
    for j in range(V.shape[1]):
        (a, b), *_ = np.linalg.lstsq(A, V[:, j], rcond=None)
        out.append(float(-(a @ b) / (b @ b)) if b @ b > 1e-12 else float("nan"))
    return out


# --------------------------------------------------------------------------- energy kink record

def ladder_kinematics(labs: list[dict], x: np.ndarray) -> dict | None:
    """A group's post-event velocities as functions of the dose, from its rungs' labels: v'_i(x) = a_i + b_i x
    (in-plane, least squares over the rungs; affine by construction), and the masses (None when absent).
    None without physlaw.post_velocity."""
    try:
        objs = scored_objects(labs[0])
        V = np.array([[np.asarray(l["physlaw"]["post_velocity"][o], float)[:2] for o in objs] for l in labs])
    except (KeyError, TypeError, ValueError):
        return None
    coef = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), V.reshape(len(x), -1), rcond=None)[0]
    ob = labs[0]["physlaw"]["objects"]
    m = [ob[o].get("mass") for o in objs] if isinstance(ob, dict) else None
    return {"a": coef[0].reshape(-1, 2), "b": coef[1].reshape(-1, 2),
            "m": None if m is None or any(v is None for v in m) else np.asarray(m, float)}


def _grid_speeds(k: dict, xs: np.ndarray) -> np.ndarray:
    """|v'_i(x)| on the doses xs: [len(xs), n_obj]."""
    return np.linalg.norm(k["a"][None] + xs[:, None, None] * k["b"][None], axis=-1)


def _grid(x: np.ndarray) -> np.ndarray:
    return np.linspace(x[0], x[-1], 100 * (len(x) - 1) + 1)


def kinematic_minima(k: dict, x: np.ndarray) -> dict:
    """Doses (on the ladder's range) where the total post speed S = Σ|v'_i| and the post KE are minimal."""
    xs = _grid(x)
    sp = _grid_speeds(k, xs)
    out = {"S": float(xs[np.argmin(sp.sum(1))])}
    if k["m"] is not None:
        out["KE"] = float(xs[np.argmin((0.5 * k["m"] * sp ** 2).sum(1))])
    return out


def speed_match_doses(k: dict, x: np.ndarray, x_hat: float) -> list[float]:
    """Doses x != x_hat (on the ladder's range) where S(x) = S(x_hat): the second V of an observer that only
    expects the total post speed of its expected dose x_hat."""
    if not np.isfinite(x_hat):
        return []
    xs = _grid(x)
    S = _grid_speeds(k, xs).sum(1)
    d = S - _grid_speeds(k, np.array([x_hat])).sum()
    sg = np.where(d >= 0, 1, -1)
    i = np.flatnonzero(sg[1:] != sg[:-1])
    roots = xs[i] - d[i] * (xs[i + 1] - xs[i]) / (d[i + 1] - d[i])
    return [float(r) for r in roots if abs(r - x_hat) > 2 * (xs[1] - xs[0])]


def logo_x_hat(curves: dict, fit_groups: list, x) -> dict:
    """{group: expected dose x̂} from per-group curves (standardised within group, as :func:`kink_stat`): a group of
    ``fit_groups`` gets the argmin of the mean curve of the OTHER fit groups (in-sample when it is the only one), any
    other group that of all fit groups (NaN without fit groups)."""
    S = {g: _standardise(c)[0] for g, c in curves.items()}
    F = np.array([S[g] for g in fit_groups]) if fit_groups else None
    out = {}
    for g in curves:
        if F is None:
            out[g] = float("nan")
        elif g in fit_groups and len(fit_groups) > 1:
            out[g] = fit_x_hat(np.delete(F, fit_groups.index(g), 0), x)
        else:
            out[g] = fit_x_hat(F, x)
    return out


def _near(doses, x0: float, rad: float) -> bool:
    return any(np.isfinite(v) and abs(v - x0) < rad for v in doses or [])


def _kink_distances(gs: list[str], twins: list[str], x, x0: float, rad: float, surprise: dict, pixel,
                    n_boot: int, seed: int, vdoses: dict, kin: dict, excluded: list[str]) -> dict:
    """{distance: kink record (log surprise; :func:`kink_compare`)} over the groups of ``gs`` not in ``excluded``
    and not excluded by the speed-match rule at this distance's x̂ (``excluded_speed_match`` {group: doses}); the
    excluded groups (both rules) get their own record ``excluded_groups``. vdoses: group -> its kinematic-V doses
    (stops, minima); with the speed-match doses they are left out of the curvature-trend fit.
    ONE expected dose per group and source, used both for the speed-match exclusion and inside K (x̂ neighbourhood
    left out of the trend fit): leave-one-group-out over the groups kept by the stop / minimum rules (all groups if
    none is kept; :func:`logo_x_hat`). The pixel baselines get theirs the same way from their own curves."""
    out = {}
    clips = {g: [f"{g}/{t}" for t in twins] for g in gs}
    metrics = sorted({m for row in clips.values() for c in row for m in surprise[c]})
    lg = lambda v: np.log(v) if v is not None and np.isfinite(v) and v > 0 else np.nan
    for m in metrics:
        S = {g: np.array([lg(surprise[c].get(m)) for c in clips[g]]) for g in gs}
        pix = {bn: {g: np.array([lg((pixel or {}).get(c, {}).get(pixel_key(bn, m))) for c in clips[g]]) for g in gs}
               for bn in PIXEL_BASELINES}
        k0 = [g for g in gs if g not in excluded]
        vd = {g: list(vdoses.get(g) or []) for g in gs}
        fit = k0 or list(gs)
        xh = logo_x_hat({g: S[g] for g in gs}, fit, x)       # the model's x̂: exclusion AND K (audit item)
        xh_pix = {bn: logo_x_hat(v, fit, x) for bn, v in pix.items()}
        for g in k0:
            vd[g] += speed_match_doses(kin[g], x, xh[g]) if kin.get(g) else []
        match = {g: [v for v in vd[g] if v not in (vdoses.get(g) or [])] for g in k0}
        match = {g: d for g, d in match.items() if _near(d, x0, rad)}

        def rec(sel):
            P = {bn: np.array([v[g] for g in sel]) for bn, v in pix.items()}
            P = {bn: B for bn, B in P.items() if np.isfinite(B).all(1).any()}
            r_ = kink_compare(np.array([S[g] for g in sel]), x, x0, rad, P, n_boot, seed,
                              stops=[vd[g] for g in sel], x_hat=[xh[g] for g in sel],
                              x_hat_baselines={bn: [xh_pix[bn][g] for g in sel] for bn in P})
            r_["model"]["x_hat_fit"] = ("leave-one-group-out over the groups kept by the stop / minimum rules "
                                        "(the same x̂ as the speed-match exclusion)")
            return r_
        kept = [g for g in k0 if g not in match]
        if not any(np.isfinite(S[g]).all() for g in gs):
            continue
        r = rec(kept) if kept else {"model": None, "verdict": "no group left after the exclusions"}
        r["n_groups_kept"], r["excluded_speed_match"] = len(kept), match
        ex = [g for g in gs if g not in kept]
        if ex:
            r["excluded_groups"] = {"groups": ex, **rec(ex)}
        out[m] = r
    return out


def energy_kink(spec: dict, groups: list[str], lab, surprise: dict, model: str | None, pixel: dict | None = None,
                n_boot: int = 1000, seed: int = 0, log=print) -> dict:
    """One ``energy_kink`` entry over a family's twin groups.

    spec: {twins, doses?, x0, exclude_radius}; groups: group ids; lab: clip -> labels; surprise: clip -> {metric:
    value} of the model (clean-window restricted); pixel: clip -> pixel-baseline surprises.
    Kinematic-V exclusion (from labels, per group; a kinematic correlate with a V or a minimum within exclude_radius
    of x0 can kink there without any law): ``stop`` a ball's |v'| minimal (stop_dose), ``speed_min`` the total post
    speed or KE minimal, and per distance ``speed_match`` S(x) = S(x̂) at x != x̂ (x̂ = the model's leave-one-group-out
    expected dose: the second V of a speed-novelty observer). Returns the l2.json ``energy_kink[<name>]`` record: per
    model distance a kink record (:func:`kink_compare`) over the kept groups and ``excluded_groups`` over the rest;
    ``excluded.{stop, speed_min}`` {group: doses}; ``groups_scored`` = the groups kept by stop / speed_min."""
    twins = [str(t) for t in spec["twins"]]
    x = ladder_doses(twins, spec.get("doses"))
    x0 = float(spec.get("x0", 1.0))
    rad = float(spec.get("exclude_radius", 0.2))
    out = {"kind": KINK_KIND, "config": spec, "twins": twins, "doses": x.tolist(), "x0": x0, "exclude_radius": rad,
           "transform": "log surprise, standardised within group", "baselines_to_beat": list(KINK_BASELINES),
           "untrained": "separate run: results/<untrained model>/<family>/l2.json energy_kink (its K CI must hold 0)",
           "exclusion_rule": f"group excluded when a ball stops, the total speed or KE is minimal, or S(x) = S(x̂) "
                             f"(x != x̂) within {rad:g} of {x0:g}; doses within {rad:g} of a stop dose are also left "
                             "out of the curvature-trend fit",
           "distances": {}, "groups_scored": []}
    gs = [g for g in groups if all(f"{g}/{t}" in surprise for t in twins)]
    out["n_groups_complete"] = len(gs)
    if not gs or model is None:
        out["status"] = "no complete twin groups (inactive until the ladder is rendered and scored)"
        return out
    try:
        eps = np.array([[float(lab(f"{g}/{t}")["physlaw"]["epsilon"]) for t in twins] for g in gs])
        out["epsilon_by_dose"] = dict(zip(twins, np.mean(eps, 0).tolist()))
    except (KeyError, TypeError) as e:
        log(f"[L2] energy_kink: labels carry no physlaw.epsilon ({e!r}); doses from the twin names")
    labs = {g: [lab(f"{g}/{t}") for t in twins] for g in gs}
    sd = {g: stop_doses(labs[g], x) for g in gs}
    kin = {g: ladder_kinematics(labs[g], x) for g in gs}
    mins = {g: kinematic_minima(kin[g], x) for g in gs if kin[g]}
    excl = {"stop": {g: sd[g] for g in gs if _near(sd[g], x0, rad)},
            "speed_min": {g: mins[g] for g in mins if _near(list(mins[g].values()), x0, rad)}}
    unknown = [g for g in gs if sd[g] is None or kin[g] is None]
    if unknown:
        log(f"[L2] energy_kink: {len(unknown)} groups without post_velocity / stop_dose labels: speed rules not applied")
    static = sorted(set(excl["stop"]) | set(excl["speed_min"]))
    keep = [g for g in gs if g not in static]
    out.update(excluded=excl, stop_doses=sd, kinematic_minima=mins, n_groups_unscreened=len(unknown), groups_scored=keep,
               n_groups_scored=len(keep))
    out["status"] = ("ok" if len(keep) >= 2 else "one group only (x_hat in-sample)" if keep
                     else "every complete group excluded by the kinematic-V rule")
    vdoses = {g: [v for v in (sd[g] or []) + list(mins.get(g, {}).values()) if np.isfinite(v)] for g in gs}
    out["distances"] = _kink_distances(gs, twins, x, x0, rad, surprise, pixel, n_boot, seed, vdoses, kin, static)
    log(f"[L2] energy_kink: {len(gs)} groups, excluded stop {len(excl['stop'])}, speed_min {len(excl['speed_min'])}, "
        "speed_match " + str({m: len(r["excluded_speed_match"]) for m, r in out["distances"].items()}))
    return out


def kink_table(ek: dict) -> str:
    """md section for the energy_kink records."""
    rows = ["## Energy kink (surprise along the ε ladder; kink at the law boundary ε = 1)", "",
            "K = Δ²s(x0) − the smooth curvature trend at x0 (linear / quadratic fit of Δ²s over the interior doses outside "
            "the radius of x0 and of the source's expected dose x̂, fitted leave-one-group-out over the groups kept by the stop / "
            "minimum rules: the same x̂ as the speed-match exclusion); s = log surprise (clean window) standardised within group. "
            "Mean over groups [group-bootstrap 95% CI]; share of groups with K > 0. The model's kink counts only if it "
            "exceeds both pixel baselines (paired CIs) and the untrained model's K CI holds 0 (its own run). Groups "
            "with a kinematic V near 1 (a ball stops, total speed / KE minimal, or S(x) = S(x̂)) are excluded (own "
            "rows).", ""]
    dci = lambda d: f"{d['diff']:+.2f} {_ci(d['diff_ci95'])}" if d else ""
    for name, e in ek.items():
        ex = e.get("excluded", {})
        rows.append(f"### {name}: twins {e['twins'][0]} … {e['twins'][-1]}, x0 = {e['x0']:g}, exclusion radius "
                    f"{e['exclude_radius']:g}; {e['n_groups_complete']} complete groups; excluded: stop "
                    f"{len(ex.get('stop', {}))}, speed/KE minimum {len(ex.get('speed_min', {}))}, speed match per "
                    f"distance ({e.get('status', '')})")
        rows.append("")
        if "epsilon_by_dose" in e:
            rows += ["Labelled ε (mean over groups): " + ", ".join(f"{t} {v:.2f}" for t, v in e["epsilon_by_dose"].items()), ""]
        if not e["distances"]:
            continue
        rows += ["| distance | groups | x̂ | K model [CI] | share K>0 | copy_last K | const_vel K | model − copy_last [CI] | "
                 "model − const_vel [CI] | verdict |", "|---" * 10 + "|"]
        for m, r0 in e["distances"].items():
            for tag, r in (("", r0), (" [excluded]", r0.get("excluded_groups"))):
                if not r or not r.get("model"):
                    continue
                mo, b, vb = r["model"], r["baselines"], r["vs_baselines"]
                n = len(r["groups"]) if "groups" in r else r0["n_groups_kept"]
                bk = lambda k: f"{b[k]['K']:+.2f}" if k in b else ""
                rows.append(f"| {m}{tag} | {n} | {mo['x_hat']:.2f} | {mo['K']:+.2f} {_ci(mo['K_ci95'])} | {mo['share_K_pos']:.2f} | "
                            f"{bk('copy_last')} | {bk('const_velocity')} | {dci(vb.get('copy_last'))} | "
                            f"{dci(vb.get('const_velocity'))} | {r['verdict']} |")
        rows.append("")
    return "\n".join(rows)


# --------------------------------------------------------------------------- across-group kink (F11 statics)
# Each F11 group sits at one overhang x (constants.overhang_x, a grid symmetric about the boundary x = 1) and has two
# twins, stay and tip; the violating one switches at x = 1 (tip below, stay above). Per group the contrast
# c = log S(violating) − log S(legal) (the labels' ``legal`` flags pick the twins). The law predicts c > 0 on BOTH
# sides of 1 (a V or a positive level); any motion correlate (the tip twin moves, the stay twin does not, and the tip
# kinematics are smooth in |x − 1|) gives c = sign(1 − x) m(|x − 1|): linear through 1 when m ~ |x − 1|, a step when
# m(0) > 0. c is a LOG RATIO of surprises (scale-free: a source whose surprise is k times larger has the same c; the
# spec's "S(violating) − S(legal)" means this log ratio). The curve c̄(x) (mean over the groups at each x) is
# STANDARDISED over the dose grid (zero mean, unit sd: K in curve-sd units, comparable across sources) and goes
# through :func:`kink_stat` with novelty=False (no
# expected dose: the dose is not an outcome); x0 = 1 falls between grid points, so the curvature is the slope change
# across it (midpoint stencil: a step cancels, a linear trend gives 0, a V gives > 0). Also the side means of c below
# and above x0 (raw c, not standardised: their sign is the law's prediction). Group bootstrap (curve rebuilt per draw); pixel baselines through the same statistics, paired.


def _dose_curve(c: np.ndarray, ix: np.ndarray, n: int, w: np.ndarray) -> np.ndarray:
    """Weighted mean of the per-group values c at each grid index (NaN where a dose has no weight)."""
    ok = np.isfinite(c)
    num = np.bincount(ix[ok], weights=(w * c)[ok], minlength=n)
    den = np.bincount(ix[ok], weights=w[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def group_kink_stats(c, dose, x, x0: float = 1.0, exclude_radius: float = 0.15, w=None) -> dict:
    """K of the across-group curve and the side means of the per-group contrast c (dose: each group's dose on the
    grid x). w: group weights (bootstrap multiplicities; default 1)."""
    c, dose, x = np.asarray(c, float), np.asarray(dose, float), np.asarray(x, float)
    w = np.ones(len(c)) if w is None else np.asarray(w, float)
    ix = np.abs(dose[:, None] - x[None]).argmin(1)
    curve = _dose_curve(c, ix, len(x), w)
    st = kink_stat(curve[None], x, x0, exclude_radius, novelty=False)
    side = lambda m: (float((w * c)[m].sum() / w[m].sum()) if w[m].sum() > 0 else float("nan"))
    ok = np.isfinite(c)
    return {"K": st["K_g"][0], "below": side(ok & (dose < x0)), "above": side(ok & (dose > x0)),
            "curve": curve.tolist(), "stencil": st["stencil"], "reason": st["reason"]}


def group_kink_compare(c, dose, x, x0: float = 1.0, exclude_radius: float = 0.15, baselines: dict | None = None,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """The model's K and side means with group-bootstrap CIs, the same for each baseline (name -> per-group c), and
    the paired differences model − baseline. Verdicts: ``kink`` (K CI > 0, beyond every baseline when all their K
    difference CIs are > 0) and ``sides`` (c > 0 on both sides of x0: both CIs > 0)."""
    c = np.asarray(c, float)
    W = _weights(len(c), n_boot, seed)
    keys = ("K", "below", "above")

    def summary(cc):
        full = group_kink_stats(cc, dose, x, x0, exclude_radius)
        boots = [group_kink_stats(cc, dose, x, x0, exclude_radius, w) for w in W]
        return full, {k: np.array([b[k] for b in boots], float) for k in keys}

    fm, bm = summary(c)
    rec = {"model": {**fm, **{f"{k}_ci95": _q(bm[k]) for k in keys}, "n_groups": int(np.isfinite(c).sum())},
           "baselines": {}, "vs_baselines": {}}
    for name, cb in (baselines or {}).items():
        cb = np.asarray(cb, float)
        if cb.shape != c.shape or not np.isfinite(cb).any():
            continue
        fb, bb = summary(cb)
        rec["baselines"][name] = {**fb, **{f"{k}_ci95": _q(bb[k]) for k in keys}}
        rec["vs_baselines"][name] = {f"{k}_diff": fm[k] - fb[k] for k in keys}
        rec["vs_baselines"][name].update({f"{k}_diff_ci95": _q(bm[k] - bb[k]) for k in keys})
    lo = rec["model"]["K_ci95"][0]
    beats = (set(rec["vs_baselines"]) == set(KINK_BASELINES)
             and all(v["K_diff_ci95"][0] > 0 for v in rec["vs_baselines"].values()))
    rec["verdict"] = ("inconclusive: " + fm["reason"] if fm["reason"] else
                      "kink beyond baselines" if lo > 0 and beats else
                      "kink, not beyond every baseline" if lo > 0 else "no kink")
    both = rec["model"]["below_ci95"][0] > 0 and rec["model"]["above_ci95"][0] > 0
    rec["sides_verdict"] = ("violation more surprising on both sides of x0" if both else
                            "not more surprising on both sides of x0")
    return rec


def group_kink(spec: dict, groups: list[str], lab, surprise: dict, model: str | None, pixel: dict | None = None,
               n_boot: int = 1000, seed: int = 0, log=print) -> dict:
    """One ``group_kink`` entry: spec {twins: [two twins], dose: constants key, x0, exclude_radius}. Per group the
    violating twin is the one labelled not legal (groups without exactly one are skipped); the dose is
    ``physlaw.constants[dose]``; the grid = the sorted distinct doses (equal steps). Returns per model distance a
    :func:`group_kink_compare` record (pixel baselines through the same statistic)."""
    twins = [str(t) for t in spec["twins"]]
    dkey = str(spec.get("dose", "overhang_x"))
    x0, rad = float(spec.get("x0", 1.0)), float(spec.get("exclude_radius", 0.15))
    out = {"kind": GROUP_KINK_KIND, "config": spec, "twins": twins, "dose": dkey, "x0": x0, "exclude_radius": rad,
           "transform": "c = log S(violating) − log S(legal) per group; curve = mean c per dose, standardised over "
                        "the dose grid before D²; side means on raw c",
           "baselines_to_beat": list(KINK_BASELINES), "distances": {}}
    gs = [g for g in groups if all(f"{g}/{t}" in surprise for t in twins)]
    out["n_groups_complete"] = len(gs)
    if not gs or model is None:
        out["status"] = "no complete twin groups (inactive until the family is rendered and scored)"
        return out
    pairs, dose = [], []
    for g in gs:
        bad = [t for t in twins if not lab(f"{g}/{t}")["physlaw"]["legal"]]
        if len(bad) != 1:
            continue
        pairs.append((f"{g}/{bad[0]}", f"{g}/{next(t for t in twins if t != bad[0])}"))
        dose.append(float(lab(f"{g}/{twins[0]}")["physlaw"]["constants"][dkey]))
    dose = np.array(dose)
    x = np.unique(np.round(dose, 6))
    out.update(n_groups=len(pairs), doses=x.tolist(), n_per_dose={f"{v:g}": int(np.sum(np.isclose(dose, v))) for v in x})
    if len(x) < 4:
        out["status"] = f"{len(x)} distinct doses (need >= 4 on an equal-step grid)"
        return out
    lg = lambda v: np.log(v) if v is not None and np.isfinite(v) and v > 0 else np.nan
    for m in sorted({k for v, l in pairs for k in surprise[v]}):
        c = np.array([lg(surprise[v].get(m)) - lg(surprise[l].get(m)) for v, l in pairs])
        if not np.isfinite(c).any():
            continue
        pix = {bn: np.array([lg((pixel or {}).get(v, {}).get(pixel_key(bn, m)))
                             - lg((pixel or {}).get(l, {}).get(pixel_key(bn, m))) for v, l in pairs])
               for bn in PIXEL_BASELINES}
        out["distances"][m] = group_kink_compare(c, dose, x, x0, rad, pix, n_boot, seed)
    out["status"] = "ok"
    log(f"[L2] group_kink {spec.get('name', '')}: {len(pairs)} groups on {len(x)} doses; "
        + "; ".join(f"{m}: {r['verdict']} / {r['sides_verdict']}" for m, r in out["distances"].items()))
    return out


def group_kink_table(gk: dict) -> str:
    """md section for the group_kink records."""
    rows = ["## Across-group kink (per-group contrast log S(violating) − log S(legal) along a scene constant)", "",
            "Law: the violating twin is more surprising on both sides of x0 (the violating twin switches there); a "
            "motion correlate gives a contrast linear through x0 (or a step). K = slope change of the mean contrast "
            "(standardised over the dose grid) across x0 minus the smooth trend (no expected dose); side means of the contrast below / above x0. "
            "Group-bootstrap 95% CIs; pixel baselines through the same statistics.", ""]
    for name, e in gk.items():
        rows += [f"### {name}: twins {' / '.join(e['twins'])} on {e['dose']}, x0 = {e['x0']:g}; "
                 f"{e.get('n_groups', 0)} groups ({e.get('status', '')})", ""]
        if not e["distances"]:
            continue
        rows += ["| distance | K [CI] | below [CI] | above [CI] | copy_last K / below / above | const_vel K / below / above "
                 "| verdict | sides |", "|---" * 8 + "|"]
        for m, r in e["distances"].items():
            mo, b = r["model"], r["baselines"]
            bl = lambda k: (f"{b[k]['K']:+.2f} / {b[k]['below']:+.2f} / {b[k]['above']:+.2f}" if k in b else "")
            rows.append(f"| {m} | {mo['K']:+.2f} {_ci(mo['K_ci95'])} | {mo['below']:+.2f} {_ci(mo['below_ci95'])} | "
                        f"{mo['above']:+.2f} {_ci(mo['above_ci95'])} | {bl('copy_last')} | {bl('const_velocity')} | "
                        f"{r['verdict']} | {r['sides_verdict']} |")
        rows.append("")
    return "\n".join(rows)


# --------------------------------------------------------------------------- L2-rep: representational kink (§5.0.4)
# Per group, a(x) = the post-event activation of rung x, frame pool, clean window, per cached layer ("cell");
# D²a(x) = a(x+h) − 2a(x) + a(x−h); K_rep = (‖D²a(x0)‖ − p(x0)) / med, p = the smooth trend of ‖D²a(x)‖ over the
# reference interior x (|x − x0| > r and |x − x̂| > r; x̂ = the model's expected dose from its L2 surprise kink, else
# none; :func:`curvature_trend`), med = their median. Inside the clean window every kinematic correlate is affine in
# ε, so a representation that is any smooth function of the kinematics has K_rep ~ 0 (curvature varies smoothly),
# while a representation that switches regime at ε = 1 has a hinge
# there. Cell selected on half the groups, scored on the other half (probes.select.heldout_select); the full-sample
# selected cell gives the group-bootstrap CI and the paired differences to the baselines, which go through the same
# statistic: ``pixels`` (rendered frames of the window, block-averaged to patch resolution, time-averaged) and
# ``kinematics`` (stacked object positions and velocities over the window: affine in ε, so K_rep ~ 0 by
# construction, a sanity check). The untrained model is a separate run (its own l2.json energy_kink_rep).
# Activations: stage 4's frame.post pool covers every post-event token-frame (unrestricted); when the .npz carries
# ``pooled_frame_t`` [T_tok, L, D] (frame pool per token-frame) and ``token_t_s`` [T_tok] (time of each token-frame's
# last master frame), the post-event token-frames inside the window are averaged instead (restricted).

REP_KIND = "energy_kink_rep"
REP_BASELINES = ("pixels", "kinematics")


def rep_kink_stat(acts, x, x0: float = 1.0, exclude_radius: float = 0.2, x_hat: float | None = None,
                  step_tol: float = 1e-6, stops=None) -> dict:
    """K_rep per group. acts: [G, L, D] or a list of [L, D_g] (doses x [L], equal steps).
    Returns {K_g, d2_norm_rel_mean (‖D²a‖ / mean ‖Δa‖ per interior x, averaged over groups), inconclusive, reason}.
    K_rep = (‖D²a(x0)‖ − trend(x0)) / med (trend: :func:`curvature_trend` over the reference points). med is floored at 1e-6 x the group's mean ‖a(x+h) − a(x)‖, so an exactly affine trajectory gives K_rep = 0."""
    x = np.asarray(x, float)
    steps = np.diff(x)
    if len(x) < 3 or not np.allclose(steps, steps[0], rtol=0, atol=step_tol) or steps[0] <= 0:
        raise ValueError(f"ladder doses must be increasing with an equal step: {x.tolist()}")
    xi = x[1:-1]
    hit = np.flatnonzero(np.isclose(xi, x0, rtol=0, atol=step_tol))
    keep = np.abs(xi - x0) > exclude_radius + step_tol
    if x_hat is not None and np.isfinite(x_hat):
        keep &= np.abs(xi - x_hat) > exclude_radius + step_tol
    K, rel = [], []
    for gi, a in enumerate(acts):
        a = np.asarray(a, float).reshape(len(x), -1)
        N = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1)
        scale = float(np.linalg.norm(np.diff(a, axis=0), axis=-1).mean())
        rel.append(N / scale if scale > 0 else np.full(len(N), np.nan))
        kg = keep.copy()
        for c in (stops[gi] if stops is not None and stops[gi] else []):
            if np.isfinite(c):
                kg &= np.abs(xi - c) > exclude_radius + step_tol
        if not len(hit) or not kg.any() or not np.isfinite(N).all():
            K.append(np.nan)
            continue
        den = max(float(np.median(N[kg])), 1e-6 * scale)
        K.append((N[hit[0]] - curvature_trend(xi[kg], N[kg], x0)) / den if den > 0 else np.nan)
    out = {"K_g": [float(k) for k in K], "x_interior": xi.tolist(), "x0": float(x0), "exclude_radius": float(exclude_radius),
           "x_hat": None if x_hat is None else float(x_hat), "n_reference_points": int(keep.sum()),
           "d2_norm_rel_mean": np.nanmean(np.array(rel), 0).tolist() if rel else []}
    reason = None
    if not len(hit):
        reason = f"ladder lacks points on both sides of x0 = {x0:g}"
    elif x_hat is not None and np.isfinite(x_hat) and abs(x_hat - x0) < exclude_radius - step_tol:
        reason = (f"|x_hat - x0| = {abs(x_hat - x0):.2f} < {exclude_radius:g}: the model's expectation sits at the "
                  "boundary")
    elif not keep.any():
        reason = "no reference interior points left"
    out.update(inconclusive=reason is not None, reason=reason)
    return out


def _rep_summary(st: dict, W: np.ndarray) -> dict:
    K = np.asarray(st["K_g"], float)
    ok = np.isfinite(K)
    boots = [float((w[ok] * K[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan for w in W]
    return {"K_rep": float(K[ok].mean()) if ok.any() else float("nan"), "K_rep_ci95": _q(boots),
            "K_rep_median": float(np.median(K[ok])) if ok.any() else float("nan"),
            "share_K_pos": float((K[ok] > 0).mean()) if ok.any() else float("nan"), "n_groups": int(ok.sum()),
            "d2_norm_rel_mean": st["d2_norm_rel_mean"]}


def _cell_name(l) -> str:
    return f"L{l}" if str(l).lstrip("-").isdigit() else str(l)


def _rep_model_acts(gs: list[str], twins: list[str], cache_dir: Path, recs: dict, windows: dict):
    """([G, L, n_cells, D] activations, cell names, {"restricted", "unrestricted"} counts)."""
    A, cells, cnt = [], None, {"restricted": 0, "unrestricted": 0}
    for g in gs:
        w = windows.get(g)
        row = []
        for t in twins:
            z = np.load(Path(cache_dir) / recs[f"{g}/{t}"]["file"])
            if cells is None:
                cells = [_cell_name(l) for l in z["layers"]]
            a = None
            if w is not None and "pooled_frame_t" in z.files and "token_t_s" in z.files:
                sel = (np.asarray(z["phases"]).astype(str) == "post") & (np.asarray(z["token_t_s"], float) <= w["t_end"] + 1e-9)
                if sel.any():
                    a = z["pooled_frame_t"][sel].astype(np.float32).mean(0)
                    cnt["restricted"] += 1
            if a is None:
                a = z["pooled"][[str(p) for p in z["pools"]].index("frame.post")].astype(np.float32)
                cnt["unrestricted"] += 1
            row.append(a)
        A.append(np.stack(row))
    return np.stack(A), cells, cnt


def _rep_kinematics(gs: list[str], twins: list[str], lab, windows: dict) -> list[np.ndarray]:
    """Per group [L, D_g]: positions and velocities of every object at each master frame of the window, stacked."""
    out = []
    for g in gs:
        rows = []
        for t in twins:
            labels = lab(f"{g}/{t}")
            objs = scored_objects(labels)
            w = windows.get(g)
            t0 = w["t_event"] if w else event_time_s(labels)
            t1 = w["t_end"] if w else np.inf
            fr = [f for f in labels["per_frame"] if t0 - 1e-9 <= f["time_s"] <= t1 + 1e-9]
            rows.append(np.array([np.r_[f[o]["position"], f[o]["velocity"]] for f in fr for o in objs], float).ravel())
        n = min(len(r) for r in rows)
        out.append(np.stack([r[:n] for r in rows]))
    return out


def _rep_pixels(gs: list[str], twins: list[str], data_dir: Path, lab, windows: dict, block: int = 16) -> list[np.ndarray]:
    """Per group [L, D]: rendered RGB frames of the window, block-averaged (block x block px), averaged over time."""
    from physlaw.data.io import load_clip
    out = []
    for g in gs:
        rows = []
        for t in twins:
            c = load_clip(Path(data_dir) / g / t, masks=False)
            fps = float(c.labels["scene_constants"]["frame_rate"])
            w = windows.get(g)
            t0 = w["t_event"] if w else event_time_s(c.labels)
            t1 = w["t_end"] if w else np.inf
            ti = np.arange(len(c.frames)) / fps
            F = c.frames[(ti >= t0 - 1e-9) & (ti <= t1 + 1e-9)][..., :3].astype(np.float32) / 255.0
            H, W_ = (F.shape[1] // block) * block, (F.shape[2] // block) * block
            F = F[:, :H, :W_].reshape(len(F), H // block, block, W_ // block, block, 3).mean((2, 4))
            rows.append(F.mean(0).ravel())
        out.append(np.stack(rows))
    return out


def energy_kink_rep(spec: dict, groups: list[str], lab, data_dir: Path, cache_dir: Path, recs: dict, windows: dict,
                    x_hat: float | None = None, n_boot: int = 1000, n_splits: int = 50, seed: int = 0,
                    log=print, stops: dict | None = None) -> dict:
    """The l2.json ``energy_kink_rep[<name>]`` record for one energy_kink entry. groups: the groups the L2 kink
    scored (stop-dose rule applied); recs: the model's index records (clip -> {"file", ...}); windows: group_windows.
    ``rep_pixels: false`` in the spec skips the pixel baseline (it decodes every ladder video)."""
    from physlaw.probes.select import heldout_select
    twins = [str(t) for t in spec["twins"]]
    x = ladder_doses(twins, spec.get("doses"))
    x0, rad = float(spec.get("x0", 1.0)), float(spec.get("exclude_radius", 0.2))
    gs = [g for g in groups if all(f"{g}/{t}" in recs and "file" in recs[f"{g}/{t}"] for t in twins)]
    out = {"kind": REP_KIND, "twins": twins, "doses": x.tolist(), "x0": x0, "exclude_radius": rad, "x_hat": x_hat,
           "x_hat_source": ("median over distances of the L2 energy kink's x̂" if x_hat is not None
                            else "none (no L2 surprise kink): only |x − x0| > r is excluded"),
           "n_groups": len(gs), "baselines_to_beat": list(REP_BASELINES),
           "untrained": "separate run: results/<untrained model>/<family>/l2.json energy_kink_rep"}
    if len(gs) < 4:
        out["status"] = f"{len(gs)} complete groups (need >= 4 for the held-out cell selection)"
        return out
    match, stops = {}, {g: list((stops or {}).get(g) or []) for g in gs}
    if x_hat is not None:          # kinematic V at the model's expected dose (the stop / minimum rules ran in L2)
        for g in gs:
            k = ladder_kinematics([lab(f"{g}/{t}") for t in twins], x)
            d = speed_match_doses(k, x, x_hat) if k else []
            stops[g] += d
            if _near(d, x0, rad):
                match[g] = d
    out["excluded_speed_match"] = match
    ex, gs = [g for g in gs if g in match], [g for g in gs if g not in match]
    out["n_groups"] = len(gs)
    if len(gs) < 4:
        out["status"] = f"{len(gs)} groups after the speed-match exclusion (need >= 4)"
        return out
    A, cells, cnt = _rep_model_acts(gs, twins, cache_dir, recs, windows)
    out["window"] = cnt
    if cnt["unrestricted"]:
        log(f"[L2-rep] WARNING: {cnt['unrestricted']} clips have no pooled_frame_t / token_t_s (cache older than the "
            "clean window): their whole post-event frame pool is used")
    sg = [stops[g] for g in gs]
    st = {c: rep_kink_stat(A[:, :, i], x, x0, rad, x_hat, stops=sg) for i, c in enumerate(cells)}
    Kc = {c: np.asarray(s["K_g"], float) for c, s in st.items()}
    gpos = {g: i for i, g in enumerate(gs)}

    def score(c, gids):
        v = Kc[c][[gpos[g] for g in gids]]
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    sel = heldout_select(score, cells, gs, n_splits=n_splits, seed=seed)
    W = _weights(len(gs), n_boot, seed)
    st_m = st[sel["selected"]]
    out["model"] = {"selection": sel, "cell": sel["selected"], "heldout_K_rep": sel["score"], **_rep_summary(st_m, W),
                    "by_cell": {c: float(np.nanmean(k)) if np.isfinite(k).any() else float("nan") for c, k in Kc.items()}}
    if ex:                         # the excluded groups, same cell, reported apart
        Ae, _, _ = _rep_model_acts(ex, twins, cache_dir, recs, windows)
        se = rep_kink_stat(Ae[:, :, cells.index(sel["selected"])], x, x0, rad, x_hat, stops=[stops[g] for g in ex])
        out["excluded_groups"] = {"groups": ex, **_rep_summary(se, _weights(len(ex), n_boot, seed))}
    out["baselines"], out["vs_baselines"] = {}, {}
    makers = {"kinematics": lambda: _rep_kinematics(gs, twins, lab, windows)}
    if spec.get("rep_pixels", True):
        makers["pixels"] = lambda: _rep_pixels(gs, twins, data_dir, lab, windows, int(spec.get("rep_pixel_block", 16)))
    for name, mk in makers.items():
        try:
            sb = rep_kink_stat(mk(), x, x0, rad, x_hat, stops=sg)
        except (KeyError, OSError, ValueError) as e:
            log(f"[L2-rep] no {name} baseline ({e!r})")
            continue
        out["baselines"][name] = _rep_summary(sb, W)
        out["vs_baselines"][name] = _kink_diff(st_m, sb, W)
    lo = out["model"]["K_rep_ci95"][0]
    beats = len(out["vs_baselines"]) == len(makers) and all(v["diff_ci95"][0] > 0 for v in out["vs_baselines"].values())
    out["inconclusive"], out["reason"] = st_m["inconclusive"], st_m["reason"]
    out["verdict"] = ("inconclusive: " + st_m["reason"] if st_m["inconclusive"] else
                      "representational kink beyond baselines" if lo > 0 and sel["score"] > 0 and beats else
                      "kink, not beyond every baseline" if lo > 0 else "no kink")
    out["status"] = "ok"
    log(f"[L2-rep] {len(gs)} groups, cell {sel['selected']}: held-out K_rep {sel['score']:+.2f}, "
        f"full-sample {out['model']['K_rep']:+.2f} {_ci(out['model']['K_rep_ci95'])}; window {cnt}; {out['verdict']}")
    return out


def rep_table(er: dict) -> str:
    """md section for the energy_kink_rep records."""
    rows = ["## L2-rep: representational kink (post-event activations along the ε ladder)", "",
            "K_rep = (‖D²a(1)‖ − trend(1)) / median ‖D²a(x)‖, trend = linear (or quadratic) fit of ‖D²a(x)‖ over interior "
            "x with |x − 1| and |x − x̂| > r; cell selected "
            "on half the groups and scored on the other half; CI: group bootstrap on the selected cell. Baselines "
            "through the same statistic: pixels (window frames, block-averaged) and label kinematics (affine in ε: "
            "K_rep ≈ 0 by construction). The untrained model is its own run.", "",
            "| entry | n groups | window (restricted / unrestricted) | cell | held-out K_rep | K_rep [CI] | share > 0 | "
            "pixels K_rep | kinematics K_rep | model − pixels [CI] | model − kinematics [CI] | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in er.items():
        if e.get("status") != "ok":
            rows.append(f"| {name} | {e['n_groups']} | | | | | | | | | | {e.get('status')} |")
            continue
        m, b, d = e["model"], e["baselines"], e["vs_baselines"]
        bk = lambda k: f"{b[k]['K_rep']:+.2f}" if k in b else ""
        dk = lambda k: f"{d[k]['diff']:+.2f} {_ci(d[k]['diff_ci95'])}" if k in d else ""
        rows.append(f"| {name} | {e['n_groups']} | {e['window']['restricted']} / {e['window']['unrestricted']} | {m['cell']} | "
                    f"{m['heldout_K_rep']:+.2f} | {m['K_rep']:+.2f} {_ci(m['K_rep_ci95'])} | {m['share_K_pos']:.2f} | "
                    f"{bk('pixels')} | {bk('kinematics')} | {dk('pixels')} | {dk('kinematics')} | {e['verdict']} |")
    return "\n".join(rows)
