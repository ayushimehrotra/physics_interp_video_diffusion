"""Leaderboard (spec §8) from results/<model>/<family>/<level>.json: one row per (model, family); per level and law ONE
primary number beside its floor / baseline and its 95% CI. Blank where a level has not been run.

Entries ``scores["<level>.<law>"] = {"score", "floor", "ci95", "metric", "floor_is", ...}``; where each comes from:
  l0.<quantity>   mean held-out R² over the targets of each kinematic quantity (velocity, angular velocity, height,
                  acceleration, angle, ...; not position / mass / size: l0.json ``targets[*].heldout``);
                  floor = mean mask-only floor R²; CI = mean of the targets' group-bootstrap CIs; ``selectivity`` = mean
                  object-control selectivity (Hewitt & Liang) where present
  l0.mass_pre / l0.kappa_pre   audit H1: the best held-out R² of the PRE-event log_mass (inertia_factor: F10 kappa,
                  solid vs hollow) targets; ``decodable`` = some target with R² CI > 0, every selectivity > 0 and R² above
                  the untrained model's (the row of a model named "untrained*" on the same family, when present)
  l1.momentum     S_mom of the held-out selected space, frame pool (``selected["energy|frame"].momentum``); baseline 0 =
                  the mass-blind rule (1 = as conserved as the probe floor). Not on F3 (one forecast per set serves
                  every hidden mass ratio: L1 does not score it)
  every l1 entry  ``claim`` = the L1 verdict (the forecast beats the three H4 negative-control forecasts and, roi,
                  the mask-only floor); ``sensitivity`` = the M1 score with gated-out clips counted as failures
  l1.energy       creation rate P(eps_hat > 1 + tau) of the forecast, floor = the same rate of the real-frame decode,
                  at the space with the best floor instrument; ``status`` measured / unmeasurable
  l1.<law>        ratio laws (gravity, rolling, friction, oscillation, statics, angular_momentum, momentum_inertia
                  = Newton I on the F6 glide, friction_stop = stop_hold on the F6 slide): S of the law's primary statistic (PRIMARY_STAT; statics: tip_consistency
                  accuracy), floor = the real-frame decode's S; mass independence beside them (MASS_STAT: gravity_mass,
                  friction_mass = mass_ratio, oscillation_mass = mass_period_ratio)
  uninformative   (H1) every mass-independence score (MASS_KEYS: l1.*_mass, l3 mass independence, l4 friction /
                  oscillation log-mass steering) of a model / family whose l0.mass_pre is not decodable, and every
                  rolling score (KAPPA_KEYS) whose l0.kappa_pre is not: a model blind to mass passes "mass does not
                  matter" without knowing the law
  l2.<law>        win rate of each law's primary contrast (l2.json ``primary_by_law``: momentum, gravity, friction,
                  oscillation, rolling, angular_momentum; model, primary distance), baseline = the best pixel baseline's win
                  rate on the same pairs
  l2.energy       kink K (first energy_kink spec, primary distance), baseline = the largest pixel-baseline K
  l2.statics      F11 across-group kink K on overhang x (``group_kink``), baseline = the largest pixel-baseline K
  l2rep.energy    held-out K_rep of the post-event activations, baseline = the largest baseline K_rep
  l2b.<law>       held-out AUC of the first non-control contrast per law, baseline = the kinematic baseline's AUC
  l3.galilean     d_epsilon at the largest boost level (decoded; law 0), floor = the base level's median |eps error|
  l3.<law>        mass-independence S (forecast vs floor vs real violator) under its law (gravity F7 drops, friction
                  F6 pairs, oscillation F9 pair), floor = the probe's log sensitivity
  l4.<law>        the steered readout's slope (energy: plateau; statics: tip step) of the decoded (else generated) variant,
                  baseline = the random-direction control; ``iia`` = swap-test IIA; ``wrong_way`` share of scenes
  outA.momentum / outA.energy   track A S_mom (pooled legal twins; baseline 0 = mass-blind) / share of generated
                  eps <= 1 + tol (floor = the real legal clips' share)
  outA.<law>, outB.<law>   ratio laws (both tracks; B: strict gate, else loose): the median of the law's primary
                  statistic (statics: the mean tip_consistency), floor = the same readout on the real clips; track B
                  also momentum antiparallel mean and energy eps share
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRIMARY_DISTANCES = ("mse_obj", "clean_mse_obj", "fut_mse_obj", "l1_obj", "mse_all", "clean_mse_all", "fut_mse_all")
FAMILY_LAW = {"f1": "momentum", "f2": "energy", "f3": "momentum", "f4": "galilean", "f6": "friction", "f7": "gravity",
              "f8": "energy", "f9": "oscillation", "f10": "rolling", "f11": "statics", "f12": "angular_momentum"}
# law -> (primary statistic, summary of the per-clip readouts): every ratio law's legal value is 1 except
# inertia_turn and stop_hold (0); tip_consistency is 0 / 1 per clip, so its summary is the mean (an accuracy).
# friction_stop: the friction law's second row (F6 slide: stops and stays; decel_const is its primary row)
PRIMARY_STAT = {"gravity": ("g_ratio", "median"), "rolling": ("order", "median"), "friction": ("decel_const", "median"),
                "oscillation": ("period_ratio", "median"), "statics": ("tip_consistency", "mean"),
                "angular_momentum": ("L_ratio", "median"), "momentum_inertia": ("inertia_speed", "median"),
                "friction_stop": ("stop_hold", "median")}
MASS_STAT = {"gravity_mass": "mass_ratio", "friction_mass": "mass_ratio", "oscillation_mass": "mass_period_ratio"}
MASS_KEYS = ("l1.gravity_mass", "l1.friction_mass", "l1.oscillation_mass", "l3.gravity", "l3.friction",
             "l3.oscillation", "l4.friction", "l4.oscillation")
KAPPA_KEYS = ("l1.rolling", "l2.rolling", "l2b.rolling", "l4.rolling")
LEGAL_ZERO = ("inertia_turn", "stop_hold")
L0_SKIP = ("position", "log_mass", "mass", "radius", "size", "direction")
NAN = float("nan")


def _num(x):
    return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None


def entry(score, floor=None, ci=None, metric: str = "", floor_is: str = "", **extra) -> dict | None:
    """One leaderboard cell; None when the score is undefined."""
    if _num(score) is None:
        return None
    ci = [_num(c) for c in ci] if ci is not None and len(ci) == 2 else None
    return {"score": float(score), "floor": _num(floor), "ci95": ci if ci and None not in ci else None,
            "metric": metric, "floor_is": floor_is, **{k: v for k, v in extra.items() if v is not None}}


def _law(family: str) -> str:
    return FAMILY_LAW.get(family.split("_")[0], "momentum")


def _mass_pre(tg: dict, q: str):
    """{target: (held-out R², CI lower, min selectivity)} of the PRE-event ``q`` targets (l0.json targets)."""
    out = {}
    for b, t in tg.items():
        if b.endswith(f".pre.{q}") and "heldout" in t:
            sel = [v.get("selectivity", NAN) for v in (t.get("selectivity") or {}).values()]
            out[b] = (float(t["heldout"]["score"]), float((t.get("r2_ci95") or [NAN])[0]),
                      float(min(sel)) if sel else NAN)
    return out


def _l0(d, fam):
    out = {}
    tg = d.get("targets", {})
    for key, q in (("mass_pre", "log_mass"), ("kappa_pre", "inertia_factor")):
        mp = _mass_pre(tg, q)
        if mp:
            best = max(mp, key=lambda b: mp[b][0])
            out[key] = entry(mp[best][0], None, None, f"held-out R² of pre-event {q} (best of {len(mp)})", "",
                             targets={b: v[0] for b, v in mp.items()},
                             decodable=any(v[1] > 0 and v[2] > 0 for v in mp.values()))
    for q in sorted({b.rsplit(".", 1)[-1] for b in tg} - set(L0_SKIP)):
        ts = [t for b, t in tg.items() if b.endswith("." + q) and "heldout" in t]
        if not ts:
            continue
        sel = [t["selectivity"]["object"]["selectivity"] for t in ts if "object" in t.get("selectivity", {})]
        cis = [t["r2_ci95"] for t in ts if t.get("r2_ci95")]
        out[q] = entry(np.mean([t["heldout"]["score"] for t in ts]), np.mean([t["mask_floor"]["r2"] for t in ts]),
                       list(np.mean(cis, 0)) if cis else None, f"held-out R² ({len(ts)} targets)", "mask-only floor",
                       selectivity_object=float(np.mean(sel)) if sel else None)
    return out


def _l1(d, fam):
    out, vs = {}, d.get("variants", {})
    for key, sel in (d.get("selected") or {}).items():
        kind, pool = key.split("|")
        if pool != "frame":
            continue
        claim = lambda x: (x.get("verdict") or {}).get("claim_allowed")
        if "momentum" in sel:
            m = vs[sel["momentum"]["selected"]]["momentum"]
            out["momentum"] = entry(m.get("S"), 0.0, m.get("S_ci95"), "S_mom", "mass-blind rule",
                                    heldout=_num(sel["momentum"].get("score")), eta_floor=_num(m["eta"]["floor"]),
                                    claim=claim(m), sensitivity=_num(m.get("S_sensitivity")))
        v = vs.get(sel.get("floor_best"), {})
        e = v.get("energy", {})
        if "forecast" in e:
            out["energy"] = entry(e["forecast"]["creation_rate"], e["floor"]["creation_rate"], e["forecast"]["creation_ci95"],
                                  "creation rate P(eps > 1 + tau)", "real-frame decode", status=e.get("status"),
                                  claim=claim(e), sensitivity=_num(e["forecast"].get("creation_rate_sensitivity")))
        rat = v.get("ratio", {})
        for law, (st, _) in list(PRIMARY_STAT.items()) + [(k_, (v_, "median")) for k_, v_ in MASS_STAT.items()]:
            r = rat.get(st)
            if r and law not in out and (law not in MASS_STAT or law.split("_")[0] in kind):
                out[law] = entry(_s(r["forecast"]), _s(r["floor"]), r["forecast"].get("S_ci95"), f"S {st}",
                                 "real-frame decode", claim=claim(r))
    return out


def _s(rec: dict):
    """A ratio statistic's score: S, or the accuracy of a 0 / 1 statistic (statics tip_consistency)."""
    return rec.get("S", rec.get("accuracy"))


def _model_src(by_src: dict):
    return next((s for s in by_src if s != "pixel_baselines"), None)


def _l2(d, fam):
    out = {}
    cons = d.get("contrasts", {})
    by_law = dict(d.get("primary_by_law") or {})       # one primary contrast per law (F6: glide momentum + friction)
    if not by_law:
        prim = d.get("primary_contrast") or next((c for c in cons if not c.startswith(" vs ".join(d.get("calibration") or []))), None)
        by_law = {_law(fam) if _law(fam) != "energy" else "momentum": prim}
    for law, prim in by_law.items():
        if prim not in cons:
            continue
        src = _model_src(cons[prim])
        rec = cons[prim].get(src, {}) if src else {}
        m = next((m for m in PRIMARY_DISTANCES if m in rec), next(iter(rec), None))
        if m:
            r = rec[m]
            base = max((v.get("baseline_win", NAN) for v in (r.get("vs_pixel") or {}).values()), default=None)
            out[law] = entry(r["win_rate"], base, r.get("win_rate_ci95"), f"win rate {prim} ({m})", "best pixel baseline")
    for spec, gk in (d.get("group_kink") or {}).items():          # F11 statics: across-group kink on overhang x
        dist = gk.get("distances") or {}
        m = next((m for m in PRIMARY_DISTANCES if m in dist), next(iter(dist), None))
        if m and dist[m].get("model"):
            r = dist[m]
            base = max((b.get("K", NAN) for b in r.get("baselines", {}).values()), default=None)
            out["statics"] = entry(r["model"]["K"], base, r["model"].get("K_ci95"), f"group kink K {spec} ({m})",
                                   "largest pixel-baseline K", verdict=r.get("verdict"), sides=r.get("sides_verdict"))
            break
    for spec, ek in (d.get("energy_kink") or {}).items():
        dist = ek.get("distances") or {}
        m = next((m for m in PRIMARY_DISTANCES if m in dist), next(iter(dist), None))
        if m and dist[m].get("model"):
            r = dist[m]
            base = max((b.get("K", NAN) for b in r.get("baselines", {}).values()), default=None)
            out["energy"] = entry(r["model"]["K"], base, r["model"].get("K_ci95"), f"kink K {spec} ({m})",
                                  "largest pixel-baseline K", verdict=r.get("verdict"))
            break
    return out


def _l2rep(d, fam):
    for spec, er in (d.get("energy_kink_rep") or {}).items():
        mo = er.get("model")
        if mo:
            base = max((b.get("K_rep", NAN) for b in er.get("baselines", {}).values()), default=None)
            return {"energy": entry(mo.get("heldout_K_rep", mo.get("K_rep")), base, mo.get("K_rep_ci95"),
                                    f"K_rep {spec} (held-out layer)", "largest baseline K_rep", verdict=er.get("verdict"))}
    return {}


def _l2b(d, fam):
    out = {}
    for c, rec in d.get("contrasts", {}).items():
        law = rec.get("law") or _law(fam)
        if rec.get("control") or law in out or not isinstance(rec.get("overall"), dict):
            continue
        out[law] = entry(rec["overall"].get("score"), (rec.get("kinematic") or {}).get("auc_mean"),
                         rec["overall"].get("auc_ci95"), f"held-out AUC {c}", "kinematic baseline")
    return out


def _l3(d, fam):
    out = {}
    g = d.get("galilean") or {}
    dec = (g.get("galilean") or {}).get("decoded") or {}
    if dec:
        lv = "ularge" if "ularge" in dec else sorted(dec)[-1]                  # the largest boost
        base = next((v for k, v in (g.get("floor") or {}).items() if k not in dec), {})
        out["galilean"] = entry(dec[lv].get("d_epsilon"), base.get("eps_abs_err_median"), dec[lv].get("d_epsilon_ci95"),
                                f"d_epsilon at {lv} (law 0)", "base-level |eps error|", measurable=dec[lv].get("measurable"))
    mi = d.get("mass_independence") or {}
    sp = next((s for s in mi.get("spaces", {}) if not s.endswith("_mm")), None)
    if sp:
        sc = mi["spaces"][sp]["score"]
        out[mi.get("law", "gravity")] = entry(sc.get("S"), sc.get("sensitivity_log"), None, "mass-independence S",
                                              "log sensitivity", measurable=sc.get("measurable"))
    return out


def _l4(d, fam):
    vs = d.get("variants") or {}
    k = next((k for k in ("frame", "frame_t", "generated") if k in vs), next(iter(vs), None))
    if k is None:
        return {}
    v = vs[k]
    law, ctl = v.get("law") or {}, v.get("random_dir") or {}
    # the law's scoring: saturation (energy: plateau at 1), step (statics: tip indicator steps at x = 1), else a slope
    key = "saturation" if "saturation" in law else "step" if "step" in law else "slope"
    what = {"saturation": "plateau (law 1)", "step": "tip step at x = 1 (law 1)",
            "slope": f"slope (law {v.get('law_value')})"}[key]
    sw = (v.get("pairs") or {}).get("swap") or {}
    return {d["law"]: entry(law.get(key), ctl.get(key), law.get(f"{key}_ci95"), f"{what}; {k}",
                            "random direction", iia=_num(sw.get("iia")), wrong_way=_num(law.get("wrong_way")),
                            law_like=(v.get("verdict") or {}).get("law_like"))}


def _out_a(d, fam):
    out = {}
    m = (d.get("momentum") or {}).get("legal") or next(iter((d.get("momentum") or {}).values()), None)
    if m:
        out["momentum"] = entry(m.get("S_mom"), 0.0, m.get("S_mom_ci95"), "S_mom (generated)", "mass-blind rule")
    e = d.get("energy") or {}
    if e.get("generated"):
        out["energy"] = entry(e["generated"]["share_le1"], (e.get("real_legal") or {}).get("share_le1"),
                              e["generated"].get("share_le1_ci95"), f"share eps <= 1 + {e.get('tol')}", "real legal clips")
    real = [s_ for s_ in (d.get("real") or {}).values() if isinstance(s_, dict)]     # the real legal twins (track A)
    out.update(_ratio_laws(d.get("generated") or {}, real, "track A"))
    return out


def _ratio_laws(gen: dict, real: list[dict], tag: str) -> dict:
    """Each ratio law's primary statistic summarised over the generated readouts, floor = the same summary over the
    real clips (median over the real twins' summaries)."""
    out = {}
    for law, (st, how) in PRIMARY_STAT.items():
        x = gen.get(f"{how}_{st}")
        if x is not None:
            fl = [r[f"{how}_{st}"] for r in real if r.get(f"{how}_{st}") is not None]
            out[law] = entry(x, float(np.median(fl)) if fl else None, None,
                             f"{how} {st} ({tag}; law {0 if st in LEGAL_ZERO else 1})", "real clips")
    return out


def _out_b(d, fam):
    gates = d.get("gates") or {}
    g = next((x for x in ("strict", "loose") if x in gates), None)
    if g is None:
        return {}
    G, out = gates[g], {}
    mo = G.get("momentum") or {}
    if mo.get("generated"):
        out["momentum"] = entry(mo["generated"].get("antiparallel_mean"), (mo.get("real_legal") or {}).get("antiparallel_mean"),
                                mo["generated"].get("antiparallel_mean_ci95"), f"antiparallel mean ({g})", "real legal clips")
    e = G.get("energy") or {}
    if (e.get("generated") or {}).get("n"):
        out["energy"] = entry(e["generated"]["share_le1"], (e.get("real_legal") or {}).get("share_le1"),
                              e["generated"].get("share_le1_ci95"), f"share eps <= 1 + {e.get('tol')} ({g})", "real legal clips")
    real = (G.get("real") or {}).get("legal")
    out.update(_ratio_laws(G["generated"], [real] if real else [], g))
    return out


READERS = {"l0": ("l0", _l0), "l1": ("l1", _l1), "l2": ("l2", _l2), "l2rep": ("l2", _l2rep), "l2b": ("l2b", _l2b),
           "l3": ("l3", _l3), "l4": ("l4", _l4), "outA": ("outputs_trackA", _out_a), "outB": ("outputs_trackB", _out_b)}
LEVELS = tuple(READERS)


def flag_uninformative(rows: list[dict]) -> None:
    """H1 (in place): l0.mass_pre / l0.kappa_pre ``decodable`` also needs R² above the untrained model's on the same
    family (rows of a model whose name starts with "untrained"); every MASS_KEYS (KAPPA_KEYS) score of a model /
    family whose pre-event mass (kappa) is not decodable gets ``uninformative`` with the reason."""
    untr = {r["family"]: r["scores"] for r in rows if r["model"].startswith("untrained")}
    for r in rows:
        sc = r["scores"]
        for key, keys, what in (("l0.mass_pre", MASS_KEYS, "mass"), ("l0.kappa_pre", KAPPA_KEYS, "solid vs hollow")):
            e = sc.get(key)
            if e is None:
                continue
            u = (untr.get(r["family"]) or {}).get(key)
            if u is not None and not r["model"].startswith("untrained"):
                e["above_untrained"] = any(v > u["targets"].get(b, -np.inf) for b, v in e["targets"].items())
                e["decodable"] = bool(e["decodable"] and e["above_untrained"])
            if not e["decodable"]:
                for k_ in keys:
                    if k_ in sc:
                        sc[k_]["uninformative"] = f"{what} not decodable from pre-event frames (l0)"


def collect(results: Path) -> list[dict]:
    rows = []
    for fam_dir in sorted(p for p in results.glob("*/*") if p.is_dir() and p.parent.name != "figures"):
        row = {"model": fam_dir.parent.name, "family": fam_dir.name, "scores": {}}
        for level, (fname, fn) in READERS.items():
            f = fam_dir / f"{fname}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            for law, e in fn(d, fam_dir.name).items():
                if e is not None:
                    row["scores"][f"{level}.{law}"] = {**e, **({"stand_in": True} if d.get("stand_in") else {})}
        if row["scores"]:
            rows.append(row)
    flag_uninformative(rows)
    return rows


def _fmt(v, s="+.2f"):
    return "" if v is None else format(v, s)


def cell(e: dict | None) -> str:
    """'score (floor) [lo, hi]' + flags."""
    if not e:
        return ""
    s = _fmt(e["score"])
    if e.get("floor") is not None:
        s += f" ({_fmt(e['floor'])})"
    if e.get("ci95"):
        s += f" [{_fmt(e['ci95'][0])}, {_fmt(e['ci95'][1])}]"
    for k in ("iia", "selectivity_object"):
        if e.get(k) is not None:
            s += f" {k} {_fmt(e[k], '.2f')}"
    if e.get("status") == "unmeasurable" or e.get("measurable") is False:
        s += " (unmeasurable)"
    if e.get("claim") is False:
        s += " ✗claim"                          # L1: a negative control (or the mask floor) scores as well
    if "decodable" in e:
        s += " decodable" if e["decodable"] else " NOT decodable"
    if e.get("uninformative"):
        s += " (uninformative)"
    return s + (" *" if e.get("stand_in") else "")


def markdown(rows: list[dict]) -> str:
    """One table per family; one column per level.law present; cell = score (floor / baseline) [95% CI]."""
    out = ["# Leaderboard (spec §8)", "",
           "Cell = primary score (its floor / baseline) [95% CI]; blank = not run; * = stand-in run. Sources and floors: "
           "physlaw/analysis/aggregate.py. ✗claim = an L1 negative-control forecast (or the mask-only floor) scores "
           "as well as the forecast, so no law claim. (uninformative) = a mass-independence (rolling) score of a model "
           "that does not encode mass (solid vs hollow) before the event: l0.mass_pre (l0.kappa_pre) beside it.", ""]
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam]
        cols = sorted({k for r in fr for k in r["scores"]}, key=lambda k: (LEVELS.index(k.split(".")[0]), k))
        out += [f"## {fam}", "", "| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        out += [f"| {r['model']} | " + " | ".join(cell(r["scores"].get(c)) for c in cols) + " |" for r in fr]
        notes = {c: next(r["scores"][c] for r in fr if c in r["scores"]) for c in cols}
        out += [""] + [f"* {c}: {e['metric']}; floor = {e['floor_is']}" for c, e in notes.items()] + [""]
    return "\n".join(out)


def run(results: Path) -> dict:
    rows = collect(results)
    (results / "leaderboard.json").write_text(json.dumps({"levels": list(LEVELS), "rows": rows}, indent=1, default=float))
    (results / "leaderboard.md").write_text(markdown(rows))
    return {"rows": rows}
