"""The eight standard figures (spec §8), from the leaderboard rows and results/; each drawn only when its inputs exist.

1 L0 vs L1          mean L0 R² vs L1 S (law family, else outcome) per model, F1
2 L2 adj vs raw     raw vs motion-adjusted win rate per contrast and model
3 L2 vs L2b         adjusted win rate vs trained-detection AUC (encoded but unused below the diagonal's mirror)
4 Galilean          invariance error vs boost speed per model and target (F4)
5 internal/output   L1 S vs output score (track A S_outcome, else track B loose measurable fraction) per model
6 pairs             base vs physics-post-trained deltas on every level (PAIRS)
7 objectives        predictive vs reconstruction vs denoising across L1 / L2 scores
8 positive control  positive controls vs the large models on the L1 / L2 axes
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Categorical order (reference palette, light mode): fixed slots, never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#7d7a73")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OBJECTIVE = (("vjepa", "predictive"), ("videomae", "reconstruction"), ("untrained", "control"),
             ("positive", "positive control"), ("pc", "positive control"))
PAIRS = (("wan2.2-ti2v-5b", "physrvg"), ("cogvideox-5b", "videorepa-5b"))   # (base, physics post-trained)


def objective(model: str) -> str:
    return next((o for p, o in OBJECTIVE if model.startswith(p)), "denoising")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _colors(models):
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(sorted(models))}


def _l1_score(r):
    l1 = r.get("l1") or {}
    fam = [v for k, v in l1.items() if k.startswith("S_family.") and v is not None]
    if fam:
        return float(np.mean(fam)), "L1 S_family (mean over laws)"
    v = l1.get("S_outcome.conserve_e1")
    return (v, "L1 S_outcome (conserve_e1, frame_mm)") if v is not None else (None, None)


def _scatter(ax, pts, colors):
    for m, x, y in pts:
        ax.scatter([x], [y], s=64, color=colors[m], edgecolor="white", linewidth=2, zorder=3, label=m)
        ax.annotate(m, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)


def draw(rows: list[dict], results: Path, out: Path, log=print) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    made = []
    colors = _colors({r["model"] for r in rows})
    by = {(r["model"], r["family"]): r for r in rows}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150, facecolor="white")
        plt.close(fig)
        made.append(name)

    # 1 L0 vs L1
    pts = [(r["model"], r["l0"]["mean_r2"], _l1_score(r)[0]) for r in rows
           if r["family"] == "f1" and r.get("l0") and _l1_score(r)[0] is not None]
    if len(pts) >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        _scatter(ax, pts, colors)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(0, 1.05); ax.set_ylim(-1.05, 1.05)
        _style(ax, "1. States decodable, laws weak (F1)", "L0 mean R²", _l1_score(by[pts[0][0], "f1"])[1])
        save(fig, "fig1_l0_vs_l1")

    # 2 L2 raw vs adjusted
    raw_adj = []
    for r in rows:
        f = results / r["model"] / r["family"] / "l2.json"
        if r["family"] != "f1" or not r.get("l2") or not f.exists():
            continue
        d = json.loads(f.read_text())
        cs, dist = r["l2"]["_covariates"], r["l2"]["_distance"]
        for c, v in d["motion_adjusted"]["contrasts"][cs].items():
            if dist in v:
                raw_adj.append((r["model"], c, v[dist]["raw_win_rate"], v[dist]["adj_win_rate"]))
    if raw_adj:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for m in sorted({x[0] for x in raw_adj}):
            xs = [(a, b) for mm, _, a, b in raw_adj if mm == m]
            ax.scatter(*zip(*xs), s=48, color=colors[m], edgecolor="white", linewidth=2, zorder=3, label=m)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color=GRID, linewidth=1)
        ax.legend(fontsize=7, frameon=False)
        _style(ax, "2. The motion confound (F1, every contrast)", "raw win rate", "motion-adjusted win rate")
        save(fig, "fig2_l2_adjusted_vs_raw")

    # 3 L2 vs L2b
    pts3 = [(r["model"], c, r["l2"][c], a) for r in rows if r.get("l2b") and r.get("l2")
            for c, a in r["l2b"].items() if isinstance(r["l2"].get(c), float)]
    if pts3:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for m in sorted({x[0] for x in pts3}):
            xs = [(a, b) for mm, _, a, b in pts3 if mm == m]
            ax.scatter(*zip(*xs), s=48, color=colors[m], edgecolor="white", linewidth=2, zorder=3, label=m)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        ax.legend(fontsize=7, frameon=False)
        _style(ax, "3. Encoded but unused", "L2 adjusted win rate", "L2b AUC")
        save(fig, "fig3_l2_vs_l2b")

    # 4 Galilean invariance error vs boost size
    gal = []
    for r in rows:
        f = results / r["model"] / r["family"] / "l3.json"
        if f.exists():
            g = json.loads(f.read_text()).get("galilean")
            if g and not json.loads(f.read_text()).get("stand_in"):
                for t, d in g["targets"].items():
                    pts_ = [(0.0, d["within_error"])] + [(v["boost_speed_mean"], v["invariance_error"]) for v in d["by_level"].values()
                                                        if v["invariance_error"] is not None and v["boost_speed_mean"] is not None]
                    gal.append((r["model"], t, sorted(pts_)))
    if gal:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m, t, p in gal:
            ax.plot(*zip(*p), color=colors[m], linewidth=2, marker="o", markersize=7,
                    linestyle="-" if t == "restitution" else "--", label=f"{m} {t}")
        ax.legend(fontsize=7, frameon=False)
        _style(ax, "4. Galilean invariance error vs boost (F4; u = 0: within-condition error)", "boost speed |u|",
               "error / sd(target)")
        save(fig, "fig4_galilean")

    # 5 internal vs output
    pts5 = []
    for r in rows:
        x = _l1_score(r)[0]
        oa, ob = r.get("outputs_trackA") or {}, r.get("outputs_trackB") or {}
        y = oa.get("S_outcome.conserve_e1")
        if y is None and "measurable.loose" in ob:
            k, n = (int(v) for v in ob["measurable.loose"].split("/"))
            y = k / n if n else None
        if x is not None and y is not None:
            pts5.append((r["model"], x, y))
    if len(pts5) >= 2:                   # a correlation across models needs at least two
        fig, ax = plt.subplots(figsize=(5, 4))
        _scatter(ax, pts5, colors)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        _style(ax, "5. Internal vs output adherence", "L1 score", "output score (track A S_outcome, else B measurable)")
        save(fig, "fig5_internal_vs_output")

    # 6 post-training pairs
    keys = lambda r: {**{f"L1 {k}": v for k, v in (r.get("l1") or {}).items() if isinstance(v, float)},
                      **{f"L2 {k}": v for k, v in (r.get("l2") or {}).items() if isinstance(v, float)},
                      **({"L0 R²": r["l0"]["mean_r2"]} if r.get("l0") else {})}
    pairs = [(b, p) for b, p in PAIRS if (b, "f1") in by and (p, "f1") in by]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
        for ax, (b, p) in zip(axes[0], pairs):
            kb, kp = keys(by[b, "f1"]), keys(by[p, "f1"])
            ks = sorted(set(kb) & set(kp))
            ax.barh(range(len(ks)), [kp[k] - kb[k] for k in ks], color=SERIES[0], height=0.6)
            ax.set_yticks(range(len(ks)), ks)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            _style(ax, f"6. {p} - {b}", "delta (physics-trained minus base)", "")
        save(fig, "fig6_post_training_pairs")

    # 7 objectives, 8 positive control: the same per-score dot plot, models grouped by objective
    def dotplot(sel, name, title):
        ms = [r for r in rows if r["family"] == "f1" and sel(r) and keys(r)]
        ks = sorted(set().union(*[keys(r) for r in ms])) if ms else []
        if len(ms) < 2 or not ks:
            return
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(ks) + 1.5))
        objs = sorted({objective(r["model"]) for r in ms})
        seen: dict = {}
        for r in ms:                     # colour = objective, marker = model within it
            kv, o = keys(r), objective(r["model"])
            mk = "osD^v"[seen.setdefault(o, []).__len__() % 5]
            seen[o].append(r["model"])
            ax.scatter([kv.get(k, np.nan) for k in ks], range(len(ks)), s=48, color=SERIES[objs.index(o)], marker=mk,
                       edgecolor="white", linewidth=1.5, zorder=3, label=f"{r['model']} ({o})")
        ax.set_yticks(range(len(ks)), ks)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        _style(ax, title, "score", "")
        save(fig, name)
    dotplot(lambda r: objective(r["model"]) in ("predictive", "reconstruction", "denoising"), "fig7_objectives",
            "7. Training objective (F1)")
    if any(objective(r["model"]) == "positive control" for r in rows):
        dotplot(lambda r: objective(r["model"]) != "control", "fig8_positive_control", "8. Positive control vs large models (F1)")
    log(f"[figures] drew {made or 'nothing'} -> {out}")
    return made
