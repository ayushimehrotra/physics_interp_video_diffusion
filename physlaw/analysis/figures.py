"""The standard figures (spec §8), from the leaderboard rows (``row["scores"]["<level>.<law>"]``, physlaw/analysis/
aggregate.py) and results/; each drawn only when its inputs exist.

1 L0 vs L1          L0 velocity held-out R² vs L1 S_mom per model (F1)
2 L2 vs pixels      L2 win rate of the primary contrast vs the best pixel baseline's, per model and family
3 L2 vs L2b         L2 win rate vs L2b held-out AUC (encoded but unused below the diagonal)
4 Galilean          decoded d_epsilon per boost level per model (F4; law 0)
5 internal/output   L1 S_mom vs the output score (track A S_mom, else track B antiparallel mean) per model (F1)
6 pairs             base vs physics-post-trained deltas on every score (PAIRS)
7 objectives        predictive vs reconstruction vs denoising across every score of every family / law
8 positive control  positive controls vs the large models across every score of every family / law
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


def score(r: dict, key: str):
    e = (r.get("scores") or {}).get(key)
    return e["score"] if e else None


def scores(r: dict) -> dict:
    return {k: e["score"] for k, e in (r.get("scores") or {}).items()}


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

    def scatter_fig(pts, name, title, xl, yl, lim=None, diag=False):
        if len(pts) < 2:
            return
        fig, ax = plt.subplots(figsize=(5, 4))
        _scatter(ax, pts, colors)
        if diag:
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        if lim:
            ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
        _style(ax, title, xl, yl)
        save(fig, name)

    f1 = [r for r in rows if r["family"] == "f1"]
    # 1 L0 vs L1
    scatter_fig([(r["model"], score(r, "l0.velocity"), score(r, "l1.momentum")) for r in f1
                 if score(r, "l0.velocity") is not None and score(r, "l1.momentum") is not None],
                "fig1_l0_vs_l1", "1. States decodable vs forecast law (F1)", "L0 velocity held-out R²", "L1 S_mom",
                ((0, 1.05), (-1.05, 1.05)))
    # 2 L2 vs pixel baselines
    p2 = [(f"{r['model']}/{r['family']}", r["scores"]["l2.momentum"]["floor"], r["scores"]["l2.momentum"]["score"])
          for r in rows if (r["scores"].get("l2.momentum") or {}).get("floor") is not None]
    if p2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cs = _colors({m for m, _, _ in p2})
        _scatter(ax, p2, cs)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color=GRID, linewidth=1)
        _style(ax, "2. Model surprise vs pixel baselines (primary contrast)", "best pixel-baseline win rate",
               "model win rate")
        save(fig, "fig2_l2_vs_pixels")
    # 3 L2 vs L2b
    scatter_fig([(r["model"], score(r, "l2.momentum"), score(r, "l2b.momentum")) for r in f1
                 if score(r, "l2.momentum") is not None and score(r, "l2b.momentum") is not None],
                "fig3_l2_vs_l2b", "3. Encoded but unused (momentum)", "L2 win rate", "L2b AUC", diag=True)
    # 4 Galilean d_epsilon per boost level
    gal = []
    for r in rows:
        f = results / r["model"] / r["family"] / "l3.json"
        if f.exists():
            d = json.loads(f.read_text())
            dec = ((d.get("galilean") or {}).get("galilean") or {}).get("decoded") or {}
            pts = [(lv, v["d_epsilon"]) for lv, v in sorted(dec.items()) if v.get("d_epsilon") is not None]
            if pts and not d.get("stand_in"):
                gal.append((r["model"], pts))
    if gal:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m, p in gal:
            ax.plot([x for x, _ in p], [y for _, y in p], color=colors[m], linewidth=2, marker="o", markersize=7, label=m)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.legend(fontsize=7, frameon=False)
        _style(ax, "4. Galilean invariance (F4): d_epsilon vs its u = 0 partner (law 0)", "boost level", "d_epsilon")
        save(fig, "fig4_galilean")
    # 5 internal vs output
    out_score = lambda r: score(r, "outA.momentum") if score(r, "outA.momentum") is not None else score(r, "outB.momentum")
    scatter_fig([(r["model"], score(r, "l1.momentum"), out_score(r)) for r in f1
                 if score(r, "l1.momentum") is not None and out_score(r) is not None],
                "fig5_internal_vs_output", "5. Internal vs output adherence (F1)", "L1 S_mom",
                "output score (track A S_mom, else B antiparallel)", ((-1.05, 1.05), (-1.05, 1.05)))
    # 6 post-training pairs
    pairs = [(b, p) for b, p in PAIRS if (b, "f1") in by and (p, "f1") in by]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
        for ax, (b, p) in zip(axes[0], pairs):
            kb, kp = scores(by[b, "f1"]), scores(by[p, "f1"])
            ks = sorted(set(kb) & set(kp))
            ax.barh(range(len(ks)), [kp[k] - kb[k] for k in ks], color=SERIES[0], height=0.6)
            ax.set_yticks(range(len(ks)), ks)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            _style(ax, f"6. {p} - {b}", "delta (physics-trained minus base)", "")
        save(fig, "fig6_post_training_pairs")

    # 7 objectives, 8 positive control: the same per-score dot plot over every family and law (F1 momentum and
    # energy, F6 friction / glide inertia, F7 gravity, F9 oscillation, F10 rolling, F11 statics, F12 angular
    # momentum, ...), models grouped by objective
    per_model: dict = {}
    for r in rows:
        per_model.setdefault(r["model"], {}).update({f"{r['family']} {k}": v for k, v in scores(r).items()})

    def dotplot(sel, name, title):
        ms = [m for m, kv in sorted(per_model.items()) if sel(m) and kv]
        ks = sorted(set().union(*[per_model[m] for m in ms])) if ms else []
        if len(ms) < 2 or not ks:
            return
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(ks) + 1.5))
        objs = sorted({objective(m) for m in ms})
        seen: dict = {}
        for m in ms:                     # colour = objective, marker = model within it
            kv, o = per_model[m], objective(m)
            mk = "osD^v"[len(seen.setdefault(o, [])) % 5]
            seen[o].append(m)
            ax.scatter([kv.get(k, np.nan) for k in ks], range(len(ks)), s=48, color=SERIES[objs.index(o)], marker=mk,
                       edgecolor="white", linewidth=1.5, zorder=3, label=f"{m} ({o})")
        ax.set_yticks(range(len(ks)), ks)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        _style(ax, title, "score", "")
        save(fig, name)
    dotplot(lambda m: objective(m) in ("predictive", "reconstruction", "denoising"), "fig7_objectives",
            "7. Training objective (every family and law)")
    if any(objective(r["model"]) == "positive control" for r in rows):
        dotplot(lambda m: objective(m) != "control", "fig8_positive_control",
                "8. Positive control vs large models (every family and law)")
    log(f"[figures] drew {made or 'nothing'} -> {out}")
    return made
