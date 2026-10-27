#!/usr/bin/env python
"""FIGURE: norm-route AUC across depth on `conservation_set_v1`, elastic vs inelastic.

Reads `docs/artifacts/norm_route_cons.json`. The solid line per model is that model's
BEST POOL for that stratum -- named on the plot, because the selection is part of the
claim -- and the faint lines behind it are the other eight pools, so the reader can see
how much of the peak is pool choice. The null band corrects over the FULL grid the
winner was chosen from (all pools x all sites), so it is comparable to §34's.

    .venv/bin/python scripts/figures/fig_norm_route_cons.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "outputs/cons_videomae-base": dict(label="VideoMAE-base", colour="#1F6F78", marker="o"),
    "outputs/cons_vjepa2-vitl":   dict(label="V-JEPA2-ViT-L", colour="#C4622D", marker="s"),
}
STRATA = [("elastic", "elastic collisions", "restitution e = 1"),
          ("inelastic", "inelastic collisions", "e < 1, energy lost to deformation")]


def depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def curve(cells: dict, pool: str):
    rows = [(s.split("/")[0], v) for s, v in cells.items() if s.split("/")[1] == pool]
    rows.sort(key=lambda r: depth(r[0]))
    x = np.array([depth(s) for s, _ in rows], float)
    return x / x.max(), np.array([v for _, v in rows])


def main() -> None:
    rep = json.load(open("docs/artifacts/norm_route_cons.json"))
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": .8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    for ax, (key, title, sub) in zip(axes, STRATA):
        for outdir, st in MODELS.items():
            if outdir not in rep or key not in rep[outdir]:
                continue
            r = rep[outdir][key]
            pools = sorted({c.split("/")[1] for c in r["cells"]})
            for pool in pools:                                   # context, faint
                if pool == r["best_pool"]:
                    continue
                x, y = curve(r["cells"], pool)
                ax.plot(x, y, color=st["colour"], lw=.6, alpha=.20, zorder=1)
            x, y = curve(r["cells"], r["best_pool"])             # the winner
            ax.plot(x, y, color=st["colour"], lw=2.0, zorder=3,
                    label=f"{st['label']}  ({r['best_pool']})")
            b = int(np.argmax(y))
            ax.plot(x[b], y[b], st["marker"], ms=5.5, color=st["colour"], zorder=5,
                    markeredgecolor="white", markeredgewidth=.7)
            corner = x[b] > .70
            ax.annotate(f"{y[b]:.3f}", (x[b], y[b]), textcoords="offset points",
                        xytext=(-8, -3) if corner else (0, 8),
                        ha="right" if corner else "center",
                        fontsize=7.5, color=st["colour"])
            ax.axhline(r["null_p95"], color=st["colour"], lw=.7, alpha=.40,
                       ls=(0, (1, 2)), zorder=2)

        ax.axhline(.5, color="#888888", lw=.8, zorder=2)
        ax.set_xlim(-.03, 1.03)
        ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=9.5, va="bottom")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5,
                color="#777777", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
        n = rep[list(MODELS)[0]][key]["n"]
        ax.text(.985, .035, f"n = {n}", transform=ax.transAxes, fontsize=7.5,
                color="#777777", ha="right")
        ax.legend(loc="upper left", fontsize=7.5, handlelength=1.8, labelspacing=.35)

    axes[0].set_ylabel("norm-route AUC\n(regress $\\Delta p$, take $|\\cdot|$ outside)")
    axes[0].set_ylim(.34, .86)
    extra = [plt.Line2D([], [], color="#888888", lw=.8),
             plt.Line2D([], [], color="#888888", lw=.7, ls=(0, (1, 2))),
             plt.Line2D([], [], color="#888888", lw=.6, alpha=.35)]
    axes[1].legend(*zip(*[(h, t) for h, t in zip(
        extra, ["chance (0.5)", "full-grid null, 95th pct", "the other 8 pools"])]),
        loc="lower left", fontsize=7, handlelength=1.8, labelspacing=.3)
    fig.supxlabel("relative depth   (0 = patch embedding, 1 = final block)",
                  fontsize=9, y=.06)
    fig.tight_layout(rect=(0, .08, 1, 1))

    out = Path("figures"); out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_norm_route_elastic_inelastic.{ext}", bbox_inches="tight")
    print("wrote figures/fig_norm_route_elastic_inelastic.pdf and .png\n")
    print("caption numbers:")
    for outdir, st in MODELS.items():
        for key, title, _ in STRATA:
            if outdir in rep and key in rep[outdir]:
                r = rep[outdir][key]
                print(f"  {st['label']:15s} {title:22s} AUC {r['auc']:.3f}  "
                      f"{r['best_cell']:<32} null p95 {r['null_p95']:.3f}  "
                      f"p = {r['p_corrected']:.3f}  n={r['n']}")


if __name__ == "__main__":
    raise SystemExit(main())
