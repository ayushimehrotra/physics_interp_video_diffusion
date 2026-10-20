#!/usr/bin/env python
"""FIGURE: norm-route score across depth, elastic vs inelastic collisions.

The y-axis follows the artifact's `stat` field -- "paired accuracy" for the
twin-vs-twin statistic, "AUC" when all clips are ranked and the pairing is
used only to keep twins in the same fold and to build the null.

Reads `docs/artifacts/norm_route_cons_layers.json`. Statistic and null are
`probe_conservation_set.py`'s, so this curve and the published per-cell table are the
same measurement: paired accuracy over twins, with the exact twin-exchange null.

    .venv/bin/python scripts/fig_norm_route_cons_layers.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "outputs_cons_videomae-base": dict(label="VideoMAE-base", colour="#1F6F78", marker="o"),
    "outputs_cons_vjepa2-vitl":   dict(label="V-JEPA2-ViT-L", colour="#C4622D", marker="s"),
}
STRATA = [("elastic", "elastic collisions", "$e = 1.0$, kinetic energy conserved"),
          ("inelastic", "inelastic collisions", "$e = 0.15$, energy lost to deformation")]


def depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/artifacts/norm_route_cons_layers_allpre.json")
    ap.add_argument("-o", "--out", default="figures/fig_norm_route_elastic_inelastic")
    a = ap.parse_args()
    rep = json.load(open(a.data))
    pool = rep.get("pool", "all.all")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": .8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    for ax, (key, title, sub) in zip(axes, STRATA):
        npairs = None
        for outdir, st in MODELS.items():
            if outdir not in rep or key not in rep[outdir]:
                continue
            r = rep[outdir][key]
            npairs = r["n_pairs"]
            sites = sorted(r["layers"], key=depth)
            x = np.array([depth(s) for s in sites], float); x /= x.max()
            y = np.array([r["layers"][s]["mean"] for s in sites])
            sd = np.array([r["layers"][s]["sd"] for s in sites])
            # the grid-max null band is no longer drawn; the corrected p it produces
            # is still reported in the legend and in the caption numbers below
            # +-1 sd over the fold seeds; shown, but not called out in the legend
            ax.fill_between(x, y - sd, y + sd, color=st["colour"], alpha=.20,
                            lw=0, zorder=2)
            ax.plot(x, y, color=st["colour"], lw=2.0, zorder=3,
                    label=f"{st['label']}   {r['best']:.3f}"
                          f"  ($p$ = {r['p_corrected']:.3f})")
            b = int(np.argmax(y))
            ax.plot(x[b], y[b], st["marker"], ms=5.5, color=st["colour"], zorder=5,
                    markeredgecolor="white", markeredgewidth=.7)
            # peak VALUES go in the caption, not on the axes: the argmax over
            # layers is a winner's curse and the two models peak within 0.001
            # of each other, so inline labels collide as well as mislead
        ax.axhline(.5, color="#888888", lw=.8, zorder=2)
        ax.set_xlim(-.03, 1.03)
        ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=9.5, va="bottom")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5,
                color="#777777", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
        # each panel carries its OWN best values and corrected p
        # headroom is opened above the curves (max 0.702) so the legend clears them
        ax.legend(loc="upper left", fontsize=8, handlelength=1.8, labelspacing=.3)

    axes[0].set_ylabel("AUC" if rep.get("stat") == "auc" else "paired accuracy")
    # headroom above the curves (max 0.702) so the legend cannot touch them
    axes[0].set_ylim(.43, .87)
    fig.supxlabel("relative depth   (0 = patch embedding, 1 = final block)",
                  fontsize=9, y=.06)
    fig.tight_layout(rect=(0, .08, 1, .97))

    Path(a.out).parent.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight")
    print(f"wrote {a.out}.pdf and .png\n")
    for outdir, st in MODELS.items():
        for key, title, _ in STRATA:
            if outdir in rep and key in rep[outdir]:
                r = rep[outdir][key]; L = r["layers"][r["best_site"]]
                nl = len(r["layers"])
                above = sum(1 for v in r["layers"].values()
                            if v["mean"] > r["gridmax_null_p95"])
                L = r["layers"][r["best_site"]]
                print(f"  {st['label']:15s} {title:22s} best {r['best']:.3f} "
                      f"+- {L['sd']:.3f} at {r['best_site']:<22}")
                print(f"                  grid-max null p95 {r['gridmax_null_p95']:.3f}"
                      f"   p_corrected = {r['p_corrected']:.3f}"
                      f"   ({above}/{nl} layers above the grid-max null)")


if __name__ == "__main__":
    raise SystemExit(main())
