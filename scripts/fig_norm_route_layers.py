#!/usr/bin/env python
"""FIGURE: §34a's norm-route AUC across depth, both encoders, both datasets.

    stage 1  ridge:  activations -> (dpx, dpy)     linear
    stage 2  numpy:  score = |prediction|          the norm, applied OUTSIDE the probe
    stage 3  AUC of that score against conserved / violating

Two panels because §38g is the live question: v1 (`probe_set`) is where §34's
positive was measured, and its violating scenes are separable by one mass-free
scalar at AUC 1.000 (§38); v4 (`probe_set_v4_legacy_fixed`) is the leak-free
regeneration, on which Delta-p is barely decodable because the fixed mechanism
forces Delta-p exactly along the contact normal.

Depth is plotted as a FRACTION so a 12-block and a 24-block encoder are
comparable -- and so the x-axis is the one Joseph et al.'s "Physics Emergence
Zone at ~1/3 depth" is stated on.

    .venv/bin/python scripts/fig_norm_route_layers.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "videomae-base": dict(label="VideoMAE-base", colour="#1F6F78", marker="o"),
    "vjepa2-vitl":   dict(label="V-JEPA2-ViT-L", colour="#C4622D", marker="s"),
}
PANELS = [
    ("docs/artifacts/norm_route_v1.json", "outputs_{m}",
     "probe_set (v1)", "original violation mechanism"),
    ("docs/artifacts/norm_route_v4.json", "outputs_v4_{m}",
     "probe_set_v4_legacy_fixed", "leak-free regeneration"),
]
POOLS = {"all.pre": dict(ls="-", lw=2.0, alpha=1.0, z=3),
         "all.post": dict(ls="--", lw=1.1, alpha=.55, z=2),
         "all.all": dict(ls=":", lw=1.1, alpha=.55, z=2)}


def depth_index(site: str) -> int:
    """embed.out -> 0; block.k.resid_post -> k+1."""
    if site.startswith("embed"):
        return 0
    return int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": .8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    for ax, (path, tmpl, title, sub) in zip(axes, PANELS):
        if not Path(path).exists():
            ax.text(.5, .5, f"missing\n{path}", ha="center", va="center",
                    transform=ax.transAxes, color="#999999")
            continue
        rep = json.load(open(path))
        for key, style in MODELS.items():
            outdir = tmpl.format(m=key)
            if outdir not in rep:
                continue
            for pool, ps in POOLS.items():
                if pool not in rep[outdir]:
                    continue
                layers = rep[outdir][pool]["layers"]
                idx = sorted(layers, key=depth_index)
                x = np.array([depth_index(s) for s in idx], float)
                x = x / x.max()
                y = np.array([layers[s]["auc"] for s in idx])
                ax.plot(x, y, color=style["colour"], ls=ps["ls"], lw=ps["lw"],
                        alpha=ps["alpha"], zorder=ps["z"],
                        label=style["label"] if pool == "all.pre" else None)
                if pool == "all.pre":
                    b = int(np.argmax(y))
                    ax.plot(x[b], y[b], style["marker"], ms=5.5,
                            color=style["colour"], zorder=5,
                            markeredgecolor="white", markeredgewidth=.7)
                    # a peak in the upper-right corner would sit under the pool
                    # legend, so label those to the LEFT of the marker instead
                    corner = x[b] > .70 and y[b] > .60
                    ax.annotate(f"{y[b]:.3f}", (x[b], y[b]),
                                textcoords="offset points",
                                xytext=(-8, -3) if corner else (0, 8),
                                ha="right" if corner else "center",
                                fontsize=7.5, color=style["colour"])
            # grid-max null band for the headline pool
            if "all.pre" in rep[outdir]:
                p95 = rep[outdir]["all.pre"]["null_p95"]
                ax.axhline(p95, color=style["colour"], lw=.7, alpha=.35, ls=(0, (1, 2)))

        ax.axhline(.5, color="#888888", lw=.8)   # chance; labelled in the legend
        ax.set_xlim(-.03, 1.03)
        # title on two lines, set as text so the two sizes do not collide
        ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=9.5, va="bottom")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5,
                color="#777777", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("norm-route AUC\n(regress $\\Delta p$, take $|\\cdot|$ outside)")
    axes[0].set_ylim(.36, .86)
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#888888", lw=.8),
          plt.Line2D([], [], color="#888888", lw=.7, ls=(0, (1, 2)))]
    l += ["chance (0.5)", "grid-max null, 95th pct"]
    line_key = [plt.Line2D([], [], color="#555555", ls=st["ls"], lw=st["lw"],
                           alpha=st["alpha"]) for st in POOLS.values()]
    axes[0].legend(h, l, loc="upper left", fontsize=7.5, handlelength=1.8,
                   labelspacing=.35)
    axes[1].legend(line_key, list(POOLS), loc="upper right", fontsize=7.5,
                   handlelength=2.2, title="pool", title_fontsize=7.5)
    fig.supxlabel("relative depth   (0 = patch embedding, 1 = final block)",
                  fontsize=9, y=.06)

    fig.tight_layout(rect=(0, .08, 1, 1))
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_norm_route_layers.{ext}", bbox_inches="tight")
    print(f"wrote figures/fig_norm_route_layers.pdf and .png")

    # the numbers the caption has to quote
    print("\ncaption numbers:")
    for path, tmpl, title, _ in PANELS:
        if not Path(path).exists():
            print(f"  {title}: MISSING"); continue
        rep = json.load(open(path))
        for key, style in MODELS.items():
            o = tmpl.format(m=key)
            if o in rep and "all.pre" in rep[o]:
                v = rep[o]["all.pre"]
                print(f"  {title:28s} {style['label']:15s} all.pre AUC {v['auc']:.3f} "
                      f"@ {v['best_site']:<22} null p95 {v['null_p95']:.3f}  p={v['p']:.3f} "
                      f" (dp R2 {v['delta_p_r2']:+.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
