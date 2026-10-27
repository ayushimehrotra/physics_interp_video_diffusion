#!/usr/bin/env python
"""FIGURE: log KE read within family M of `ke_factor_set`, across depth (§63a).

Family M scales both masses by c at fixed velocities, so the clip is the same video at
every level and only the reduced mass moves KE. Left: out-of-fold R2 of a linear probe
on log KE, M clips only. Right: within-group slope of predicted on true log KE -- 1 means
the readout steps by the right amount when only mass changes.

Each model is drawn at its own best pool for family M (`all.all` for both), the curve
`scripts/ke_reduced_mass.py` stores as `within_family.M.depth_curve_best_pool`. Depth is
relative because the giant has 40 blocks and V-JEPA2-L 24.

    .venv/bin/python scripts/fig_ke_family_m_depth.py
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "docs/artifacts/ke_reduced_mass_videomaev2-giant.json":
        dict(label="VideoMAEv2-giant", colour="#2a78d6", marker="o"),
    "docs/artifacts/ke_reduced_mass_vjepa2-vitl.json":
        dict(label="V-JEPA2-ViT-L", colour="#eb6834", marker="s"),
}


def depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="figures/fig_ke_family_m_depth")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": .8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, (ax_r2, ax_sl) = plt.subplots(1, 2, figsize=(7.6, 3.2))

    for path, st in MODELS.items():
        M = json.load(open(path))["within_family"]["M"]
        curve = M["depth_curve_best_pool"]
        sites = sorted(curve, key=depth)
        x = np.array([depth(s) for s in sites], float); x /= x.max()
        r2 = np.array([curve[s]["r2"] for s in sites])
        sl = np.array([curve[s]["within_slope"] for s in sites])
        pool = M["best_cell"].split(" @ ")[0]
        ax_r2.plot(x, r2, color=st["colour"], lw=2, label=f"{st['label']}  ({pool})")
        ax_sl.plot(x, sl, color=st["colour"], lw=2)
        b = int(np.argmax(r2))
        for ax, y in ((ax_r2, r2), (ax_sl, sl)):
            ax.plot(x[b], y[b], st["marker"], ms=6, color=st["colour"], zorder=5,
                    markeredgecolor="white", markeredgewidth=.8)
        print(f"  {st['label']:17s} [{pool}] best R2 {r2[b]:.3f} at {sites[b]} "
              f"(slope {sl[b]:.3f});  embed.out R2 {r2[0]:.3f}, slope {sl[0]:.3f}")

    ax_r2.set_ylabel("out-of-fold $R^2$,  log KE")
    ax_r2.set_ylim(0, 1.04)
    ax_r2.legend(loc="lower right", fontsize=8, handlelength=1.8)
    ax_r2.set_title("Linear probe on log KE, family M", fontsize=9.5, loc="left")

    ax_sl.axhline(1.0, color="#888888", lw=.8, ls=":", zorder=1)
    ax_sl.text(1.02, 1.0, "energy\ncode = 1", fontsize=7, color="#666666",
               va="center", ha="left", transform=ax_sl.get_yaxis_transform())
    ax_sl.set_ylabel("within-group slope, pred vs true log KE")
    ax_sl.set_ylim(0.6, 1.1)
    ax_sl.set_title("Does it step correctly when only mass moves?", fontsize=9.5, loc="left")

    for ax in (ax_r2, ax_sl):
        ax.set_xlim(-.03, 1.03)
        ax.grid(axis="y", color="#e6e6e6", lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("relative depth   (0 = patch embedding, 1 = final block)", fontsize=9, y=.04)
    fig.tight_layout(rect=(0, .06, 1, 1))

    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.pdf and .png")


if __name__ == "__main__":
    main()
