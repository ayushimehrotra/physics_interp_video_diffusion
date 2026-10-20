#!/usr/bin/env python
"""FIGURE: a frame-wide pool holds the SUM of the two objects, not either part."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("docs/artifacts/pool_sum_sweep.json"))
ORDER = ["$v_A + v_B$", "$v_A$", "$v_B$", "$v_A - v_B$"]
COL = {"$v_A + v_B$": "#1F6F8B", "$v_A$": "#D97706",
       "$v_B$": "#8E44AD", "$v_A - v_B$": "#B03A2E"}
LW  = {"$v_A + v_B$": 2.0, "$v_A$": 1.4, "$v_B$": 1.4, "$v_A - v_B$": 2.0}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": .8, "axes.edgecolor": "#444444",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 200,
})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, (model, rec) in zip(axes, R.items()):
    sw = rec["depth_sweep"]
    dmax = max(v["depth"] for v in sw["$v_A$"].values())
    for name in ORDER:
        pts = sorted(sw[name].values(), key=lambda v: v["depth"])
        x = np.array([p["depth"] for p in pts]) / dmax
        y = np.array([p["mean"] for p in pts]); sd = np.array([p["sd"] for p in pts])
        ax.plot(x, y, color=COL[name], lw=LW[name], label=name, zorder=3)
        ax.fill_between(x, y - sd, y + sd, color=COL[name], alpha=.18, lw=0, zorder=2)
    ax.text(0, 1.10, model, transform=ax.transAxes, fontsize=9.5, va="bottom")
    ax.text(0, 1.02, "frame-wide pool, pre-contact velocities", transform=ax.transAxes,
            fontsize=7.5, color="#777777", va="bottom")
    ax.set_ylim(-0.08, 1.06); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("out-of-fold $R^2$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.4, handlelength=1.8,
           columnspacing=1.6, bbox_to_anchor=(.5, .085))
fig.text(.5, .015, "relative depth   (0 = patch embedding, 1 = final block)",
         ha="center", fontsize=9)
fig.subplots_adjust(bottom=.30, top=.86, wspace=.07)
for e in ("png", "pdf"):
    fig.savefig(f"figures/pool_sum_sweep.{e}", bbox_inches="tight")
print("wrote figures/pool_sum_sweep.png and .pdf")
