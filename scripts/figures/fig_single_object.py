#!/usr/bin/env python
"""FIGURE: per-object attributes across depth, under two poolings.

Object-local masking makes these quantities linearly available at the patch embedding, so
no emergence is visible. Frame-wide mean pooling -- the protocol of the single-object
interpretability literature, and equivalent to object-local when only one object is in
frame -- recovers the mid-depth transition.
"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOCAL = json.load(open("docs/artifacts/single_object_attrs.json"))
WIDE  = json.load(open("docs/artifacts/single_object_attrs_framewide.json"))
ORDER = ["velocity", "direction", "speed"]
COL = {"velocity": "#1F6F8B", "direction": "#D97706", "speed": "#4E9F3D"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": .8, "axes.edgecolor": "#444444",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 200,
})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, model in zip(axes, LOCAL):
    for src, ls, lw in ((LOCAL, "-", 1.6), (WIDE, "--", 1.4)):
        sw = src[model]["depth_sweep"]
        dmax = max(v["depth"] for v in sw["velocity"].values())
        for name in ORDER:
            pts = sorted(sw[name].values(), key=lambda v: v["depth"])
            x = np.array([p["depth"] for p in pts]) / dmax
            y = np.array([p["mean"] for p in pts]); sd = np.array([p["sd"] for p in pts])
            ax.plot(x, y, color=COL[name], lw=lw, ls=ls, zorder=3)
            ax.fill_between(x, y - sd, y + sd, color=COL[name], alpha=.15, lw=0, zorder=2)
    ax.text(0, 1.10, model, transform=ax.transAxes, fontsize=9.5, va="bottom")
    ax.text(0, 1.02, "pre-contact quantities, object A", transform=ax.transAxes,
            fontsize=7.5, color="#777777", va="bottom")
    ax.set_ylim(-0.08, 1.06); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("out-of-fold $R^2$")

h = [plt.Line2D([], [], color=COL[n], lw=1.6, label=n) for n in ORDER]
h += [plt.Line2D([], [], color="#555555", lw=1.6, ls="-",  label="object-local (A.pre)"),
      plt.Line2D([], [], color="#555555", lw=1.4, ls="--", label="frame-wide (all.pre)")]
axes[1].legend(handles=h, loc="lower right", fontsize=7.4, handlelength=2.0,
               labelspacing=.28, borderpad=.2, ncol=2, columnspacing=1.0)
fig.text(.5, .02, "relative depth   (0 = patch embedding, 1 = final block)",
         ha="center", fontsize=9)
fig.subplots_adjust(bottom=.20, top=.86, wspace=.07)
for e in ("png", "pdf"):
    fig.savefig(f"figures/single_object_attrs.{e}", bbox_inches="tight")
print("wrote figures/single_object_attrs.png and .pdf")
