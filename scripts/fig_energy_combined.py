#!/usr/bin/env python
"""FIGURE: the two energy panels side by side -- the probe sweep and the predictor's
surprise -- so the section carries one float instead of two."""
import json, re, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
from math import comb

DEPTH = json.load(open("docs/artifacts/energy_voe_depth.json"))
VJ    = json.load(open("docs/artifacts/voe_energy_vjepa2.json"))
VM    = json.load(open("docs/artifacts/voe_energy_videomae_all.json"))
STY = {"outputs_evoe_videomae-base": ("VideoMAE-base", "#1F6F8B", "o"),
       "outputs_evoe_vjepa2-vitl":   ("V-JEPA2-ViT-L", "#D97706", "s")}
C_LEG, C_ANC, C_IMP = "#2E6F40", "#5A6B7C", "#B03A2E"
depth = lambda s: 0 if s.startswith("embed") else int(re.search(r"block\.(\d+)", s).group(1)) + 1

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": .8, "axes.edgecolor": "#444444",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 200,
})
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9),
                         gridspec_kw={"width_ratios": [1.25, 1, 1]})

# ---- (a) probe across depth -------------------------------------------------
ax = axes[0]
for key, (lab, col, mk) in STY.items():
    rec = DEPTH[key]["layers"]
    pts = sorted(rec.items(), key=lambda kv: depth(kv[0]))
    dmax = depth(pts[-1][0])
    x = np.array([depth(k) for k, _ in pts]) / dmax
    m = np.array([v["mean"] for _, v in pts]); s = np.array([v["sd"] for _, v in pts])
    ax.plot(x, m, color=col, lw=1.8, label=lab, zorder=3)
    ax.fill_between(x, m - s, m + s, color=col, alpha=.18, lw=0, zorder=2)
    b = int(np.argmax(m)); ax.plot(x[b], m[b], mk, ms=5, color=col, zorder=5)
ax.axhline(1.0, color="#888888", lw=.9, ls=":", zorder=1)
ax.text(.98, 1.0, "observer given the\ncontact geometry", ha="right", va="top",
        fontsize=6.6, color="#777777", transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_ylabel("energy created vs conserved, AUC")
ax.set_xlabel("relative depth"); ax.set_ylim(.42, 1.07)
ax.legend(loc="lower right", fontsize=7.6, handlelength=1.6, labelspacing=.28)
ax.text(0, 1.04, "(a) linear probe", transform=ax.transAxes, fontsize=9, va="bottom")
ax.spines[["top", "right"]].set_visible(False)

# ---- (b,c) the predictor's own surprise -------------------------------------
order = ["legal", "anchor", "impossible"]
labels = ["loses\nenergy", "conserves\nenergy", "creates\nenergy"]
D = np.array([[r["balls"][c]["mse"] for c in order] for r in VJ["rows"]])
V = np.array([[r["legal"], r["anchor"], r["impossible"]] for r in VM["rows"]])
for ax, A, model, chan, tag in ((axes[1], D, "V-JEPA2-ViT-L", "predictor error", "(b)"),
                                (axes[2], V, "VideoMAE-base", "reconstruction loss", "(c)")):
    A = A - A.mean(axis=1, keepdims=True)
    mu = A.mean(axis=0); ci = 2.093 * A.std(axis=0, ddof=1) / np.sqrt(len(A))
    span = float(np.max(np.abs(mu) + ci)); exp = int(np.floor(np.log10(span))) if span > 0 else 0
    if -2 <= exp <= 2: exp = 0
    unit = "" if exp == 0 else f"  ($\\times10^{{{exp}}}$)"
    mu, ci = mu / 10.0**exp, ci / 10.0**exp
    xs = np.arange(3)
    ax.axhline(0, color="#CCCCCC", lw=.7, zorder=0)
    ax.bar(xs, mu, width=.6, color=[C_LEG, C_ANC, C_IMP], alpha=.9, zorder=2)
    ax.errorbar(xs, mu, yerr=ci, fmt="none", ecolor="#222222", elinewidth=1.0,
                capsize=3, capthick=1.0, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_xlim(-.62, 2.62); ax.spines[["top", "right"]].set_visible(False)
    ax.text(0, 1.04, f"{tag} {model}", transform=ax.transAxes, fontsize=9, va="bottom")
    ax.set_ylabel(chan + unit, fontsize=7.8)
fig.tight_layout(w_pad=1.6)
for e in ("png", "pdf"):
    fig.savefig(f"figures/energy_combined.{e}", bbox_inches="tight")
print("wrote figures/energy_combined.png and .pdf")
