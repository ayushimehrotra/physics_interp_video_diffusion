#!/usr/bin/env python
"""FIGURES: violation of expectation -- does the predictor mind impossible physics?

V-JEPA2's own predictor, no fitting anywhere. It sees ONLY the pre-contact frames,
emits one forecast, and we ask which real ending that forecast matches better.

Writes two independent figures:

  fig_voe_energy              the three endings of the energy set. Each scene carries
                              an inelastic ending (e = 1/k, loses energy), an elastic
                              one (e = 1, conserves it) and an impossible one (e = k,
                              CREATES it, which no simulator produces). Momentum is
                              conserved to 3.9e-16 in all three. Distances are centred
                              per scene, because the matched design exists to remove
                              between-scene variance and the raw scale is dominated by
                              it. THE ELASTIC BAR IN THE MIDDLE IS THE CONTROL: a model
                              merely tracking departure from elastic would put it
                              lowest, since the impossible ending sits exactly k times
                              further from elastic than the inelastic one.

  fig_voe_energy_vs_momentum  the identical measurement on momentum-violating twins.
                              Null at 150 pairs, where 86/150 would reach significance,
                              while the energy effect is 95%. Not a power problem.

    .venv/bin/python scripts/figures/fig_voe_surprise.py
"""
from __future__ import annotations
import json
from math import comb
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENERGY = "docs/artifacts/voe_energy_vjepa2.json"
MOMENTUM = "docs/artifacts/voe_twins_vjepa2_masked.json"
TAG = "balls"          # ball-occupied tokens; ~95% of the frame is identical background
METRIC = "mse"         # the JEPA loss shape; L1 and cosine agree (see the other figure)
C_IMP, C_LEG, C_ANC = "#B23A2E", "#2E6B41", "#5B6B78"
OUT = Path("figures")

RC = {"font.family": "serif", "font.size": 9,
      "axes.linewidth": .8, "axes.edgecolor": "#444444",
      "xtick.direction": "out", "ytick.direction": "out",
      "xtick.major.size": 3, "ytick.major.size": 3,
      "legend.frameon": False, "figure.dpi": 200}


def sign_p(k, n):
    return sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def threshold(n, alpha=.05):
    for k in range(n // 2, n + 1):
        if sign_p(k, n) <= alpha:
            return k
    return n


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    print(f"wrote figures/{stem}.pdf and .png")


# --------------------------------------------------------------- figure 1
VIDEOMAE = "docs/artifacts/voe_energy_videomae_all.json"


def figure_energy(E):
    """Both surprise channels side by side, each on its own scale.

    V-JEPA2's predictor forecasts representations of the unseen aftermath, so it has an
    expectation that a continuation can violate. VideoMAE's pretraining head reconstructs
    masked pixels; asked to rebuild the whole aftermath from the pre-contact frames it
    sits at the trivial baseline, so its comparison has no forecast behind it. The two
    panels are not on a common y-axis for that reason -- the magnitudes differ by three
    orders.
    """
    order = ["legal", "anchor", "impossible"]
    labels = ["loses\nenergy", "conserves\nenergy", "creates\nenergy"]

    D = np.array([[r[TAG][c][METRIC] for c in order] for r in E["rows"]])
    M = json.load(open(VIDEOMAE))
    V = np.array([[r["legal"], r["anchor"], r["impossible"]] for r in M["rows"]])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    panels = [(D, "V-JEPA2-ViT-L", "predictor error", axes[0]),
              (V, "VideoMAE-base", "reconstruction loss", axes[1])]
    out = []
    for A, model, channel, ax in panels:
        A = A - A.mean(axis=1, keepdims=True)
        n = len(A)
        mu = A.mean(axis=0)
        ci = 2.093 * A.std(axis=0, ddof=1) / np.sqrt(n)      # t(.975, df = 19)

        # Fold the decade into the axis LABEL rather than letting matplotlib park a
        # floating "x10^-4" in the top-left corner, where it lands on the subtitle.
        span = float(np.max(np.abs(mu) + ci))
        exp = int(np.floor(np.log10(span))) if span > 0 else 0
        if -2 <= exp <= 2:            # only rescale when the decade is genuinely far out
            exp = 0
        unit = "" if exp == 0 else f"  ($\\times10^{{{exp}}}$)"
        mu_raw = mu.copy()
        mu, ci = mu / 10.0 ** exp, ci / 10.0 ** exp
        xs = np.arange(3)
        ax.axhline(0, color="#CCCCCC", lw=.7, zorder=0)
        ax.bar(xs, mu, width=.6, color=[C_LEG, C_ANC, C_IMP], alpha=.9, zorder=2)
        ax.errorbar(xs, mu, yerr=ci, fmt="none", ecolor="#222222", elinewidth=1.1,
                    capsize=4, capthick=1.1, zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-.62, 2.62)
        ax.text(0, 1.10, model, transform=ax.transAxes, fontsize=9.5, va="bottom")
        ax.text(0, 1.02, channel + unit, transform=ax.transAxes, fontsize=7.5,
                color="#777777", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
        k = int((A[:, 2] > A[:, 0]).sum())
        anc = int(((A[:, 1] < A[:, 0]) & (A[:, 1] < A[:, 2])).sum())
        mono = int(((A[:, 0] < A[:, 1]) & (A[:, 1] < A[:, 2])).sum())
        out.append((model, channel, mu_raw, ci, k, n, anc, mono))
    axes[0].set_ylabel("surprise, centred within scene")
    fig.tight_layout()
    save(fig, "fig_voe_energy")

    for model, channel, mu, ci, k, n, anc, mono in out:
        print(f"  {model} ({channel})")
        print(f"    means   lost {mu[0]:+.6f}   conserved {mu[1]:+.6f}   created {mu[2]:+.6f}")
        print(f"    created > lost {k}/{n}  p = {sign_p(k, n):.4f}   "
              f"conserved least surprising {anc}/{n}   monotone {mono}/{n}")


# --------------------------------------------------------------- figure 2
def figure_contrast(E, M):
    e_sum = E["summary"][TAG]
    mets = ["l1", "mse", "cos"]
    e_n, m_n = e_sum["l1"]["n"], len(M)
    e_frac, m_frac = [], []
    for met in mets:
        e_frac.append(e_sum[met]["n_impossible_worse"] / e_n)
        h = np.array([r[TAG]["honest"][met] for r in M])
        v = np.array([r[TAG]["viol"][met] for r in M])
        m_frac.append(((v < h) if met == "cos" else (v > h)).sum() / m_n)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = np.arange(len(mets))[::-1]
    h_ = .36
    ax.barh(y + h_ / 2, e_frac, height=h_, color=C_IMP, alpha=.9,
            label=f"energy  ($n$ = {e_n} scenes)", zorder=3)
    ax.barh(y - h_ / 2, m_frac, height=h_, color="#3D6E8F", alpha=.9,
            label=f"momentum  ($n$ = {m_n} scenes)", zorder=3)
    for yy, f, nn in list(zip(y + h_ / 2, e_frac, [e_n] * 3)) + \
                     list(zip(y - h_ / 2, m_frac, [m_n] * 3)):
        ax.text(1.115, yy, f"{int(round(f * nn))}/{nn}", va="center", ha="right",
                fontsize=7.5, color="#333333", zorder=7)
    for yy, nn in ((y + h_ / 2, e_n), (y - h_ / 2, m_n)):
        ax.plot([threshold(nn) / nn] * len(y), yy, marker="|", ms=11, ls="none",
                color="#111111", zorder=6, markeredgewidth=1.3)
    ax.axvline(.5, color="#888888", lw=.8, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(["L1", "MSE", "cosine"], fontsize=8.5)
    ax.set_xlim(.35, 1.13)
    ax.set_xlabel("fraction of scenes where the impossible ending is more surprising")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([], [], marker="|", ms=11, ls="none", color="#111111",
            markeredgewidth=1.3, label="$p$ = 0.05 threshold at that $n$")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3,
              fontsize=7.5, handlelength=1.4, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig_voe_energy_vs_momentum")

    for met, ef, mf in zip(mets, e_frac, m_frac):
        ke, km = int(round(ef * e_n)), int(round(mf * m_n))
        print(f"    {met:4s} energy {ke}/{e_n} p={sign_p(ke, e_n):.4f}"
              f"   momentum {km}/{m_n} p={sign_p(km, m_n):.4f}")
    print(f"    p=0.05 thresholds: n={e_n} needs {threshold(e_n)}/{e_n};  "
          f"n={m_n} needs {threshold(m_n)}/{m_n}")


def main() -> None:
    plt.rcParams.update(RC)
    E = json.load(open(ENERGY))
    M = json.load(open(MOMENTUM))["rows"]
    figure_energy(E)
    print()
    figure_contrast(E, M)


if __name__ == "__main__":
    raise SystemExit(main())
