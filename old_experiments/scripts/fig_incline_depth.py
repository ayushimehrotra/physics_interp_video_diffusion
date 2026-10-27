#!/usr/bin/env python
"""FIGURE: friction sweep on `incline_set_v2` -- mu, acceleration and theta across depth.

Three targets in one panel per model, because §35's claim is about the RELATION between
them. theta is scene geometry visible in frame 0. `a` is kinematics. mu is neither: it
is recovered from the two, and over the sampled theta range mu is 0.997 linear in
(a, theta), so a direct-vs-composed tie there is algebra rather than measurement (§35b).

Bands are +- 1 sd over fold seeds; the data, targets and split protocol are identical
across seeds, so the spread is fold-assignment variance alone.

    .venv/bin/python old_experiments/scripts/fig_incline_depth.py
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Each model gets ITS OWN best pool, from the §35 grid. `all.all` is a poor choice
# here -- it costs VideoMAE a factor of four on mu (0.108 vs 0.423 at A.pre).
# Both models prefer a `.pre` pool, and per §35a `.pre` is a MISNOMER on this set:
# there is no collision, so it is simply a fixed crop of the first third of the clip.
MODELS = {
    "docs/artifacts/incline_depth_videomae_Apre.json":
        dict(key="outputs/incline_videomae-base", label="VideoMAE-base", pool="A.pre"),
    "docs/artifacts/incline_depth_vjepa2_allpre.json":
        dict(key="outputs/incline_vjepa2-vitl", label="V-JEPA2-ViT-L", pool="all.pre"),
}
# mu oracles, measured on the 150 ground-truth label files under the SAME protocol
# as the probes -- `cell_oof_regression`, grouped folds, nested alpha, 12 seeds --
# so probe and oracle are scored identically. Scoring an out-of-fold probe against
# an in-sample oracle would stack the comparison against the probe (the in-sample
# values are 0.416 and 0.997; the optimism is small here because n = 150 and the
# bar is a one-feature fit, but the check is the point). §44b's kinematic observer
# is the same discipline.  docs/artifacts/incline_mu_oracles.json
MU_BAR     = 0.4024   # mu from acceleration ALONE.  corr(mu, a) = -0.645
MU_CEILING = 0.9970   # mu from (a, theta). The closed form reaches 0.9997.
STYLE = {"theta":        dict(colour="#7A5AA6", marker="^", label=r"$\theta$  (incline angle)"),
         "acceleration": dict(colour="#1F6F78", marker="o", label=r"$a$  (acceleration)"),
         "mu":           dict(colour="#C4622D", marker="s", label=r"$\mu$  (friction coeff.)")}


def depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="figures/fig_incline_depth")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": .8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)

    printed = []
    for ax, (path, meta) in zip(axes, MODELS.items()):
        if not Path(path).exists():
            ax.text(.5, .5, f"missing\n{Path(path).name}", ha="center", va="center",
                    transform=ax.transAxes, color="#999999", fontsize=8)
            continue
        rep = json.load(open(path))
        r = rep[meta["key"]]
        # the mu ceiling only. The acceleration-only baseline (MU_BAR) and its
        # shaded region are computed and reported but no longer drawn.
        ax.axhline(MU_CEILING, color="#C4622D", lw=1.1, ls=":", zorder=1)
        for tname, st in STYLE.items():
            L = r["targets"][tname]["layers"]
            sites = sorted(L, key=depth)
            x = np.array([depth(s) for s in sites], float); x /= x.max()
            mu = np.array([L[s]["mean"] for s in sites])
            sd = np.array([L[s]["sd"] for s in sites])
            ax.fill_between(x, mu - sd, mu + sd, color=st["colour"], alpha=.16,
                            lw=0, zorder=2)
            ax.plot(x, mu, color=st["colour"], lw=1.9, zorder=3, label=st["label"])
            b = int(np.argmax(mu))
            ax.plot(x[b], mu[b], st["marker"], ms=5.2, color=st["colour"], zorder=5,
                    markeredgecolor="white", markeredgewidth=.7)
            printed.append((meta["label"], meta["pool"], tname, mu[b],
                            sd[b], sites[b], L["embed.out"]["mean"]))
        ax.axhline(0, color="#888888", lw=.8, zorder=1)
        ax.set_xlim(-.03, 1.03)
        ax.text(0, 1.09, meta["label"], transform=ax.transAxes, fontsize=9.5, va="bottom")
        ax.text(0, 1.01, f"pool = {meta['pool']}", transform=ax.transAxes,
                fontsize=7.5, color="#777777", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("out-of-fold $R^2$")
    axes[0].set_ylim(-.08, 1.06)
    # oracle lines go in a legend, not inline: the ceiling sits on top of the theta
    # curve (both ~1.0) and the bar label lands on the mu curve in either panel
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, .055), fontsize=8,
                   handlelength=1.8, labelspacing=.35)
    axes[1].legend([plt.Line2D([], [], color="#C4622D", lw=1.1, ls=":")],
                   [rf"$\mu$ ceiling: from $(a,\theta)$   {MU_CEILING:.3f}"],
                   loc="lower right", bbox_to_anchor=(1.0, .055), fontsize=7.5,
                   handlelength=2.2, labelspacing=.35,
                   title="oracle (out-of-fold)", title_fontsize=7.5)
    fig.supxlabel("relative depth   (0 = patch embedding, 1 = final block)",
                  fontsize=9, y=.04)
    fig.tight_layout(rect=(0, .06, 1, 1))

    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.pdf and .png\n")
    print(f"  mu oracles:  a alone {MU_BAR:.3f}   (a, theta) {MU_CEILING:.3f}")
    for lab, pool, t, m, sd, site, emb in printed:
        flag = ""
        if t == "mu":
            flag = "  <- AT/BELOW the a-alone bar" if m <= MU_BAR + .02 else "  <- clears the bar"
        print(f"  {lab:15s} [{pool:8s}] {t:13s} {m:+.3f} +- {sd:.3f} at {site:<22}"
              f" embed {emb:+.3f}{flag}")


if __name__ == "__main__":
    raise SystemExit(main())
