#!/usr/bin/env python
"""FIGURE: can the encoder tell an ELASTIC collision from an inelastic one?

`energy_loss_honest` on `conservation_set_v1`'s honest half: 75 clips at e = 1.00
(kinetic energy conserved exactly, KE ratio 1.000 on all 75) against 75 at e = 0.15.
A 75/75 binary with no threshold to argue about.

ERROR BARS ARE FOLD-ASSIGNMENT VARIANCE. The seed sets `grouped_folds` and therefore the
alpha chosen per fold, so a single-seed depth curve carries resampling wiggle that reads
as structure. Each layer is re-fit under several seeds; the line is the mean and the band
is +- 1 sd. The data, the pairing and the label are identical across seeds.

Reads `docs/artifacts/elasticity_depth.json` (written by `probe_elasticity_depth.py`).

    .venv/bin/python scripts/figures/fig_elasticity_probe.py
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "outputs/cons_videomae-base": dict(label="VideoMAE-base", colour="#1F6F78", marker="o"),
    "outputs/cons_vjepa2-vitl":   dict(label="V-JEPA2-ViT-L", colour="#C4622D", marker="s"),
}


def depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/artifacts/elasticity_depth.json")
    ap.add_argument("-o", "--out", default="figures/fig_elasticity_probe")
    args = ap.parse_args()
    rep = json.load(open(args.data))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.linewidth": .8, "axes.edgecolor": "#444444",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(5.1, 3.4))

    for outdir, st in MODELS.items():
        r = rep[outdir]
        sites = sorted(r["layers"], key=depth)
        x = np.array([depth(s) for s in sites], float); x /= x.max()
        mu = np.array([r["layers"][s]["mean"] for s in sites])
        sd = np.array([r["layers"][s]["sd"] for s in sites])
        ax.fill_between(x, mu - sd, mu + sd, color=st["colour"], alpha=.18, lw=0, zorder=2)
        b = int(np.argmax(mu))
        ax.plot(x, mu, color=st["colour"], lw=2.0, zorder=3,
                label=f"{st['label']}   peak {mu[b]:.3f}")
        ax.plot(x[b], mu[b], st["marker"], ms=5.5, color=st["colour"], zorder=5,
                markeredgecolor="white", markeredgewidth=.7)

    ax.axhline(.5, color="#888888", lw=.8, zorder=1)
    # right side: the legend now occupies the upper left
    ax.text(.985, .506, "chance", fontsize=7.5, color="#777777", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(which="grid"))
    ax.set_xlim(-.03, 1.03); ax.set_ylim(.44, 1.02)
    ax.set_xlabel("relative depth   (0 = patch embedding, 1 = final block)")
    ax.set_ylabel("elastic vs inelastic, AUC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, handlelength=1.8, labelspacing=.35)
    fig.tight_layout()

    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.pdf and .png\n")
    for outdir, st in MODELS.items():
        r = rep[outdir]
        sites = sorted(r["layers"], key=depth)
        L = r["layers"][r["best_site"]]
        print(f"  {st['label']:15s} peak {L['mean']:.3f} +- {L['sd']:.3f} at "
              f"{r['best_site']:<22} seed range [{L['min']:.3f}, {L['max']:.3f}]")
        print(f"                  n={r['n']} ({r['n_pos']} inelastic)  "
              f"embed.out {r['layers']['embed.out']['mean']:.3f}  "
              f"ceiling {r['mass_free_ceiling']:.3f}  "
              f"mean sd {np.mean([r['layers'][s]['sd'] for s in sites]):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
