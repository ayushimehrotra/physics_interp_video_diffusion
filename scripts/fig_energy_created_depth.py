#!/usr/bin/env python
"""FIGURE: probing energy CREATED vs energy CONSERVED, across depth.

`energy_voe_pilot`, `impossible` (e = k, kinetic energy created out of nothing) against
`anchor` (e = 1, energy conserved exactly, a bit-exact no-op). Momentum is held to
3.9e-16 in BOTH arms, which is what makes this the clean cell: on
`conservation_set_v1` every energy-creating clip also breaks momentum, so nothing there
separates "created energy" from "broke momentum".

ERROR BARS ARE FOLD-ASSIGNMENT VARIANCE. At n = 40 the grouped-fold split and the
nested alpha choice move the number by ~0.03, so a single-seed curve shows wiggles that
are resampling noise. Each layer is re-fit under 12 seeds; the band is +- 1 sd and the
line is the mean. Nothing about the data, the pairing or the label changes across seeds.

    .venv/bin/python scripts/fig_energy_created_depth.py
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "outputs_evoe_videomae-base": dict(label="VideoMAE-base", colour="#1F6F78", marker="o"),
    "outputs_evoe_vjepa2-vitl":   dict(label="V-JEPA2-ViT-L", colour="#C4622D", marker="s"),
}


def depth(site: str) -> int:
    return 0 if site.startswith("embed") else int(re.search(r"block\.(\d+)", site).group(1)) + 1


def main() -> None:
    rep = json.load(open("docs/artifacts/energy_voe_depth.json"))
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
    ax.text(.985, .506, "chance", fontsize=7.5, color="#777777", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(which="grid"))
    ax.set_xlim(-.03, 1.03); ax.set_ylim(.44, .85)
    ax.set_xlabel("relative depth   (0 = patch embedding, 1 = final block)")
    ax.set_ylabel("energy created vs conserved, AUC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5, handlelength=1.8, labelspacing=.35)
    fig.tight_layout()

    out = Path("figures"); out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_energy_created_depth.{ext}", bbox_inches="tight")
    print("wrote figures/fig_energy_created_depth.pdf and .png\n")
    for outdir, st in MODELS.items():
        r = rep[outdir]
        sites = sorted(r["layers"], key=depth)
        mu = {s: r["layers"][s]["mean"] for s in sites}
        b = max(mu, key=mu.get)
        L = r["layers"][b]
        print(f"  {st['label']:15s} peak {L['mean']:.3f} +- {L['sd']:.3f} at {b:<22}"
              f" range over seeds [{L['min']:.3f}, {L['max']:.3f}]")
        print(f"                  n={r['n']}  ceiling {r['mass_free_ceiling']:.3f}  "
              f"mean sd across layers {np.mean([r['layers'][s]['sd'] for s in sites]):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
