#!/usr/bin/env python
"""FIGURE: example frames from each of the five two-body regimes.

Rows 1-2 and 3-4 are TWINS -- the same seed with and without the momentum edit -- so the
reader can see directly that the pair is identical up to contact and diverges only after.
Row 5 is the impossible arm, which conserves momentum and creates kinetic energy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CONS = Path("momentum_dataset/conservation_set_v1")
EVOE = Path("momentum_dataset/energy_voe_pilot")
ELASTIC_SEED, INELASTIC_SEED, ENERGY_SEED = 101, 100, 9004
FRAMES = [0, 14, 20, 26, 40]          # contact lands at 20.03

ROWS = [
    (CONS / f"elastic_mixed_honest_seed{ELASTIC_SEED}",      "elastic\nmomentum conserved",   "#2E6F40"),
    (CONS / f"elastic_mixed_violating_seed{ELASTIC_SEED}",   "elastic\nmomentum violated",    "#B03A2E"),
    (CONS / f"inelastic_mixed_honest_seed{INELASTIC_SEED}",  "inelastic\nmomentum conserved", "#2E6F40"),
    (CONS / f"inelastic_mixed_violating_seed{INELASTIC_SEED}","inelastic\nmomentum violated", "#B03A2E"),
    (EVOE / f"super1.5_seed{ENERGY_SEED}",                   "energy created\nmomentum conserved", "#8E44AD"),
]

def bbox(dirs, frames, pad=95, aspect=1.5):
    """Tightest box holding both spheres over `frames`, padded, at a fixed aspect ratio.
    Twins share a box so the pair stays visually comparable."""
    xs, ys = [], []
    for d in dirs:
        pf = {q["frame"]: q for q in json.load(open(d / "labels.json"))["per_frame"]}
        for f in frames:
            for o in ("A", "B"):
                x, y = pf[f][o]["pixel"]; xs.append(x); ys.append(y)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max((max(xs) - min(xs)) / 2 + pad, ((max(ys) - min(ys)) / 2 + pad) * aspect)
    half_h = half_w / aspect
    l, t = max(0, cx - half_w), max(0, cy - half_h)
    r, b = min(720, cx + half_w), min(480, cy + half_h)
    return (int(l), int(t), int(r), int(b))


# twins share a crop; the energy row gets its own
BOXES = [bbox([ROWS[0][0], ROWS[1][0]], FRAMES)] * 2 + \
        [bbox([ROWS[2][0], ROWS[3][0]], FRAMES)] * 2 + \
        [bbox([ROWS[4][0]], FRAMES)]

fig, axes = plt.subplots(len(ROWS), len(FRAMES),
                         figsize=(1.85 * len(FRAMES), 1.35 * len(ROWS)))
for r, (d, label, colour) in enumerate(ROWS):
    L = json.load(open(d / "labels.json"))
    ke = L["collision"]["kinetic_energy_ratio"]
    for c, f in enumerate(FRAMES):
        ax = axes[r, c]
        im = Image.open(d / "frames" / f"frame_{f:04d}.png").crop(BOXES[r]).resize((330, 220), Image.LANCZOS)
        ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(1.4)
        if r == 0:
            ax.set_title(("contact" if f == 20 else f"frame {f}"),
                         fontsize=8.5, color=("#222222" if f == 20 else "#777777"),
                         fontweight=("bold" if f == 20 else "normal"), pad=4)
    axes[r, 0].set_ylabel(label, fontsize=8, color=colour, rotation=0,
                          ha="right", va="center", labelpad=8)
    axes[r, -1].text(1.03, 0.5, f"$KE_{{post}}/KE_{{pre}}$\n{ke:.2f}",
                     transform=axes[r, -1].transAxes, fontsize=7,
                     color="#777777", ha="left", va="center")

fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.13, right=0.90, top=0.94, bottom=0.02)
Path("figures").mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig_dataset_grid.{ext}", dpi=200, bbox_inches="tight")
print("wrote figures/fig_dataset_grid.png and .pdf")
for d, label, _ in ROWS:
    L = json.load(open(d / "labels.json"))
    print(f"  {label.replace(chr(10),' / '):38s} e={L['restitution']:.2f}  "
          f"KE ratio {L['collision']['kinetic_energy_ratio']:.3f}  "
          f"viol={L.get('violates_momentum')}")
