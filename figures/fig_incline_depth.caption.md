# Figure: friction sweep on incline_set_v2

**Files:** `figures/fig_incline_depth.pdf` / `.png`
**Script:** `old_experiments/scripts/fig_incline_depth.py`
**Data:** `docs/artifacts/incline_depth_videomae_Apre.json`,
`incline_depth_vjepa2_allpre.json`, `incline_mu_oracles.json`
**Sweep:** `old_experiments/scripts/probe_incline_depth.py`, 12 fold seeds per layer, n = 150.

## Draft caption

> **The ingredients of friction decode; the coefficient does not follow.** Out-of-fold
> R² against relative depth for the incline angle θ, the down-slope acceleration *a*,
> and the friction coefficient μ, on `incline_set_v2` (n = 150). Bands are ±1 sd over 12
> fold seeds. Each encoder is shown at its own best pool from the §35 grid — `A.pre` for
> VideoMAE, `all.pre` for V-JEPA2 — because pool choice alone moves VideoMAE's μ by a
> factor of four (0.423 at `A.pre` vs 0.108 at `all.all`). Horizontal lines are μ oracles
> fitted through the **identical** protocol on ground-truth labels: the dashed **bar** is
> μ from acceleration alone (0.402), the dotted **ceiling** is μ from (a, θ) jointly
> (0.997). Shading marks the region where a μ probe has learned nothing that decoded
> acceleration does not already supply.

## Numbers

| | pool | θ | *a* | **μ** |
|---|---|---|---|---|
| VideoMAE-base | `A.pre` | +0.991 ± 0.001 | +0.791 ± 0.013 | **+0.512 ± 0.037** |
| V-JEPA2-ViT-L | `all.pre` | +1.000 ± 0.000 | +0.905 ± 0.006 | **+0.765 ± 0.013** |

`embed.out` (patch embedding, pre-transformer): θ **+0.957 / +1.000**, *a* +0.456 / +0.524,
μ +0.029 / −0.051.

## μ oracles, fitted through the probes' own protocol

`cell_oof_regression`, grouped folds, nested alpha, 12 seeds — so probe and oracle are
scored identically. In-sample values are given for comparison; the optimism is small
because n = 150 and the bar is a one-feature fit.

| oracle | in-sample | out-of-fold |
|---|---|---|
| μ from *a* alone | 0.4160 | **0.4024 ± 0.0073** |
| μ from θ alone | 0.0007 | **−0.0135 ± 0.0053** |
| μ from (*a*, θ) | 0.9972 | **0.9970 ± 0.0001** |
| μ from `tan θ − a/(g cos θ)` | 0.9997 | 0.9997 |

Target correlations on this set: **corr(μ, a) = −0.645**, corr(*a*, θ) = +0.746,
corr(μ, θ) = +0.026.

## What to claim

* **θ is not a finding.** It reads +0.957 / +1.000 *at the patch embedding* — static scene
  geometry visible in frame 0, not something the transformer constructs. It belongs in the
  figure as a reference line, not as a result.
* **`a` is constructed**, rising from ~0.46/0.52 at the embedding to 0.79/0.91.
* **μ clears the acceleration bar on both encoders** — 0.512 and 0.765 against 0.402 — so
  neither is merely re-reading acceleration.
* **But both fall far short of 0.997.** The two ingredients decode at 0.79–0.91 and
  0.99–1.00, and μ is recoverable from them at 0.997, yet the direct readout reaches only
  0.51 / 0.77. The parts are present near ceiling; the combination is not formed.

## Caveats

* **Not comparable to §35's numbers.** §35 used `fit_probe`: a single 60/40 split, in-sample
  alpha selection, no seed averaging (μ 0.423 / 0.779). This uses grouped 5-fold OOF with
  nested alpha over 12 seeds. Same data, different protocol.
* **`.pre` is a misnomer on this set (§35a).** There is no collision — a cube is released and
  slides. `.pre` is a fixed crop of roughly the first third of the clip, aligned to no
  physical event; `labels.json` carries a nominal `collision.actual_frame = 20.0` purely as a
  compatibility shim.
* **μ is 0.997 linear in (a, θ)** over the sampled θ range, so a direct-vs-composed comparison
  here is an algebraic identity rather than a measurement (§35b). The claim that survives is
  §35c's cross-geometry transfer, which is empirical.
