# Figure: norm-route AUC across depth

**File:** `figures/fig_norm_route_layers.pdf` (vector, for LaTeX) / `.png`
**Script:** `scripts/fig_norm_route_layers.py` — regenerates from the two JSONs below.
**Data:** `docs/artifacts/norm_route_v1.json`, `docs/artifacts/norm_route_v4.json`

## Draft caption

> **Momentum-violation detection via the norm route, across encoder depth.** Conserving
> scenes sit at Δp = (0,0) and violating scenes on a ring around it, so no linear
> readout separates them — a linear classifier handed the *exact* Δp vector scores
> AUC 0.559. The norm route instead regresses the Δp **vector** with ridge and applies
> |·| *outside* the probe (§34a). Left: `probe_set` (v1), where §34's positive was
> measured. Right: `probe_set_v4_legacy_fixed`, the leak-free regeneration, on which the
> fixed mechanism forces Δp exactly along the contact normal (|cos| 0.636 → 1.000) and
> removes the direction degree of freedom. Solid lines are the `all.pre` pool, which wins
> on both encoders; dashed and dotted are `all.post` and `all.all`. Dotted horizontals are
> each model's grid-max permutation null (95th percentile, 200 draws). Depth is plotted as
> a fraction so a 12-block and a 24-block encoder are comparable. n = 150 scenes.

## Numbers quoted

| set | model | `all.pre` AUC | best site | null p95 | p | Δp R² |
|---|---|---|---|---|---|---|
| v1 | VideoMAE-base | **0.765** | block.8 | 0.684 | 0.000 | +0.414 |
| v1 | V-JEPA2-ViT-L | **0.707** | block.11 | 0.704 | 0.030 | +0.165 |
| v4 | VideoMAE-base | 0.631 | block.9 | 0.658 | 0.130 | +0.255 |
| v4 | V-JEPA2-ViT-L | 0.664 | block.11 | 0.688 | 0.110 | +0.139 |

## TWO CAVEATS BEFORE QUOTING THE p-VALUES

1. **These nulls are LESS CONSERVATIVE than §34's.** `norm_route.py` takes its grid max
   over *sites within one pool* (13 or 25 cells). §34 took it over the *whole* cell grid
   (all pools × all sites). Same observed AUCs to three decimals — 0.765 and 0.707 — but
   §34 reports p = 0.006 / 0.106 where this reports 0.000 / 0.030. **Quote §34's p-values
   as the headline**; this figure's nulls are for the shape of the curve, not for
   significance. The V-JEPA2 v1 cell in particular is a near-miss under the full-grid
   correction, not the significant result the per-pool p suggests.

2. **Δp R² = +0.414 here vs §34's 0.517** at the same pool. That is another point inside
   §37's unresolved reproduction gap (0.517 under one fitting protocol, 0.205–0.261 under
   another). Settle §37 before either number goes in the paper.

## What the figure shows

* **v1, VideoMAE:** flat-to-noisy through the first third, rising from ~0.4 depth to a
  broad plateau at 0.72–0.77. Consistent with Δp being *constructed*, not read off pixels.
* **v1, V-JEPA2:** peaks sharply at half depth (block 11 of 24) and falls away — the same
  mid-depth location Joseph et al. call the Physics Emergence Zone.
* **v4, both:** no depth structure survives. Isolated spikes at single layers, all inside
  or barely above the null band. This is §38g, and it is the open question the paper's
  momentum section turns on.
