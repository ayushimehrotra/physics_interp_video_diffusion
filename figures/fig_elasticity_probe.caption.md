# Figure: elastic vs inelastic, binary probe

**Files:** `figures/fig_elasticity_probe.pdf` / `.png`
**Script:** `scripts/fig_elasticity_probe.py` (no recomputation — reads existing artifacts)
**Data:** `docs/artifacts/energy_probes.json`, `energy_probes_maskfloor.json`
**Target:** `energy_loss_honest` on `conservation_set_v1`, honest half — 75 clips at
e = 1.00 (KE ratio 1.000 on all 75) vs 75 at e = 0.15 (KE ratio 0.098–0.992 on all 75).

## Draft caption

> **Both encoders read elasticity, and neither beats a camera.** Linear probes classify
> elastic from inelastic collisions from the `all.post` pool, grouped 5-fold by seed,
> alpha nested inside training, 200-draw permutation nulls. AUC is near chance at the
> patch embedding (0.522 / 0.487) and is built through the transformer, peaking at
> **0.945** (VideoMAE, block 8 of 12) and **0.983** (V-JEPA2, block 20 of 24). Dotted
> horizontals are each model's max-over-grid null (95th pct). The solid line at the top is
> the **mass-free ceiling**: on honest clips the label *is* `e < 1`, and
> `e = |(v_A − v_B)_post · n| / |(v_A − v_B)_pre · n|` is a ratio of two visible speeds
> that separates the classes at **AUC 1.000** with no notion of mass. n = 150.

## Numbers

Plotted pool, `all.post`:

| model | best | site | `embed.out` | null max p95 | ceiling |
|---|---|---|---|---|---|
| VideoMAE-base | **0.945** | block.8 of 12 | 0.522 | 0.664 | **1.000** |
| V-JEPA2-ViT-L | **0.983** | block.20 of 24 | 0.487 | 0.668 | **1.000** |

Grid best over all nine pools, for the text (p_corrected = 0.000 for both):

| model | grid best | cell | bal. acc |
|---|---|---|---|
| VideoMAE-base | 0.946 | block.10 / `all.pre` | 0.873 |
| V-JEPA2-ViT-L | 0.983 | block.20 / `all.post` | 0.920 |

## Pool structure, for the text rather than the figure (§44a)

1. **Frame-wide beats object-local everywhere**, by 0.05–0.15. Elasticity is a property
   of the *pair* (`|v_rel_post| / |v_rel_pre|`); a single-object pool cannot hold a
   relation between two objects.
2. **`.all` is the worst phase in every row.** Averaging tokens across the collision mixes
   the two sides of the very contrast being probed. Pooling over more frames is not
   pooling over more information.
3. **`.pre` ties `.post`.** Both encoders attend bidirectionally, so a pre-contact token
   has already attended to the aftermath. **No cell in this grid tests prediction** — the
   `all.pre` win is the model reading the future through attention.

The `all.*` mask floors are 0.500 **by construction, not by measurement**: those pooling
rows are all-ones and identical in every scene, so the plotted pool needs no floor marker.
Object-local floors run 0.43–0.61 and none survives the grid correction (best
p_corrected = 0.140). Full per-pool numbers are in `energy_probes.json`.

**Why `all.post` and not `all.pre`.** They tie (0.945 vs 0.946 on VideoMAE; 0.983 vs 0.982
on V-JEPA2), and `all.post` avoids inviting a prediction reading. Note that neither pool
tests prediction: both encoders attend bidirectionally, so `all.pre` tokens have already
seen the aftermath. Measured on twins that are bit-identical through frame 20, encoding
the full clip makes their *pre-contact* tokens diverge to cos **0.919**, and the same
clip's pre-contact tokens shift by cos **0.72** when the model is allowed to see the
future.

## What the figure must not be read as saying

The depth curve looks like a capability result and is not one. §44b fitted a kinematic
observer through the identical pipeline — same grouped folds, same ridge, same nested
alpha — on a quadratic in five visible mass-free scalars. Across **8 probe-vs-observer
comparisons not one 95% CI excludes zero in the probe's favour**, and six of eight put the
probe significantly *below* an observer that also sees the true kinetic energies. The
probe also carries a winner's curse worth roughly +0.05 (its cell is the argmax over 117
or 225 cells, scored on the same data) and still does not clear the camera.

**Claim to make:** both encoders carry, linearly and well, *whether a collision looked
elastic* — a mass-free ratio of two visible speeds, constructed through depth rather than
read off the pixels. **Claim not to make:** that either represents kinetic energy, which
cannot be formed without mass.
