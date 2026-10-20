# DATASETS — what each set can and cannot answer

Every design decision in this project is a decision about **what is visible**.
A probe target that is an exact function of what a camera can see measures
arithmetic on kinematics, not physical knowledge (§12), so the only thing that
makes a set interesting is what it *hides*. This table is that, per set.

Read the "cannot answer" column before designing an experiment. Most of the
project's dead ends were designs that a glance at this column would have killed.

---

## The sets

| set | n | mass/mu | violations | state |
|---|---|---|---|---|
| `pilot` | 1 | visible | none | superseded |
| `probe_set` (v1) | 150 | **visible** | common kick | live, but see the giveaway |
| `probe_set_v2_latent` | 300 | latent | antiparallel | **SUPERSEDED — do not probe (§11)** |
| `probe_set_v3_latent` | 300 | **latent** | antiparallel | live |
| `slide_pilot` | 8 | latent mu | n/a | pilot |
| `slide_set_v1` | 300 | latent mu | n/a | superseded by v2 |
| `slide_set_v2` | 300 | latent mu | n/a | live, but mu ≡ deceleration |
| `incline_pilot` | 6 | latent mu | n/a | pilot |
| `incline_set_v1` | 150 | latent mu | n/a | **VOID (§32b)** — keep as negative control |
| `incline_set_v2` | 150 | **latent mu** | n/a | live — the friction set |
| `collide_pilot` | 6 | latent mu | n/a | pilot |
| `collide_set_v1` | 21/150 | latent mu | n/a | **PAUSED mid-render** |
| `conservation_set_v1` | 300 | visible | matched twins | live — the 2x2 (§42) |
| `energy_voe_pilot` | 60 | visible | **energy only** | live — 20 seeds, k=1.5 (§46d) |
| `energy_voe_v1` | 420 | visible | **energy only** | planned — full k-ladder, ~36 h CPU render |

---

## `conservation_set_v1` — the matched-twin 2x2

150 seeds x 2 clips: an **honest** and a **violating** twin of the same seed, which
share their entire pre-contact segment bit for bit. Crossed with elasticity, PINNED
at `e = 1.00` or `e = 0.15`, and with material (120 seeds `mixed`, 30 `same`).
Design and the leak audit are §42; do not probe it with ungrouped folds, which
manufacture 0.787 discrimination out of nothing.

**What it can answer that no other set here can: ENERGY.** Elasticity was pinned so
the energy axis would have an exact positive control, and it is exact — the elastic
honest cell reads `KE_post/KE_pre` = 1.000 on all 75 clips, the inelastic honest
cell 0.098-0.992 on all 75. So "is this clip losing energy" is a 75/75 binary.
The violating twins then **decouple** energy from elasticity: their post velocities
are overwritten, so 90/150 lose energy, 57/150 *create* it, and `e` predicts which
at only 0.636.

**Cannot answer — and read §44 before designing anything on it.** Whether the model
has a notion of *energy*. On the honest half the label is exactly `e < 1`, and `e`
is a mass-free ratio of two visible speeds that separates the classes at **AUC
1.000**. §12's rule in full: the ceiling is an oracle, not `shuf*`. Separating the
two needs a set where the mass ratio varies widely at fixed `e`, so that
`KE_lost/KE_pre = (mu/M)(1-e^2)` moves while the mass-free scalar does not. That set
does not exist yet.

---

## `energy_voe_v1` — the cell `conservation_set_v1` is missing

**`energy_voe_pilot` is RENDERED, CAPTURED AND PROBED** (20 seeds x 3 clips = 60,
k = 1.5 only). The full 420-clip version below is still planned. All six validator
checks pass on the pilot's real render — see §46d.

60 seeds x 7 clips = 420. Every clip **conserves momentum to 1e-15**; the only thing
that varies is the energy. Built by `apply_energy_edit`, which rescales BOTH
deflections by `lambda = (1+e_target)/(1+e_sim)` — a common rescale, so the split
`b/a = m_A/m_B` that conservation fixes is untouched and only the sum that restitution
fixes moves. Realised `e` lands on target to 1e-15 and `dKE` matches
`1/2 mu (e^2 - 1) u_n^2` to 1e-14.

| cell | per seed | `e` | legal? |
|---|---|---|---|
| anchor | 1 | 1.00 | yes — energy conserved exactly, a verified bit-exact no-op |
| legal | 3 | 1/1.25, 1/1.5, 1/2 | yes — energy lost, ordinary inelastic physics |
| **impossible** | 3 | 1.25, 1.5, 2.0 | **no — kinetic energy created from nothing** |

**Why it exists.** Cross what is conserved in `conservation_set_v1` and one box is
empty: zero clips conserve momentum while creating energy. Its 57 energy-creating
clips all break momentum too, because the violation edit targets momentum and moves
energy only as a side effect — so surprise there cannot be attributed to the invented
energy. And the honest half does not fill the gap, because **losing energy is legal**:
an inelastic collision violates nothing.

**The control that makes it work.** Each seed carries `e = k` and `e = 1/k`. Both are
the same distance from elastic in log space, one separates the balls k times too fast
and the other k times too slow, so both are equally unusual as kinematics —
**exactly one is impossible**. Asymmetric surprise between them is an energy prior;
symmetric surprise is novelty detection.

**Can answer.** Whether a model treats physically impossible energy creation
differently from an equally-unusual legal alternative — conservation of energy as a
LAW, which nothing else in this project tests. Read by the model's own surprise
(VideoMAE reconstruction loss, V-JEPA2 predictor error) against the matched twin.

**CANNOT answer, and this is not a defect.** Anything by probing. `e` is mass-free and
visible, so a probe separates impossible from legal at AUC 1.000 by reading separation
speed and demonstrates nothing (§44, §44c-bis). The violation *has* to be visible or
the model could not react to it at all. **Do not probe this set.** The validator prints
that AUC every run so the point cannot be missed.

---

## `probe_set` — v1, the `legacy` arm

150 scenes. Two spheres, **no gravity, no friction, no rotation** — they float,
collide once, and fly apart. Contact is pinned at frame 20 ± 0.017. Velocities
are read at frame 18 (pre) and 22 (post).

**What is visible.** Everything. Density takes two values rendered as
unmistakable materials (6.0 and 1.5) and radius varies freely, so
`mass = density · (4/3)πr³` is fully determined by the pixels. Measured:
`mass_A` decodes at R² 0.934, `radius_A` at 0.975 — and `radius_A` reads
**0.952–0.965 at `embed.out`**, before a single transformer block runs.

**The violation mechanism, and its giveaway.** Post-contact velocities are
overwritten with ones that do not conserve momentum. This version added the
*same* velocity kick to both balls, which breaks the sum but also breaks the
geometry: a frictionless impulse is along the contact normal, so `dv_A` and
`dv_B` are exactly opposed, and a common kick tilts them.
`cos(dv_A, dv_B)` runs −1.000000 on every conserving scene and −0.997 to +0.994
on violating ones, so **thresholding that one visible scalar separates the
classes at balanced accuracy 1.000** (§34f). 75 conserving / 75 violating.

**Can answer.** Per-object velocity and mass decodability; the composition test
for products (§26); whether Δp is *constructed* rather than present in the
pixels (§34); object addressability and per-object steering (§16, §18).

**Cannot answer.** Anything about conservation — a probe "detecting violations"
here is doing geometry with no notion of mass or momentum anywhere. Any target
built on `p = m·v` or `m_A/m_B = |dv_B|/|dv_A|`: both are identities, so the
probe answers a definition (§12).

**One stratum worth knowing about.** With a wide mass range and a narrow speed
range, momentum and mass are nearly the same variable (the original pilot had
`corr(|p|, mass_A) = 0.97`). A `matched_momentum` stratum fixes `|p_A|` to a
band independent of mass — it is the only subset on which a momentum probe is
distinguishable from a mass probe. Select it with `--where`.

---

## `probe_set_v2_latent` — SUPERSEDED

300 scenes, `latent_mass`, `violate_frac 0.25`. Superseded by v3 over the
antiparallel leak (§11). **Captures exist but must not be probed.**

---

## `probe_set_v3_latent` — the latent-mass set

300 scenes, `--arm latent_mass --violate-frac 0.25` → **225 conserving / 75
violating**. Same floating-sphere geometry as v1.

**What is hidden.** Both spheres render **identically** whatever their
densities — one neutral albedo, and **radii are equal within a scene**, so the
mass ratio is a pure density ratio with size removed as a cue. The redesign
worked: `mass_A` fell from R² 0.934 to **0.28**.

**The violation mechanism.** Impulse magnitudes are scaled unequally along the
*same* contact normal (`dv_A' = (1−δ)dv_A`, `dv_B' = (1+δ)dv_B`, δ capped at
0.9), so `cos(dv_A', dv_B') = −1` exactly while the momentum sum breaks.
Injected `|dp|/|p_pre|` ∈ (0.35, 0.75).

**Can answer.** Whether the mass ratio is recoverable from the collision alone
— §13's answer is no: 0.49–0.58 against a 0.516 kinematic oracle, i.e. exactly
what a linear readout of the model's own noisy velocity estimates gives with no
notion of mass anywhere. Also the error-structure tests in
`scripts/lawful_errors.py`, for which the theorem below is an *asset*.

**Cannot answer — and this is a theorem, not a measurement.** Any antiparallel
pair of deflections is consistent with conservation under *some* mass ratio,
namely `m_A/m_B = |dv_B|/|dv_A|`. Combine that with invisible masses and **no
observer, however capable, can beat chance on `momentum_conserved`** (§14).
Measured: 0.500 / 0.599 balanced accuracy, both *below* what selection noise
alone produces. Report it as a property of the design, never as evidence about
the models.

The same theorem voids any statistic that requires the model to *register* the
violation — e.g. the `beta` and `pull` arms of `lawful_errors.py`. Those need a
set where Δp is inferable. Only the `rho` arm, which uses conserving scenes
alone, is informative here.

**Two caveats.**

* `log_mass_ratio` must be fit with `--where momentum_conserved=true`. In a
  violating scene the aftermath is edited, so the label no longer equals the
  kinematic ratio (median discrepancy ~10×) and those 75 labels behave as noise.
* `identifiability.min_delta_displacement_px` in `labels.json` records how far
  the *less*-deflected ball departs from its undeflected path. The ratio needs
  both deflections, so the smaller one sets the precision. Median is 13.5 px but
  **33 of 300 scenes are under 2 px and 7 under 0.5 px** — at or below tracker
  noise, so the ratio is unrecoverable there for information-theoretic reasons.
  `select_scenes()` only does equality, so `--where` cannot filter on this
  without a change.

---

## `slide_set_v1` / `slide_set_v2` — flat-floor friction

300 scenes each, `--arm latent_friction`, no violations. One cube decelerating
on a friction floor. v2 adds a **banded floor** and a **side view** and
supersedes v1. Both pass all six validation checks.

**The fatal property, found late (§32).** `g = 1.0` on a level floor, so
`a = mu·g` means **mu and the deceleration are literally the same number**.
Measured on v2: `corr(mu, decel) = 0.99976`, and predicting mu from the
deceleration with *no fitting at all* gives R² = 0.995.

**So mu decoding at 0.93 / 0.97 here is an acceleration result.** §30 and §30a
must not be cited as friction results. Use `incline_set_v2` instead.

**Mask-leak floor: the worst in the project.** `cube_size` is 99% recoverable
from the pooling mask alone, and mu from the mask-only floor reads 0.796. Every
object-local number from this set needs `scripts/mask_only_floor.py` beside it.

**Still answers one thing cleanly.** Whether mu binds to the *surface* or the
*object* — §31 patches floor tokens into cube tokens and back, on disjoint
pools where the passive channel computes to exactly 0.000, and finds mu is
object-bound (floor→cube 0.011 vs cube→floor 0.097, 300–800× per token).

---

## `incline_set_v1` — VOID, keep as a negative control

150 scenes, all seven validation checks green, and **unusable**. With the start
pinned and the cube released from rest, position in *any single frame* is the
acceleration — so mu reads off one frame's cube mask at **R² 0.877 (centroid) /
0.988 (+ mask area)**, against the 0.459 the set exists to impose (§32b).

The frame-0 check passed because frame 0 was the one frame that was clean. The
fix for the *previous* leak (§32a, pinning `s0`) is what moved the leak from
frame 0 to every later frame. Third distinct route into the identity trap, and
the third to arrive through the rendering rather than the physics.

**Do not capture on it.** It is a good worked example of a leak no validator
caught.

---

## `incline_set_v2` — the friction set

150 scenes, `--arm latent_friction --script generate_incline_scene.py`. The
§32b fix: **start jitter + launch speed**, which takes the single-frame route to
0.077 while leaving the 3-frame route at 0.997. All eight validation checks pass;
mask leak for mu falls from 0.988 to **0.066**. Zero rejections over 80k draws.

**Why it exists.** On an incline `a = g(sinθ − mu·cosθ)`, so mu and the
acceleration finally come apart: `corr(mu, a) = −0.645`, which caps a
readout of acceleration alone at R² 0.416.

**Results (§35).** mu 0.423 (VideoMAE) / 0.779 (V-JEPA2); acceleration
0.827 / 0.929; theta **1.000 at `embed.out`** in both — theta is in the pixels.
Direct vs composed +0.002 / −0.003.

**The known limitation.** mu is **0.997 linear in (a, θ)** over the sampled
theta range, so the quotient that §32 was built to test never materialises.
Error propagation from decoded acceleration alone predicts R²(mu) of
0.584 / 0.829 against the observed 0.423 / 0.779 — **mu reads below what
acceleration alone affords, so there is no independent mu representation here.**
Widening theta to 15–70° would fix it, and needs a shape that does not topple
(a sphere with rolling friction, or a low puck).

---

## `collide_set_v1` — two cubes, PAUSED at 21/150

Generator and validator done and audited; render stopped in favour of §32.
One shared mu, two equal cubes, A (rear, fast) catching B (front, slow) at
frame 14.5.

**Three things one cube cannot ask.** Is mu represented once or per object;
does a mu readout fit pre-collision still work post; and the composition test
four times over the same scalar.

**Traps already found and fixed — worth not rediscovering.** Head-on geometry
does not work (friction and restitution eat the speed, leaving no observable
post-collision deceleration); same-direction catch-up fixes it and makes the
relative motion frictionless. The collision pitches the cubes 9–33° with tilt
scaling in the impulse, so the silhouette becomes a second channel carrying mu
— **angular damping** is the fix, not `localInertiaDiagonal`. Contact must land
in a temporal cell's *interior*, not on its boundary. Segmentation files are
`segmentation_%04d.png`, and a mismatch does not degrade gracefully.

---

## The arm that is implemented and has never been rendered

`ARMS = ("legacy", "latent_mass", "visible_mass")`. The **`visible_mass`** arm
ramps **albedo** log-linearly with density and keeps radii equal within a scene,
so mass is knowable from appearance *without* size as a cue and *without* the
dynamics.

Run with the v3 antiparallel violation it is the one cell of the design space
never tested, and the only one where the conservation label is decidable but
**only** by combining an appearance-derived mass with observed deflections:

|  | mass visible | mass latent |
|---|---|---|
| **common kick** | geometric giveaway, AUC 1.000 (was v1) | same giveaway |
| **antiparallel** | **never rendered** | theorem-null (§14), is v3 |

```bash
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 300 --output-dir momentum_dataset/probe_set_v4_visible \
    --seed-offset 0 --violate-frac 0.25 --workers 3 --arm visible_mass
```

Two cautions. `CLAUDE.md`'s "do not render the `visible_mass` arm" was written
about the *mass-ratio* target, where §13's oracle bounds it for free; it does
not apply to the conservation question. And the tracker erodes the bright
sphere's blob, so tracker radius carries a small density-dependent bias in this
arm — check it before probing.

---

## Facts that hold across every set

* **Contact is pinned at frame 20 ± 0.017 in every existing set**, so
  time-to-contact has no variance anywhere on disk. Any TTC experiment needs a
  new render.
* v1 and v3 have `gravity = (0,0,0)` and all friction terms zero — no material
  property is present in them at all.
* Spheres do not rotate in v1/v3; cubes are angularly damped in the slide,
  incline and collide sets. No set carries angular momentum.
* Pools are `{A, B, all} × {pre, post, all}` — nine. A summary that filters to
  `all.*` will report nulls that are not there: `log_mass_ratio` reads ~0 in
  every frame-wide pool and 0.75–0.81 in object-local ones on v1.
* Every object-local number needs its mask-only floor
  (`scripts/mask_only_floor.py`) printed beside it, or it is not reportable.
