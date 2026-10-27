# Kinetic energy

Split out of `CLAUDE.md` on 2026-10-27, verbatim apart from paths updated to the new layout. Section numbers are unchanged; the index in
`CLAUDE.md` says which file each § lives in. §44's finding (the models read *elasticity*,
not energy) is the bridge to [`elasticity.md`](elasticity.md).

---

### 44. ENERGY — the models read ELASTICITY, which is mass-free, and that accounts for all of it

`scripts/kinetic_energy/probe_energy.py`, run 2026-08-25 on `conservation_set_v1`, both encoders,
the full 9-cell pool grid the captures already store: region {all, A, B} x phase
{pre, post, all}, every site. 200-draw permutation nulls throughout.

**Why this set could answer it at all.** §42c pinned elasticity as a factor
precisely so the energy axis would have a positive control. It delivers one that is
exact rather than approximate: the elastic honest cell reads `KE_post/KE_pre` =
**1.000 on all 75 clips**, the inelastic honest cell **0.098-0.992 on all 75**. So
"is this clip losing energy" is a 75/75 binary with no threshold to argue about.

#### 44a. THE GRID — it decodes, and it is not the mask

`energy_loss_honest`, grouped 5-fold by seed, alpha nested inside train, AUC:

| pool | VideoMAE | site | V-JEPA2 | site | mask floor VM / VJ |
|---|---|---|---|---|---|
| `all.pre` | **0.946** | b10 | 0.982 | b20 | 0.500 / 0.500 |
| `all.post` | 0.945 | b8 | **0.983** | b20 | 0.500 / 0.500 |
| `all.all` | 0.884 | b11 | 0.953 | b19 | 0.500 / 0.500 |
| `A.pre` | 0.864 | b11 | 0.939 | b15 | 0.540 / 0.526 |
| `A.post` | 0.847 | b8 | 0.954 | b17 | 0.609 / 0.575 |
| `A.all` | 0.796 | b9 | 0.864 | b18 | 0.578 / 0.488 |
| `B.pre` | 0.937 | b9 | 0.948 | b17 | 0.582 / 0.507 |
| `B.post` | 0.892 | b9 | 0.948 | b16 | 0.428 / 0.520 |
| `B.all` | 0.823 | b9 | 0.943 | b19 | 0.545 / 0.536 |

Balanced accuracy at the grid best **0.873 / 0.920**. Max-over-grid null 95th
percentile 0.664 / 0.668, `p_corrected` = 0.000. `embed.out` reads 0.621 / 0.595,
so roughly 0.33-0.39 of this is built by the transformer and not by the patch
embedding.

**Three structural facts in that table, none of them about energy.**

* **Frame-wide beats object-local everywhere**, by 0.05-0.15. Consistent with §39e:
  a frame-wide pool superposes the two balls, and elasticity is a property of the
  PAIR (`|v_rel_post| / |v_rel_pre|`), not of either ball. A single-object pool
  cannot hold a relation between two objects, and the numbers say so.
* **`.all` is the WORST phase in every row.** Averaging tokens across the collision
  mixes the two sides of the very contrast being probed and washes it out. Pooling
  over more frames is not pooling over more information.
* **`.pre` ties `.post`.** Both encoders attend bidirectionally, so a pre-contact
  token has already attended to the aftermath — §4's warning, reproduced. **No cell
  in this grid tests prediction.** The `all.pre` win at 0.946 is the model reading
  the future through attention.

The `all.*` mask floors are 0.500 **by construction, not by measurement**: those
pooling rows are all-ones and identical in every scene, so a probe on them alone
scores every clip the same. Object-local floors run 0.43-0.61 and none survives the
grid correction (best `p_corrected` = 0.140).

#### 44b. AND THE WHOLE THING IS §12 AGAIN — the label is a mass-free scalar

On honest clips the label is exactly `e < 1`, and

    e = |(v_A - v_B)_post . n| / |(v_A - v_B)_pre . n|

is **mass-free**: any observer who can see the two balls separate computes it with
no notion of mass, energy or momentum anywhere. Measured on this set it separates
the honest classes at **AUC 1.000**. A probe at 0.946 has demonstrated it can see
relative speed. It has demonstrated nothing about energy.

So the test is not "does the probe beat chance" but "does it beat a camera". A
KINEMATIC OBSERVER was fitted through the identical pipeline — same grouped folds,
same ridge, same nested alpha — on a full quadratic in five visible mass-free
scalars (`e`, `|v_rel|` pre and post, sum of speeds pre and post). Paired bootstrap
over seeds, 2000 draws:

| target | probe (best of grid) | mass-free observer | probe − observer, 95% CI | + true KE, probe − it |
|---|---|---|---|---|
| `energy_loss_honest` VM | 0.946 | **1.000** | −0.054 [−0.090, −0.024] | −0.054 [−0.090, −0.024] |
| `energy_loss_honest` VJ | 0.983 | **1.000** | −0.017 [−0.036, −0.005] | −0.017 [−0.036, −0.005] |
| `energy_loss_all` VM | 0.830 | 0.864 | −0.034 [−0.077, +0.007] | −0.100 [−0.148, −0.051] |
| `energy_loss_all` VJ | 0.819 | 0.864 | −0.044 [−0.086, −0.002] | −0.110 [−0.157, −0.063] |
| `energy_loss_violating` VM | 0.795 | 0.779 | +0.016 [−0.057, +0.084] | −0.102 [−0.176, −0.031] |
| `energy_loss_violating` VJ | 0.762 | 0.779 | −0.017 [−0.103, +0.067] | −0.135 [−0.215, −0.057] |
| `energy_created` VM | 0.798 | 0.758 | +0.040 [−0.053, +0.126] | −0.092 [−0.180, −0.012] |
| `energy_created` VJ | 0.793 | 0.758 | +0.034 [−0.060, +0.130] | −0.098 [−0.173, −0.023] |

**Eight comparisons, not one interval excluding zero in the probe's favour.** Four
exclude zero against it. Meanwhile the probe is significantly BELOW an observer
that also sees the true kinetic energies — the quantity that needs the masses — in
six of eight.

**The bias runs the other way.** The probe's cell is the argmax over 117 or 225
cells, chosen on the very data the comparison is scored on; the observer gets no
search at all. The probe was handed a winner's curse worth roughly +0.05 and still
did not clear a camera.

#### 44c. THE DECOUPLED CELLS, and why they were needed

`energy_loss_honest` cannot tell an energy detector from a separation-speed
detector, because there the two are the same function. The violating twins break
that: their post-contact velocities are overwritten, so 90 of 150 lose energy while
`e` predicts which ones at only 0.636 and 57 of 150 **create** energy out of
nothing. Three readings on that decoupled half:

* **within-violating** — 0.795 / 0.762 against a 0.779 mass-free observer. A tie.
* **energy created** (impossible physics, plainly visible) — 0.798 / 0.793 against
  0.758. The largest margin anywhere, and its CI still contains zero.
* **transfer, honest → violating**, grouped so a scene's own twin is never in
  train. Direct (binary label) **0.688 / 0.685**; graded (fit the continuous honest
  KE ratio, then rank by predicted energy) **0.769 / 0.760**, against a mass-free
  transfer ceiling of **0.775**. The graded route also reaches **R2 +0.303 / +0.304**
  on the violating clips' actual KE ratio — a real readout, landing exactly where
  `|v_A| + |v_B|` alone lands.

#### 44c-bis. THE MASS RATIO IS VISIBLE — and it makes the MAGNITUDE mass-free too

A fair objection to 44b: masses vary enormously on this set (ratio 0.059-24.2, both
radius and density moving), so surely *how much* energy is lost needs them, even if
the sign does not. It does not, and the reason is an identity, not a measurement.

Momentum conservation says `m_A dv_A = -m_B dv_B`, so on any conserving clip

    m_A / m_B = |dv_B| / |dv_A|

and both deflections are visible. **Verified on all 150 honest clips at
|implied - true| / true, max 1.02e-15.** Substituting that ratio for the true masses
(the overall scale cancels out of a ratio of energies) reconstructs the KE ratio from
**visible velocities alone, R2 = 1.000000, max absolute error 4.44e-16.**

So there is no mass-requiring target anywhere on the honest half. The sign is mass-free
because the reduced mass is positive; the magnitude is mass-free because conservation
hands you the ratio. Measured, at `e` pinned to 0.15 (n = 75, KE ratio 0.098-0.992):

| | VideoMAE | V-JEPA2 |
|---|---|---|
| probe, best of 117 / 225 cells | +0.511 | +0.673 |
| observer K1 — kinematics only, **no size, no mass** | **+0.785** | **+0.785** |
| observer K2 — K1 + the two radii | +0.749 | +0.749 |
| observer K3 — K2 + true masses and KE_pre | +0.610 | +0.610 |
| probe − K1, 95% CI | −0.281 [−0.477, −0.115] | −0.119 [−0.278, +0.020] |

The probe loses outright on VideoMAE and ties at best on V-JEPA2, against an observer
holding **no mass information at all**. The descending K1 > K2 > K3 ordering is a
capacity artifact, not an information one — the quadratic expansion runs 54 / 77 / 119
columns against n = 75, so each rung overfits harder. Do not read it as "mass hurts".

**Mass is not the bottleneck, and this rules that out directly.** On these same
captures `log mass_A` reads **+0.977 / +0.990** and `log mass_B` **+0.982 / +0.989**
(VideoMAE / V-JEPA2, grouped 5-fold, best cell — both at `A.all` / `B.all`, as the
legacy arm's visible radii and materials would predict). The model has the masses, has
the velocities, and still does not produce their energy combination — the §26 pattern
again, now for `KE = ½mv²`.

> **THE GENERAL POINT, which outlives this dataset.** For a **fully observed**
> two-body collision, no observer ever needs the masses to compute the energy ratio.
> Conservation makes the mass ratio a function of what the camera already saw. So no
> experiment of this shape — two balls, both visible throughout, momentum conserved —
> can separate an energy concept from kinematics, however wide the mass range is made.
> Breaking that requires the collision to be **incompletely observed**: one ball
> leaving frame after contact, an occluder over one aftermath, an immovable wall whose
> `dv` cannot be read, or a third body. That, not a wider mass range, is the next set.

#### 44d. THE CLAIM

Both encoders carry, linearly and well, **whether a collision looked elastic**.
That is a mass-free ratio of two visible speeds, and every energy number in this
section is accounted for by it. Nothing here requires an internal notion of kinetic
energy, which cannot be formed without mass — and §26/§34e already say the models
do not store the products that would take.

This is the third law to come out this way, and the pattern from §4 holds
unchanged: **quantities the camera shows read at 0.75-0.98; the relations between
them read at the level of whatever visible scalar happens to correlate.**

**What would move it — and 44c-bis rules out the obvious answer.** Widening the mass
range does nothing: conservation makes the mass ratio visible, so the energy ratio is
already a mass-free function of the deflections at R2 = 1.000000. The next set has to
hide part of the aftermath, not vary more of it — one ball exiting frame after contact,
an occluder, a wall with no readable recoil, or a third body. Only then does computing
the energy require holding a mass, and only then is any of 44b a real test rather than
a ceiling check.

#### 44e. Two protocol bugs this section had to fix first, both worth remembering

* **The `std + 1e-6` guard is not a zero.** For a feature that is CONSTANT across
  scenes, `X - mean` is float noise near 1e-19; dividing by 1e-6 lifts it to 1e-13
  and it then ranks the scenes. The mask floor's `all.*` rows — identical in every
  scene, so they MUST read 0.500 — came out at **0.399**. Dead features are now
  zeroed explicitly.
* **A pooled out-of-fold score carries each fold's intercept.** A ridge prediction
  is `Kte @ w + mean(y_train)`, and pooling raw scores across folds makes that
  intercept a comparison BETWEEN folds: the fold whose TEST half holds more
  positives has fewer left in train, so its whole block shifts down. Pooled AUC then
  reads **systematically below chance** for any cell with no signal — the same
  `all.*` rows read 0.364 on one target. The intercept IS the training mean, so
  subtracting it uses no test information and fixes it exactly. **Any pooled-OOF AUC
  elsewhere in this project has this bias**; it is small where signal is large and
  it is the whole number where signal is zero.

#### 44f. Files

* `scripts/kinetic_energy/probe_energy.py` — the grid, `--mask-floor`, `--transfer`,
  `--vs-kinematics`, `--fixed-e`. No GPU and no torch: numpy throughout, plus Pillow for
  `--mask-floor`. Under 20 min for the full grid on 8 cores, measured with the
  other three modes running alongside it.
* `docs/artifacts/energy_probes.json`, `energy_probes_maskfloor.json`,
  `energy_probes_transfer.json`, `energy_vs_kinematics.json`, `energy_fixed_e.json`.
* Positive controls on the same captures: `kinetic_energy_pre` R2 **+0.745 /
  +0.690**, `ke_ratio` R2 **+0.348 / +0.382**, grouped 5-fold, best cell.

---

### 45. ENERGY, ASKED WHERE THE IDENTITY DOES NOT REACH — the friction sets

§44c-bis closed the collision route with a theorem: for a fully observed two-body
collision, momentum conservation makes the mass ratio visible, so the energy ratio
is a mass-free function of the deflections at R2 = 1.000000 and no probe there can
separate an energy concept from kinematics. The fix it named was an **incompletely
observed** interaction. `slide_set_v2` already is one, and was already captured.

**Why the slide set escapes the identity.** One cube, one friction floor, no second
body. The momentum goes into the floor and the floor's recoil is unobservable, so
nothing hands the observer a mass. And the factors are clean:

| | |
|---|---|
| `mass = size^3` **exactly** | density pinned at 1.0 — mass visible, but only through a cubed length |
| `corr(log size, log speed) = -0.053` | the two factors are independent by design |
| log-KE variance from speed alone 0.207, from size alone 0.749 | neither factor is a single-scalar shortcut |

`scripts/kinetic_energy/energy_composition.py`, run 2026-08-25, both encoders.

#### 45a. TEST A — CROSS-SCENARIO TRANSFER. The design that works.

Fit the readout on `slide_set_v2` (cubes, side camera, energy to friction heat),
apply it UNCHANGED to `conservation_set_v1` (spheres, different camera, energy to
deformation). The kinematic shortcut differs completely between the two —
deceleration x distance versus `(1-e^2)` — so a single direction reading KE in both
is reading energy rather than a scenario-specific trick. Cell and alpha are chosen
**on the source set only**, so the target contributes nothing to selection and there
is no winner's curse to correct. Scored by SPEARMAN: the domains put KE on different
scales and raw R2 would punish an affine mismatch that says nothing.

| | VideoMAE | V-JEPA2 |
|---|---|---|
| control — apparent size (segmentation pixels) | +0.631 | +0.585 |
| control — speed | +0.244 | +0.428 |
| **kinetic energy** | **+0.022** (p = 0.725) | **+0.275** (p < 0.005) |
| composed: size and speed transferred SEPARATELY, then multiplied | +0.291 | +0.324 |
| direct − composed | **−0.269** | **−0.050** |

**Both controls cross the gap on both models**, so a KE failure here is a negative
and not a void test — that gate was written in before the numbers were seen.

* **VideoMAE's energy direction is entirely scenario-bound.** Chance transfer, while
  reading KE at R2 0.971 *within* the slide set. Whatever it learned is slide-specific.
* **V-JEPA2's transfers**, at 0.275 with p < 0.005. This is the only thing in §44 or
  §45 that looks like a scenario-invariant energy quantity.
* **Neither beats its own factors.** Transferring apparent size and speed separately
  and multiplying wins on both models. The factors travel; the product adds nothing.

`B.*` pools are excluded from the transfer: object B is the FLOOR in the slide set
and the second sphere in the collision set. The pool name matches, the content does not.

#### 45b. TEST B — COMPOSITION. At ceiling, and it fails in a way worth recording.

Does a linear readout give KE better than decoding mass and speed separately and
multiplying? §26's criterion, in LINEAR space — in log space `log KE = log(1/2) +
log m + 2 log v` is a linear combination, ridge is linear in Y, and direct-vs-composed
becomes an algebraic identity rather than a measurement (§34b, which already cost this
project the mass-ratio result once).

The gaps are tiny, because everything saturates — mass decodes at **0.994** and speed
at **0.95-0.99**, so both routes land near 0.98 with no room to resolve. §26's momentum
version had teeth precisely because its factors sat at 0.73-0.97 while the product FELL
to 0.43-0.66.

| model | target read at | gap | 95% CI | says |
|---|---|---|---|---|
| VideoMAE | first pre frame | −0.010 | [−0.015, −0.006] | composed |
| VideoMAE | mean over pre window | **+0.004** | **[+0.000, +0.008]** | **held as a unit** |
| V-JEPA2 | first pre frame | −0.029 | [−0.041, −0.019] | composed |
| V-JEPA2 | mean over pre window | −0.001 | [−0.008, +0.006] | inconclusive |

> **THE TRAP, AND A CONFIDENCE INTERVAL DOES NOT CATCH IT.** VideoMAE's two rows are
> both significant and they point OPPOSITE WAYS — same captures, same cells, differing
> only in where inside the pre window the target is read. With n = 300 the bootstrap is
> tight enough to certify a 0.004 gap, so **significance at the ceiling certifies
> precision, not robustness.** Without a guard this run hands you a quotable "ENERGY IS
> HELD AS A UNIT" off four thousandths.
>
> The guard now in the script is a **cross-convention sign-agreement check** that
> overrides the interval and prints INCONCLUSIVE when the sign flips. Gating on "the CI
> excludes zero" was tried first and was not enough. **Any best-vs-best comparison in
> this project that runs near ceiling needs this check, not just a null.**

Taking the four rows together: three point to composed-or-null, the fourth points the
other way and is the one that flips. **No evidence energy is held as a unit**, and the
set cannot be pushed further — mass here is `size^3` on a plainly visible cube.

#### 45c. WHAT THIS ADDS TO §44

§44 said the collision sets cannot ask the question. §45 asks it where they cannot and
gets the same answer from a different direction: **the factors are represented, the
energy is assembled at readout.** VideoMAE's energy direction does not survive a change
of scenario at all; V-JEPA2's does, and is still fully explained by its factors.

**What is left to try, in order.** (1) A set where mass is NOT a visible length —
the `latent_mass` arm crossed with friction, so the floor hides the recoil AND the
appearance hides the mass. Nothing on disk does both. (2) Violation of expectation on
the 57 energy-CREATING clips in `conservation_set_v1`, measured as the model's own
reconstruction or predictor error against each clip's matched honest twin, controlling
against the energy-LOSING violators matched on `|Delta-p|`. That is the only design
discussed here that tests conservation as a **law** rather than energy as a quantity,
and it needs a GPU forward pass but no new render.

#### 45d. Files

* `scripts/kinetic_energy/energy_composition.py` — `--test A`, `--test B`. numpy only, no GPU.
* `docs/artifacts/energy_composition_A.json`, `energy_composition_B.json`.

---

### 46. `energy_voe_v1` — the missing cell, generator written and verified, RENDER NOT RUN

§45c named the one design that tests conservation of energy as a **law** rather than
energy as a quantity, and it needs a cell that does not exist on disk. Cross what is
actually conserved in `conservation_set_v1`'s 300 clips:

| | KE created | KE lost | KE kept |
|---|---|---|---|
| **momentum conserved** (< 1e-9) | **0** | 73 | 77 |
| **momentum broken** | 57 | 90 | 3 |

**Zero clips conserve momentum while creating energy.** All 57 energy-creating clips
break momentum too, because the violation edit targets momentum and moves energy only
as a side effect, so surprise on them cannot be attributed to the invented energy. And
the honest half does not fill the gap: **losing kinetic energy is legal physics** — an
inelastic collision converts it to heat and violates nothing. §44 measured whether the
model NOTICES energy changed (0.946 / 0.983, and mass-free). Whether it holds
conservation as a LAW needs clips that are *impossible*, which means energy CREATED,
which no physics engine will produce.

#### 46a. THE EDIT — and it is the exact complement of §42's

`apply_energy_edit` in `generate_scene.py`. Write the deflections as magnitudes along
the contact normal: `a + b = (1+e)|u_n|` is set by restitution, `b/a = m_A/m_B` is set
by conservation. `apply_violation_ratio` holds the SUM and moves the SPLIT — momentum
breaks. This holds the SPLIT and moves the SUM:

    dv_A' = lambda * dv_A,   dv_B' = lambda * dv_B,   lambda = (1 + e_target)/(1 + e_sim)

A COMMON rescale, so `m_A dv_A' + m_B dv_B' = lambda * 0 = 0` — momentum conserved
identically, not approximately. Verified on all 75 elastic honest collisions of
`conservation_set_v1`:

| | |
|---|---|
| realised `e` vs target | max error **1.8e-15** (scored against the INDEPENDENT measured contact normal) |
| momentum relative error | max **8.6e-16** |
| `dKE` vs `1/2 mu (e^2-1) u_n^2` | max **1.9e-14** |
| `e_target = 1.0` | a bit-exact no-op, KE ratio 1.000–1.000 |

> **ONE BUG WORTH KEEPING.** The first version took the normal from
> `geo["contact_normal_analytic"]`. Realised restitution still looked exact at 1e-16 —
> because the same wrong normal sat on both sides of that check — while `dKE` came out
> up to **3.6% off** the closed form. The collision is frictionless, so `dv_A` lies
> along the true normal by construction; deriving `n` from `dv_A` removes the free
> parameter and both agree at 1e-14. **A self-consistent check on a derived quantity
> certifies nothing; score it against something the edit did not touch.**

#### 46b. THE SET, and the control it is shaped around

60 seeds x 7 clips = **420**. Per seed: one anchor at `e = 1.0`, and for each
`k` in {1.25, 1.5, 2.0} a pair at `e = k` (impossible) and `e = 1/k` (legal).

> Both members of a pair sit the same distance from elastic in log space — one
> separates the balls `k` times too fast, the other `k` times too slow — so both are
> equally unusual as kinematics and **exactly one is impossible**. A model more
> surprised by `k` than by `1/k` holds an energy prior. A model equally surprised by
> both is detecting novelty. That asymmetry is the whole measurement.

Seeds start at 9000 so the set never shares a scene with `conservation_set_v1`.
Material policy `mixed`, matching that set's main arm.

#### 46c. DO NOT PROBE THIS SET

`e` is mass-free and visible, so a probe separates impossible from legal at AUC
**1.000** by reading separation speed and demonstrates nothing — §44 and §44c-bis, still
true. For violation of expectation that is not a flaw but a requirement: the violation
must be visible or the model cannot react to it at all. The question is whether the
model treats *impossible* differently from merely *unusual*, and only the `k` / `1/k`
pairing isolates that. `validate_energy_voe_set.py` prints the AUC on every run so the
distinction cannot be lost.

Read it with the model's own surprise — VideoMAE's decoder reconstruction loss,
V-JEPA2's predictor error — against each clip's matched twin. Needs a GPU forward pass,
no new capture code.

#### 46d. RENDERED AND PROBED — the pilot, and a null that was a no-op

**`energy_voe_pilot`: 20 seeds x 3 clips = 60, rendered 2026-08-26.** CPU only (the
box's GPU was held by another tenant, and Cycles OOMs in the 936 MB left), 274 min at
~548 s/clip, zero failures. `validate_energy_voe_set.py` passes all six checks on the
real render: momentum relative error max **3.94e-16**, realised `e` on target to
**6.7e-16**, `dKE` matching the closed form to **7.6e-15**, twins sharing pre-contact
positions to **0.0**, and the impossible cell creating energy in **20/20** clips
against 0/20 in each of the other two.

Both encoders captured on CPU (VideoMAE 98 s; V-JEPA2-ViT-L ~4 h, since the other
tenant's 44.8 GB left too little GPU to even load ViT-L's weights) and probed. Impossible
vs legal, n = 40, grouped by seed:

| | VideoMAE | V-JEPA2 |
|---|---|---|
| grid best | **0.830** `block.8/all.pre` | **0.825** `block.11/all.pre` |
| max-over-grid null p95 | 0.743 | 0.735 |
| p corrected | **0.005** | **0.000** |
| **mass-free ceiling (`e` alone)** | **1.000** | **1.000** |
| paired within seed, best cell | 1.000 (20/20) | 1.000 (20/20) |
| paired p corrected | **0.060** | **0.060** |

Per-pool the two models agree closely and `all.pre` wins on both — which is §4 again:
attention is bidirectional, so a pre-contact pool has already seen the aftermath.

**The result is the one §46c predicted, and it is not evidence of anything about
energy.** The probe separates the classes well below what thresholding `e` does, so it
has read separation speed and nothing more. The paired 20/20 looks striking and does
NOT survive the grid correction: with 117 cells and 20 pairs the null's max reaches
1.000 on its own.

> **THE NULL WAS A NO-OP AND EVERY p CAME OUT 1.000.** `probe_energy.permuted_labels`
> permutes whole GROUPS. On this set every seed contributes exactly one `impossible`
> and one `legal` clip in the same order, so the label pattern is identical in every
> group and permuting groups reproduces the observed vector **bit for bit** — measured
> at 20/20 draws identical. The symptom was diagnostic once seen: every p-value read
> exactly 1.000 and the max-over-grid null p95 sat exactly on the observed AUC.
>
> The fix is the **exact paired null** — flip which twin carries the label,
> independently per pair with probability 1/2, the same construction §42 already uses.
> `probe_energy_voe.paired_permuted_labels`. **A balanced paired design silently
> disables group-level permutation; check that the null actually moves before reading
> a p-value off it.** The §44/§45 nulls are unaffected — their groups carry differing
> label patterns, and their nulls were measured moving (means 0.49-0.51, p-values
> spanning 0.000-1.000).

#### 46e. STATE, and what is NOT done

**Done: generator, planner, runner, validator, a 60-clip pilot render, VideoMAE capture
and probe.** The environment was rebuilt to get there — python3.11 from apt (which also
revived the project's own `.venv`), a kubric venv on **bpy 4.5.4** because the pinned
4.3.0 has been withdrawn from PyPI, and kubric installed `--no-deps` from the local tree.

**NOT DONE: the full 420-clip set.** Rendering is CPU-only here at ~9 min per clip with
two workers, so the full set is a ~36 h job; the pilot is 20 seeds at k = 1.5 only, with
no k-ladder. Both encoders ARE captured and probed on the pilot.

**NOT DONE: the VoE measurement itself.** Everything above is a PROBE, which §46c says
cannot answer the question. The set exists for the model's own surprise on `e = k`
versus `e = 1/k`, which needs VideoMAE's decoder head or V-JEPA2's predictor and has
not been run.

* `datasets/plan_energy_voe_set.py` — physics-only feasibility, writes `plan.json`
* `datasets/run_energy_voe_set.py` — resumable render from the manifest
* `datasets/validate_energy_voe_set.py` — six checks, run it before probing

#### 46f. THE VoE MEASUREMENT, RUN — and the momentum contrast that explains it

Run 2026-08-26. `scripts/kinetic_energy/voe_energy_twins.py`, V-JEPA2 predictor on the 60-clip
pilot. **Nothing is fit**: no labels, no readout, no probe. §46c said a probe cannot
answer the question and named this measurement instead; it is now run.

**THE CONTEXT COMES FROM THE ANCHOR**, the one design change from `voe_twins.py`.
All three clips of a seed share a bit-identical pre-contact segment, so any of them
yields the same expectation — but taking it from the `e = 1` anchor means the
prediction belongs to NEITHER twin and no asymmetry can enter through the choice of
context. Identity control: predictions from the anchor's, the impossible twin's and
the legal twin's own pre-frames agree at cos **1.00000** (min, not mean).

impossible (`e = k`) vs legal (`e = 1/k`), 20 twin pairs, exact sign test:

| metric | all 3584 target tokens | ball-occupied only |
|---|---|---|
| l1 | 16/20, p = 0.0059 | **19/20, p = 0.0000** |
| mse | 16/20, p = 0.0059 | **19/20, p = 0.0000** |
| cos | 15/20, p = 0.0207 | **18/20, p = 0.0002** |

Ball tokens are **4.7%** of the target, so a frame-wide average dilutes by roughly
the reciprocal of that — which is exactly the gap between the columns, and
`voe_twins_masked.py`'s premise reproduced on a second set. Substrate: the predictor
beats persistence **20/20** on mse in both token sets. Room: `cos(anchor, impossible)`
0.7716 / 0.7209 against `cos(anchor, legal)` 0.7924 / 0.7778, both < 1.

#### 46f-i. THE ANCHOR ARM KILLS THE CONFOUND THIS SET WAS OPEN TO

§46b justified the pairing as "the same distance from elastic in log space", and that
is exactly true of `e`: `log(e_imp)` = +0.40547, `log(e_leg)` = -0.40547. **It is NOT
true of what the camera sees.** The edit rescales by `lambda = (1+e)/2`, so

    |lambda_imp - 1| / |lambda_leg - 1| = ((k-1)/2) / ((k-1)/2k) = k

exactly, with no scene dependence. Measured: `||v_post - v_post(anchor)||` is 0.2882
against 0.1922, a ratio of **1.500 in 20/20 seeds**. A detector of raw departure from
elastic separates the classes at AUC **0.743** pooled and **20/20 paired** — which is
§46d's probe result reproduced by a confound alone.

**The anchor, scored as a future, refutes it.** A monotone-in-departure account
requires the ANCHOR to be the least surprising of the three. It is least surprising in
**0/20** seeds (mse and cos, ball tokens; 3/20 on l1). The observed ordering is
`legal < anchor < impossible` in 17/20, with `d(legal) < d(anchor)` in **19/20**.

> **ADD A THIRD ARM TO ANY TWIN DESIGN THAT CAN CARRY ONE.** The anchor cost one extra
> encode per seed and it is the only reason this result is attributable. The `k` / `1/k`
> pairing alone would not have been — its symmetry holds in the coordinate the DESIGN
> chose and fails by exactly a factor `k` in the coordinate the PIXELS live in.

#### 46f-ii. WHAT IT SHOWS, AND WHAT IT DOES NOT

The three futures are collinear in velocity space at `lambda in {0.833, 1.0, 1.25}`, so
`d(legal) < d(anchor)` places the projection of the expectation BELOW `lambda = 0.917`
— an expected restitution under roughly **e = 0.83**. The model expects collisions to
LOSE energy, which is right for nearly every collision in natural video, and energy
creation is surprising because it runs the other way.

**SAFE: V-JEPA2 holds a DISSIPATION PRIOR**, strong enough to separate impossible from
equally-unusual-but-legal at p = 0.0000 on 20 pairs, with every control passing.

**NOT SAFE: that it holds conservation as a LAW.** A clip *more* dissipative than the
prior expects is also legal and should also be surprising under this account.
Separating a dissipation prior from an impossibility detector needs an arm at `e` BELOW
the expectation — the k-ladder the full 420-clip set specifies and the pilot lacks.

#### 46f-iii. MOMENTUM, THE SAME MEASUREMENT — NULL AT 150 PAIRS

Re-analysed from `docs/artifacts/voe_twins_vjepa2_masked.json` under identical
statistics; no new forward passes. `conservation_set_v1`, honest (legal) vs violating
(impossible), V-JEPA2 predictor, ball-occupied tokens:

| metric | momentum, n = 150 | energy, n = 20 |
|---|---|---|
| l1 | 80/150, p = 0.231 | **19/20, p = 0.0000** |
| mse | 84/150, p = 0.083 | **19/20, p = 0.0000** |
| cos | 71/150, p = 0.769 | **18/20, p = 0.0002** |
| mse effect size | +0.0121 | **+0.1346** |

Null in every cell: `elastic_mixed` 36/60 (p 0.078), `elastic_same` 6/15 (p 0.849, sign
NEGATIVE), `inelastic_mixed` 33/60 (p 0.259), `inelastic_same` 9/15 (p 0.304).

**The null is not a power problem.** n = 150 reaches significance at 86/150 = **57.3%**
pair agreement; n = 20 needs 75%. Energy shows **95%**. An energy-sized effect would be
unmissable at 150 pairs.

**And the trend test kills the residual lean.** Honest and violating trajectories
diverge MONOTONICALLY after contact (4.9% of pixels differ at frame 22, 17.1% at frame
48), so a model that notices the violation must find it MORE surprising the further past
contact it looks. All three metrics run BACKWARDS: `fraction_trend` -0.0036 / -0.0045 /
-0.0024, slope t = -1.66 / -1.85 / -1.89. The 84/150 is noise with a lucky sign.

Substrate passes on both sets (150/150 and 147/150), so both nulls are interpretable —
there IS an expectation to violate in each.

> **THE CONTRAST IS THE RESULT.** Same model, same predictor, same protocol, same
> masking, same simulator. Surprised by impossible ENERGY; NOT surprised by impossible
> MOMENTUM, with 7.5x the pairs. **§26 predicts exactly this**: `e` is a mass-free ratio
> of two visible speeds, so a prior over it is learnable from video; a momentum violation
> is visible only by relating the deflection SPLIT to the MASSES, which is the
> mass-weighted product the representation does not form. This is the **behavioural
> confirmation of the format claim**, and the first result here that tests physics as an
> EXPECTATION rather than as a decodable quantity.

#### 46f-iv. §34's INSTRUMENT DOES NOT PORT — the two targets are mirrors

`scripts/kinetic_energy/probe_energy_voe_graded.py`. §34a's fix is "regress the vector, put the
nonlinearity OUTSIDE". Applied here it **loses**:

| | binary label | graded (regress dKE, rank outside) |
|---|---|---|
| VideoMAE | **0.830** | 0.765 |
| V-JEPA2 | **0.825** | 0.780 |

Because the two conservation targets are structurally opposite. Hand a linear classifier
the exact, noiseless quantity:

| target | given | AUC |
|---|---|---|
| `momentum_conserved` | exact Delta-p vector | 0.559 |
| | exact \|Delta-p\| | **1.000** |
| `energy_conserved` (\|dKE\| < eps) | exact signed dKE | 0.738 |
| | exact \|dKE\| | **1.000** |
| `energy_created` (dKE > 0) | exact signed dKE | **1.000** |
| | exact \|dKE\| | 0.634 |

`momentum_conserved` is a threshold on a NORM and needs one applied outside.
`energy_created` is ONE-SIDED and LINEAR, and the norm DESTROYS it. `energy_conserved`
— the two-sided version — does have §34's structure and would have hit §34's trap, but
**it is not a law**: losing KE violates nothing, so that target is §44's elasticity
result under another name.

#### 46f-v. Files and state

* `scripts/kinetic_energy/voe_energy_twins.py` — the measurement. **~178 s/seed on CPU** (17.1 s per
  20-frame encode, 10.0 s predictor, 31.8 s per 48-frame encode); 59 min for 20 seeds.
* `scripts/kinetic_energy/probe_energy_voe_graded.py` — §34's instrument, kept for the record that it
  loses. One bug worth remembering: `cell_oof_regression` takes a COLUMN target;
  `RidgeSplit.predict` documents `Y [n_train, P]` and a 1-D `y` broadcasts to
  `[n_train, n_train]` inside the eigen-solve rather than erroring on shape.
* `docs/artifacts/voe_energy_vjepa2.json`, `energy_voe_graded.json`.
* **NOT DONE: the VideoMAE side.** Its decoder reconstruction loss is the second,
  independent surprise measure; the cached checkpoint carries the pretraining head
  (86.2M encoder + 7.7M decoder, 2.19 s per forward on CPU). Needs IDENTICAL mask draws
  for both twins, as `voe_cogvideox.py` uses identical `eps`. 40 clips x 32 draws ~ 47
  min. Script does not exist.
* **NOT DONE: the k-ladder**, which is what 46f-ii says the interpretation needs.

---

### 63. THE KINETIC-ENERGY READOUT WITH THE REDUCED MASS VARIED — `ke_factor_set`

Run 2026-10-23, same two encoders, all 1000 clips, full-clip captures `outputs/ke_<model>/`.

**The set** (`plan_ke_set.py`): 200 groups, each one scene at five KE levels x {1/2,
1/sqrt2, 1, sqrt2, 2}. **Family M** (100 groups) scales both masses by c — so the reduced
mass mu, the total KE and `K_rel = 1/2 mu v_rel^2` all scale by c while the motion is
the SAME video at every level. **Family S** (100 groups) scales both speeds. e = 1
throughout. Checked on the labels: within-group dlog mu / dlogKE = 1.000 (M) and 0.000
(S); dlog|p| / dlogKE = 1.000 (M) and 0.500 (S); corr(radius, level) = 0.04 in M, but
**corr(#metal, level) = 0.77** — the mass level flips material in many M groups.

#### 63a. Within a route, KE reads at ceiling — including when only mu moves

| | giant | V-JEPA2 |
|---|---|---|
| log KE within S, R2 (best cell) | 0.989 (`all.all` b18) | 0.990 (`all.all` b15) |
| log KE within M, R2 | 0.988 (`all.all` b15) | 0.987 (`all.all` b15) |
| `embed.out` best, S / M | 0.40 / 0.44 | 0.37 / 0.31 |
| **within-M group slope, cell chosen on S** (only mu moves) | **1.003** [0.990, 1.016] | **0.997** [0.980, 1.012] |
| same, partialling the material count | 1.004 [0.984, 1.026] | 0.987 [0.960, 1.012] |
| log mu itself, within-M slope | 0.998 | 0.994 |

Material alone explains 58.5% of the within-M variance in log KE, so the partial slope is
well posed and says **the readout follows graded mass, not "metal or rubber"**. (A first
version also partialled radius; with material and radius both fixed m = rho r^3 is
determined, the covariates explained nearly everything, and the partial slope was
uninterpretable. Do not reintroduce it.)

#### 63b. THE CROSS-ROUTE TEST — and the version of it that is an identity

The set's designed test: fit log KE on one family, apply it unchanged to the other, read the
within-group slope. Energy code -> 1; momentum code -> 2 (S->M) or 0.5 (M->S); speed code
(S->M) or mass code (M->S) -> 0. Cell chosen on the source only.

> **THE RAW VERSION IS §34b AGAIN.** S groups sit at five mass levels and M groups at five
> speed levels, so the source family's BETWEEN-group variation already carries the other
> route, and `log K_rel = log mu + 2 log v_rel - log 2` is linear in the factors. Raw
> transfer gives **0.979–0.994** — and transferring log mu and log v_rel SEPARATELY and
> adding them gives the same number (K_rel: 0.948 vs 0.951 giant, 1.005 vs 0.996 V-JEPA2,
> 0.975 vs 0.975, 0.979 vs 0.979). Slope 1 here follows from decodable factors alone. **The
> test is the ROUTE-CALIBRATED readout**: trained on group-demeaned source data, so it sees
> speed only (S) or mass only (M), read against a same-route split-half ceiling.

| | giant | V-JEPA2 |
|---|---|---|
| **S->M route-calibrated** (speed-calibrated, applied where only mu moves) | 0.160 | -0.111 |
| — refit on bootstrap-resampled source groups | 0.16 [-0.08, 0.37] | -0.07 [-0.27, 0.16] |
| — median over blocks at depth >= 0.25 (IQR) | -0.02 [-0.12, 0.05] | -0.05 [-0.15, 0.09] |
| — partialling the material count | 0.42 [0.40, 0.44] * | 0.46 [0.44, 0.48] * |
| **M->S route-calibrated** (mass-calibrated, applied where only speed moves) | **0.233** | **0.169** |
| — refit on bootstrap-resampled source groups | **0.24 [0.17, 0.32]** | **0.20 [0.15, 0.26]** |
| — median over blocks at depth >= 0.25 (IQR) | 0.22 [0.20, 0.25] | 0.16 [0.11, 0.19] |
| same-route split-half ceiling, S / M | 0.981 / 0.991 | 0.974 / 0.990 |
| gate: mass readout S->M / speed readout M->S | 0.991 / 0.959 | 0.975 / 0.964 |

\* target-group bootstrap only; it understates the uncertainty, which the source refit
shows is ~0.2 wide at this cell.

* **No single KE direction serves both routes.** Calibrated on speed, the readout does not
  follow KE when only the reduced mass changes: the source-refit CIs contain 0 and exclude
  1 (energy) and 2 (momentum), and across depth it scatters around 0 with sign flips
  block to block (single blocks reach 0.57 / 0.97 among neighbours near 0 — never quote a
  single cell). Calibrated on mass, it follows speed at **0.17–0.24, stable at every
  block** — excluding energy (1), momentum (0.5) and a pure mass code (0).
* **Both gates pass** (0.96–0.99), so the factors cross the gap and this is a negative,
  not a void test.
* **The material partial is the open thread.** Holding material fixed, the speed-calibrated
  readout follows mass at **0.42 / 0.46**: material flips push it DOWN, same-material mass
  increases push it UP. Within one material a heavier ball is a BIGGER ball, so a
  frame-wide speed feature that grows with the area of moving pixels predicts exactly this
  without any energy concept. This set cannot separate the two (radius is decorrelated
  from level overall, not within material). A size-matched M family — mass via density
  only, the `latent_mass` arm's trick — would.

#### 63c. THE CLAIM

**KE, including KE moved by the reduced mass alone, is read near-perfectly within a route,
and is not held as one quantity across routes.** The mass route and the speed route are
read off largely different directions; a readout calibrated on one transfers 0–0.24 of the
energy-code slope to the other. The naive cross-family slope of ~1 is the composition of
two separately decodable factors, measured to agree with it to the second decimal. This
is §26 / §45 for `KE = 1/2 mu v_rel^2`, now with the mass varied in isolation: **the
factors are represented; the energy is assembled at readout.**

#### 63d. Files

```bash
python scripts/kinetic_energy/ke_reduced_mass.py --model <m> --captures outputs/ke_<m>/momentum_probe/activations
```

* `scripts/kinetic_energy/ke_reduced_mass.py` — within-family grid, route-calibrated and raw transfer,
  composed-from-factors identity check, split-half ceiling, source-refit bootstrap, depth
  curve, gates, material partial. ~8 min per model at 40 workers, numpy only.
* `docs/artifacts/ke_reduced_mass_{videomaev2-giant,vjepa2-vitl}.json`.

---
