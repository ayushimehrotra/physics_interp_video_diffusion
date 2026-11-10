# Momentum — and whether it is conserved

Split out of `CLAUDE.md` on 2026-10-27, verbatim apart from paths updated to the new layout. Section numbers are unchanged, so every
`§N` cross-reference still resolves; the index in `CLAUDE.md` says which file each § lives in.
Momentum is `p = m v`, so the mass and velocity lines live here too.
Read `CLAUDE.md` Part 1 (methodology) before designing a new probe target.

---

## Momentum and conservation

### 1. Probing — physical variables are decodable, conservation is not

Capture: 150/150 scenes, 30 blocks x 9 noise levels x 9 pooled regions, 36
s/scene, ~90 min, zero failures. 8.6 MB per scene (not the ~3 MB an earlier
revision of this file claimed).

**Variables, compared within one pool** (`all.pre`) — comparing across pools is
misleading, because appearance wins in object-local pools and momentum wins in
frame-wide ones:

| target | R2 | VAE baseline |
|---|---|---|
| momentum magnitude (pre) | **0.849** | 0.563 |
| kinetic energy (pre) | **0.820** | 0.725 |
| momentum (pre, 2-vector) | **0.805** | 0.421 |
| momentum (post) | **0.819** (`all.post`) | 0.545 |
| *mass_A* (control) | 0.497 | 0.441 |
| *radius_A* (control) | 0.462 | 0.268 |

Cleanest single result: `velocity_B_pre` R2 **0.686 with VAE at −0.002** — the
autoencoder carries nothing there. `restitution` is essentially not decodable
(R2 ~0.10). `speed_A_pre` hits 0.990 but the VAE gets 0.902, so that one is
mostly already in the latent and is not evidence about the transformer.

Depth structure reproduces the reference paper: momentum peaks at **blocks
15–24**, appearance (mass, radius) at **blocks 6–13**.

**Conservation: 0.683 accuracy, against a selection-corrected null of 0.633**
and a VAE baseline of 0.583, on 60 held-out scenes. This is *not* a result — see
the next subsection.

### 3. Momentum is not an independently steerable factor

> **REVISED BY §26 — read that first.** The claim below that the probe "beat the
> oracle" (0.805 vs 0.695) rests entirely on giving that oracle only the raw
> factors. The excess survives a quadratic basis but not the composition test:
> decoding mass and velocity separately and multiplying beats the direct probe.
> The object-level version of the claim is refuted outright.

This is the most important negative, and it is structural rather than empirical.

* `R2(p_vector | m_A*v_A, m_B*v_B) = 1.000000`. Momentum is an **identity**, not
  a latent factor. Hold mass and velocity and momentum is held; there is no
  residual degree of freedom to steer.
* Consequently a "momentum direction disentangled from mass and velocity" does
  not exist *for any model*, and any test framed as "is there momentum
  information beyond mass and velocity" is answering a definition, not a
  question about CogVideoX. An earlier version of this analysis made exactly
  that mistake and concluded far too much from it.

**What can still be asked, and the answer:** does the representation make
momentum *linearly accessible* beyond what the raw factors linearly afford?

| | R2 on \|p\|, held out |
|---|---|
| oracle: best **linear** function of the **true** labels (m, r, \|v\|) | 0.695 |
| **linear probe on block-15 activations** | **0.805** |

The probe never sees the labels and still beats a linear combination of them.
Since `|p|` is a product-and-norm of mass and velocity, a linear model on those
scalars tops out around 0.695 — so reaching 0.805 means the representation has
already combined them into something momentum-shaped. That is what an internal
momentum concept looks like operationally. It is *not* proof: some other
non-linear feature correlated with `|p|` would look the same.

### 4. The probes were reading, not predicting

CogVideoX's transformer attends **bidirectionally** over all 13 latent frames
(`is_causal=False` in `CogVideoXAttnProcessor2_0`). Latent frames 0–5 are
strictly pre-contact and 6–12 strictly post — but a "pre" token has already
attended to the post frames. Measured best-over-grid:

| target | from `all.pre` | from `all.post` |
|---|---|---|
| `velocity_A_post` | 0.596 | 0.680 |
| `momentum_post` | **0.792** | **0.807** |
| `ke_ratio` | 0.615 | 0.633 |
| `momentum_delta` | −0.015 | 0.021 |
| `momentum_conserved` | 0.517 | 0.533 |

`momentum_post` reads equally well from either side. **So decoding a
post-collision quantity is the model looking at the aftermath, not inferring
it** — no probe here tests prediction, and neither does the reference paper's.
Note the pattern: everything *observable* sits at 0.6–0.8, while the two
*relational* quantities (`momentum_delta`, `momentum_conserved`) sit at zero and
chance. The model encodes what it sees, not the relation between the parts.

### 26. PRODUCTS ARE NOT STORED — and this revises §3

> **PARTLY WITHDRAWN BY §34.** The **momentum** results here stand — `p = m*v` is a
> *product* of the factor targets, so the composition test is valid, and §34e
> strengthens it best-vs-best (direct 0.825/0.728 against composed 0.934/0.936).
> The **mass-ratio** rows do not: `log(m_A/m_B) = log|dv_B| - log|dv_A|` is a
> *linear combination* of two probed targets, and ridge is linear in Y, so
> `w_ratio = w_B - w_A` to 1e-15 and direct-vs-composed is an identity rather than
> a measurement. See §34b. The **fidelity-matched oracle** used here is also
> withdrawn as unfalsifiable — §34c shows its answer swings from +0.12 to -0.14
> with the analyst's choice of basis.

The single most important result of 2026-08-15/16, and it changes how the
project's one positive claim should be stated.

**The test.** For any target that is a product or quotient of factors, fit two
readouts from the *same* pooled vector: a **direct** probe on the target, and a
**composed** one — probe each factor separately, then combine the predictions
arithmetically. If composed >= direct, nothing in the representation stored the
combination; the model holds the factors and the readout does the algebra.

**Object-local per-object momentum, v1** (read from that object's own pool, best
over blocks, VideoMAE / V-JEPA2):

| pool | m alone | v alone | `m*v` DIRECT | `m*v` COMPOSED | quadratic oracle |
|---|---|---|---|---|---|
| A.pre | 0.875 / 0.928 | 0.974 / 0.964 | **0.663 / 0.654** | 0.894 / 0.930 | 0.893 |
| B.pre | 0.731 / 0.839 | 0.959 / 0.939 | **0.515 / 0.434** | 0.910 / 0.903 | 0.683 |

Mass reads at 0.73-0.93 and velocity at 0.94-0.97 **out of the same vector**, and
their product reads at 0.43-0.66 — below what a quadratic function of the true
factors achieves. Multiply the two decoded quantities and you recover 0.89-0.93.
**A linear probe cannot multiply two things it can each read perfectly well**, and
that is exactly what is happening.

**Frame-wide total momentum, v1:**

| model | direct | composed | RBF kernel probe | quadratic oracle |
|---|---|---|---|---|
| VideoMAE | 0.825 | 0.842 | 0.873 | 0.682 |
| V-JEPA2 | 0.706 | 0.815 | 0.791 | 0.682 |

Composition matches or beats the direct probe here too, and a *generic nonlinear*
probe (RBF kernel ridge, gamma/alpha on an inner split) lands on the composed
number rather than above it. Nothing indicates the total is stored as a unit.

**Same result a third time, on the v3 mass ratio** (10 random splits, four-pool
concat, fixed cell):

| readout | VideoMAE | V-JEPA2 |
|---|---|---|
| direct linear | 0.382 +- 0.110 | 0.225 +- 0.133 |
| **structured: decode Δv, then log-ratio of norms** | **0.642 +- 0.023** | **0.650 +- 0.035** |
| generic nonlinear (RBF) | 0.500 | 0.476 |
| matched norm-fidelity oracle | 0.532 +- 0.054 | 0.536 +- 0.045 |

The structured readout beats its matched oracle by +0.110 +- 0.034 and
+0.114 +- 0.025, **positive on 10/10 splits** — the only place in this project
where a probe cleanly beats a properly matched oracle. Note the generic nonlinear
probe does NOT: at n=135 training scenes and 3072-4096 dimensions it cannot
discover `log||.||` on its own. **Imposing the known nonlinearity costs no
capacity; learning it costs more than the data affords.**

**THE ORACLE IS A PROPERTY OF ITS BASIS, NOT OF THE INFORMATION.** Every target
here is an exact function of the visible factors, so a sufficiently expressive
oracle scores 1.000 by construction. What an oracle measures is *how far from
linear-in-the-chosen-basis* the target is. Measured on v1, n=150:

| oracle basis | momentum_pre | mass_A | mass_B |
|---|---|---|---|
| raw factors `[v_A, v_B, r_A, r_B, rho_A, rho_B]` | 0.565 | 0.782 | 0.339 |
| + all pairwise products (quadratic) | 0.682 | **0.988** | **0.967** |
| with `r^3` instead of `r` | 0.566 | 0.794 | 0.238 |
| with the true masses | 0.599 | 1.000 | 1.000 |
| with the exact products `m*v` | 1.000 | 1.000 | 1.000 |
| *observed probe* | 0.825 / 0.728 | 0.919 / 0.960 | 0.840 / 0.920 |

So **the mass "excess" over its oracle was an artifact of feeding raw `r` to a
target cubic in `r`** — a quadratic basis reaches 0.988, above both models. Any
claim of the form "probe beats oracle" must name the basis. Momentum's excess
does survive a quadratic basis (0.825 vs 0.682), because `p = rho*(4/3)pi r^3 v`
is quartic — but composition beats the direct probe there too, so the most likely
reading is that the quadratic oracle is a weak baseline for a quartic target,
not that the model built a momentum feature.

**What this does to §3.** "The representation has already combined mass and
velocity into something momentum-shaped" is **not supported at the object level**,
where the combination demonstrably has not happened. The defensible statement is
narrower and different in kind:

> This representation stores physical factors **separately and linearly**. Every
> product or quotient must be composed by the readout rather than read off. It is
> a claim about representational FORMAT, not about physical knowledge — any
> observer holding mass and velocity computes the product with one multiplication.

That is now supported three times over (mass ratio, per-object momentum, total
momentum) and is a better headline than §3's, even though it is a negative one.

**One mundane mechanism ruled out along the way.** "Per-object products are
computed locally and mean-pooling performs the sum" was the natural story for why
frame-wide pools win on momentum. The object-local table above refutes its first
half, and pooling cannot create a product either — it averages token vectors, so
if tokens carry `m` and `v` separately, the pool carries averages of `m` and `v`.

### 42. `conservation_set_v1` — the leak-free 2x2, and the mechanism that makes it possible

Built 2026-08-23/24. First set where **momentum conservation is the only route to the
label**, and the first 2x2 crossing elasticity with conservation. 300/300 clips in
207 min, 0 failures, 4.4 GB.

**All nine gates pass on the finished set:**

| gate | result |
|---|---|
| deflections exactly antiparallel | max \|cos+1\| = 4.4e-16 |
| realised e == pinned e | max dev 8.9e-16 |
| honest conserve / violating do not | honest 4.9e-16, violating min 0.080 |
| joint mass-free classifier, grouped CV | 0.514 / 0.486 / 0.444 / 0.484 across the four cells |
| worst single mass-free scalar | **0.561** (`speedB`), null sd 0.053 at n=60/60 |
| KE ratio, the competing law | 0.501 |
| frame-exit balance | 0.513 |

Two further checks beyond the gates. The plan's analytic model predicts the rendered
physics exactly — over all 300 clips, max deviation **8.9e-16** in `|dp|/|p_pre|` and
**6.7e-16** in KE ratio. And the permutation guarantee survived rendering: sorting the
implied mass ratios of the honest and violating halves of each cell and differencing
them elementwise gives **max 3.9e-14** — the two classes hold the *same multiset*, which
is what makes the kinematic leak surface provably empty rather than merely small.

#### 42a. The mechanism: e-preserving, ratio-transplanted violation

`apply_violation_ratio()` in `generate_scene.py`. Every frictionless collision obeys

```
a = |dv_A|,  b = |dv_B|,   a + b = (1 + e) * u_n     <- set by RESTITUTION
                           b / a = m_A / m_B         <- set by CONSERVATION
```

so the *sum* is the restitution law and the *split* is the conservation law. The edit
keeps the sum and moves the split:

```
a' = S / (1 + rho'),   b' = S * rho' / (1 + rho'),   S = a + b
```

Verified against the rendered sets to machine precision:

| property | result |
|---|---|
| realised e vs pinned e | 6.7e-16 |
| `cos(dv_A, dv_B)` | -1 to 2.2e-16 |
| relative velocity vs the honest twin | **bit-identical, 2.2e-16** |
| the edit itself | a COMMON drift `Delta-p / M` on both balls |

That last row is the point: the twins differ ONLY in how the fixed sum is split, so
separation rate, closest approach, contact geometry and interpenetration are provably
equal. **The entire leak surface is one scalar, the implied ratio `b/a`.**

This is v1's common-kick mechanism with the kick pinned to the contact normal, which
is why it inherits v1's exact preservation of `e` while avoiding v1's `cos` leak.
Contrast v4, whose `alpha = 1-delta, beta = 1+delta` gives `e' - e = (1+e)*delta*(w_A - w_B)`
(verified 1.4e-15) — the source of the `e > 1` leak, and proportional to mass asymmetry.

#### 42b. Closing the one remaining channel

`rho'` is another scene's mass ratio, assigned by a **permutation** within the cell. A
permutation preserves a multiset, so the implied-ratio distribution is identical in the
two classes by construction. Rejection sampling cannot build it — requiring all n
violations to land in a usable size range has vanishing probability — so
`plan_conservation_set.py` builds the admissibility graph and draws **perfect matchings**
from it, spending the restarts on also matching the KE-ratio distribution (free, since
every matching preserves the ratio marginal).

Audited on simulated twins, all mass-free scalars: implied ratio 0.500, restitution
0.500, `log|dv_A|` 0.504, `log|dv_B|` 0.507, centroid 0.501, relative speed 0.500,
both-faster 0.500, frame-exit 0.500. **Joint 18-feature classifier, grouped CV: 0.498.**
Energy, the competing law, 0.506 after matching (0.561 unmatched).

> **THE TRAP THAT ALMOST GOT REPORTED AS A LEAK.** That joint classifier reads **0.213**
> — 0.787 discrimination — if the folds are NOT grouped by seed. A scene's honest twin
> in train lets the classifier memorise its deflection split and flag the test twin.
> **Grouped: 0.498.** This applies to probe evaluation on this set too, not just the
> audit. It is why Joseph et al. specify "5-fold *grouped* cross-validation".

Second trap: `collision.contact_normal` is not always oriented along the approach.
Flip it so `u_n > 0` or `S` goes negative and every log-based statistic becomes NaN.
Third: scalars pinned by construction differ only in the last float bits; rank those
and you invent a leak. `validate_conservation_set.py` quantises before ranking.

#### 42c. Design of the set

150 seeds x 2 clips (honest + violating twin of the same seed) = **300 clips**.

| factor | levels | why |
|---|---|---|
| elasticity | `e = 1.00` / `e = 0.15`, PINNED | the elastic honest cell conserves energy exactly, so it is the positive control for the energy axis; the inelastic row loses energy in BOTH classes, so detection surviving there is not an energy detector |
| conservation | honest / violating | matched pairs: verified 51/51 seeds give a bit-identical scene under `violate=True/False` |
| material | 120 seeds `mixed`, 30 `same` | see 39d |

Per cell: mixed/elastic 60 pairs, mixed/inelastic 60, same/elastic 15, same/inelastic 15.
Violation sizes median 0.27-0.49 of `|p_pre|`, and the matching incidentally equalised
them across the elasticity factor (0.49 vs 0.46 in the mixed arm), removing a confound
that `Delta-p = lambda*J` with `J = (1+e)*mu*u_n` would otherwise force.

#### 42d. `--material-pair`, and a correction

The legacy sampler draws each sphere's material **independently**, so half the scenes
get a matched pair. That was never a design decision about pairs — it falls out of two
independent `rng.choice` calls.

**A first reading of this — that matched pairs hide the mass ratio because the only cue
is `r^3` on two similar spheres — is wrong.** Decomposing R2 on the v4 captures:

| | R2 | **RMSE** | target sd |
|---|---|---|---|
| VideoMAE, mixed (n=35) | +0.557 | **0.304** | 0.471 |
| VideoMAE, matched (n=40) | +0.171 | **0.308** | 0.338 |
| V-JEPA2, mixed | +0.619 | **0.282** | 0.471 |
| V-JEPA2, matched | +0.254 | **0.292** | 0.338 |

The absolute error is the same. The R2 gap is entirely the target's variance. Mixed wins
for a different reason: **fixed noise, larger signal** — it widens the ratio spread, so
the same 0.29 error floor buys ~1.4x the margin.

That makes the `same` arm a real control with a quantitative prediction: if the model
uses a single mass-ratio readout, matched-pair detection should degrade **in proportion
to the spread (~0.72x), not collapse**. Collapse would mean it is keying on material
rather than on mass.

#### 42e. Files and commands

```bash
datasets/plan_conservation_set.py       # physics-only; writes plan.json + probe cache
datasets/run_conservation_set.py        # resumable renderer, --workers
datasets/validate_conservation_set.py   # 9 gates, incl. the leak audit
datasets/conservation_set_v1/plan.json  # 300 clips, donor seed per violating clip

python datasets/plan_conservation_set.py --n-mixed 120 --n-same 30
KUBRIC_USE_GPU=1 .venv/bin/python datasets/run_conservation_set.py --workers 6
.venv/bin/python datasets/validate_conservation_set.py
```

**Throughput, measured on this box, and it contradicts the v4 figure.** 831 s for a
single scene alone; ~80 s/clip at 4 workers and ~76 s/clip at 6, i.e. **the ceiling is
~75 s/clip and more workers do not help**. 300 clips ~= 6.5 h. Do not exceed 6 workers:
8 concurrent Blender processes exhausted the 32 GB GPU during OptiX BVH build and three
scenes died with OOM. The runner retries failures serially and skips `_COMPLETE`, so a
killed batch resumes for free — but **check for `_FAILED` markers**, since a silent OOM
would otherwise leave a class-imbalanced set.

#### 42f. What this set can and cannot settle

It can decide whether violation detection is possible when **no kinematic statistic
distinguishes the classes** — the guarantee is that no readout using only pre-contact
velocities, the contact normal and the post-contact deflections can beat chance, since
their joint law is identical between classes. A readout that converts radius and
material into a mass ratio and compares it against `b/a` **can** succeed, and that
comparison IS conservation. It remains an appearance-plus-kinematics claim, not proof
the model *applies* a law; §29's test 9 (double dissociation) and an INLP ablation of
the mass subspace are what would upgrade it.

---

### 38. `probe_set_v4_legacy_fixed` — the leak-free v1, and what it costs §34

Run 2026-08-23. This section adds a fourth dataset, answers the object-local
pooling question, diagnoses the binary null to its root cause, and **puts §34's
headline positive in question** — not by finding an error in it, but by measuring
the same target on a set whose violation mechanism is fixed.

#### 38a. The mechanism swap, and why v1 needed one

v1 `probe_set` was rendered with the **committed** `apply_violation`: an
equal-and-*same*-direction kick, `v' = v + dp/(m_A+m_B)`, along a freely sampled
`violation_direction_rad`. That is the mechanism §11 already condemned on v2.
**It is also what v1 has**, and nothing in the document said so.

Measured on the label files, not inferred:

| | v1 `probe_set` | v4 `probe_set_v4_legacy_fixed` |
|---|---|---|
| `cos(dv_A, dv_B)`, conserving | −1.000000 | −1.000000 |
| `cos(dv_A, dv_B)`, violating | −0.9999 .. **+0.9993** | **−1.000000 .. −1.000000** |
| discrimination AUC from that scalar alone | **1.000** | 0.492 (chance) |

So every §1/§2/§34 number computed on v1 sat on a dataset where **one mass-free
scalar separates the classes perfectly**. That is the §34f observation, promoted
from a remark to a property of the set the whole momentum line was measured on.

`probe_set_v4_legacy_fixed` is v1 regenerated with the current
antiparallel-preserving mechanism and the **`legacy` arm kept**, so mass stays
visible (`mass_A` R2 0.934, `radius_A` 0.975). It is the first set with both
properties: the model *could* apply the law, and no kinematic shortcut certifies
the class.

* 150 scenes, 35.9 min on OPTIX. The 75 conserving scenes are **copied from v1
  unchanged** — verified bit-identical under the current generator (max |pos diff|
  0.00e+00 over frames 0/18/22/48), so the halves are consistent and there is no
  generator-drift confound.
* Validation: **5 of 6 hard checks PASS.** `violations_actually_violate` median
  0.438 min 0.118 (the `MAX_VIOLATION_ASYMMETRY` clamp did not bite on this arm;
  v3's min was 0.0073).

#### 38b. THE FIX OPENED A NEW MASS-FREE CHANNEL — restitution

`restitution_realised` **FAILS** on v4 at |measured − target| max 1.23, where v1
passed at 8.88e-16. This is mechanical, not a bug:

`e = |rel_post| / |rel_pre|` along the contact normal depends only on
`v_A − v_B`. Adding a **common** vector cancels there, so v1 preserved `e`
exactly and put the damage in the deflection *angle*. Scaling the two deflections
by different factors **is** a change in relative velocity, so v4 preserves the
angle and puts the damage in `e`.

| | conserving | violating |
|---|---|---|
| measured `e` | 0.079 – 0.999 | 0.030 – **2.056** |
| `e > 1` (super-elastic, impossible) | **0/75** | **12/75** |

**16% of violations are certifiable from a mass-free kinematic scalar.** Same
shape as the `cos` leak, one sixth the coverage.

> **RULE.** A violation edit cannot leave the aftermath unchanged; the question is
> only *which channel* it disturbs. Classify the channel by whether reading it
> needs the masses. `cos(dv_A,dv_B)` and `e` do **not** — they are leaks.
> KE ratio and `Delta-p` **do** — they are the intended route (though KE tests a
> different law and should not be read as evidence about momentum).

**Energy creation is NOT new and not a defect.** 34/75 violating scenes create
kinetic energy in v1 and 29/75 in v4 (discrimination AUC 0.621 / 0.572). You
generally cannot break momentum without changing energy. It needs both masses, so
it is not a shortcut.

**Unrun fix:** bound `delta` so the edited relative speed never exceeds the
incoming one, capping `e <= 1` by construction. ~36 min re-render.

#### 38c. Joseph et al.'s probe, replicated exactly — and it changes nothing

`old_experiments/scripts/joseph_intphys_probe.py`. Their IntPhys probe is **not** `fit_probe`
(§3.2, §4.1, App B, App C.11): binary **logistic** regression with cross-entropy,
Adam, a **20-config** sweep (lr {1e-4,3e-4,1e-3,3e-3,5e-3} x wd {0.01,0.1,0.4,0.8}),
**5-fold grouped CV**, 50 epochs, scored by accuracy **and AUC**, on
spatiotemporally pooled activations.

On v1 it lands within noise of the ridge probe — VideoMAE **0.684** vs 0.708,
V-JEPA2 **0.625** vs 0.683. Swapping the estimator does not move the result.

**One thing cannot be replicated and it is structural.** Their groups are *matched
possible/impossible pairs* differing at a single breakpoint frame. Our scenes are
independent draws differing in everything. Verified that matched pairs are **free**
here: `sample_params` with the same seed and `violate=True/False` returns an
identical physical scene (32/32 seeds, all 17 physical fields). Rendering seed S
twice would give their design exactly. **Not done — needs a 150-scene render.**

#### 38d. Object-local pooling does NOT help this target

Masking is a *deviation* from Joseph et al. (§33a: no masks anywhere in the
paper), and it carries §20's leak. Every cell run against its own mask-only floor.
v1, VideoMAE, all 10 pool specs:

| pool | best AUC | layer | floor (norm) | margin |
|---|---|---|---|---|
| **all.post** | **0.697** | 8 | 0.500 | **+0.197** |
| all.all | 0.684 | 8 | 0.500 | +0.184 |
| all.pre | 0.668 | 7 | 0.500 | +0.168 |
| A.post | 0.686 | 4 | 0.532 | +0.154 |
| A.pre | 0.616 | 2 | 0.492 | +0.124 |
| A.post+B.post | 0.604 | 6 | 0.492 | +0.112 |
| A.pre+A.post+B.pre+B.post | 0.568 | 11 | 0.502 | +0.046 |
| B.pre | 0.564 | 7 | 0.505 | +0.060 |
| A.pre+B.pre | 0.547 | 4 | 0.474 | +0.073 |
| **B.post** | 0.551 | 3 | **0.538** | **+0.013** |

* **The three frame-wide pools take the top three margins.** Masking buys nothing.
* **`B.post` is almost entirely leak** (+0.013). Reported without its floor it
  would read as a modest positive.
* **Concatenation actively hurts.** The four-pool spec holds every ingredient of
  `m_A dv_A + m_B dv_B` and scores 0.568, *below* `A.pre` alone. At n=150 each
  concat multiplies input dimension without adding a usable direction. Ingredients
  being present does not make the relation linearly available.
* Sanity check passed: all three `all.*` floors are **exactly 0.500** (all-ones
  masks, identical every scene).

V-JEPA2 partial (5 of 10 specs; run killed to free the GPU) shows the same
ordering. `scratchpad/mask_sweep_partial.log`.

#### 38e. WHY THE AUC IS ~0.6 — four candidates, three ruled out

`old_experiments/scripts/why_low_auc.py`, on v4, `all.post`.

**C. The label shape is the binding constraint.** Same classifier, exact noiseless
inputs:

| linear classifier given | AUC |
|---|---|
| exact **visible ingredients** `[m_A, m_B, dv_A, dv_B]` (6d) | **0.490 — chance** |
| exact Delta-p vector (2d) | 0.533 |
| exact Delta-p + quadratic terms (5d) | **0.893** |
| exact \|Delta-p\| (1d) | 1.000 |

**A linear classifier holding every exact ingredient scores at chance**, because
`Delta-p = m_A dv_A + m_B dv_B` is a **product** and a linear map emits only
weighted sums. The label then needs a **norm** on top. Two nonlinear steps.
**0.893 with quadratic terms is inside Joseph et al.'s 0.85-0.95.** The gap to
their result is *functional form*, not model capacity.

**A + B ruled out.** Same pipeline, same n, same pool:

| target | VideoMAE | V-JEPA2 | n=40 | n=70 | n=100 | n=150 |
|---|---|---|---|---|---|---|
| `b_moving` (visible) | **0.976** | **0.999** | 0.938 | 1.000 | 0.994 | 0.997 |
| `radius_A > median` | 0.866 | 0.841 | 0.925 | 0.820 | 0.860 | 0.844 |
| `mass_A > median` | 0.851 | 0.828 | 0.700 | 0.792 | 0.742 | 0.793 |
| `momentum_conserved` | 0.604 | 0.673 | 0.613 | 0.584 | 0.596 | 0.651 |

*(curve columns are V-JEPA2)*

A real linear signal **announces itself at n=40** — V-JEPA2 reads `b_moving` at
0.938 with 40 scenes. `momentum_conserved` is flat at every n in both encoders.
Neither more scenes nor more parameters is the binding constraint.

**D not binding.** V-JEPA2-L is 3.4x VideoMAE-base and *better* on the control
(0.999 vs 0.976), but reaches only 0.673 on the target. The ceiling is readout
algebra, which a larger encoder is subject to identically.

> **RULE.** Binarising costs even a perfectly visible quantity. `radius_A`
> regresses at R2 0.975 and its median split reads AUC 0.866. `b_moving` beats both
> binarised continuous controls because it is *natively* categorical — which is
> exactly what IntPhys violations are and ours is not.

#### 38f. THE PROBES ARE NOT READING DELTA-P — proof from the ceiling

A linear classifier on the **exact** Delta-p vector scores **0.533**. Both
encoders score **above** that from noisy activations — VideoMAE 0.604, V-JEPA2
0.673. No probe can extract more from a noisy estimate of a quantity than from
the quantity itself. **Therefore the binary probes are not reading Delta-p.**

They are reading correlates of the edit. Confirmed directly
(`old_experiments/scripts/leak_vs_probe.py`, out-of-fold per-scene predictions, v4):

| model | pool | AUC | specificity | recall LEAKED (n=12) | recall UNLEAKED (n=63) | gap |
|---|---|---|---|---|---|---|
| VideoMAE | all.post | 0.580 | 0.533 | **0.917** | 0.571 | **+0.346** |
| V-JEPA2 | all.post | 0.668 | 0.600 | 0.667 | 0.556 | +0.111 |
| VideoMAE | all.all | 0.572 | 0.547 | 0.583 | 0.587 | −0.004 |
| V-JEPA2 | all.all | 0.602 | 0.547 | 0.583 | 0.540 | +0.043 |

**VideoMAE `all.post` wins substantially through the `e > 1` leak** — 11 of 12
impossible scenes called violating against 36 of 63 clean ones (p ~ 0.012 under a
shared-rate null). It shows up on `all.post` and not `all.all` because `e` is a
*post-contact* quantity: averaging the phases dilutes it. V-JEPA2 does not exploit
it, consistent with its AUC being unchanged between v1 and v4.

Specificity is 0.533-0.600 everywhere: these probes barely recognise a *conserving*
scene, the class with an exact defining property.

> **CORRECTION, recorded so it is not repeated.** A `|dp|/|p_pre|` tercile control
> looked like recall failing to rise with violation size in 4/4 cells. It is
> mis-specified: across terciles the *absolute* violation moves only 0.578 -> 0.712
> while `|p_pre|` nearly halves (2.054 -> 1.099), `corr(log rel, log |p_pre|) =
> −0.300`. It is largely a momentum-scale split. No clean fix exists in this data:
> `corr(log |dp|_abs, log |p_pre|) = +0.849`.

#### 38g. THE NORM ROUTE FAILS ON v4 — and this puts §34 in question

§34a's fix — regress the Delta-p vector, take the norm *outside* the probe — was
predicted to reach ~0.75. `old_experiments/scripts/norm_route.py`, 200-permutation **grid-max**
nulls over the same layer selection the observed value enjoyed:

| model | pool | AUC | Delta-p R2 | null 95th | p |
|---|---|---|---|---|---|
| VideoMAE | all.pre | 0.631 | **0.255** | 0.658 | 0.130 |
| VideoMAE | all.post | 0.541 | −0.155 | 0.656 | 0.700 |
| VideoMAE | all.all | 0.486 | 0.026 | 0.665 | 0.885 |
| V-JEPA2 | all.pre | 0.664 | **0.139** | 0.688 | 0.110 |
| V-JEPA2 | all.post | 0.492 | 0.024 | 0.694 | 0.985 |
| V-JEPA2 | all.all | 0.605 | −0.007 | 0.695 | 0.480 |

**Six cells, zero significant.** The route needs a decodable Delta-p to take the
norm of, and on v4 **Delta-p is barely decodable**: R2 **0.255 / 0.139** against
§34's **0.517** at the same cell type on v1.

The mechanism changed the target's structure:

| | v1 | v4 |
|---|---|---|
| \|cos(Delta-p, contact normal)\| | 0.636 | **1.000** |

On v4 Delta-p lies **exactly along the contact normal** in every violating scene —
forced by the mechanism. On v1 the common kick had a freely sampled direction, an
extra degree of freedom that no longer exists. (Checked and rejected: common-drift
magnitude and its correlation with |Delta-p| are similar in both sets, 0.266/+0.59
vs 0.203/+0.63, so the differentiator is direction structure, not visible drift.)

> **THE OPEN QUESTION THIS LEAVES.** §34's `momentum_delta` positive — R2 0.517,
> p = 0.000 against 500-permutation grid-max nulls in three architectures, the
> best-supported result in this document — was measured on the set whose mechanism
> is now known to leak `cos` at AUC 1.000 and to give Delta-p a free direction. On
> v4 the same target reads 0.255. **This is not sufficient to call §34 an
> artifact**: 0.255 is still far above the ~0.014 pre-transformer baseline, and a
> freely-sampled Delta-p direction is not obviously the same thing as the `cos`
> giveaway. But its effect size is **mechanism-dependent**, and §34 must be
> re-measured on v4 under its full protocol before being carried forward.

#### 38h. What to do next, in order

1. **Re-run §34's full `momentum_delta` protocol on v4** — all pools, all layers,
   500-permutation grid-max. Settles whether the headline positive survives.
   Captures exist; no GPU render needed.
2. **Close the `e` leak** — bound `delta` so `e <= 1`, re-render (~36 min).
   Nothing should rest on v4 until this is done.
3. **Then scale.** `OpenGVLab/VideoMAEv2-Base` is implemented (`VideoMAEv2Adapter`)
   and **cached but never captured** — Joseph et al. probe that family directly.
   V-JEPA2 Huge/Giant need a small adapter subclass; `VJEPA2Adapter` reads geometry
   from the model config. Note the ordering CHANGED: scale is no longer "the fix
   after the readout fix", because the readout fix failed. It is now a test of
   whether a stronger encoder carries Delta-p at all on a clean set.
4. **Matched pairs** (§38c) if the binary framing is kept — the one structural
   difference from Joseph et al. that remains, and it is free to construct.
5. **Lawful errors on v4** (`old_experiments/scripts/lawful_errors.py`). v3 was chosen for it
   because violations are undetectable there — but v3 also hides mass, so the model
   cannot apply the law even if it has it. v4 is the first set with visible mass AND
   no geometric shortcut, which is the setting where that test is meaningful.

#### 38i. Files

```
datasets/probe_set_v4_legacy_fixed/   150 scenes, 5/6 checks pass
outputs/v4_{videomae-base,vjepa2-vitl}/       150 captures each
old_experiments/scripts/joseph_intphys_probe.py               §38c/§38d — their protocol + §20 floors
old_experiments/scripts/leak_vs_probe.py                      §38f — out-of-fold recall by leak status
old_experiments/scripts/why_low_auc.py                        §38e — ceilings, controls, learning curves
old_experiments/scripts/norm_route.py                         §38g — §34a's two-stage readout
docs/artifacts/joseph_intphys_{allall,v4}.json
docs/artifacts/leak_vs_probe_all.{post,all}.json
docs/artifacts/why_low_auc.json
docs/artifacts/norm_route_v4.json
```

**Known gap:** V-JEPA2's mask sweep covered 5 of 10 pool specs before the run was
killed to free the GPU; the combined JSON was never written. Re-running is cheap
and reads only existing v1 captures.

### 34. §28 RESOLVED — Delta-p IS represented, and the binary label was the wrong instrument

Run 2026-08-17. This section **closes §28**, **withdraws results in §26/§13/§29**,
and adds two rules about which probe targets a linear readout can reach at all.
Every number is v1 (`probe_set`, 150 scenes, 75 conserving / 75 violating) unless
stated. Scripts are analysis one-offs, described inline.

**The headline: `momentum_delta` is a real, replicated positive.** 500-permutation
grid-max nulls — permute labels, refit every cell, take the max, repeat 500x:

| | VideoMAE-base | CogVideoX-2b | V-JEPA2-ViT-L |
|---|---|---|---|
| `momentum_delta` R2 | **0.517** | **0.318** | **0.232** |
| null grid-max, 95th pct | 0.028 | 0.055 | 0.033 |
| **p** | **0.000** | **0.000** | **0.000** |
| best cell | `all.pre` / blk 9 of 12 | `all.pre` / blk 10 of 30, step 4 | `all.pre` / blk 17 of 24 |
| pre-transformer baseline | 0.014 (`embed.out`) | **0.000** (`vae.latent`) | 0.022 (`embed.out`) |

Zero of 500 permutations came close, in **three architectures** — two encoders and
the diffusion model. And the depth curves say it is *constructed*:

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VideoMAE | -0.02 | -0.00 | 0.01 | 0.03 | 0.16 | 0.30 | 0.38 | 0.48 | 0.41 | **0.52** | 0.44 | 0.37 |
| CogVideoX | 0.04 | 0.04 | 0.05 | 0.15 | 0.16 | 0.14 | 0.06 | 0.09 | 0.05 | 0.21 | **0.32** | 0.25 |
| V-JEPA2 | -0.04 | -0.00 | -0.03 | -0.01 | 0.01 | 0.05 | 0.01 | 0.04 | 0.13 | 0.14 | 0.11 | 0.17 |

Flat at ~0 before the transformer and through the early blocks, rising to a
mid-to-late peak. Compare `radius_A`, which reads **0.952-0.965 at `embed.out`** —
that one is in the pixels; Delta-p is not.

**Violation detection, via the norm of the predicted vector.** Regress Delta-p,
then take `|prediction|` *outside* the probe and score AUC against the
conserving/violating label. Same 500-permutation treatment:

| | AUC | null 95th | p |
|---|---|---|---|
| VideoMAE-base | **0.765** | 0.715 | **0.006** |
| CogVideoX-2b | 0.749 | 0.766 | 0.096 |
| V-JEPA2-ViT-L | 0.707 | 0.724 | 0.106 |

One clear hit (0.006, still 0.018 under Bonferroni for three models) and two
near-misses. All three land below 0.11, which under H0 has probability 0.0013;
Fisher's method gives combined p ~ 0.0035. **Treat that as indicative only** — the
three tests share the same 150 scenes and the same train/test split, so they are
not independent and the combined figure is optimistic.

#### 34a. WHY THE BINARY LABEL FAILED — and it is not about the models

§1/§2's `momentum_conserved` null (0.683 against a 0.650 grid-max null) was read as
"conservation is not represented". **It was substantially an artifact of the
target's shape.** The classes are *concentric* in Delta-p space:

| | mean Delta-p | mean \|Delta-p\| |
|---|---|---|
| conserving (75) | (0, 0) | 1.7e-16 |
| violating (75) | (-0.070, -0.066) | **0.818** |

Conserving scenes are a single **point at the origin**; violating scenes are a
**ring around it**, direction near-uniform. Class means are 0.096 apart while the
violating class has radius 0.818 — the between-class separation is ~8.5x *smaller*
than the within-class spread.

A linear classifier computes `w.x + b`. On the conserving class `x = 0`, so the
output is `b` for all 75; on the violating class `x` lies on a ring, so `w.x` is
symmetric about 0 and roughly half fall either side of any threshold.

**The proof it is the label and not the model** — hand a linear classifier the
*exact, noiseless, true* Delta-p and ask it to classify:

| linear map from | accuracy | AUC |
|---|---|---|
| the exact Delta-p **vector** | 0.433 | **0.559** |
| the exact Delta-p + quadratic terms | 0.717 | 0.912 |
| the exact **\|Delta-p\|** | 0.833 | **1.000** |

**AUC 0.559 with perfect information.** The separating statistic is
`|dp| = sqrt(dpx^2 + dpy^2)` — a norm, not in the linear span. A perfect model with
perfect activations and infinite data would still fail this framing.

On activations the gap decomposes cleanly (VideoMAE):

| readout | AUC |
|---|---|
| binary probe, scored by accuracy | 0.633 |
| binary probe, scored by AUC | 0.708 |
| regress Delta-p, then take the norm | **0.765** |

First step is the scoring metric (thresholding at 0.5 discards ranking); second is
the structural fix.

> **RULE.** Before probing a target, ask whether it is a **linear functional of
> quantities the model represents**. `Delta-p = p_post - p_pre` is (both momenta
> decode at 0.83-0.85). `|Delta-p|` is not — it is a norm. `momentum_conserved` is a
> *threshold on a norm*, two nonlinear steps out. This is §26's own finding applied
> to the methodology: **linear functionals are readable, nonlinear combinations are
> not.** Where a target needs a nonlinearity, put it in the readout *after* a linear
> probe, and permute the whole pipeline for the null.

**What this does NOT show.** On v1 both masses and both velocity changes are
visible, so `Delta-p = m_A dv_A + m_B dv_B` is an exact function of what a camera
can see. This is decodability and violation-discrimination, **not** evidence that
the model applies the law. §12 still applies.

#### 34b. WITHDRAWN — the mass-ratio subspace and INLP results are ALGEBRAIC

§29's test 4 (subspace containment) and test 5 (INLP), and the composition test,
were run on `log_mass_ratio` and reported as evidence. **They are vacuous for that
target and are withdrawn.**

Ridge is `W = Xc'(Xc Xc' + aI)^-1 Yc` — a linear map applied to `Yc`. And
`log(m_A/m_B) = log|dv_B| - log|dv_A|` exactly. So at matched alpha:

```
||w_ratio - (w_B - w_A)|| / ||w_ratio||  =  1.2e-15  (alpha 1)
                                            1.1e-15  (alpha 100)
                                            8.3e-16  (alpha 1000)
```

`w_ratio` **is** `w_B - w_A`, whatever the model contains. Consequences:

* The reported **16.5 deg / 17.7 deg** containment was an artifact of the three
  targets selecting *different* alpha (100, 1000, 100). Across 5 split seeds the
  angle ranges 0.0-17.7 deg and is exactly **0.0 on three of them**.
* INLP's factor arm removes `span(w_A, w_B)`, which contains `w_ratio` by
  construction, so its collapse is guaranteed.
* Direct and composed agreeing to three decimals is the same fact.

> **RULE, generalising.** Before comparing fitted weight vectors across targets,
> check whether the target is a **linear combination of the other targets**. If it
> is, composition / subspace-containment / INLP all return their "confirming" answer
> by construction. This is the second time an estimator property masqueraded as
> geometry here (§30b was ridge-alpha selection).

**What survives, and it is sharper than what was withdrawn:** because
`w_ratio = w_B - w_A` to 1e-15, "the model computed the mass ratio" and "the model
stored two log-magnitudes and the readout subtracted them" are **the same
estimator**. No linear probe can ever separate them — at any R2, including 1.000.
That is what the identity trap claims: *not* that the model lacks the law, but that
**the experiment cannot decide**.

**Momentum is unaffected.** `p = m*v` is a *product* of the factor targets, not a
linear combination, so no identity exists and §26's composition result stands.

#### 34c. THE FIDELITY-MATCHED ORACLE IS UNFALSIFIABLE — stop using it

§13/§26/§30 lean on an oracle built by corrupting true constituents to the model's
own measured fidelity. **The answer is entirely determined by choices the analyst
makes.** On the bounded ratio `(m_A-m_B)/(m_A+m_B)`, same data, same model:

| oracle basis / fidelity | VideoMAE excess | V-JEPA2 excess |
|---|---|---|
| raw magnitudes, same-cell | **+0.104** | **+0.120** |
| log magnitudes, same-cell | +0.049 | +0.042 |
| log magnitudes, best-grid fidelity | -0.007 | +0.006 |
| both raw and log, best-grid | **-0.137** | **-0.095** |

The excess swings from +0.12 to -0.14. It is not a measurement.

**Use the composition test instead** — it decodes the model's *own* factors, so it
inherits the real error structure and has no noise parameter or basis to pick. On
the bounded ratio at `A.pre+B.pre`:

| | VideoMAE | V-JEPA2 |
|---|---|---|
| direct probe | 0.540 | 0.646 |
| composed `tanh((lb-la)/2)` | 0.542 | 0.615 |
| direct - composed | **-0.002** | **+0.031** |

A tie. No evidence the model computes the ratio beyond tracking two deflections.
(Composing from *raw* magnitudes gives 0.257 / -4.073 — division by near-zero
predicted sums, a numerical artifact, not a result.)

**The oracle's basis-dependence is not monotone in fidelity.** The bounded target is
`tanh(0.5 log r)`, near-linear in logs (exact-log oracle 0.950) and strongly
nonlinear in the quotient (exact-quotient oracle 0.336) — the reverse of the raw
ratio, where the quotient scores 1.000. Never read an oracle row as if the basis
were neutral.

#### 34d. The raw and bounded mass ratio — the versions that escape the collapse

The raw ratio is a *quotient*, so §34b does not apply. Verified: residual of
`w_ratio` after the best linear fit to `[w_|dvA|, w_|dvB|]` is **0.883 / 0.923**,
against 1.1e-15 for the log form.

| target | VideoMAE | V-JEPA2 | note |
|---|---|---|---|
| raw `m_A/m_B` | 0.123 | 0.195 | Spearman 0.493 / 0.517 |
| **bounded `(m_A-m_B)/(m_A+m_B)`** | **0.540** | **0.646** | best pool `A.pre+B.pre` |
| `log_mass_ratio` | 0.536 | 0.600 | **algebraically vacuous, §34b** |

**The raw ratio is a wretched regression target** — range 0.010 to 97.8, median
3.32, so R2 is dominated by a handful of huge values while Spearman shows the rank
information is present. **Use the bounded form**: same quotient, same information,
bounded in (-1, 1), and it escapes the collapse.

#### 34e. Momentum, best-vs-best — the product is still not stored

An earlier revision claimed momentum is "not linearly accessible". **That was wrong
as stated** — `momentum_pre` reads **0.825** frame-wide, above even a quadratic
oracle (0.711). The correct comparison is best direct readout anywhere on the grid
against composition from each factor's own best cell:

| selection | model | best DIRECT | best COMPOSED | gap |
|---|---|---|---|---|
| best cell on test | VideoMAE | 0.825 (`all.pre`) | **0.934** | -0.110 |
| best cell on test | V-JEPA2 | 0.728 (`all.all`) | **0.936** | -0.208 |
| **cell picked on validation** | VideoMAE | 0.783 | **0.929** | -0.146 |
| **cell picked on validation** | V-JEPA2 | 0.616 | **0.926** | -0.310 |

The gap **widens** under honest selection, so it is not a winner's curse. And the
frame-wide readout does not transfer — split on total log mass, train on one half:

| regime | VideoMAE direct / composed | V-JEPA2 direct / composed |
|---|---|---|
| random split (ceiling) | 0.778 / 0.813 | 0.609 / 0.688 |
| train HIGH -> test LOW | **0.303** / 0.739 | **-0.265** / 0.686 |

**Beware the within-pool comparison.** At `all.pre` direct (0.825) ties composed
(0.824) — but `mass_B` decodes at **-0.627** there, so *composition* is crippled,
not direct advantaged. Always compare at a pool where the factors are decodable, or
best-vs-best.

#### 34f. Nonlinear probing does not rescue conservation — and the ingredient is missing

RBF kernel ridge, hyperparameters tuned per cell, **200-permutation grid-max null**:

| model | observed | null mean +- sd | null 95th | p |
|---|---|---|---|---|
| V-JEPA2 | 0.717 | 0.631 +- 0.050 | **0.717** | **0.060** |
| VideoMAE | 0.667 | 0.614 +- 0.056 | 0.700 | 0.200 |

A single-permutation null had given 0.550 and made this look like +0.167. The true
null grid-max averages **0.631** — the earlier draw was a low outlier.

**The nonlinear null sits 0.131 above naive chance** (0.631 vs 0.500) against
0.05-0.15 for linear grids. That gap is the per-cell hyperparameter tuning inflating
each cell before the grid max applies.

> **RULE.** The `Bin(60, 1/2)/60` derivation is valid only for a *fixed-alpha*
> linear probe; it was validated by agreeing with the measured grid-max (0.650 both
> ways). Any probe with **per-cell hyperparameter selection** needs an empirical
> multi-permutation null — one permutation estimates the null max with too much
> variance to compare against.

**And the ingredient is not there.** On v1 the violation adds the *same* velocity
kick to both balls, tilting the deflections off antiparallel — so `cos(dv_A, dv_B)`
separates the classes **perfectly**:

| class | min | median | max |
|---|---|---|---|
| conserving | -1.0000 | -1.0000 | -1.0000 |
| violating | -0.9999 | -0.7212 | **+0.9993** |

**AUC 1.000 from one visible scalar, no mass needed.** Yet the models encode it at
only **0.237 / 0.199 linear and 0.197 / 0.204 nonlinear**. They do not hold the
quantity that would make violation detection trivial — which explains the binary
null without any claim about conservation, and is robust to the linear/nonlinear
question.

#### 34g. The attentive-probe re-capture — data on disk, INSTRUMENT NOT WORKING

`--patch-sites` was already wired into `run_experiment.py`. Re-captured v1 with raw
per-token activations at four strided blocks per model:

```bash
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
    --mode capture --model videomae-base --dataset-dir datasets/probe_set \
    -o outputs/patch_videomae-base \
    --patch-sites "block.1.resid_post,block.5.resid_post,block.8.resid_post,block.11.resid_post"
# same for vjepa2-vitl with blocks 1,7,13,22 -> outputs/patch_vjepa2-vitl
```

150/150 both, no failures. **830 MB** (VideoMAE, 1568x768 per site) and **6.6 GB**
(V-JEPA2, 6144x1024). The point was to remove §20's mask leak — an attentive probe
learns its own pooling and never touches the ground-truth segmentation.

**`fit_attentive_probe` as configured does not work at n=150.** The sanity check is
what shows it — VideoMAE, best of four blocks:

| target | attentive | linear, for reference |
|---|---|---|
| `velocity_A_pre` *(sanity)* | **0.413** | **0.974** object-local / 0.577 frame-wide |
| `momentum_delta` | 0.079 | 0.517 |
| `cos(dv_A, dv_B)` | 0.207 | 0.237 |
| `momentum_conserved` (acc) | 0.600 (shuf 0.483) | 0.633 |

A trivially visible quantity reads 0.41 where a linear probe on a *frame-wide mean*
reads 0.58. **The instrument is broken, so none of its nulls are evidence.** Cause is
almost certainly capacity vs data: learned attention over 1568 tokens x 768 dims
trained on 90 scenes for 120 epochs.

**Next step, unrun:** push epochs and check whether `velocity_A_pre` recovers toward
0.9. If it plateaus at ~0.4 the probe is data-limited at n=150 and needs the
300-scene v3 set. **The captures are reusable — do not re-render or re-capture.**

### 28. Conservation through the delta vector — **RESOLVED IN §34, READ THAT INSTEAD**

> **THIS SECTION IS SUPERSEDED.** §34 settled it with 500-permutation grid-max
> nulls: `momentum_delta` decodes at 0.517 / 0.318 / 0.232 with **p = 0.000 in all
> three models**, from a pre-transformer baseline of ~0, and `|predicted dp|`
> detects violations at AUC 0.765, **p = 0.006** (VideoMAE; 0.096 / 0.106 in the
> other two). The "collapses under honest selection" worry below was an artifact of
> estimating the null from a single permutation. Keep this section only for the
> decomposition showing the R2 is about the *vector* rather than the class.

`momentum_delta` decodes at **0.517** on v1 (VideoMAE, block.9/`all.pre`, grid-max
shuffled null −0.055; V-JEPA2 0.232). The doc previously recorded it as ~0 for
CogVideoX and did not cover the encoders. Since `|dp|` is exactly 0 in all 75
conserving scenes and median 0.63 in the violating ones, that looked like a crack
in the project's central negative.

**It is not, or at least not demonstrably.** Decomposed at the winning cell: the
probe gets the Δp *direction* right (mean cosine **0.83** on violating scenes), an
oracle that knows *only* the conserved/violating class scores **0.007**, and the
class means nearly cancel because the deflection directions are uniformly
distributed. So the R² is about the vector, not about detecting violations.

The norm of the predicted vector does separate the classes, and the evidence is
genuinely mixed:

| test | VideoMAE | V-JEPA2 | CogVideoX |
|---|---|---|---|
| AUC at the fixed cell, permutation p | 0.761, **p<0.001** | 0.686, p=0.005 | 0.682, p=0.015 |
| cell selected on VALIDATION -> test AUC | 0.460 | 0.538 | 0.370 |
| grid-max test AUC vs grid-max shuffled | 0.777 vs 0.663 | 0.757 vs 0.644 | 0.751 vs 0.685 |
| balanced accuracy, grid-max vs its null | 0.670 vs 0.650 | 0.666 vs 0.650 | 0.702 vs 0.717 |

The fixed-cell permutation test is strong but that cell was itself the argmax of a
279-cell R² search. Honest validation-based selection collapses to chance — though
with only ~22 scenes in the inner validation slice, that selection is itself too
noisy to trust. **Neither direction is established.** The clean resolution is
k-fold cross-validated AUC per cell with a grid-max permutation null, or a
pre-registered cell; ~20 minutes, not yet run.

**Related, and settled: violating scenes do NOT distort the other v1 targets.**
Measured at a fixed cell, 5 splits, matched n:

| target | all 150 | conserving 75 | random 75 (mixed) | violating 75 |
|---|---|---|---|---|
| momentum_pre | 0.825 | 0.662 | 0.656 | 0.762 |
| velocity_A_pre | 0.974 | 0.948 | 0.962 | 0.960 |
| delta_v_A | 0.848 | 0.702 | 0.736 | 0.675 |
| mass_A | 0.919 | 0.802 | 0.799 | 0.799 |

Conserving-75 and random-75 agree in all 16 rows measured; the drop from 150 is
**sample size**, visible equally in all three 75-scene arms. So keeping the
violating half is free statistical power for every target except those recoverable
only *through* the conservation identity — `log_mass_ratio`, which must use
`--where momentum_conserved=true` (0.417/0.544 on all 300 vs 0.488/0.584 on 225).

### 14. Conservation on v3 is unidentifiable by construction — a *predicted* null

The v3 violation mechanism keeps `dv_A` and `dv_B` exactly antiparallel (§11).
Its own docstring states the consequence, which is easy to miss and decisive:

> a violation is **undetectable from kinematics alone**. Any antiparallel pair of
> deflections is consistent with momentum conservation under *some* mass ratio,
> namely `m_A/m_B = |dv_B|/|dv_A|`.

That was deliberate — it kills the geometry shortcut. But combine it with
`latent_mass`, where the masses are **invisible**, and an observer has no
independent handle on mass at all. Therefore **no observer, however capable, can
beat chance on `momentum_conserved` in v3.** The null is a theorem about the
dataset, not a measurement of the models.

Measured, and it lands exactly there (v3 is 225 conserved / 75 violating, so the
majority-class baseline is 0.750):

| model | best balanced acc | best raw acc | grid-max shuffled null |
|---|---|---|---|
| VideoMAE-base | **0.500** | 0.742 | 0.750 |
| V-JEPA2-ViT-L | **0.599** | 0.792 | 0.758 |

Under H0 a cell's balanced accuracy has `sd = 0.5*sqrt(1/90 + 1/30) = 0.105` on
the 120-scene test split, and `E[max of 144 iid N(0,1)] = 2.50`, so the expected
grid maximum is `0.5 + 0.105*2.50 = 0.764`. **Both observed values sit below what
selection noise alone produces.** Raw accuracy clears its shuffled null by
+0.033, under one standard error.

**Do not report this as evidence about the models.** Report it as a property of
the design.

**Also: `log_mass_ratio` must be fit on conserving scenes only.** In a violating
scene the aftermath is edited, so the label no longer equals the kinematic ratio
— median discrepancy `e^2.33 ~ 10x`. Those 75 labels are unrecoverable by any
observer and behave as noise. Confirmed: fitting on all 300 gives 0.417 / 0.544;
restricted to the 225 conserving scenes it rises to **0.488 / 0.584**. Use
`--where momentum_conserved=true`.

**One more caveat, already recorded per scene.** `identifiability` in
`labels.json` stores `min_delta_displacement_px`, how far the *less*-deflected
ball departs from its undeflected path over the post-contact window. The ratio
needs both deflections, so the smaller one sets the precision, and it shrinks as
`1/(1+kappa)` in the mass ratio. On v3: median 13.5 px, but **33 of 300 scenes
are under 2 px and 7 under 0.5 px** — at or below the tracker's ~0.1 px centroid
noise, so the ratio is unrecoverable there for information-theoretic reasons.
`select_scenes()` only does equality, so `--where` **cannot** filter on a
threshold; stratifying by identifiability needs a small change there.

### 23. The four-velocity concatenated probe — run, and it changes nothing

The natural reading of the mass-ratio experiment is "hand the probe the four
object-local velocity vectors (A and B, before and after) and ask for the ratio".
Every probe run before this one fit **one pool at a time**, so that had never
actually been tested — and `B.pre` was not even in the config's pool list. It is
now run, on both encoders, v3, 225 conserving scenes, `--tag ratio_concat`.

`probe_grid` already supported a `'+'`-joined pool spec (feature-axis
concatenation); the only thing missing was a way to ask for one, so
`run_experiment.py` grew a `--pools` flag.

**Result: the concat does not beat the best single pool.** `log_mass_ratio`,
best over blocks:

| pool spec | VideoMAE | V-JEPA2 |
|---|---|---|
| `A.pre` | 0.459 | **0.584** |
| `A.post` | 0.488 | 0.437 |
| `B.pre` (never probed before) | 0.333 | 0.492 |
| `B.post` | 0.310 | 0.447 |
| `A.pre+B.pre` | **0.536** | **0.600** |
| `A.post+B.post` | 0.435 | 0.472 |
| **`A.pre+A.post+B.pre+B.post`** | **0.520** | **0.576** |
| `all.pre` / `all.post` | 0.039 / 0.166 | 0.100 / 0.291 |

All of it still straddles §13's degraded oracle of **0.516**. Giving the probe
every ingredient of the identity at once buys nothing over one object-local pool,
which is the §13 conclusion again on the design that was supposed to escape it.

**The concat plumbing is working — the control proves it.** Δv is a *linear*
function of the four velocities, and exactly those targets jump when the four
pools are supplied together:

| target | best single pool | 4-pool concat | why |
|---|---|---|---|
| `delta_v_A` | 0.708 / 0.670 | **0.865 / 0.865** | linear in the four velocities |
| `delta_v_B` | 0.759 / 0.770 | **0.903 / 0.920** | linear in the four velocities |
| `log_mass_ratio` | 0.488 / 0.584 | 0.520 / 0.576 | a **quotient of norms** — not linear |

(VideoMAE / V-JEPA2. Single-pool column is the max over the six per-object pools.)

So the concat demonstrably adds the information it should add, and the ratio is
precisely the target it fails to help. **That is the identity trap showing its
mechanism rather than just its consequence.**

**The oracle on the user's exact proposal.** A linear readout of the four
*exact* velocity vectors — a perfect model, no representation error at all:

| oracle input (n = 225, same split protocol) | R2 on `log_mass_ratio` |
|---|---|
| **exact four velocity 2-vectors (8 features)** | **−0.016** |
| the same 8 plus all pairwise products (quadratic) | 0.346 |
| exact Δv vectors | −0.020 |
| exact \|Δv\| magnitudes | 0.716 |
| exact log\|Δv\| | 1.000 |
| magnitudes degraded to the models' own fidelity (0.71) | 0.516 |

**A linear probe on perfect velocity components cannot reach the mass ratio at
all.** So the observed 0.52-0.58 is not the probe composing four velocities — the
representation must already hold something magnitude-like, and the probe is
reading *that*. This also means a negative result from this design would have
been uninterpretable: the readout, not the model, would have been the binding
constraint.

**The mask-leak floor had to be recomputed, and it is clean.** §20 found
`log_mass_ratio` safe *because* a single pool's mask holds neither both objects
nor both phases — an argument that dies for a spec whose mask holds exactly that.
Measured with `scripts/common/mask_only_floor.py` (new, no GPU, rebuilds the pooling
geometry from the capture metadata):

| target, 4-pool concat mask only | VideoMAE | V-JEPA2 |
|---|---|---|
| `log_mass_ratio` | 0.006 / −0.007 | −0.507 / −0.432 |
| `delta_v_A` | 0.075 / 0.238 | 0.017 / −0.018 |
| `delta_v_B` | 0.003 / 0.178 | **0.430 / 0.341** |

(normalised weights / raw occupancy — see below. `delta_v_B`'s 0.92 clears its
0.43 floor comfortably, but quote it *with* the floor.)

Two things about that script worth knowing:

* It reports **two** floors. The pooling rows the recorder actually uses are
  normalised to sum 1, which **removes the mask's area** — and area is how
  `radius_A` leaks. Normalised vs occupancy on VideoMAE `A.pre`: −0.020 vs
  **0.252**. §20's published table is the *occupancy* convention; the script
  reproduces every one of its VideoMAE and V-JEPA2 numbers exactly
  (0.252 / −0.076, −2.470, 0.092, 0.356, ...), which is what validates it.
  Occupancy is the stronger control — use it.
* The `all.*` masks are identical in every scene and score −0.001 to −0.032,
  the same built-in sanity check §20 specifies.

**What this does not settle.** The ratio is still an exact function of visible
kinematics, so none of this is evidence about a represented mass concept (§12).
The one thing the concat *is* good for is what §16 used it for implicitly:
`delta_v_A/B` at 0.87-0.92 is the cleanest per-object kinematic readout measured
in this project, and it is the natural readout for any future patching work that
needs to score a *change* across the collision.

### 27. The pool list was incomplete, and it cost a published number

`configs/momentum_probe.yaml` lists **six** pools while every capture stores
**nine**. `B.pre`, `A.all` and `B.all` were never in a standard probe run — `B.pre`
appeared only in §16's one-off script and §23's concat run. The omission has no
recorded justification; the config comment only argues for including the *post*
pools.

Re-running v1 with all nine plus the concat specs (`probes_gridfull.json`,
seconds, no GPU) moves several numbers:

| target | old best (6 pools) | new best | pool |
|---|---|---|---|
| `velocity_B_pre`, VideoMAE | 0.740 (from **A**.pre) | **0.959** | **B.pre** |
| `velocity_B_pre`, V-JEPA2 | 0.686 (from **A**.pre) | **0.939** | **B.pre** |
| `log_mass_ratio`, VideoMAE | 0.752 | **0.910** | A.post+B.post |
| `log_mass_ratio`, V-JEPA2 | 0.813 | **0.967** | A.post+B.post |
| `mass_A`, V-JEPA2 | 0.937 | 0.960 | **A.all** |
| `radius_A`, V-JEPA2 | 0.967 | 0.974 | **A.all** |

**The `velocity_B_pre` correction matters most**: it was being reported at
0.69-0.74 read out of **A's** tokens because B's own pre-collision pool was not in
the list. B is encoded just as sharply as A (0.94-0.96 vs 0.96-0.97), so the
apparent A/B asymmetry in every earlier table was an artifact of the pool list,
not of the models. This sharpens §16's own-object diagonal.

`A.all`/`B.all` win **6 of 12 rows** — every mass and radius target — which makes
sense: those are static properties, so averaging the whole clip is pure noise
reduction. **Probe all nine pools by default.** The config's six are now a bug.

The full 12-spec grid for v1 and v3, and the raw per-block `log_mass_ratio`
matrices, were produced in this session; regenerate with `--pools` (§23) rather
than trusting any six-pool summary.

Two structural facts visible only in the raw v3 matrices, both worth keeping:

* **Frame-wide pools are dead for relational targets.** `all.pre` never exceeds
  0.100 for `log_mass_ratio` across all 37 blocks of both models, while
  object-local pools reach 0.49-0.60.
* **Depth profiles differ by architecture.** VideoMAE peaks at block 7 of 12 and
  decays; V-JEPA2 rises monotonically to block 16-17 of 24 and plateaus. Neither
  has anything at `embed.out` (-0.05 to 0.07), so the ratio is *built*, not in the
  pixels. At blocks 0-3, `A.post`/`B.post` already read 0.09-0.36 while
  `A.pre`/`B.pre` sit at ~0.00 — the pre pools only catch up by block 5-7, which
  is §4's bidirectionality developing with depth.

### 5. Steering — negative, and consistent with (3)

Directions are built by `build_directions()` (seven estimators, all unit-length)
and written back with `ActivationSteer` at `block.N.resid_post`, which is the
same site the reference paper probes and perturbs.

**Geometry of the candidates** (this table is the useful artefact):

| direction | cos vs `probe` | cos vs `mass_A` |
|---|---|---|
| `probe` (ridge weights, rescaled by 1/sd) | 1.00 | **0.35–0.42** |
| `residual` (mass/radius projected out) | 0.91–0.95 | **0.007–0.063** |
| `diff_means` | **0.03–0.14** | 0.013–0.051 |
| `temporal` (phase-mean removed) | 0.23–0.32 | 0.016–0.085 |

Two things worth keeping. The mass confound is **visible in the geometry** — the
naive probe direction is 0.35–0.42 aligned with the mass readout — and
residualising removes it while keeping 91–95% of the direction. And `probe` and
`diff_means` are **nearly orthogonal while both decode at R2 ~0.85**, which is
expected at `d >> n`: under `x = mu + y*a + eta`, difference-of-means estimates
the generative direction `a` and ridge estimates the discriminative `Sigma^-1 a`,
and with a rank-<=150 sample covariance in 1920 dimensions those can be almost
perpendicular. Never assume the two agree.

**Perturbation size.** `||dh||/||h|| ~ alpha/sqrt(D) = alpha/43.8`, which
predicts the damage threshold and was confirmed: alpha=2 -> 4.6% (nothing
happens), alpha=20 -> 46% (ball radius shifts 2.3 px), alpha=40 -> 91% (video
destroyed, 0 frames tracked).

**Result of the 90-run sweep** (3 blocks x 5 directions x 2 alphas x 3 scenes),
median momentum change, fitted directions vs controls at matched alpha:

| block | alpha | fitted | control |
|---|---|---|---|
| 6 | 10 | 6.46% | 4.63% |
| 15 | 10 | 1.00% | 1.34% |
| 15 | 20 | 2.65% | 9.76% |
| 24 | 10 | 1.14% | **73.59%** |
| 24 | 20 | 2.53% | 4.56% |

Controls match or beat the fitted directions everywhere except block 6, where
every condition had 4–8 px of radius damage. Restricting to runs where appearance
survived (`dR < 0.5 px`) leaves 17 of 90, and there the **shuffled-label control
produces the largest momentum change** (median 1.34%, against `residual` 0.99%
and `diff_means` 0.31%, with the round-trip floor at 0.53%).

**No candidate direction produces a clean, control-beating momentum edit.** The
window between "too weak to do anything" (<5% perturbation) and "destroys the
video" (>45%) contains no clean effect. Given (3), that is what should have been
expected: there is no independent factor to move.

### 7. Generative conservation test — not answerable on this checkpoint

`runner.inpaint()` holds latent frames 0–5 (strictly pre-contact) to a real scene
and regenerates 6–12, re-imposing the held frames at every step correctly
renoised. This is the only setting where conservation is **identifiable**: in a
fixed clip `delta-p` is variance *we* authored, but here the aftermath is a free
variable the model controls.

It does not work, in any configuration tried:

| setup | generated-region pixel variance | blobs @t24 | outcome |
|---|---|---|---|
| unguided, empty prompt, t=999 | 381.9 | 3–5 | structured but **radii off by 7–36 px** |
| prompt, guidance 1.0 | 27.4 | 1 | degenerate |
| prompt, guidance 2.0 / 3.0 | 14.2 / 17.0 | 1 | degenerate |
| prompt, guidance 6.0 | 9.1 | 1 | degenerate (0 blobs by t40) |
| any prompt, t=599 | 19.5 | 1 | flat by construction |

*(source region variance for comparison: 150.8)*

**CogVideoX-2b cannot inpaint a coherent two-ball continuation here.** It fails
object permanence — a prerequisite — before conservation becomes measurable, so
the momentum errors it produces (41%, 511%, ...) are uninterpretable and must not
be reported as violations. The `dR` appearance gate is what caught this.

Two counter-intuitive findings inside that table, both worth not re-deriving:

* **CFG made it worse, and the prompt is what kills it** — not the guidance
  scale. Text conditioning at *any* strength collapses the generation and more
  guidance collapses it further. The likely cause is a domain gap: Kubric renders
  of two small spheres on flat grey are far outside CogVideoX's real-video prior,
  so a prompt pulls the model toward its own notion of the scene and fights the
  held frames. The empty prompt is the closest thing to "no opinion".
* **`t_start` must sit at the top of the ladder.** The `max_timestep=599`
  truncation that inversion *needs* is actively wrong for generation: at t=599,
  alpha=0.215, so the model expects a signal component pure noise does not have,
  reads it as `x0 ~ 0` and returns the dataset mean.

**Implication.** This checkpoint is not generatively competent on this domain, so
generation-time questions — steering with a visible effect, conservation under
generation — cannot be posed on it at all. Probing still works because it rides
on *inverted real videos* rather than on the model's generative prior. It also
qualifies every result above: "CogVideoX represents momentum" means "in synthetic
Kubric renders", a distribution it was never trained on.

### 37. `momentum_delta` — within-class check and a reproduction gap

Run 2026-08-21, VideoMAE, `probe_set` v1, cell `all.pre` / block 9, standardised
features, fixed alpha 10, `Delta-p` computed as `total_momentum[22] - total_momentum[18]`.

**Class geometry.** Conserving (n=75) mean `|Delta-p|` = 1.7e-16; violating (n=75)
mean 0.818, range 0.016-2.572. **Between-class share of the variance in Delta-p:
0.005.**

| target | 1 split (seed 0) | mean of 20 splits |
|---|---|---|
| `momentum_delta`, all 150 | +0.205 | +0.261 |
| `momentum_delta`, violating only (n=75) | +0.229 | +0.114 |
| `\|Delta-p\|` magnitude, violating only | **+0.798** | **+0.616** |
| Delta-p direction (cos, sin), violating only | +0.020 | +0.120 |
| the binary label fit as a regression, all 150 | -0.844 | -0.783 |

**Reproduction gap.** §34 reports `momentum_delta` at **0.517** at this cell.
Re-fitting it here gives **0.205** on one split and **0.261** averaged over 20.
The pipelines differ — this one standardises features, fixes alpha at 10, and takes
`total_momentum` at frames 18/22 — so this is "does not reproduce under a different
protocol", not "is wrong". **Resolve before the number goes in a paper.**

---

### 62. THE VALID-COLLISION SUBSPACE FROM PRE-COLLISION CLIPS, AND HOW VIOLATIONS LEAVE IT — `conservation_set_1000`

Run 2026-10-23 on the A100 box, VideoMAEv2-giant (40 blocks) and V-JEPA2-ViT-L (24
blocks). Numbered §62 to stay clear of the register's §47–§61 (`docs/register_gap.md`);
this is the register's §47 (valid-collision subspace on truncation rungs) and §49
(subspace membership, `--fit-honest`) rebuilt and extended on the 1000-clip set.

**The set.** 500 seeds x {honest, violating} twins, e pinned to 1.00 / 0.15, `mixed` /
`same` material: 200/50/200/50 pairs per cell. **997 of 1000 clips are usable — three
violating clips (`elastic_mixed_violating_seed255/262/264`) have only 41–42 of 49 frames
on disk despite carrying `_COMPLETE`**; `load_frames` refuses them. 497 complete pairs.

**Captures.** Full clips (`outputs/cons_<model>/`) and resample-truncation rungs
`--resample-to 10,14,17,19,20,22,25,30,38,48` (`outputs/construnc_<model>_a100/`). Contact
is at frame 20, so **k <= 20 is a genuinely pre-collision clip** — unlike the `.pre`
pools of a full clip, which have attended to the aftermath (§4). Only the `*.all` pools
are exact on resampled rungs. The `_a100` suffix keeps these apart from the partial
`outputs/construnc_*` the other box captured on an RTX 5090 (§58: never mix capture GPUs).
Twins must encode identically before contact and do: max relative activation difference
**0.1–0.7% at k <= 20**, jumping to **6–27% at k >= 22**.

**The geometry.** For a smooth frictionless pair the legal outcome `(dv_A, dv_B)` lies on
ONE line per scene, `J (-n/m_A, +n/m_B)`. The scene's basis {normal impulse, tangential
impulse, drift along n, drift across n} spans R^4; only the first is allowed. A
violation in this set is a **pure common drift along n** (verified: the true twin
difference has every other coefficient below 1e-14 of the drift).

> **LABEL BUG, fixed in the script and not in the library.** `pre_state_from_labels`
> prefers `contact_normal_analytic`, which on this set sits up to **0.66 deg** off the true
> impulse direction: honest outcomes then miss their own legal line by up to 2% and
> score membership up to 0.012 instead of 0. The solver's `contact_normal` agrees with
> `dv_A`'s direction to 1e-6 deg and makes the honest legal outcome exact (5e-15). It is
> recorded at the solver event, before any edit, so it is also every violating clip's
> pre-contact normal. Same bug as §46a, second instance.

#### 62a. PART A — what the pre-collision representation holds

Decode the legal outcome from each rung with a readout fit on honest clips only
(grouped 5-fold by seed, alpha nested), best cell by honest R2 (always `A.all+B.all`).
Two brackets: the **ceiling** is a perfect observer of everything except restitution,
which is latent until impact (it is pinned per cell, not per material) — the legal outcome
at the mean e, **R2 0.930**. The **shuffled-residual reference** re-deals the decoded
outcome's own errors across scenes (200 draws): same error sizes, no relation to each
scene's line. "Structured" = the decoded outcome stays on its legal line more than that.

| k (contact = 20) | giant R2 | on-line, obs / shuffled (p) | V-JEPA2 R2 | on-line, obs / shuffled (p) |
|---|---|---|---|---|
| 10 | 0.636 | 0.792 / 0.831 (1.00) | 0.658 | 0.815 / 0.847 (1.00) |
| 14 | 0.658 | 0.818 / 0.830 (0.86) | 0.678 | 0.824 / 0.831 (0.73) |
| 17 | 0.665 | 0.800 / 0.825 (0.98) | 0.685 | 0.845 / 0.836 (0.19) |
| 19 | 0.675 | 0.821 / 0.820 (0.48) | 0.709 | **0.868 / 0.838 (0.005)** |
| 20 | 0.684 | 0.835 / 0.827 (0.22) | 0.710 | **0.875 / 0.839 (0.005)** |
| 22 | 0.769 | **0.880 / 0.861 (0.02)** | 0.758 | **0.888 / 0.858 (0.005)** |
| 25 | 0.837 | **0.915 / 0.892 (0.005)** | 0.830 | **0.920 / 0.889 (0.005)** |
| 30 | 0.878 | **0.930 / 0.913 (0.005)** | 0.880 | **0.932 / 0.916 (0.01)** |
| 48 | 0.892 | 0.934 / 0.926 (0.02) | 0.900 | 0.937 / 0.926 (0.005) |

("on-line" = median share of the decoded outcome's energy on the legal line, 1 - membership^2.)

* **Before contact both models hold the LINE and not the POINT on it.** The decoded
  restitution is flat — median implied e 0.44 (true 0.15) vs 0.48 (true 1.00) on the giant
  at k = 20 — and corr(implied e, true e) is +0.01 to +0.09 at every k <= 20, rising to
  **+0.64 / +0.77 at k = 48**. A theorem-null, measured: e is invisible pre-contact.
* **V-JEPA2's pre-collision prediction is shaped by the collision geometry; the giant's is
  not.** V-JEPA2 is more on-line, less tangential (0.192 vs 0.237) and has less momentum
  drift (0.843 vs 1.028) than shuffled errors, all p = 0.005, at k = 19 and 20 — and not at
  k <= 17. The giant passes only the drift statistic before contact (p = 0.005 at every k);
  on-line and tangential are at or BELOW shuffled (p 0.16–1.00) until k = 22.
* **Fidelity is not the difference** — both sit at R2 0.68–0.71 at k = 20 against the 0.93
  ceiling. The difference is where the error goes.
* 200 draws floor p at 0.005; across 10 rungs x 3 statistics treat isolated p = 0.02–0.05
  cells as noise. V-JEPA2's k = 19/20 hold on all three statistics together.

#### 62b. PART B — how violating collisions deviate (full clips)

Decode the OBSERVED outcome with a readout fit on honest clips only, so it has never seen a
violation; cell chosen by honest R2 only (`A.post+B.post`, giant block 26 / V-JEPA2 block 16;
honest R2 0.856 / 0.833, violating outcomes decoded at 0.826 / 0.776). The detection
statistic is membership (§49: distance to the legal line / |outcome|). **The paired sign
test over twins is the null** — group permutation is a no-op on a balanced twin design
(§46d).

| | VideoMAEv2-giant | V-JEPA2-L |
|---|---|---|
| membership AUC, true outcome (ceiling) | 1.000 | 1.000 |
| **membership AUC, decoded outcome + true masses** | **0.628** | **0.575** |
| paired: violating twin further off the line | **330/497, p = 1e-13** | **309/497, p = 3e-8** |
| decoded outcome + the MODEL'S OWN decoded masses | 0.629 (331/497) | 0.576 (310/497) |
| true outcome + model's decoded masses (mass R2 0.988 / 0.992) | 0.959 | 0.975 |
| by cell: elastic-mixed / elastic-same / inelastic-mixed / inelastic-same | 0.64 / 0.56 / 0.67 / 0.52 | 0.57 / 0.59 / 0.59 / 0.56 |

**The direction of the deviation** — decoded twin difference against the true one (a pure
drift along n), medians with bootstrap CIs:

| | giant | V-JEPA2 |
|---|---|---|
| cos(decoded, true) | **0.61** [0.58, 0.65] | **0.56** [0.52, 0.60] |
| share of the drift recovered | 0.49 [0.45, 0.55] | 0.35 [0.31, 0.41] |
| read as an impulse change instead | -0.002 [-0.017, 0.009] | +0.002 [-0.009, 0.016] |
| tangential | +0.007 | -0.007 |
| implied-e shift, violating - honest | -0.002 [-0.029, 0.017] | +0.005 [-0.014, 0.034] |

* **The deviation is represented as what it is** — a momentum drift — and not as a
  different collision (impulse change and implied-e shift are zero within their CIs). It is
  under-recovered (35–49% of its size), which is ridge shrinkage plus the representation's
  own fidelity.
* **The models' own masses are as good as the true ones** for this: 0.629 vs 0.628. The
  bottleneck is decoding the outcome precisely enough, not the mass.
* **In activation space** the twin difference loads on the rank-4 span of the honest
  outcome readout **2.2x (giant) / 1.6x (V-JEPA2)** more than on random rank-4 subspaces
  (p = 0.005): the edit moves the representation preferentially along outcome-coding
  directions.
* **The models do not EXPECT the violation.** Distance between the full-clip decoded outcome
  and the k = 20 pre-collision prediction separates violating from honest at AUC **0.536**
  (288/497, p = 2e-4) / **0.509** (258/497, p = 0.21). The pre-collision prediction is too
  coarse (R2 0.68–0.71) relative to violations of median size 0.27–0.49 of |p_pre|.

#### 62c. THE CLAIM, and what it is not

Before contact, both encoders hold the collision's valid line but not where on it the
outcome lands (restitution is unknowable, and they correctly do not know it). V-JEPA2's
prediction respects the geometry of valid collisions in the last frames before contact;
the giant's does not until the aftermath is on screen. After contact, a violation is
represented as a momentum drift in the right direction, at about half its size, and the
models' own masses suffice to certify it at AUC 0.58–0.63.

**This is §42f's ceiling and not more.** The deviation is in the visible aftermath; the
readout and the legal line (true or decoded masses) do the comparison. Nothing here shows
the model APPLIES conservation, and the expectation-gap row says its own pre-collision
prediction barely registers the violation — consistent with §46f-iii's momentum VoE null.

#### 62d. Files and commands

```bash
# captures (A100 box, from a local-disk copy of the set; see Reference for the env)
python scripts/pipeline/capture_encoder.py --model <m> --dataset datasets/conservation_set_1000 \
    --out outputs/cons_<m>/momentum_probe/activations --keep-going
python scripts/pipeline/capture_encoder.py --model <m> --dataset datasets/conservation_set_1000 \
    --out outputs/construnc_<m>_a100/activations --resample-to 10,14,17,19,20,22,25,30,38,48 --keep-going
# analysis, CPU only, ~6 min per model at 48 workers
python scripts/momentum/momentum_valid_subspace.py --model <m> \
    --full outputs/cons_<m>/momentum_probe/activations --trunc outputs/construnc_<m>_a100/activations
```

* `scripts/momentum/momentum_valid_subspace.py` — Parts A and B; `scripts/common/_fast_cv.py` — the same
  grouped-CV ridge as `stats.cv_predict` with every alpha read off one eigendecomposition
  per split (matches it to ~1e-6; constant features zeroed per §44e).
* `docs/artifacts/momentum_valid_subspace_{videomaev2-giant,vjepa2-vitl}.json`, logs in
  `docs/artifacts/logs_2026-10-23/`.

---

### 64. §62 WITHOUT LABELS — a pooled conditional subspace, and why it is not the legal line

Run 2026-10-23/24 on the same captures as §62 (`outputs/cons_<m>`, `outputs/construnc_<m>_a100`),
same 497 twin pairs. No readout, no physics label in the construction: fit an
activation-to-activation map `f: pre -> post` on honest clips (pre = rung `k*/`
`A.all+B.all`, post = full-clip `A.post+B.post` or `all.post`; PCA 64 each side, ridge,
alpha by inner grouped CV on reconstruction error), PCA the OUT-OF-FOLD honest residuals,
pick the rank by parallel analysis, and score a held-out clip by Mahalanobis distance of its
residual outside the top-r directions. Cell chosen by honest reconstruction R2 (label-free).
Labels used only to evaluate. Paired sign test over twins; grid-max null flips which twin is
violating, per pair.

| | giant k=10 | giant k=20 | giant k=25 | V-JEPA2 k=10 | V-JEPA2 k=20 | V-JEPA2 k=25 |
|---|---|---|---|---|---|---|
| chosen cell | `all.post` b16 | `all.post` b16 | `all.post` b16 | `A.post+B.post` b10 | b10 | b14 |
| recon R2 (honest) | 0.672 | 0.714 | 0.783 | 0.675 | 0.719 | 0.802 |
| residual rank (PA, per fold) | 3–6 | 2–6 | 2–6 | 5–11 | 7–10 | 8–11 |
| **AUC, off-subspace** | **0.551** | **0.560** | **0.561** | **0.545** | **0.550** | **0.541** |
| paired | 278/497, p 5e-3 | **291/497, p 8e-5** | 300/497, p 2e-6 | 270/497, p 0.03 | 276/497, p 8e-3 | 266/497, p 0.06 |
| same cell, NO pre clip (plain PCA), r=0 | 0.537 | 0.537 | 0.537 | 0.528 | 0.528 | 0.517 |
| top residual dir reads e (AUC) | 0.670 | 0.537 | 0.541 | 0.533 | 0.510 | 0.531 |
| twin diff share in span(U_r) / random | 0.41 / 0.05 | 0.30 / 0.04 | 0.40 / 0.07 | 0.24 / 0.13 | 0.27 / 0.15 | 0.35 / 0.13 |

Grid-max paired fraction 0.60–0.62 against null 95th 0.567–0.575 on every rung, both models.
Twins encode identically pre-contact (max rel diff 0.07% V-JEPA2 / 0.3% giant at k = 20).
Giant rung trend for the record: k = 14 0.554 (290/497), 17 0.558 (292/497, top dir ~ e at
0.775), 19 0.547, 22 0.563 (299/497).

* **Weak detection, about the size of §62's supervised readout** (0.63 / 0.58 there with
  true masses; 0.56 / 0.55 here with nothing). Conditioning on the pre-collision clip buys
  +0.02–0.03 AUC over a plain PCA of valid post-collision activations at the same cell.
* **Rank does not matter** (r = 0, 1, 8, PA all within 0.01). The score is "any residual",
  not "residual off a valid subspace". The residual spectrum is flat (top direction 10–16%).
* **The violation moves the representation ALONG the residual directions, 2–10x more than a
  random subspace.** Opposite of what an off-subspace score needs.
* **The top residual direction is NOT a stable restitution axis.** It reads e at 0.51–0.78,
  inconsistent across rungs.

**WHY — THE DESIGN CANNOT GIVE A PER-SCENE SUBSPACE WITHOUT LABELS.** Every seed has ONE
honest future: e is pinned per cell and never varied within a seed. §62's legal line
`J (-n/m_A, +n/m_B)` is 1-D per scene, traced out by e, and §62 got it from the labels. Pooled
across scenes, the honest residual is mostly the pre->post map's own error, with a binary e
somewhere inside it. So this is a conditional anomaly score, not a recovered valid subspace.
**Do not report it as "the unsupervised valid-collision subspace".**

**What would make it work: several honest futures per pre-clip.** Render each seed at 3–4
restitutions with a bit-identical pre-contact segment, plus the violating twin. Each scene's
valid subspace is then `span{Q_e - Q_e'}` over its own honest clips, label-free and per scene.
Score the violating clip's distance from that affine span against the honest clips'
leave-one-out distance. `energy_voe_pilot` already has 2 legal futures per seed (e = 1,
1/1.5; momentum conserved) and can test the construction, but has no momentum violations.
The full set is ~200 seeds x 5 clips, ~21 h at §42e's throughput.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/momentum/momentum_unsup_subspace.py --model <m> \
    --full outputs/cons_<m>/momentum_probe/activations --trunc outputs/construnc_<m>_a100/activations --workers 56
```

* **Set `OMP_NUM_THREADS=1`.** Without it the workers oversubscribe BLAS on the 120-core box
  and a 2-site smoke test ran >10 min instead of 108 s.
* Loading is the bottleneck: ~3.5 MB/s off `/workspace` (geesefs), ~40 min for the giant's
  captures. Fit is 25 / 43 min (V-JEPA2 / giant) at 56 workers.
* The unconditioned control's label-free cell choice is degenerate (recon R2 = 0 everywhere,
  picks `embed.out`); compare it at the conditioned cell, as in the table.
* `docs/artifacts/momentum_unsup_subspace_{videomaev2-giant,vjepa2-vitl}.json`, logs in
  `docs/artifacts/logs_2026-10-23/`.

---

## Mass

### 10. The v3 `latent_mass` arm — what it is, and that it worked

`generate_scene.py` grew an `--arm` flag with three values (`ARMS`):

* `legacy` — v1. Two densities `{1.5, 6.0}` as visually distinct materials,
  **radii sampled independently** per sphere.
* `latent_mass` — **radii equal within a scene** (`r_a = r_b`), both spheres
  rendered with the same neutral albedo (0.22) whatever their density. Density
  log-uniform over `(0.2, 20.0)`, with the *ratio* drawn first, log-uniform over
  `+-log(100)`, and the geometric mean absorbing what is left.
* `visible_mass` — identical to `latent_mass` except albedo ramps log-linearly
  with density. **Implemented, never rendered.** See "What to do next" for why
  it is no longer worth the 4 GPU-hours.

Equal radii is the load-bearing detail. With `m = rho*(4/3)*pi*r^3` and
`r_A = r_B`, the mass ratio collapses to a **pure density ratio**:

```
m_A / m_B = rho_A / rho_B          (radii cancel exactly)
```

so no size cue can shortcut it. In v1, by contrast, `m_A/m_B` carries a factor
`(r_A/r_B)^3`, and radius decodes at 0.975 — v1's "visible mass" is visible
mostly *through size*.

**The arm worked.** Measured on v3 (300 scenes, both encoders), against v1:

| target | v1 (mass visible) | v3 (mass latent) — VideoMAE / V-JEPA2 |
|---|---|---|
| `mass_A` | 0.934 | **0.281 / 0.268** |
| `momentum_pre` | 0.825 | **0.540 / 0.461** |
| `radius_A` | 0.975 | 0.993 / 0.990 |
| `velocity_A_pre` | — | 0.986 / 0.984 |

Mass went from nearly perfectly readable to barely readable, and momentum
degraded with it — exactly what must happen when `p = m*v` loses access to `m`.
Radius and velocity stay pinned near 1.0, so nothing in the pipeline broke.

### 11. `probe_set_v2_latent` is superseded — the antiparallel leak

v2 is the same 300 seeds and the same `latent_mass` arm as v3, but with the
**first** violation mechanism, which added the *same* velocity kick to both
balls. That breaks the momentum sum, but it also breaks the geometry: in a
frictionless collision the impulse is along the contact normal, so `dv_A` and
`dv_B` are exactly antiparallel, and a common added vector tilts them.

Measured on the two sets:

| set | `cos(dv_A, dv_B)`, violating scenes | leak |
|---|---|---|
| v2 | −0.9974 to **+0.9944**, 75/75 off −1 | thresholding one scalar separates the classes at **balanced accuracy 1.000** |
| v3 | −1.000000 for **74/75** | none |

So on v2 a probe could "detect violations" by pure geometry with no notion of
mass or momentum anywhere. **Do not probe v2 for conservation.** Its 300-scene
captures for both encoders still exist (`outputs/v2_*`, 192 MB) and are of a
flawed set; the pixels for 225 of its 300 scenes were deleted, only the 75
violating ones remain on disk.

`apply_violation()` now scales the two impulse *magnitudes* instead:

```
dv_A' = (1 - delta) * dv_A          dv_B' = (1 + delta) * dv_B
```

Both stay scalar multiples of the original antiparallel pair, so
`cos(dv_A', dv_B') = -1` exactly, while
`m_A dv_A' + m_B dv_B' = 2*delta*J != 0`. `delta` is clipped to
`MAX_VIOLATION_ASYMMETRY = 0.9` so `(1 - delta) > 0` and ball A never reverses.

**One residue:** `scene_00299` of v3 has `cos = +0.240`, the only leak in 300.
The clamp should prevent sign reversal, so that scene indicates a path the clamp
does not cover. One scene in 300 cannot move a probe, but it is unexplained.

### 13. v3 results, and the oracle that explains them

> **Two numbers below need §20's mask-only floor.** `radius_A` (0.992 / 0.986)
> sits on a 0.252 / 0.094 floor — the pooling mask's *area* encodes the sphere's
> size — and `delta_v_B` (0.704 / 0.683) sits on a 0.356 floor for V-JEPA2, since
> B's post-contact mask partly encodes its own deflection. Both margins survive
> comfortably. **`log_mass_ratio` is clean** (floors −2.470 / −0.024): the ratio
> needs both objects across both phases, and a single A-pool mask holds neither.
> It was safe for the same structural reason that makes it uninteresting.

Probes on the 225 **conserving** scenes only (see §14 for why violating scenes
must be excluded from this target):

| target | VideoMAE | V-JEPA2 | note |
|---|---|---|---|
| `log_mass_ratio` | **0.488** | **0.584** | grid-max nulls −0.093 / −0.073 |
| `log_density_scale` | 0.025 | 0.031 | unrecoverable by anyone -> **control passes** |
| `delta_v_A` | 0.708 | 0.670 | the constituents |
| `delta_v_B` | 0.704 | 0.683 | |
| `log_speed_change_B` | 0.623 | 0.575 | |
| `radius_A` | 0.992 | 0.986 | appearance, trivially present |

Two things to read off this.

**The ratio decodes no better than the pieces it is made of.** In VideoMAE the
ratio (0.488) reads *worse* than `delta_v_A` (0.708), `delta_v_B` (0.704) and
even `log_speed_change_B` (0.623). An explicitly represented mass-ratio feature
should survive at least as well as its own noisy constituents. This one does not.

**The oracle settles it.** Ask what a linear readout of *nothing but the two
deflection magnitudes* achieves — no mass, no momentum, no conservation. Same
split protocol (`train_fraction 0.6`, `split_seed 0`), n = 225:

| what is available to the oracle | R2 on `log_mass_ratio` |
|---|---|
| exact `norm(dv_A)`, `norm(dv_B)` | 0.716 |
| exact `dv` **vectors** (no magnitudes) | −0.020 |
| **the same norms, degraded to the fidelity the models have (~0.71)** | **0.516** |
| exact **log**-magnitudes | 1.000 |
| observed probe, VideoMAE | 0.488 |
| observed probe, V-JEPA2 | 0.584 |

The models decode `delta_v_A`/`delta_v_B` at 0.67–0.71. Feed magnitudes of
exactly that quality into a linear map and you get **0.516** — and the two
observed probes straddle it (0.488, 0.584).

**So the mass-ratio result is quantitatively accounted for by "the model tracks
two deflection magnitudes imperfectly, and the probe divides them."** There is no
residual requiring a represented mass or conservation concept.

Contrast this with the one genuinely positive result in the whole project (§3):
for momentum, probe 0.805 against an oracle of 0.695 — there the probe **beat**
what the true factors linearly afford. Here it does not. That asymmetry is the
cleanest summary of what was and was not found.

*Method caveat:* the degraded oracle injects Gaussian noise on magnitudes, which
only approximates the models' error structure, and at n = 225 the rows jitter
(fidelity 0.50 scored 0.424, above fidelity 0.60's 0.413). Directionally solid,
not a precise identity. The reproduction script is not committed — it is 20 lines
against `labels.json` plus `fit_probe`, described fully by this section.


---

## Velocity and collision dynamics

### 16. Object addressability — how the two objects are actually encoded

This is the multi-object representation question, and unlike everything in §12 it
is **identity-trap-free**: velocity is directly observable, and the question is
not "does the model have a concept" but *where in the representation each
object's information lives*. It needs no new capture — all 9 region x phase pools
are already stored.

Decode each sphere's pre-contact velocity from each sphere's own token pool
(best over blocks, v3, 150 scenes):

| pool the probe reads | `velocity_A_pre` | `velocity_B_pre` |
|---|---|---|
| **V-JEPA2** A.pre | **0.984** | 0.801 |
| **V-JEPA2** B.pre | 0.827 | **0.968** |
| **VideoMAE** A.pre | **0.986** | 0.825 |
| **VideoMAE** B.pre | 0.847 | **0.979** |

**Objects are selectively but not exclusively represented.** The diagonal wins by
0.14-0.22 in both models, so there is a genuine own-object advantage — but the
off-diagonal sits at 0.75-0.83, meaning each sphere's tokens carry most of the
*other* sphere's velocity as well. The effect replicates across two different
architectures and training objectives.

The full matrices, both models, best over blocks (150 scenes, v3):

| pool | `velocity_A_pre` | `velocity_B_pre` | `delta_v_A` | `delta_v_B` |
|---|---|---|---|---|
| **V-JEPA2** A.pre | 0.984 | 0.801 | 0.738 | 0.707 |
| B.pre | 0.827 | 0.968 | 0.642 | 0.718 |
| A.post | 0.681 | 0.523 | 0.644 | 0.602 |
| B.post | 0.592 | 0.628 | 0.541 | 0.706 |
| *all.pre* | *0.548* | *0.346* | *0.199* | *0.271* |
| *all.post* | *0.516* | *0.287* | *0.159* | *0.282* |
| **VideoMAE** A.pre | 0.986 | 0.825 | 0.709 | 0.604 |
| B.pre | 0.847 | 0.979 | 0.544 | 0.685 |
| A.post | 0.714 | 0.422 | 0.679 | 0.610 |
| B.post | 0.519 | 0.644 | 0.510 | 0.702 |
| *all.pre* | *0.548* | *0.391* | *0.200* | *0.298* |
| *all.post* | *0.535* | *0.308* | *0.194* | *0.288* |

> **REFRAMED BY §39e.** Frame-wide pools do not *lose* the objects, they
> **superpose** them: `v_A + v_B` reads 0.878 in `all.pre` against 0.389/0.152 for
> the parts, and `v_A - v_B` reads -0.353. A mean over tokens of an
> object-agnostic per-token code is a sum, and a sum cannot be un-mixed. The
> practical advice below (steer object-local) is unchanged.

**Object-local pooling is not a refinement, it is the whole signal.** Frame-wide
`all.*` pools lose most of it — `velocity_A_pre` falls 0.984 -> 0.548 and
`delta_v_A` collapses 0.738 -> 0.199, in both models. This is the quantitative
version of the "Pooling is by segmentation mask, not frame-wide mean" design note
below, and it has a direct consequence for steering: **a steer must be written to
object-local tokens.** A frame-wide direction spends most of its norm where the
quantity is not represented, which is the §5 failure mode (damage without
control) in a new guise.

**This makes a falsifiable prediction for steering**, which is the point of
measuring it: if A-local tokens carry B's velocity at 0.80, then a steer applied
through A-local tokens should move B nearly as much as A. Per-object
controllability should be poor, and poor *for a reason visible in the
representation before any steering is run*. That is a far better-posed steering
hypothesis than the one CogVideoX was tested against in §5.

> **The prediction was run and is REFUTED — see §18.** Measured leak is 5-14%, not
> parity, on both encoders and with an intervention that demonstrably engages the
> model. The inference itself was invalid: **co-location in a token pool implies
> nothing about shared directions.** A's and B's velocity subspaces sit ~75° apart
> inside A's tokens and are separately removable — strip 8 of A's velocity
> directions and B's velocity still reads 0.829 against a 0.814 baseline. This is
> the general lesson, and it is the one the field gets wrong casually:
> decodability does not imply causal coupling.
>
> Two numbers here also need §20's mask-only floor: `velocity_A_pre` from `A.pre`
> at 0.984 sits on a 0.258 floor rather than zero. The key off-diagonals leak
> little (0.013-0.117), so the central claim of this section is unaffected.

### 18. Per-object velocity steering — POSITIVE, and §16's prediction is refuted

This is the answer to part 2, and the second real positive in the project. Run on
v3, both encoders, 2026-08-13/14. Code: `scripts/momentum/steer_velocity.py`,
`steer_multidim.py`, `patch_velocity.py`. Results in `outputs/steer_*/`.

**Why velocity rather than momentum.** §12 kills momentum as a steering target:
`p = m*v` is an identity, so there is no residual degree of freedom to move, *for
any model*. The selection rule that follows is **"is it an independent degree of
freedom"**, not "is it decodable" — those come apart completely. Velocity is a
primitive, is sampled independently per sphere, decodes at 0.98, and §16 had
already pre-registered a prediction about it.

**Shared intervention mechanics.** All four experiments below use the same rig.

* Written at `block.N.resid_post` through `ActivationSteer`, restricted to a
  **binary** A-local ∩ pre-contact token mask. **Not** the normalised pooling
  row — that sums to 1, so using it scales the edit by `1/n_tokens` and silently
  makes alpha mean something different in every scene.
* `scale="sigma"` adds `alpha * sigma_token * d_hat`, so alpha means the same
  relative perturbation at every depth. Measured `||dh||/||h|| ~ alpha/sqrt(D)`:
  0.323 at alpha=8 for VideoMAE (D=768), 0.255 for V-JEPA2 (D=1024).
* Directions **and** readouts fit on the 180 train scenes; every intervention is
  evaluated on held-out test scenes only.
* Readout changes are reported in **training-label standard deviations**, so
  on-target, leak and appearance numbers are directly comparable.
* `fit_readout()` in `steer_velocity.py` is a parallel implementation of
  `fit_probe`'s protocol that **keeps its coefficients** — `fit_probe` returns
  only scores, and a steering readout has to be applied to new activations.

**18a. Single-direction steering along the probe direction — INERT.**

Confirmation run, all 120 test scenes, alpha=8, read at block 11:

| steer block | direction | on-target | passive | leak | appearance |
|---|---|---|---|---|---|
| 1 | probe | 0.252 | 0.181 | 0.076 | 0.110 |
| 6 | **probe** | **1.330** | 1.278 | **0.079** | 0.114 |
| 6 | shuffled | 0.106 | 0.091 | 0.055 | 0.229 |
| 6 | random | 0.082 | 0.044 | 0.036 | 0.086 |
| 9 | probe | 2.872 | 2.602 | 0.139 | 0.429 |

The **passive-propagation control is what makes this readable, and it is not
optional**. A residual stream carries an added vector forward, so a readout at
block M > N moves whether or not the model *read* the direction. The control
predicts the readout change assuming the pooled delta measured at the steered
block arrives untouched. **observed/passive = 1.04 (block 6), 1.10 (block 9),
1.39 (block 1).** The effect is arithmetic. The apparent 12x margin over the
shuffled control is definitional: the probe direction and the readout are both
velocity directions on near-identical representations, so adding one moves the
other, while a shuffled direction is orthogonal to it.

Readout-free checks agree. `cos` between the injected vector and the observed
change at block 11 falls 0.73 (steer at 9) -> 0.17 (steer at 1) — the network does
rotate what you inject given depth — but the **shuffled direction is rotated
identically** (0.77 -> 0.14). Cross-object bleed in representation space is
likewise generic: `relB/relA` ~ 0.34 for the fitted direction and the same for
random.

*A null leak from an inert intervention proves nothing.* That is why 18b-18d had
to happen before §16's prediction could be called either way.

**18b. Why it is inert: velocity is redundantly encoded.**

Iterative nullspace projection (fit readout, project its directions out, re-fit).
Each round removes 2 directions for the 2-vector targets, 1 for radius.

| rounds | VideoMAE blk 6: vel_A / vel_B / rad_A | V-JEPA2 blk 7: vel_A / vel_B / rad_A |
|---|---|---|
| 0 | 0.986 / 0.814 / 0.974 | 0.984 / 0.695 / 0.985 |
| 2 | 0.930 / 0.492 / 0.930 | 0.889 / 0.394 / 0.923 |
| 4 | 0.961 / 0.062 / 0.879 | 0.845 / 0.038 / 0.962 |
| 8 | 0.944 / 0.021 / 0.888 | 0.599 / 0.004 / 0.928 |
| 12 | **0.803** / 0.013 / −0.015 | −0.014 / −0.001 / 0.946 |

**A's own velocity survives 12 rounds — 24 dimensions removed — at R2 0.80 on
VideoMAE**, and to ~20 dimensions on V-JEPA2. You cannot destroy it by deleting
directions, so a rank-1 edit cannot overwrite it; it only moves the one readout
that shares its direction. B's velocity, read from the same tokens, collapses
after 3-4 rounds.

The curve is **non-monotonic** (0.930 at k=2, 0.960 at k=3, 0.889 at k=5, 0.956 at
k=6).

> **REVISED BY §30b — this paragraph used to attribute that to the sawtooth
> Joseph et al. document, i.e. sinusoidal feature pairs. It is wrong.** Every dip
> is a round where `fit_readout` selected ridge alpha 1000 and every recovery a
> round where it selected 100; the numbers above reproduce exactly under that
> explanation. The reading originally dismissed here — noise — was the right one.
> Do not cite this as corroborating §22.

**Subspace geometry — the mechanism for everything below.** Principal angles
inside `A.pre`, 90° = orthogonal:

| pair | VideoMAE | V-JEPA2 |
|---|---|---|
| velocity_A vs velocity_B | **74.3°, 75.6°** | **76.4°, 77.9°** |
| velocity_A vs radius_A | 87.9° | 88.6° |
| velocity_B vs radius_A | 88.0° | 84.7° |

And the direct test — strip A's velocity subspace out of A's tokens, then
re-probe for **B's** velocity:

| velocity_A directions removed | VideoMAE vel_B | V-JEPA2 vel_B |
|---|---|---|
| 0 (baseline) | 0.814 | 0.695 |
| 2 | 0.831 | 0.734 |
| 8 | **0.829** | **0.733** |

B's velocity is *completely untouched*. The two objects are co-located in the
same tokens and carried by different directions. **This already shows §16's
inference is invalid**: co-location in a token pool implies nothing about shared
directions.

**18c. The generative direction works where the discriminative one does not.**

Stop asking "which direction best *predicts* velocity" and ask "how does the
activation *move* when velocity changes". Ridge finds `Sigma^-1 a`; an OLS
regression of activations **on** labels finds `a`. Measured
`cos(probe, gen_x) = 0.268` — largely different objects, exactly as §5's geometry
note predicts. `cos(gen_x, gen_y) = 0.016`, so the two velocity components are
independently addressable.

VideoMAE, 60 test scenes, alpha=8, block 6 -> 11:

| direction | dv_x | dv_y | obs/passive | speed resp. | r(speed) | leak |
|---|---|---|---|---|---|---|
| probe *(discriminative)* | 1.794 | −0.062 | **1.02** | 0.263 | 0.740 | 0.082 |
| **gen_x** *(generative)* | 1.973 | −0.138 | **1.40** | **1.110** | 0.895 | 0.097 |
| gen_y | 0.121 | 1.467 | — | 0.792 | 0.892 | −0.048 |
| gen_xy | 1.387 | 1.069 | 1.29 | 1.026 | 0.906 | 0.027 |
| gen_radius *(specificity)* | −0.398 | −0.286 | — | 0.398 | 0.700 | −0.034 |
| random *(null)* | 0.051 | −0.076 | — | 0.497 | **−0.160** | −0.033 |

**The speed-consistency test is the closest thing here to a functional check, and
it is worth reusing.** Steering `v_x` should change speed by `(v . dv)/|v|`, which
**flips sign with the sign of `v_x`** — and `v_Ax` is 52% positive in this
dataset, so the hypotheses separate cleanly. An additive edit is linear, so a
purely passive network gives the *same* speed-readout change in every scene; a
network that re-derives speed must give a scene-dependent one. Generative: r =
0.895 with magnitude 1.110. Discriminative: r = 0.740 but magnitude only 0.263.
As a fraction of the velocity change, **0.56 vs 0.15**. The probe direction was
fooling the velocity readout without moving the model's notion of speed.

V-JEPA2, 40 scenes, alpha=8, block 7 -> 23: `probe` on-target 0.490 with
**obs/passive 0.78** (actively attenuated); `gen_x` 1.299 with **obs/passive
1.43** — the generative-beats-discriminative result replicates near-exactly, and
the gap is wider.

*Control caveat:* `gen_shuffled` is **contaminated**. OLS on shuffled labels still
loads on the activations' dominant variance directions, which velocity also loads
on, so it comes out at cos −0.42 with `gen_x` and behaves as a scaled
anti-`gen_x` rather than a null. Use `random` and `gen_radius`.

**18d. Activation patching — the strongest result.**

The intervention the steering runs could not be: target values come from
activations the model actually produced, so the edited state is on-manifold by
construction, no fitted direction can manufacture an effect, and the whole
redundant code changes at once. A-local tokens are shifted by exactly
`x_j - x_i`, the difference of two scenes' **captured pooled** `A.pre`
activations; because pooling weights sum to 1, a uniform token shift moves the
pooled mean by exactly that, so A's pooled representation lands on the donor's.
Donors are drawn at random from the test set — not max-contrast pairs, which
would inflate transfer.

Note the projected variant is *mathematically a subspace replacement*:
`P(x_i) + P(x_j - x_i) = P(x_j)`.

VideoMAE, 60 test scenes, block 6 -> 11 (transfer 1.0 = the model now represents A
as moving at the donor's velocity):

| variant | \|delta\| | transfer | passive | leak | r(speed) | r(radius) |
|---|---|---|---|---|---|---|
| **full** | 30.24 | **0.856 ±0.021** | 0.537 | 0.111 | **0.936** | 0.930 |
| **velocity subspace (16d)** | 18.76 | **0.761 ±0.023** | 0.542 | 0.069 | 0.443 | 0.168 |
| radius subspace (16d) | 9.76 | 0.137 ±0.041 | 0.107 | 0.028 | 0.069 | 0.968 |
| random subspace (16d) | 3.99 | 0.026 ±0.003 | 0.007 | 0.009 | 0.596 | 0.404 |

**Clean double dissociation**: the 16-dim velocity subspace transfers velocity
(0.761) and not radius (0.168); the radius subspace transfers radius (0.968) and
not velocity (0.137). **16 of 768 dimensions carry essentially all the
causally-relevant velocity content.** The network *amplifies* the patch 1.6x
beyond passive, against 1.0x for the rank-1 steer. And the full patch's speed
readout tracks the pair-dependent, sign-flipping `|v_j| - |v_i|` at r = 0.936.

V-JEPA2, 40 scenes, block 7 -> 23: full transfer **0.390** (passive 0.160,
obs/passive **2.44**), velocity subspace 0.273, radius −0.001, random 0.005,
r(speed) **0.923**, leak/on-target 14.0%.

**The V-JEPA2 transfer is much lower while amplification is higher.** Likely
reading: with 16 downstream blocks instead of 5, the network partially
**restores** A's original velocity after the patch — it can, because A's velocity
is also encoded in B's tokens (§16 has velocity_A at 0.83 from B's tokens) and in
the frame-wide pool, neither of which the patch touches. That makes the redundant
cross-object encoding an **error-correcting channel**, which would explain both
the decay with depth and why leak stays low. **Untested prediction: patching A
and B jointly should transfer far more than patching A alone.** Cheap to run.

**The result.** Across all four interventions, including the ones that provably
engage the model, installing a donor's velocity into A leaves B's velocity
representation **86-93% intact**, on both architectures. §16's prediction that
"a steer through A-local tokens should move B nearly as much as A" is refuted.
Objects are **separately addressable**, and the ~75° subspace separation says why.

**A bug worth not rediscovering.** Activations are stored fp16 and V-JEPA2's reach
`|x| ~ 264`, so a sum of squares overflows fp16's 65504 ceiling and
`np.linalg.norm` silently returns `inf` — turning every perturbation ratio into a
clean, plausible-looking **0.000**. Element-wise arithmetic and all readouts go
through float64, so only the diagnostic columns were affected, but the failure is
silent and looks like a real result. Cast to float64 before any norm.

### 19. Polar vs Cartesian factorisation — correlational only

> **EXTENDED BY §39g.** The subspace version of this question is now measured and
> is unambiguous: `speed` and `heading` occupy **orthogonal** subspaces (align
> -0.01 to -0.04 against ceilings of 0.38-0.51) in both encoders and both objects,
> on labels the §34b audit certifies as independent (R2 0.003-0.016). §39g also
> runs the mirror of the axis-stability test below — heading axis across SPEED
> bins — which this section never did; it comes out 0.77-0.98 against the speed
> axis's 0.24-0.84, but is under-powered and should not be cited alone.

Velocity has two degrees of freedom, coordinatisable as `(v_x, v_y)` or
`(speed, heading)`. This is **not** an identity trap — it is a change of
coordinates on the same information, so the question is which factorisation the
geometry is aligned with. Both decode near ceiling either way (VideoMAE speed
0.962 / heading 0.970 / velocity 0.986; V-JEPA2 0.981 / 0.988 / 0.984), and the
two occupy distinct subspaces (principal angle **62.3°** VideoMAE, **71.7°**
V-JEPA2), so decodability cannot settle it.

**Correlational test (this is the usable result).** Fit the generative speed axis
separately within each heading quadrant and ask whether it stays put. Noise alone
lowers agreement, so the comparison needs a ceiling: fit twice within the *same*
bin on disjoint halves, at matched sample size.

| model | within-bin (ceiling) | across-bin | ratio |
|---|---|---|---|
| VideoMAE-base | 0.614 ±0.067 | 0.338 ±0.192 | **0.55** — axis rotates, Cartesian-leaning |
| V-JEPA2-ViT-L | 0.741 ±0.076 | 0.635 ±0.092 | **0.86** — axis roughly fixed, polar-leaning |

V-JEPA2 agrees with Joseph et al. §5.3 (polar dominates). VideoMAE-base does not —
but their App C.1 notes that **smaller VideoMAE-v2 variants fail to exhibit
reliable emergence**, and they used VideoMAE-v2-*Giant*. Capacity is the more
likely explanation than architecture. `videomaev2-base` is registered and cached,
so this is testable without a download.

**Causal test (`old_experiments/scripts/steer_polar.py`) — INCONCLUSIVE, two design flaws.**

1. *The heading conditions were duplicates.* `cos(gen_hdg_x, gen_v_x) = 0.997` on
   both models. Since `heading_x = v_x/|v|` and speed varies with cv 0.40, the two
   are nearly collinear and their generative directions coincide.
2. *A rank-1 additive edit cannot implement a polar operation.* "Scale the
   magnitude, preserve the heading" is inherently scene-dependent — the required
   change points along each scene's *own* heading — and one fixed vector added to
   every scene cannot do that. High concentration is partly forced by the
   intervention's form. Joseph et al.'s central finding (20-50 coordinated
   dimensions needed) predicts a rank-1 probe of factorisation would be muddy.

What survives: `gen_speed` is distinct from the Cartesian axes (cos 0.129 /
0.055), and steering along it produces a change more aligned with each scene's own
heading (par 0.278 / 0.378) than a Cartesian axis does (0.150 / 0.060) — same
ordering as the correlational test, concentration still 0.88-0.93 against a
0.98-1.00 reference. **The correct test is a subspace intervention with per-scene
target coordinates** (Joseph et al. App C.12 adapted); `inlp_subspace()` in
`patch_velocity.py` already builds the basis.

### 39. THE GEOMETRY OF PHYSICAL FACTORS — v1's conserving 75, both encoders

Run 2026-08-23. `scripts/momentum/geometry_v1.py`, six stages, CPU only, on the existing
`outputs/{videomae-base,vjepa2-vitl}` captures plus §34g's never-analysed raw
per-token capture. This asks a different question from everything above it: not
*is* a factor decodable, but how the factors are **arranged** relative to each
other. Every number replicates on both encoders.

The dataset is `probe_set` v1 restricted to the **75 momentum-conserving**
scenes. Violating scenes are excluded because their aftermath was overwritten
after the fact, so post-contact kinematics there are not what the physics
produced. §38's `cos` leak is irrelevant here — nothing in this section reads
the conserved/violated label.

#### 39a. The measurement, and why a raw principal angle is worthless

Three things corrupt an angle between two fitted weight subspaces, and each
needs its own bracket. This is the transferable part of the section.

* **ceiling** — the SAME factor fit twice on different scenes does not give the
  same subspace. Measured here at overlap 0.33-0.58, i.e. 40-55 deg. Nothing can
  beat that, so the unit is the ceiling, **not 0 deg**.
* **null** — ridge on meaningless labels does **not** return a random direction;
  it loads on the activation's dominant variance directions, which is where real
  factors also load (§18c's `gen_shuffled` contamination). The null is therefore
  a **shuffled-label fit**, measured at 0.005-0.017. A random subspace scores
  0.0006-0.003 and is the wrong bar.
* **shared noise** — fitting two factors on the SAME scenes lets one noise
  realisation push both weight vectors the same way. Every cross-factor number
  here is fit on **DISJOINT HALVES** of the scenes (factor f on H1, factor g on
  H2, averaged both ways), which removes that channel and matches the ceiling's
  sample size at the same time.

Reported as `align = (overlap - null) / (ceiling - null)`. 0 = as unrelated as
the estimator allows; 1 = the same subspace measured twice. Alpha is **pinned**
(§30b) and the headline is unchanged across alpha 1 -> 10000 (align 1.01, 1.01,
1.00, 1.00, 1.02). `align` is suppressed when the ceiling fails to clear its own
null by 0.05, which is what happens in frame-wide pools and is itself a result.

> **RULE.** Never report a subspace angle without its ceiling and a
> shuffled-label null. A bare "75 deg apart" is uninterpretable: the same factor
> measured twice is already 40-55 deg apart at n=75.

> **SECOND RULE, added after §39j fell over.** A fitted weight vector is a
> property of the ESTIMATOR as much as of the representation — ridge demonstrably
> puts 4% of its energy in a component that alone predicts the target at R2 0.54.
> Angles between two probes fit the same way on similar data are still comparable
> (both are distorted identically), which is what keeps §39c/§39g standing, but
> **an angle can never license "the information is / is not in subspace S"**. Pair
> every geometric claim with a functional one.

#### 39b. The label-algebra audit, which changed what could be asked

§34b's guard, run on labels before any activation. R2 of predicting one factor
from another by OLS, 75 scenes:

| pair | R2 | consequence |
|---|---|---|
| `heading <- velocity` | **0.868 / 0.876** | **partly forced.** Heading is a quotient so it is not a §34b collapse, but 87% of it IS linear in velocity here, and a heading-vs-velocity subspace comparison inherits that |
| `speed <- velocity` | **0.024 / 0.028** | free. Speed is essentially NOT linear in velocity, because heading is near-uniform over the full circle |
| `speed <-> heading` | 0.003-0.016 | **free in both directions** — the polar question is clean |
| `velocity_B <- velocity_A` | 0.034 | free — the object question is clean |
| `mass <- radius` | 0.169 / 0.247 | free |

So `speed vs heading` and `A vs B` are the two comparisons this dataset can
actually make, and both are the ones the section rests on. Everything involving
`velocity vs heading` is reported but reads mostly as label algebra.

The rest of the label geometry is unusually clean: all |corr| < 0.31 between
kinematic and appearance factors on these 75 scenes, so no subspace result here
is driven by correlated labels.

#### 39c. THE HEADLINE — the code is per-factor and OBJECT-AGNOSTIC

Every *same-factor, different-object* pair sits **at its own ceiling**, and every
*different-factor* pair sits at zero. VideoMAE block 6 / V-JEPA2 block 8:

| pair | overlap | null | ceiling | **align** |
|---|---|---|---|---|
| `velocity_A` \| `velocity_B` | 0.484 / 0.428 | 0.012 | 0.482 / 0.428 | **1.00 / 1.00** |
| `heading_A` \| `heading_B` | 0.425 / 0.542 | 0.012 | 0.431 / 0.539 | **0.99 / 1.01** |
| `radius_A` \| `radius_B` | 0.458 / 0.440 | 0.007 | 0.472 / 0.447 | **0.97 / 0.98** |
| `mass_A` \| `mass_B` | 0.332 / 0.359 | 0.005 | 0.336 / 0.360 | **0.99 / 1.00** |
| `speed_A` \| `speed_B` | 0.348 / 0.298 | 0.007 | 0.389 / 0.432 | 0.89 / 0.68 |
| `velocity_A` \| `radius_A` | 0.005 / 0.008 | 0.014 | 0.488 / 0.430 | **-0.02 / -0.01** |
| `velocity_A` \| `mass_A` | 0.005 / 0.010 | 0.013 | 0.379 / 0.394 | -0.02 / 0.00 |
| `mass_A` \| `radius_A` | 0.189 / 0.183 | 0.007 | 0.377 / 0.384 | 0.49 / 0.47 |

Note the two pools involved are **not the same vector** — `A.pre` and `B.pre`
share only **0.066 centred cosine** across scenes. The alignment is not pool
similarity.

**Two boring explanations are dead.** It is not alpha (flat over four orders of
magnitude) and it is not pool correlation (0.066). It is also not geometry
alone, because it makes a prediction that was then tested:

**39c-i. The transfer test.** Fit the readout on object A only — A's pool, A's
label — then apply it, weights and intercept untouched, to B's pool and score
against B's label. Nothing about B was used to build it. Held-out R2:

| role | A->B | B->A | own B | own A | swapped | shuffled |
|---|---|---|---|---|---|---|
| **VideoMAE** velocity | **0.943** | 0.943 | 0.920 | 0.946 | **-0.395** | -0.707 |
| heading | 0.898 | 0.924 | 0.910 | 0.931 | -0.522 | -0.633 |
| radius | 0.875 | 0.891 | 0.891 | 0.897 | -0.439 | -0.514 |
| mass | 0.795 | 0.811 | 0.853 | 0.797 | -0.544 | -0.449 |
| speed | 0.774 | 0.832 | 0.906 | 0.826 | -0.972 | -0.618 |
| **V-JEPA2** velocity | **0.930** | 0.911 | 0.894 | 0.937 | -0.368 | -0.928 |
| heading | 0.935 | 0.960 | 0.929 | 0.965 | -0.430 | -0.873 |

**A probe that has never seen object B reads object B's velocity as well as a
probe trained on B** (0.943 against 0.920). `swapped` — A's readout on B's
tokens scored against A's OWN label — fails at -0.4, which is the control that
makes it a locality claim rather than a global-feature artifact.

Only **speed** transfers below its own ceiling (0.774 vs 0.906), and it is the
only role whose `align` is under 0.9. Speed is the least object-invariant factor.

**39c-ii. And it holds at the level of a single token.** §34g's raw per-token
capture (`outputs/patch_videomae-base`, 1568 tokens x 768, block 5) had never
been analysed. One readout fit on **individual tokens belonging to ball A**,
labelled with that scene's `v_A`, then applied token by token to held-out scenes:

| scored on | R2 |
|---|---|
| A's own tokens vs `v_A` (ceiling) | 0.941 |
| **B's tokens vs `v_B`** | **0.920** |
| B's tokens vs `v_A` (control, must fail) | **-0.545** |
| A's tokens, shuffled-label fit | -0.442 |

**Replicated on V-JEPA2** (`patch.block.7`, 6144 tokens x 1024): ceiling **0.891**,
B's tokens vs `v_B` **0.854**, control **-0.525**, shuffled -0.593.

A single local readout reports whichever object sits under the token. This is
the strongest available form of the claim and it needs no pooling at all.

> **THE FORMAT CLAIM.** There is one code per factor and it is **object-agnostic**.
> Object identity is carried by **WHICH TOKENS you read**, not by which
> directions. This is a different statement from §18's "objects are separately
> addressable", and both are true: separate *within* a token pool, common *across*
> pools.

#### 39d. Within a single pool the two objects ARE orthogonal — and that is consistent

The reconciliation, and it is the part that took the longest to see. Both
velocities read from the *same* pool:

| pool | overlap | null | ceiling | align | R2 `v_A` | R2 `v_B` |
|---|---|---|---|---|---|---|
| A.pre | 0.007 / 0.009 | 0.013 | 0.205 / 0.264 | **-0.03 / -0.02** | 0.949 | 0.436 |
| B.pre | 0.012 / 0.010 | 0.012 | 0.282 / 0.293 | -0.00 / -0.01 | 0.623 | 0.910 |
| all.pre | 0.025 / 0.027 | 0.019 | **0.021 / 0.026** | *undefined* | 0.425 | 0.155 |

So inside A's tokens, A's velocity and B's velocity are **orthogonal** — §18b's
~75 deg on v3, sharper here. The picture that fits everything:

* the **own-object** velocity is written in the canonical, shared code;
* the **other object's** velocity also reaches these tokens (0.44-0.53) but in a
  *different, orthogonal* channel;
* and the two channels have different geometry. Ablation, against a
  rank-matched random control (VideoMAE, A.pre):

| removed | R2 `v_A` | R2 `v_B` | R2 `radius_A` |
|---|---|---|---|
| nothing | 0.948 | 0.452 | 0.845 |
| `v_A` subspace, 12d | **0.911** | 0.472 | 0.852 |
| `v_B` subspace, 12d | 0.948 | **0.242** | 0.839 |
| random, 12d | 0.948 | 0.455 | 0.845 |

A clean double dissociation — and an asymmetry worth keeping: **the own-object
code is redundant and survives its own removal (0.948 -> 0.911), while the
other-object code is fragile and nearly halves (0.452 -> 0.242).** §18b found
velocity surviving 12 INLP rounds; that resistance is a property of the
*own-object* channel specifically.

In `all.pre` the ceiling collapses to the null (0.021 vs 0.019) — neither
object's velocity is reliably enough encoded there to have a subspace at all.
That is not a measurement failure, it is the next result.

#### 39e. FRAME-WIDE POOLING COMPUTES A SUM — this reframes §16

Held-out R2, mean over 20 splits:

| pool | `v_A` | `v_B` | **`v_A + v_B`** | `v_A - v_B` | `momentum_pre` |
|---|---|---|---|---|---|
| **VideoMAE** A.pre | 0.946 | 0.413 | 0.741 | 0.739 | 0.419 |
| B.pre | 0.619 | 0.920 | 0.744 | 0.728 | 0.419 |
| **all.pre** | 0.389 | 0.152 | **0.878** | **-0.353** | 0.649 |
| all.all | 0.425 | 0.168 | 0.858 | -0.273 | 0.673 |
| **V-JEPA2** all.pre | 0.450 | 0.069 | **0.869** | **-0.336** | 0.488 |

The frame-wide pool carries the **sum** better than either object-local pool
does (0.878 vs 0.741/0.744) and carries the **difference not at all** (-0.35).

§16 read the same phenomenon as "object-local pooling is the whole signal;
frame-wide pools lose it". They do not lose it — they **superpose** it. A mean
over tokens of an object-agnostic per-token code *is* a sum over whatever
objects are in the pool, and a sum is exactly what cannot be un-mixed. This also
explains why frame-wide pools win on `momentum_pre` (0.649 vs 0.419): momentum
is a sum.

Confirmed directly. Apply the velocity readout fit on `A.pre` only to `all.pre`,
then regress its output on the two true velocities:

```
pred_x = +0.06 * v_Ax  +0.07 * v_Bx        pred_y = +0.12 * v_Ay  +0.11 * v_By
```

Near-equal weight on both objects, i.e. an unweighted average, strongly
attenuated. Scored against candidate targets: `mean` 0.221 > `sum` 0.178 >
`v_B` 0.142 > `v_A` 0.121.

#### 39f-39i. CUT — these were replications, and are recorded only as pointers

Removed 2026-08-23 on the rule that this document should carry what is NEW. Each
was measured correctly; each answers a question `interpreting_physics.pdf` or
`cav.pdf` had already answered. The code that produced them is still in
`scripts/momentum/geometry_v1.py`; the numbers are in the JSONs.

| cut | why | prior work |
|---|---|---|
| **spatial delocalisation of the code** (decay profile, ~17x the object's footprint) | the phenomenon is theirs, only the size was ours | **Joseph et al. C.5**, "Direction moves to per-patch encoding": direction becomes "broadly distributed across patches", probes generalise across frame regions, reaching "a globally accessible form that survives pooling" |
| **speed and direction occupy orthogonal subspaces** (align -0.01 to -0.04) | the polar split and the near-orthogonality are both established | **Joseph et al. §5.3** (polar decomposition) and **C.4** (subspace overlap); **cav.pdf** cites physics/direction subspaces at 69-83 deg |
| **INLP dimensionality: direction ~32-64 dims, speed ~16** | replicates their ordering on a new set | **Joseph et al. C.4 Table 3**: direction 66-136, speed 21-29. **cav.pdf** Fig 2 runs the same iterative-orthogonalisation decay |
| **"direction is in the pixels, speed is built"** | **WRONG**, not merely redundant — it was the pooling mask (floor 0.320). See below | — |

**The one thing to carry forward from the cut §39h**, because it is ours and it
cost a day: *this two-object design cannot date speed against direction at all.*
The object-local pool leaks (its mask traces the ball's path, so the mask alone
scores 0.320 on heading and -0.222 on speed) and the frame-wide pool holds the
SUM (§39e), so a per-object target is ill-posed there. In the leak-free pool
**neither** quantity is decodable at the patch embedding (-0.08 to -0.10, both
models) — both are built by the transformer — but the ORDER is not recoverable
here. With one object, frame-wide pooling *is* object-local and neither confound
exists, which is why their single-ball set can answer it and this one cannot.

#### 39j. CONCENTRATION vs REDUNDANCY — two different "dimensionalities"

> **A first version of this section claimed physics AVOIDS the top principal
> components, from the energy of the ridge weight vector per PC. That was the
> ESTIMATOR and is withdrawn.** In the PC basis
> `W = sum_i [s_i/(s_i^2+alpha)] * corr(Y, PC_i) * PC_i`, and the shrinkage factor
> peaks at `s = sqrt(alpha)` then decays as `1/s` — ridge starves the leading
> components however much they carry. Measured: PC1 of VideoMAE `A.pre` correlates
> with velocity at **|r| = 0.942** and holds 29% of the variance, and the ridge
> direction puts **3.7%** of itself there.

> **RULE, and it is the one to carry.** A fitted weight vector is a property of the
> ESTIMATOR as much as of the representation. Angles between two probes fit the
> same way are still comparable — both distorted identically, which is what keeps
> §39c/§39d standing — but **an angle can never license "the information is / is not
> in subspace S"**. Pair every geometric claim with a functional one. This applies
> to **cav.pdf's CAV extraction** (it PCAs to 64 components before fitting, then maps
> weights back) and to **Joseph et al. C.4** equally.

**The estimator-free measurement.** Regress the label on the PC *scores*, SVD refit
**inside each fold** so no test scene touches the basis, against a full 768-d ridge
probe on identical folds:

| | top-1 | top-2 | **top-3** | top-5 | top-10 | top-20 | FULL 768d ridge |
|---|---|---|---|---|---|---|---|
| **VideoMAE** velocity | 0.522 | 0.486 | **0.913** | 0.936 | 0.943 | 0.943 | 0.956 |
| direction | 0.509 | 0.485 | **0.894** | 0.899 | 0.906 | 0.935 | 0.946 |
| **speed** | -0.031 | -0.053 | **-0.019** | 0.397 | 0.645 | 0.821 | 0.887 |
| **V-JEPA2** velocity | -0.061 | 0.517 | **0.841** | 0.901 | 0.892 | 0.909 | 0.954 |
| direction | -0.059 | 0.536 | **0.897** | 0.966 | 0.969 | 0.971 | 0.968 |
| **speed** | -0.014 | -0.065 | **-0.017** | 0.569 | 0.870 | 0.911 | 0.938 |

**Three components recover ~0.90 of velocity and direction** against a 768-d probe's
0.95; **speed reads zero from those same three** and needs 10-20.

#### 39j-i. Why that is NOT the same axis as INLP dimensionality

INLP (the cut §39i, and Joseph et al. C.4, and cav.pdf Fig 2) says direction is
HIGH-dimensional and speed LOW. The PC basis says direction is concentrated in 3
components and speed is not. **Both are right**, and the check that reconciles them
is whether INLP removal reaches the top PCs at all:

| target | removed | ridge R2 | **top-3 PC R2** | top-10 PC R2 |
|---|---|---|---|---|
| **velocity** | 0 | 0.954 | **0.913** | 0.943 |
| | 16 | 0.904 | **0.317** | 0.731 |
| | 32 | 0.723 | **0.083** | 0.418 |
| | 64 | 0.281 | 0.008 | 0.037 |
| **direction** | 16 | 0.852 | 0.466 | 0.742 |
| | 32 | 0.726 | 0.142 | 0.528 |
| **speed** | 0 | 0.852 | **-0.019** | 0.645 |
| | 16 | 0.169 | -0.027 | -0.001 |

**INLP does reach the top PCs, and reaches them FIRST.** Sixteen removals cost the
ridge readout almost nothing (0.954 -> 0.904) while collapsing the top-3 readout
(0.913 -> 0.317). So the two measurements name two independent properties:

* **CONCENTRATION** — where the primary copy sits in the variance spectrum.
  Direction/velocity: the top 3 components. Speed: not there at all, spread over
  ~10-20.
* **REDUNDANCY** — how many copies exist. Direction/velocity: many, so ridge keeps
  working off low-variance duplicates long after the concentrated copy is gone.
  Speed: few, so it dies with the primary (0.852 -> 0.169 at k=16).

> **Direction is both concentrated AND redundant; speed is neither.** "How many
> dimensions does this variable need" has no single answer — it depends on whether
> you count in the **variance basis** or the **deletion basis**, and this dataset
> gives **opposite orderings** in the two. Report which one you mean. Joseph et al.'s
> 66-136 vs 21-29 and cav.pdf's "2-3 principal directions" are both deletion-basis
> numbers, i.e. redundancy — neither says where the primary copy sits.
>
> **This is the surviving contribution of the PC line**, and it is a distinction
> neither paper draws.

*Method note:* n=75, so the activation matrix has rank <= 74 and PCs beyond ~20 are
noise-dominated. The leading components are well estimated; do not read the tail.

#### 39j-bis. WHAT IS ACTUALLY NEW HERE — audited against Joseph et al.

Checked against the paper text, not from memory. Most of §39 is replication, and
the section is only worth citing for the three rows marked NEW.

| claim | their status | ours |
|---|---|---|
| direction high-dimensional, speed low-dimensional | **theirs.** C.4 Table 3: direction 66-136 dims, speed 21-29, V-JEPA2-L | §39i replicates the ordering on a collision set (~32-64 vs ~16) |
| subspace geometry by principal angles + projection overlap on QR bases | **theirs.** C.4, the same two formulae | §39a adds brackets — see below |
| polar decomposition of velocity into speed and direction | **theirs.** §5.3 | §39g measures the speed-direction pair, which their Table 3 does not report (it reports direction/speed vs IntPhys only) |
| **the code delocalises; probes generalise across frame regions; "globally accessible form that survives pooling"; "position-invariant decoding"** | **THEIRS. Appendix C.5**, including the train-on-one-half / test-on-the-other test | §39f is a **quantification** of their C.5 (decay profile, 17x extent), NOT a discovery. Cite them |
| per-object probes on multi-object video | **theirs.** C.3, CLEVRER, 48 objects — but frame-wide mean-pooled, and the 48 are object *identities* | §39c uses object-local pooling, which is what makes transfer askable |
| Physics Emergence Zone at ~1/3 depth | **theirs.** §4 | not re-tested here |
| **cross-object readout transfer, two objects present at once with uncorrelated labels** | **NOT IN THE PAPER.** Their spatial generalisation is ONE object, one label, region->region | **NEW (§39c).** Arguably a corollary of C.5, but untested there — and the `swapped` control only exists when a second object is present to confuse the readout. Upgrades "position-invariant" to "object-invariant and simultaneously multiplexed" |
| **what a frame-wide pool CONTAINS with two objects** | **NOT IN THE PAPER.** They say pooling "survives"; they never decompose it | **NEW (§39e).** `v_A+v_B` 0.878, `v_A-v_B` -0.353. Bears on their own CLEVRER protocol: per-object probes on a mean-pooled vector read a superposition |
| **factors vs the activation's variance spectrum** | **NOT IN THE PAPER** — no PCA, eigen- or explained-variance analysis anywhere in it | **NEW (§39j), but the first version of it was WRONG.** What stands: direction rides a leading PC, speed a low-variance one; and probe-weight geometry does not locate information |
| own-object vs other-object channels differ in redundancy | not askable single-object | **NEW (§39d)**, small |
| mask-leak control | they have none and need none (no masks anywhere) | ours, and forced by object-local pooling — see §20 and §39h |

**The methodological disagreement, and it bears on their central negative.** Their
subspace null is the *analytic* random-subspace expectation `k_A/d`, they report no
reliability **ceiling**, and both probes in a pair are fit on the **same** data.
Measured here, those three choices matter:

| bracket | value |
|---|---|
| random 2-d subspaces (their baseline's form) | **0.0006** |
| shuffled-label fit (ridge on noise still loads on dominant variance) | **0.012**, ~20x higher |
| split-half ceiling — the most any two estimates of the SAME factor agree | **0.33-0.58**, nowhere near 1.0 |

C.4 concludes "no representational reuse" from direction->IntPhys overlap 0.129
against an expected 0.133. With a ceiling of ~0.5 rather than 1.0 that conclusion
probably survives — but **without a ceiling the test cannot distinguish "no shared
structure" from "too noisy to resolve any"**, and their overlaps are computed from
same-data fits, which biases toward agreement rather than away. Worth raising with
them; it is cheap for them to add.

#### 39k. Controls, caveats, and what would break this

* **24 of the 75 conserving scenes have ball B at rest**, so `velocity_B` has a
  point mass at the origin and `heading_B` is undefined there — the §34a
  concentric pathology in miniature. Every result was re-run on the **51
  `b_moving` scenes** and holds: transfer A->B velocity **0.956** against own B
  **0.940** (VideoMAE), align 1.06/1.12, `speed vs heading` align -0.01 to -0.04.
* **Object-local numbers carry §20's mask leak.** Floors are recomputed for the
  conserving 75 (`mask_floor_geom.json`): `velocity_A_pre` from `A.pre` sits on
  **0.312**, `heading_A_pre` on **0.320**, `radius_A` on 0.121, and the frame-wide
  `all.*` floors are -0.06 to -0.11 as they must be. **None of §39's conclusions
  are absolute R2 claims** — transfer, ablation and angle results are all
  differences at fixed mask, which the mask cannot manufacture.
* n = 75 (45 train / 30 test). Every geometric number is averaged over 40
  disjoint-half repeats; the decodability grid is a single split and carries the
  usual winner's curse.
* **What would break the headline:** a transfer test that succeeds in the
  `swapped` condition. It fails at -0.4 to -1.1 everywhere, which is why the
  claim is about a local shared code rather than a global feature.

#### 39l. Files

```
scripts/momentum/geometry_v1.py                  six stages; --stage, --moving-only, --alpha
outputs/{videomae-base,vjepa2-vitl}/momentum_probe/
    geometry_v1.json                    all stages, 75 conserving
    geometry_v1_moving.json             the 51 b_moving replication
    probes_geom_factors.json            the decodability grid (§39h)
    mask_floor_geom.json                §20 floors for the same grid
```

`heading_A_pre`, `heading_B_pre`, `speed_A_post`, `speed_B_post` and `b_moving`
were added to `DERIVED`; no re-capture was needed.




---

### 40. WHAT TO ASK NEXT ABOUT TWO-BODY COLLISIONS

Written 2026-08-23, after §39 cut four replications. The organising question is
the one nothing in this project or in `interpreting_physics.pdf` / `cav.pdf` has
touched: **how does the model know a collision happened, and where does it put
that fact?** Both prior papers are single-object (Joseph et al.'s toy set is one
ball in a straight line; cav.pdf is IntPhys). Their own limitations name this:
"we do not isolate richer physical computations such as **contact dynamics**".

#### 40a. THE BINDING CONSTRAINT — read this before designing anything

**Every scene on disk collides, at frame 20.033-20.067, sd 0.0166.** So on
existing data:

* "did a collision occur" has **no variance** — there is no negative class;
* "when did it occur" has **no variance** — the label is a constant;
* time-to-contact has **no variance** (already noted in §33).

Any experiment about collision *detection* or *timing* therefore needs a render.
Experiments about the collision's *consequences* — the impulse, the outcome, the
propagation between objects — do not, and those come first.

Two facts checked before proposing anything on them:

* `cos(dv_A, dv_B) = -1.000000` in all 75 conserving scenes (impulse exactly
  along the contact normal), but the scale `|dv_B|/|dv_A| = m_A/m_B` ranges
  **0.141 to 14.573**, so `R2(dv_B <- dv_A, linear across scenes) = 0.560`.
  **Not a §34b collapse** — the impulse-transfer experiments below are well posed.
* `impact_parameter_frac` is **already a sampler knob** in `generate_scene.py`, so
  near-miss scenes are a one-line change, not a new generator.

---

#### 40b. TIER A — answerable on captures that already exist, no GPU render

**A1. When does the model commit to the outcome? (the truncation curve)**
*The single best experiment on this list.*

Re-encode each clip truncated at frame `k` for k in {12, 16, 18, 20, 21, 22, 24,
28, 36, 48} and probe **post**-collision velocities from each truncated encoding.

* If `velocity_A_post` becomes decodable **before** frame 20, the model is
  *predicting* the collision outcome. If it appears only after, it is *observing*.
* This is the **only prediction test available on these checkpoints**. §7 (cannot
  inpaint), §15 and §21 (predictor is a pass-through) closed generation and the
  V-JEPA2 predictor; truncation needs neither. It also sidesteps §4's
  bidirectional-attention contamination *by construction*: a frame that does not
  exist cannot be attended to.
* **Controls, all required.** (i) `radius_A` at every truncation — a static
  property, so any decline is the encoding degrading, not the physics. (ii)
  `velocity_A_pre`, which must stay flat. (iii) A mask-only floor per truncation
  (§20), since the pooling mask shortens too.
* **Cost:** one encoder forward pass per (scene x truncation) — ~6 min per encoder
  per truncation, so ~1 GPU-hour per encoder for the whole ladder. Pixels exist.
* **Prediction.** §4 found post-collision quantities decode as well from
  pre-collision *pools* as from post ones, which was attributed to bidirectional
  attention. Truncation removes that explanation. If `v_post` is still decodable
  from frames 0-18, the model genuinely anticipates; if it collapses, §4's reading
  was right and the model only ever reads the aftermath.

**A2. Does the model PROPAGATE the interaction? (causal, and the strongest test
of whether it computes the collision)**

Patch ball A's **pre**-contact representation with a donor scene's (the §18d rig,
`patch_velocity.py`, unchanged), then read **B's POST-contact** velocity.

* Physics says changing A's incoming velocity changes B's outgoing velocity by a
  computable amount. Does the readout move in the required direction, scene by
  scene, with the right sign?
* **This is not identity-trapped.** It is not asking whether a quantity is
  present, it is asking whether an intervention propagates the way the law says.
* **Controls.** (i) The same patch must NOT move B's *pre*-contact readout — a
  clean before/after dissociation. (ii) The passive-propagation control (§18a),
  which is what made §18 readable. (iii) A radius-subspace patch, which should
  move nothing.
* **Cost:** ~3 min per encoder. Existing captures supply the directions.
* **Prediction.** §18d found A->B leak of only 5-14% for *pre*-contact velocity.
  If post-contact leak is much larger, the collision is a real computation in the
  network. If it stays at 5-14%, the model is tracking two trajectories and never
  binding them.

**A3. Is the IMPULSE coded in the shared object-agnostic basis? (the direct
extension of §39c)**

§39c showed velocity, speed, mass and radius all use one object-agnostic code.
The impulse is different in kind: it is **relational and antisymmetric**
(`m_A dv_A = -m_B dv_B`). Fit a `delta_v_A` readout on A's tokens, apply it
frozen to B's tokens, and score against `delta_v_B`.

* If it transfers **with a sign flip**, the interaction rides the same shared code
  and the antisymmetry is in the representation.
* If it fails, the interaction has its own code — which would be the first factor
  in this project that is not object-agnostic, and a genuinely new object.
* **Controls.** The `swapped` control from §39c; the mask-only floor (§20 measured
  `delta_v_B` leaking 0.356 from `B.post` on v3, so this one needs its floor).
* **Cost:** minutes, existing captures. `R2(dv_B <- dv_A) = 0.560`, so not vacuous.

**A4. What re-binds object identity across the collision?**

§39c's headline creates a puzzle it does not answer: if the code is
object-agnostic and identity is only *which tokens*, then the balls swap sides at
contact and identity must be re-established from something. Candidate: appearance.

* Decode "which pre-contact ball is this" from post-contact tokens, and stratify
  by the appearance gap. v1 has the material cue in 35 of 75 scenes and a radius
  gap with median 0.075.
* **The limiting stratum is small**: only **13 of 75** scenes have both the same
  material and `|r_A - r_B| < 0.05`. Enough for a signal check, not for a
  publishable number — flag it, or render a matched-appearance arm (40c-B3).

---

#### 40c. TIER B — needs a render, and each is a small generator change

**B1. CONTACT vs NEAR-MISS, matched pairs.** *The direct answer to "how does the
model know a collision happened."*

Sample `impact_parameter_frac` just **above** and just **below** 1.0 from the same
seed, so the pair shares masses, radii, materials, speeds and directions and
differs only in whether the surfaces touch.

* Probe **pre-contact** tokens: does the representation anticipate contact?
* Probe post-contact tokens: this is the detection question proper, and matched
  pairs make it leak-free in the way §38c wanted and never built.
* **The trap to design around**, and it is fatal if missed: after the near-miss
  point the two members of a pair diverge in *pixels*, so a post-contact probe can
  separate them from geometry alone. **Restrict the pre-contact probe to frames
  where the two clips are pixel-identical**, and verify with a frame-by-frame
  pixel-difference curve — the §32b lesson, applied at design time.
* **Cost:** ~150 scenes, ~2 GPU-hours. `--violate-frac 0` and a new
  `--near-miss-frac`.

**B2. VARIABLE CONTACT TIME.** Randomise the contact frame over 12-36 instead of
pinning it at 20.

* Unlocks: "when did it happen" as a probe target, **time-to-contact** (§33's #1
  candidate, "never probed in a video world model", currently impossible because
  the label is constant), and a temporal localisation of the collision code.
* **Requires a capture change**: `tubelet_phase_split` raises if a temporal cell
  straddles contact (§31a's trap), so contact must be targeted at a cell interior
  and the phase split becomes per-scene. Also re-read §30's rule — a **per-scene
  phase boundary is itself an information channel**; with contact time varying,
  the split leaks contact time. Use fixed windows and put the timing in the label.
* **Cost:** ~150 scenes, ~2 GPU-hours, plus a day on the capture path.

**B3. MATCHED-APPEARANCE ARM.** Both balls identical in radius and material, so
the only thing distinguishing them is trajectory. Makes A4 answerable properly and
directly tests whether the object-agnostic code has *any* identity channel besides
position. ~150 scenes.

**B4. OCCLUDED COLLISION.** An occluder covering the contact region, so the
collision itself is never visible — only the before and after. Does the model still
represent the outcome? This separates a *perceptual* collision code from an
*inferential* one, and no version of it exists in either paper. Needs a new object
in the generator; ~3 GPU-hours.

---

#### 40d. TIER C — needs new capture code

**C1. WHERE IS THE COLLISION, SPATIALLY? (per-token map)** §34g's raw per-token
captures already exist for v1. Decode `delta_v_A` per token and map R2 over the
14x14x8 grid. Does the impulse concentrate at the **contact patch**, or does it
live on the objects? §39c predicts the latter; a peak at the contact point would be
evidence for a distinct event representation. **Actually Tier A** — the captures
exist — but it needs the mask geometry per token, which §39's `spatial` stage
already builds. Cheap and high value.

**C2. ATTENTION ROUTING AT CONTACT.** Does attention from B's tokens to A's tokens
spike at the contact frame? This is the mechanistic version of "where is the
information". `interp/hooks.py` taps residual streams and `block.*.attn_heads`
(head *outputs*); **attention weights are not currently captured** and need a new
hook. The deleted `outputs/heads/` was head outputs, not weights, so it would not
have helped. ~1 day of code, ~2 GPU-hours.

---

#### 40e. Ranked, with what each would settle

| # | experiment | cost | settles |
|---|---|---|---|
| 1 | **A1 truncation curve** | ~1 GPU-h/encoder | predicts vs observes — the question §4 raised and could not answer |
| 2 | **A2 causal propagation** | ~3 min/encoder | whether the collision is COMPUTED or two trajectories are tracked |
| 3 | **C1 per-token impulse map** | minutes, existing data | where the interaction sits spatially |
| 4 | **A3 impulse transfer** | minutes | whether the shared code covers relational quantities |
| 5 | **B1 contact vs near-miss** | ~2 GPU-h | collision DETECTION, the headline question |
| 6 | **B2 variable contact time** | ~2 GPU-h + a day | timing, and unlocks time-to-contact |
| 7 | A4 / B3 identity | minutes / ~2 GPU-h | what carries identity when the code is object-agnostic |
| 8 | B4 occlusion | ~3 GPU-h | perceptual vs inferential collision code |
| 9 | C2 attention routing | ~1 day code | the mechanism |

**Do 1-4 first: they are all on existing pixels or existing captures**, and 1 and 2
between them decide whether this model does collision physics at all. Only then
spend GPU on 5 and 6.

**Every one of these must clear the standing bars**: write the target as a function
of what a camera sees (§12), report object-local numbers against a mask-only floor
(§20), check the target is not a linear combination of another probed target
(§34b), check it is not readable from one frame (§32b), and pin ridge alpha for any
sequence whose shape is read (§30b). And publish a contact sheet of any new render
and **look at it** (§32a) — three of this project's five false positives were
invisible to every validator.


---

### 41. CAUSAL PROPAGATION — the model does NOT compute the collision

Run 2026-08-23. `old_experiments/scripts/causal_collision.py`. §40b-A2 executed.

**The question.** Every probe here asks whether a quantity is PRESENT. This asks
whether the network performs the interaction: patch ball A's **pre-contact**
representation with a donor scene's, then read ball B's **POST-contact** velocity.
Physics says B's outgoing velocity depends on A's incoming velocity by a
computable amount. Does the readout move by that amount?

**Why decoding cannot answer it.** `velocity_B_post` decodes at 0.90 whether the
model computed the collision or watched the aftermath — §4 showed post-contact
quantities read as well from pre-contact pools, because attention is
bidirectional. An intervention on the CAUSE that moves the EFFECT is a different
claim.

#### 41a. The counterfactual, and it reconstructs to machine precision

For a frictionless two-sphere impact with restitution `e`, impulse along the
contact normal `n`:

```
J = (1+e) * mu_red * ((v_A - v_B) . n)        v_B_post = v_B_pre + (J/m_B) n
```

Reconstructing `v_B_post` from the labels this way matches the simulator to a
**median 1.95e-16** over all 75 conserving scenes. Holding contact geometry
(`n`, `e`, masses, `v_B_pre`) fixed and swapping in a donor's incoming velocity
gives a pure linear response:

```
required dv_B_post = k * ((v_A_donor - v_A_i) . n) * n ,   k = (1+e) mu_red / m_B
```

`k` ranges **0.29-3.07**; the required change is a median **0.86 label-sd**, so it
is measurable. **Caveat, stated not hidden:** a real re-simulation would also move
the contact point, hence `n`. Holding `n` fixed is the local linearisation and is
the only well-posed version without re-rendering.

#### 41b. THE PASSIVE CHANNEL IS EXACTLY ZERO — verified, not argued

§18a needed a passive-propagation control because edit and readout shared tokens,
and a purely arithmetic effect scored obs/passive 1.02 and looked like success.
Here the patched tokens (A-local AND pre-contact) and the read tokens (B-local
AND post-contact) are **disjoint in both region and phase**. Measured per scene:

| pool | pooling weight carried by the patched tokens |
|---|---|
| **A.pre** (patched) | **1.000000** |
| **B.post** (read) | **0.000000** |
| B.pre / B.all / A.post | 0.000000 |
| all.pre | 0.023810 (= 14/588) |

**0 of 30 scenes** had any overlap. At the steered block itself B.pre and B.post
move by exactly 0.000000. Anything arriving at B.post between blocks 6 and 11 has
to have been **computed by attention**. This is §31's floor/cube trick applied to
the collision.

**The patch works.** It lands A's pooled representation on the donor's to
**0.03%** (‖patched − donor‖ = 0.0084 against ‖delta‖ = 30.2), and installs
**0.796** of the donor's velocity into A's own readout (§18d got 0.856 on v3).
This is not a weak-intervention null.

#### 41c. THE RESULT — use the error metric, not a transfer ratio

> **METHOD NOTE, and it cost a round of confusion.** `transfer = observed/required`
> divides by a quantity spanning 0.005 to 3.899, so single pairs reach -1.95 and
> the mean carries a huge SE. `r(obs, required)` is worse: `required = k * du_n`,
> so a **constant-gain leak** of A's velocity into B's readout (§16: cross-object
> decoding 0.4-0.5) predicts the same correlation as physics. Three dressings of
> the same noise gave three different stories.
>
> **Use `|required - observed|` against the do-nothing baseline `|required|`.**
> Scale-free, no division by a near-zero, and it is Joseph et al.'s
> MAE-to-target protocol (§22 flagged it as the thing to adopt).

120 patch pairs (30 held-out scenes x 4 donors), VideoMAE block 6 -> 11,
V-JEPA2 block 7 -> 23:

| | median \|required\| | median \|required − observed\| | **ratio** | pairs beating do-nothing |
|---|---|---|---|---|
| **VideoMAE** patch | 0.862 | 0.866 | **1.002** | 47.5% |
| VideoMAE random | 0.862 | 0.865 | 1.002 | 47.5% |
| **V-JEPA2** patch | 0.808 | 0.750 | **0.990** | 56.7% |
| V-JEPA2 random | 0.808 | 0.838 | 0.999 | 53.3% |

Paired sign-flip test on `err_model - err_null`, 10 000 permutations:

| | mean difference | p |
|---|---|---|
| VideoMAE patch | **-0.0018** | **0.677** |
| VideoMAE random | +0.0027 | 0.480 |
| **V-JEPA2 patch** | **-0.0243** | **0.005** |
| V-JEPA2 random | +0.0004 | 0.854 |

* **VideoMAE does not update B at all.** Patch and random agree to three decimals;
  win rate 47.5%, a coin flip on the wrong side.
* **V-JEPA2 nudges B the right way.** Statistically real, physically tiny.

**POWERED RERUN, 360 pairs (30 scenes x 12 donors), V-JEPA2** — run because the
n=120 partial test was underpowered (r = 0.145 needs n ~ 373 for 80% power):

| | value | p |
|---|---|---|
| median error ratio | **0.967** | — |
| mean(err_model − err_null) | **-0.0258** | **0.0000** |
| beat do-nothing | **62.8%** of pairs | (random 50.8%, p = 0.560) |
| **partial r(obs, req \| du_n)** | **+0.189** | **0.0005** |
| slope, low-k half (mean k 1.13) | +0.0189 +- 0.0082 | |
| slope, high-k half (mean k 2.02) | +0.0539 +- 0.0101 | |
| slope ratio observed | **2.85** | physics predicts 1.79, leak predicts 1.00 |

> **THIS REVISES THE n=120 READING.** At full power the response gain **does scale
> with the collision gain `k`** — the low-k and high-k slopes are now individually
> significant and differ by 2.85x. A constant-gain leak of A's velocity into B's
> readout predicts a ratio of 1.00 and cannot produce `partial r = +0.189 at
> p = 0.0005`. So V-JEPA2 has a **genuine, physics-dependent causal path** from A's
> incoming velocity to B's outgoing-velocity representation.
>
> **And it delivers ~3% of the required change** (error ratio 0.967). The law says
> 100%. `transfer` collapsed from 0.170 +- 0.092 at n=120 to **0.008 +- 0.025** at
> n=360, which is what that ratio metric was worth.

**Corrected verdict.** VideoMAE: **no propagation** (p = 0.677, patch = random).
V-JEPA2: **weak but real and physics-dependent propagation, ~3% of required**. The
larger model has a collision computation; it is nearly vestigial.

> **THE FINDING.** Changing what hits ball B barely changes the model's
> representation of how B leaves — not at all in VideoMAE, by ~3% of the required
> amount in V-JEPA2. The aftermath is **not causally downstream** of
> the approach; the two are computed independently. This is the causal counterpart
> of §4's "the probes were reading, not predicting", on an intervention that
> demonstrably engages the model and through a channel with zero passive leak.

#### 41d. WHAT IT DOES NOT SHOW, and this decides the next experiment

**The aftermath is visible.** Frames 21-48 show B moving, so the model never needs
to infer B's outgoing velocity from A's incoming one. A model with a perfectly
good collision model would still show nothing here. **This shows the model does
not propagate; it does not show it cannot.**

The only way to force the inference is to remove the visible answer — which is
truncation (§40b-A1). The sharp version is **§42: truncate before contact AND
patch**, then read the predicted outcome. If an effect appears there and not here,
the model has a collision model it uses only when it must.

#### 41e. Files

```
old_experiments/scripts/causal_collision.py                          the experiment
outputs/steer_{videomae-base,vjepa2-vitl}/causal_collision.json
outputs/steer_vjepa2-vitl_pow/causal_collision.json  360-pair powered rerun
```


---

### 43. TRUNCATION — does the model ANTICIPATE the collision, or read it?

Run 2026-08-23/24. `old_experiments/scripts/truncation_curve.py`. §40b-A1 executed, and it is the
experiment §41's null demanded: §41 found the model does not propagate the
collision, but the aftermath is VISIBLE there, so the model never has to infer it.
Truncation removes the visible answer.

Re-encode each clip cut at frame `k` and probe the post-collision velocity from
what the model has seen so far. **Contact is at 20.03**, so `k <= 20` is strictly
pre-contact. §7/§15/§21 closed generation and the predictor; this needs neither,
and it removes §4's bidirectional-attention confound **by construction** — a frame
that does not exist cannot be attended to.

#### 43a. TWO DATA TRICKS, both worth reusing

**The violating half is usable, and doubles n for free.** The violation edits
POST-contact velocities only; frames 0-20 are honest physics in all 150 scenes.
The recorded `post` label is corrupted there, but the collision is deterministic
given the pre-contact state and the labels store `contact_normal`, so the physical
aftermath can be RECONSTRUCTED:

```
J = (1+e) * mu_red * ((v_A - v_B) . n)      v_B_post = v_B_pre + (J/m_B) n
```

Validated on the 75 conserving scenes where the recorded aftermath IS the physics:
median relative error **1.56e-16**. On the violating half the reconstruction
differs from the recorded (edited) label by a median 0.266 — exactly the noise this
avoids. Label spread matches across halves (sd 0.512 both).

**`probe_set_v4_legacy_fixed` adds NOTHING to a pre-contact experiment.** All 150
seeds are shared with v1 and the **pre-contact state is bit-identical in 150/150**
(velocities, masses, radii, restitution, contact normal, contact frame); only
post-contact `velocity_B` differs, in the 75 violating scenes. Checked before
spending GPU on it.

#### 43b. THE CURVE — flat before contact, steep after

`velocity_B_post`, best over sites, whole-clip pools (below k=20 there are no
post-contact cells, so the pre/post split does not exist and `*.all` is the only
defined pool):

| k | 10 | 14 | 17 | 19 | **20** | 22 | 25 | 30 | 38 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| VideoMAE | 0.057 | 0.132 | 0.153 | 0.189 | **0.195** | 0.359 | 0.571 | 0.695 | 0.826 | 0.895 |
| V-JEPA2 | 0.236 | 0.265 | 0.231 | 0.264 | **0.289** | 0.316 | 0.549 | 0.635 | 0.791 | 0.863 |

**The controls are what make it readable.** `velocity_A_pre` 0.88-0.96 and
`radius_A` 0.87-0.97, FLAT at every k in both models — so resampling 16 frames
from a 10-frame window is not degrading the encoding, and the low pre-contact
numbers are information absence rather than a broken input. The `v_B_post`
mask-only floor is **negative at every pre-contact k** (-0.37 to -0.04); it only
turns positive at k=38-48, so the post-contact rise is partly mask and should not
be quoted as a pure representational number.

#### 43c. THE LADDER at k=20 — and the bar was far higher than first quoted

n=150, **5-fold CV** (train 120, every scene tested once), honest site selection.

**Oracles on TRUE pre-contact features:**

| | R2 |
|---|---|
| linear `[v_A, v_B]` | 0.526 |
| linear + masses, radii, contact normal | **0.683** |
| quadratic (all pairwise products) | 0.643 |
| RBF kernel ridge | 0.724 |
| **exact impulse law, restitution replaced by its mean** | **0.957** |
| exact impulse law, true restitution | 1.000 |

Restitution is invisible until impact and accounts for **4.3%** of the variance;
0.957 is therefore the true ceiling for any pre-contact observer. Everything else
IS available: the contact normal is a **deterministic function of frame 0** (two
straight lines, contact when `|p_A-p_B| = r_A+r_B`, a quadratic in t) — recovered
from frame 0 to a median **0.0065 deg**.

**The model:**

| | VideoMAE | V-JEPA2 |
|---|---|---|
| DIRECT linear, best pool `all.all` | **0.584** | **0.539** |
| DIRECT RBF, same pool | 0.581 | 0.550 |
| COMPOSED (decoded `v_A`,`v_B` -> linear map) | 0.500 | 0.497 |
| **STRUCTURED (decoded parts -> impulse law)** | **0.605** | **0.662** |
| *ingredients:* `v_A` / `v_B` | 0.981 / 0.962 | 0.957 / 0.938 |
| *ingredients:* `m_A` / `m_B` | 0.943 / 0.920 | 0.967 / 0.936 |
| *ingredient:* **contact normal** (`A.all+B.all`) | **0.866** | **0.925** |

**The model holds every ingredient of the impulse law before contact**, contact
geometry included, and its own direct readout delivers 0.54-0.58 of a possible
0.957 — **86% of the 0.683 a linear map of all true visible features affords**.

#### 43d. POOL CHOICE DOMINATES, and it is target-specific

Direct probe on `v_B_post`, VideoMAE, by pool: `A.all` 0.414, `B.all` 0.483,
**`all.all` 0.584**, `A.all+B.all` 0.515. Contact normal, same capture:
`A.all` 0.814, `B.all` 0.811, **`all.all` -0.005**, `A.all+B.all` 0.866.

> **RULE.** There is no single right pool. A **relational** property of two named
> objects (the contact normal) dies in a frame-wide pool and lives in object-local
> ones; a **scene-level outcome** (`v_B_post`) does the reverse. Sweep pools per
> target, and never read a null from one pool as absence.

#### 43e. RBF buys nothing; the ATTENTIVE probe is confirmed data-limited

RBF kernel ridge matches linear in **every** pool (0.414/0.414, 0.483/0.483,
0.584/0.581, 0.515/0.525) — a generic nonlinear learner finds nothing extra at
n=150 in 768 dims, reproducing §26 and §34f across five pools. But **imposing**
the nonlinearity (STRUCTURED, 0.605/0.662) beats both, which is §26's rule again:
*imposing a known nonlinearity is free; learning one costs more than the data
affords.*

**§34g's open question is now CLOSED.** It asked whether pushing epochs would
recover the attentive probe. Captured raw tokens at k=20 (150 x 1568 x 768, three
blocks) and ran the sanity target:

| `velocity_A_pre` | 120 epochs | 400 epochs |
|---|---|---|
| block.5 | 0.409 | **0.389** |
| block.8 | 0.308 | 0.318 |
| block.11 | 0.292 | — |

A **linear probe on a pooled vector reads 0.981**. Tripling epochs does not help,
so it is not underfitting — the probe is **data-limited at n=150**, exactly as
§34g suspected. Its `v_B_post` numbers (0.28-0.50) bound nothing and are not
evidence. Fixing it needs more `legacy`-arm scenes, not more epochs.

#### 43f. What this says, with §41

* **The model does anticipate the collision outcome**, at 0.54-0.58 against a
  0.683 linear-feature oracle and a 0.957 ceiling — far more than the first
  (n=75, wrong-pool) pass suggested.
* **It does not PROPAGATE it** (§41): patching A's incoming velocity leaves B's
  outgoing-velocity representation at error ratio 1.002 / 0.990.
* Those are consistent: the representation **contains** the information and the
  **causal path between the objects is not used**. Decodability and causal
  coupling come apart — §18's lesson, now on prediction.

#### 43g. Four corrections this section forced, all the same failure mode

Recorded because the pattern is the finding:

| claimed | killed by | actual |
|---|---|---|
| "direction is in the pixels, speed is built" (§39h) | mask-only floor | the `A.pre` mask traces the ball's path; floor 0.320 |
| "physics avoids the top PCs" (§39j) | ridge shrinkage algebra | PC1 correlates with velocity at 0.942; the probe puts 3.7% there |
| "the model has no representation of impact geometry" | pool sweep | it reads 0.866-0.925 object-local; I used `all.all` |
| "direct/composed gap = 0.35, the model does not compute" | learning curve | the gap is a sample-size artifact: 0.465 at n=60 -> 0.124 at n=150 |

Every one looked like a result until its control ran. **A number from a single
pool, a single split, or a fitted weight vector is a hypothesis, not a finding.**

#### 43h. Files

```
old_experiments/scripts/truncation_curve.py     the k-ladder capture + probe
old_experiments/scripts/causal_collision.py     §41
outputs/trunc_{videomae-base,vjepa2-vitl}.json
```


---
