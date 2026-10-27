# RESULTS — what is currently believed

Only live claims are in Part 1. Anything withdrawn, superseded or reclassified
is in Part 2 with the reason. `CLAUDE.md` has the detail behind every §; this
file exists so you can tell what is currently true without walking a chain of
retractions.

Scoring conventions: R² unless stated. A number means nothing without its
`shuf*` (grid-max on shuffled labels, **not** per-cell), its pre-transformer
baseline (`embed.out` for encoders, `vae.latent` for CogVideoX), and — for any
object-local pool — its mask-only floor. Read margins as `best − shuf*`, never
`best − 0`.

Models: CogVideoX-2b (30 blocks), VideoMAE-base (12 blocks, 768d),
V-JEPA2-ViT-L (24 blocks, 1024d). Numbers are VideoMAE / V-JEPA2 where paired.

---

# Part 1 — live claims

## 1. Products and quotients are not stored (§26, §34e) — the strongest claim

Physical factors are held **separately and linearly**. Every product or quotient
must be composed by the readout; nothing indicates the model stores the
combination.

Object-local, v1, best over blocks:

| pool | m alone | v alone | `m·v` DIRECT | `m·v` COMPOSED | quadratic oracle |
|---|---|---|---|---|---|
| A.pre | 0.875 / 0.928 | 0.974 / 0.964 | **0.663 / 0.654** | 0.894 / 0.930 | 0.893 |
| B.pre | 0.731 / 0.839 | 0.959 / 0.939 | **0.515 / 0.434** | 0.910 / 0.903 | 0.683 |

Mass and velocity read at 0.73–0.97 **out of the same vector**; their product
reads at 0.43–0.66, below a quadratic function of the true factors. A linear
probe cannot multiply two things it can each read perfectly.

Frame-wide total momentum, best-vs-best, and the gap **widens** under honest
cell selection — so it is not a winner's curse:

| selection | model | best DIRECT | best COMPOSED | gap |
|---|---|---|---|---|
| best cell on test | VideoMAE | 0.825 | **0.934** | −0.110 |
| best cell on test | V-JEPA2 | 0.728 | **0.936** | −0.208 |
| cell picked on validation | VideoMAE | 0.783 | **0.929** | −0.146 |
| cell picked on validation | V-JEPA2 | 0.616 | **0.926** | −0.310 |

Replicated a third time on the v3 mass ratio. A generic nonlinear probe (RBF
kernel ridge) lands on the composed number, not above it: at n=135 and 3–4k
dimensions it cannot discover `log‖·‖` on its own. **Imposing a known
nonlinearity costs no capacity; learning it costs more than the data affords.**

This is a claim about representational **format**, not about physical
knowledge — any observer holding mass and velocity computes the product with
one multiplication.

> Beware the within-pool comparison. At `all.pre` direct (0.825) ties composed
> (0.824) — but `mass_B` decodes at **−0.627** there, so composition is
> crippled, not direct advantaged. Compare best-vs-best, or at a pool where the
> factors are decodable.

## 2. Per-object velocity is causally steerable (§18)

Patching one scene's A-local representation into another **installs the donor's
velocity at 0.86 of the required change** (VideoMAE; 0.39 V-JEPA2), with a clean
double dissociation against radius, and leaves the *other* object's
representation 86–93% intact.

Mechanics that matter: the edit is written at `block.N.resid_post` through a
**binary** A-local ∩ pre-contact token mask (not the normalised pooling row,
which sums to 1 and silently rescales alpha per scene); `scale="sigma"` so
alpha means the same relative perturbation at every depth; directions and
readouts fit on train, every intervention evaluated on held-out scenes.

**§16's prediction is refuted.** From "A's tokens decode B's velocity at
0.75–0.83" it inferred that steering A should move B nearly as much. Measured
leak is **5–14%** on both encoders, with an intervention that demonstrably
engages the model. **Co-location in a token pool implies nothing about shared
directions** — the two objects' velocity subspaces sit ~75° apart and are
separately removable. This corrects an inference the field makes casually.

**Single-direction steering does not work, and why.** Velocity survives 12
rounds of INLP at R² 0.80, so it is encoded redundantly; a rank-1 discriminative
edit only fools the readout sharing its direction (observed/passive = 1.02). The
*generative* direction works (1.40). Independently confirmed by Joseph et al.

## 3. mu is bound to the object, not the surface (§31)

Patch the **floor's** tokens from a donor with a different mu, read mu from the
**cube's** tokens, and vice versa. The pools are disjoint, so the passive
channel is not argued away but *computed*: for floor→cube the pooling-weight sum
over patched tokens is exactly **0.000**.

| condition | tokens | transfer | 95% CI | per token |
|---|---|---|---|---|
| VideoMAE floor→cube | 973 | 0.011 | [+0.003, +0.019] | 1.1e−05 |
| rand→cube (norm-matched) | 973 | 0.005 | [−0.001, +0.011] | 5.1e−06 |
| **VideoMAE cube→floor** | **11** | **0.097** | [+0.060, +0.133] | **8.8e−03** |
| V-JEPA2 floor→cube | 3734 | 0.011 | [+0.005, +0.018] | 3.0e−06 |
| **V-JEPA2 cube→floor** | **50** | **0.048** | [+0.021, +0.072] | **9.6e−04** |

Replacing the entire floor moves the cube's mu readout ~1%, with a CI
overlapping the random control; replacing 11–50 cube tokens moves the floor's
readout 5–10%. Per token the cube is **300–800× more effective**. The floor
pool's high decodability is *inherited*: information flows cube → floor.

Scope: in the latent arm the floor is one flat albedo and cannot carry mu, so
this shows where a dynamically-inferred mu lives, not whether the model could
bind an advertised material property to a surface. That needs the
`visible_friction` arm, still unrendered.

## 4. Δp is represented, and it is constructed (§34)

500-permutation grid-max nulls — permute, refit every cell, take the max, ×500:

| | VideoMAE | CogVideoX-2b | V-JEPA2 |
|---|---|---|---|
| `momentum_delta` R² | **0.517** | **0.318** | **0.232** |
| null grid-max, 95th pct | 0.028 | 0.055 | 0.033 |
| **p** | **0.000** | **0.000** | **0.000** |
| pre-transformer baseline | 0.014 | **0.000** | 0.022 |

Zero of 500 permutations came close, in three architectures. The depth curves
are flat at ~0 before the transformer and through early blocks, rising to a
mid-to-late peak — so it is **built**, not present in the pixels. Compare
`radius_A`, which reads 0.952–0.965 at `embed.out`.

**This is decodability, not law-use.** On v1 both masses and both velocity
changes are visible, so `Δp = m_A dv_A + m_B dv_B` is an exact function of what
a camera can see (§12).

> **Open, and it must be settled before any writeup.** §37 reports 0.205–0.261
> at the same cell under a different fitting protocol. No GPU required.

## 5. Friction, on the set where mu and acceleration separate (§35)

`incline_set_v2`, 150 scenes, both encoders.

| target | VideoMAE | V-JEPA2 | note |
|---|---|---|---|
| `mu` | **0.423** (A.pre, blk 7) | **0.779** (all.pre, blk 21) | mask floor 0.066 |
| `accel_measured` | 0.827 | 0.929 | |
| `theta` | 1.000 | 1.000 | **at `embed.out`** — in the pixels |
| mu composed from (a, θ) | 0.425 | 0.776 | direct − composed: +0.002 / −0.003 |

**mu reads below what acceleration alone affords.** Error propagation from the
decoded acceleration predicts R²(mu) of 0.584 / 0.829 against the observed
0.423 / 0.779. There is no independent mu representation here.

The quotient test §32 was built for did not materialise: mu is **0.997 linear in
(a, θ)** over the sampled theta range. Widening theta to 15–70° is the fix.

## 6. No evidence of lawful error structure (`old_experiments/scripts/lawful_errors.py`)

Run 2026-08-22 on `probe_set_v3_latent`, both encoders. One readout tied across
time (same weights on `.pre` and `.post` pools), fit on conserving scenes only;
the statistic is the partial correlation of the two readouts' errors given the
true momentum, on held-out conserving scenes.

At the best-readout cell, with 500-permutation **grid-max** nulls:

| model | target | ρ | grid-max null p95 | margin | ρ at max cell |
|---|---|---|---|---|---|
| VideoMAE | momentum | 0.916 | 0.231 | 0.685 | 0.945 |
| VideoMAE | **KE placebo** | 0.893 | 0.299 | 0.594 | **0.964** |
| V-JEPA2 | momentum | 0.914 | 0.238 | 0.676 | 0.956 |
| V-JEPA2 | **KE placebo** | 0.937 | 0.336 | 0.601 | **0.958** |

The sharing is far above its null in all four runs — but **kinetic energy, which
is not conserved in these collisions, shows it just as strongly**, and on
VideoMAE already at 0.835 in block 0 where momentum is at 0.396. Momentum's
margin over the null is 0.06–0.08 larger than the placebo's at the best-readout
cell and *smaller* at the max-ρ cell, so the ordering flips with cell selection
— which is exactly what a grid-max null exists to punish. No separation.

What is being measured is generic scene-level error sharing between two token
pools of the same clip, not a conservation constraint.

Two things this run established about the instrument, both worth keeping:

* The *raw* error correlation is degenerate. On a conserving scene
  `p_pre == p_post` exactly, so when the readout is weak its errors *are* the
  target and ρ → 1 with nothing in the model doing any work — measured 0.998 at
  `embed.out` on a readout scoring R² 0.003. Conditioning on the true target
  fixes it, and cells below a readout floor must be excluded from selection.
* **The `beta` and `pull` arms are void on v3 for the same reason the binary
  is.** §14's theorem says no observer can register the violation, so a readout
  that fails to register it is not evidence of anything. Those arms need a set
  where Δp is inferable — i.e. the `visible_mass` arm.

## 7. Energy is read as elasticity, and elasticity is mass-free (§44)

Run 2026-08-25, `conservation_set_v1`, both encoders, `scripts/kinetic_energy/probe_energy.py`.
The full 9-cell pool grid ({all, A, B} × {pre, post, all}) × every site, grouped
5-fold by seed, 200-draw max-over-grid nulls.

**It decodes, cleanly.** "Is this clip losing kinetic energy", on the honest half
where the elastic cell conserves energy exactly (KE ratio 1.000, all 75) and the
inelastic cell loses it (0.098–0.992, all 75):

| pool | VideoMAE | V-JEPA2 | mask floor |
|---|---|---|---|
| `all.pre` | **0.946** | 0.982 | 0.500 |
| `all.post` | 0.945 | **0.983** | 0.500 |
| `all.all` | 0.884 | 0.953 | 0.500 |
| `A.pre` / `A.post` / `A.all` | 0.864 / 0.847 / 0.796 | 0.939 / 0.954 / 0.864 | 0.54 / 0.61 / 0.58 |
| `B.pre` / `B.post` / `B.all` | 0.937 / 0.892 / 0.823 | 0.948 / 0.948 / 0.943 | 0.58 / 0.43 / 0.55 |

Balanced accuracy 0.873 / 0.920, `shuf*` (grid-max) p95 0.664 / 0.668,
`embed.out` 0.621 / 0.595. The `all.*` mask floors are 0.500 **by construction** —
those pooling rows are identical in every scene.

**And it is measuring a camera, not a concept.** On the honest half the label IS
`e < 1`, where `e = |(v_A−v_B)_post·n| / |(v_A−v_B)_pre·n|` needs no mass at all.
A kinematic observer fitted through the identical pipeline on a quadratic in five
visible mass-free scalars, paired bootstrap over seeds:

| target | probe | mass-free observer | probe − observer, 95% CI |
|---|---|---|---|
| honest (VM / VJ) | 0.946 / 0.983 | **1.000** | −0.054 [−0.090, −0.024] / −0.017 [−0.036, −0.005] |
| all 300 clips | 0.830 / 0.819 | 0.864 | −0.034 [−0.077, +0.007] / −0.044 [−0.086, −0.002] |
| violating only | 0.795 / 0.762 | 0.779 | +0.016 [−0.057, +0.084] / −0.017 [−0.103, +0.067] |
| energy *created* | 0.798 / 0.793 | 0.758 | +0.040 [−0.053, +0.126] / +0.034 [−0.060, +0.130] |

**Eight comparisons; not one interval excludes zero in the probe's favour**, and
six of eight put the probe significantly *below* an observer that also sees the
true kinetic energies. The probe's cell is the argmax over 117 or 225 cells chosen
on the scored data and the observer gets no search, so the bias runs toward the
probe and it still did not clear a camera.

Transfer honest → violating (grouped, so a twin is never in train) lands at 0.769 /
0.760 against a 0.775 mass-free transfer ceiling; the graded route reaches R² +0.303
on the violating clips' KE ratio — exactly where `|v_A| + |v_B|` alone lands.

**Three structural facts in the grid, none about energy.** Frame-wide beats
object-local by 0.05–0.15 (elasticity is a property of the pair, and a single-object
pool cannot hold a relation — §39e). `.all` is the *worst* phase in every row:
averaging across the collision mixes the two sides of the contrast. `.pre` ties
`.post`, because attention is bidirectional — **no cell here tests prediction** (§4).

**And the magnitude is mass-free too, by identity.** Masses vary hugely on this set
(ratio 0.059–24.2) so *how much* energy is lost ought to need them. It does not:
momentum conservation gives `m_A/m_B = |Δv_B|/|Δv_A|`, verified on all 150 honest
clips at max relative error **1.02e−15**, and substituting it reconstructs the KE
ratio from visible velocities alone at **R² = 1.000000**. At `e` pinned to 0.15
(n = 75), an observer with **no size and no mass information at all** reads 0.785
against the probe's 0.511 / 0.673 — the probe loses outright on VideoMAE
(−0.281 [−0.477, −0.115]) and ties at best on V-JEPA2.

Mass is not the bottleneck: `log mass_A` reads **+0.977 / +0.990** and `log mass_B`
**+0.982 / +0.989** on these same captures. The model holds the masses and the velocities and still does
not produce their energy combination — §1's pattern, now for `KE = ½mv²`.

**Claim.** Both encoders carry whether a collision *looked* elastic. Nothing here
requires an internal notion of kinetic energy, and §1 already says the products that
would take are not stored.

> **Scope, and it outlives this dataset.** For a **fully observed** two-body
> collision no observer ever needs the masses to get the energy ratio — conservation
> hands over the mass ratio for free. So no experiment of this shape can separate an
> energy concept from kinematics, at any mass range. Breaking it needs an
> **incompletely observed** collision: a ball leaving frame after contact, an
> occluder, a wall with no readable recoil, or a third body.

## 8. Energy does not survive a change of scenario (§45)

Run 2026-08-25, `scripts/kinetic_energy/energy_composition.py`, both encoders. §7 showed the
collision sets cannot ask the question — conservation makes the mass ratio visible,
so the energy ratio is mass-free at R² = 1.000000. The friction sets can: one cube,
one floor, **no second body**, so the floor's recoil is unobservable and nothing
hands the observer a mass. `mass = size³` exactly, `corr(log size, log speed) =
−0.053`, and neither factor alone explains log KE (0.207 / 0.749).

**Fit on `slide_set_v2`, apply unchanged to `conservation_set_v1`.** Different
objects, different camera, different loss mechanism — deceleration×distance vs
`(1−e²)`. Cell chosen on the source only; scored by Spearman.

| | VideoMAE | V-JEPA2 |
|---|---|---|
| control — apparent size | +0.631 | +0.585 |
| control — speed | +0.244 | +0.428 |
| **kinetic energy** | **+0.022** (p = 0.725) | **+0.275** (p < 0.005) |
| composed: size and speed transferred separately, then multiplied | +0.291 | +0.324 |

Both controls cross the gap, so this is a negative rather than a void test.
VideoMAE's energy direction is **scenario-bound** — chance transfer, while reading KE
at R² 0.971 within the slide set. V-JEPA2's **does** transfer, the only
scenario-invariant energy signal anywhere in §44–§45 — and is still beaten by
composing its own transferred factors. **The factors travel; the product adds nothing.**

The within-set composition test (direct vs composed on the slide set) is **at ceiling
and cannot answer**: mass decodes at 0.994 and speed at 0.95–0.99, so both routes land
near 0.98.

> **A CI EXCLUDING ZERO IS NOT A ROBUSTNESS CHECK.** VideoMAE gave gaps of −0.010
> [−0.015, −0.006] and **+0.004 [+0.000, +0.008]** — both significant, pointing
> **opposite ways**, from the same captures, differing only in where inside the pre
> window the target was read. At n = 300 the bootstrap certifies four thousandths.
> Significance at the ceiling certifies precision, not robustness; the sign has to
> agree too. The script now enforces a cross-convention sign check. **Any best-vs-best
> comparison in this project running near ceiling needs it.**

## 9. Methodology — five distinct routes into the same false positive

Transferable, and the reason to trust everything above.

| trap | instance | fix |
|---|---|---|
| **identity in the target** (§12) | `p = m·v`, `m_A/m_B = \|dv_B\|/\|dv_A\|` | write the target as a function of what a camera sees; if exact, the bar is an oracle, not `shuf*` |
| **correlated visible scalar** (§32) | mu ≡ deceleration on a level floor (corr 0.99976) | separate them by construction (the incline) |
| **pooling-mask leak** (§20) | `cube_size` 99% recoverable from the mask | `mask_only_floor.py` beside every object-local number |
| **rendering leak** (§32a) | frame-0 position pinned by the sampler | jitter the start |
| **single-frame readability** (§32b) | released from rest ⇒ any frame's position *is* the acceleration | jitter + launch speed; check *every* frame, not frame 0 |

**Three of the five were invisible to every validator and were found by looking
at the clips.** Publish a contact sheet and look at it before probing.

Two further estimator traps, both of which produced "confirming" answers by
construction:

* **Ridge is linear in Y** (§34b). If a target is a linear combination of two
  others, `w_ratio = w_B − w_A` to 1e−15 — so composition, subspace containment
  and INLP all confirm by algebra. Check before comparing fitted weight vectors.
* **The fidelity-matched oracle is unfalsifiable** (§34c). Its excess swings
  from +0.12 to −0.14 with the analyst's choice of basis. Use the composition
  test instead: it decodes the model's own factors and has no basis to pick.

And one about label shape (§34a): `momentum_conserved` is a **threshold on a
norm**. Hand a linear classifier the exact, noiseless Δp vector and it scores
**AUC 0.559**. Where a target needs a nonlinearity, put it in the readout
*after* a linear probe and permute the whole pipeline for the null.

## 10. Capability limits

**Prediction is closed on all three checkpoints.** CogVideoX cannot inpaint a
coherent two-ball continuation (§7); VideoMAE cannot reconstruct; V-JEPA2's
predictor adds nothing over the encoding it was computed from (§15, rebuilt
with object-local pooling in §21 after finding §15's readout was 12× too blunt —
the verdict survived on stronger evidence).

**Consequence:** everything in §18 and §31 is **representation-level steering,
not video-level steering.** There is no ground truth and you cannot show a
steered clip. Closing that needs a decoder trained from frozen encoder latents
to pixels (with a mask-only-style control designed in, or it will hallucinate
plausible scenes regardless of the latent), or a generatively competent model.

---

# Part 2 — withdrawn, superseded, reclassified

| claim | status | why |
|---|---|---|
| "The representation has combined mass and velocity into something momentum-shaped" (§3) | **withdrawn** | §26: at the object level the combination demonstrably has not happened |
| "Conservation is not decodable" (§1, §2) | **withdrawn** | §34a: an artifact of the binary label's concentric class geometry, not a property of the models |
| Mass-ratio composition / subspace / INLP evidence (§29 tests 4–5, §26 rows) | **withdrawn** | §34b: `log(m_A/m_B)` is a linear combination of two probed targets, so ridge returns the confirming answer by construction |
| Fidelity-matched oracle everywhere (§13, §26, §30) | **withdrawn** | §34c: basis-dependent, swings +0.12 → −0.14 |
| "Momentum is not linearly accessible" | **wrong as stated** | §34e: `momentum_pre` reads 0.825 frame-wide |
| The INLP "sawtooth" (§30b) | **withdrawn** | tracks ridge-alpha selection exactly; also withdraws a §22 comparison row |
| mu results on `slide_set_v2` (§30, §30a) | **reclassified** | §32: `a = mu·g` on a level floor, so these are acceleration results |
| `incline_set_v1` (§32) | **void** | §32b: single-frame leak, mu at R² 0.88–0.99 |
| `probe_set_v2_latent` | **superseded** | §11, the antiparallel leak |
| Conservation null on v3 (§14) | **not evidence about models** | a theorem about the dataset |
| Per-head capture `outputs/heads/` | **deleted, never analysed** | ~110 GPU-min to redo |

---

# Part 3 — open

1. **§37's reproduction gap** — `momentum_delta` 0.517 vs 0.205–0.261 at the
   same cell under a different fitting protocol. No GPU. Settle before writeup.
2. **Render the `visible_mass` arm** with the v3 violation — the one untested
   cell, and what the `beta`/`pull` arms of `lawful_errors.py` need.
3. **Intervention-compensation** — patch A's pre-collision velocity, read B's
   post-collision velocity, ask whether it shifts by −(m_A/m_B)ΔV_A. The purest
   form of the law-use question, and the rig exists (§18, §31).
4. **Widen theta** on the incline to make mu a real quotient (15–70°, needs a
   shape that does not topple).
5. **The joint-patch prediction (§18d)** — if cross-object redundancy is an
   error-correcting channel, patching A *and* B should transfer far more than A
   alone, and would explain V-JEPA2's 0.39 against VideoMAE's 0.86. No new
   capture or rendering.
6. **Adopt Joseph et al.'s held-out evaluation probe** (§22) — trained on the
   test split only, plus their "MAE-to-target down, MAE-to-truth up" check.
7. `collide_set_v1` is paused at 21/150; the `visible_friction` arm is
   unrendered, without which §31's surface-attribution null is unfalsifiable.
