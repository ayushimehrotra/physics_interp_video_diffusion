# physics_interp_video_diffusion

Probing whether **CogVideoX-2b internally represents momentum and momentum
conservation**. Videos come from a Kubric/Bullet simulator with exact physics
labels; the model's internal activations are recovered along an inverted latent
trajectory; linear probes are fit from activations to labels.

Method follows `diffusion_interp.pdf` in this directory ("The Invisible Hand of
Physics", Esmati/Nath et al.). Section 3.1 is reverse sampling, Section 4.4 is
the quantitative-variable probing this project extends to momentum.

---

## How this file is organised

Eight parts, grouped by **the physical factor being tested** rather than by the
order things were run. Section numbers (§1-§34) are unchanged from when they were
written, so every cross-reference in the file and in the code still resolves; they
are simply no longer in numeric order.

| part | what is in it |
|---|---|
| **Methodology** | the traps every experiment here has hit. Read §12 before designing any probe target |
| **Momentum** | the original question, and the one that keeps turning out to be undecidable |
| **Mass** | the latent-mass redesign and the mass-ratio line |
| **Velocity** | the project's cleanest positive — per-object steering |
| **Friction** | the second physical domain, and the set that is currently being re-rendered |
| **Capability** | what the models can do at all: prediction, reconstruction, generation |
| **Next** | the resume block. Start here after reading CURRENT STATE |
| **Reference** | environment, layout, reproduction commands, design decisions. Everything that is not a finding |

Nearly every obvious next step has already been tried. Read the relevant part
before proposing new work — the ones that failed mostly failed for *structural*
reasons that no amount of engineering fixes.

## UPDATE — 2026-10-23 (A100 box, VideoMAEv2-giant + V-JEPA2-L, new 1000-clip sets)

* **§62 — FROM PRE-COLLISION CLIPS, BOTH ENCODERS HOLD THE VALID-COLLISION LINE BUT NOT
  THE POINT ON IT.** Legal outcome decoded at R2 0.68 / 0.71 at k = 20 against a 0.93
  ceiling (restitution is latent until impact, and corr(implied e, e) is ~0 until contact).
  **V-JEPA2's pre-collision prediction is structured toward the valid subspace at k = 19–20
  (p = 0.005 on all three statistics); the giant's is not until the aftermath is visible.**
  Violations are represented as a momentum DRIFT in the right direction (cos 0.61 / 0.56,
  35–49% of its size, zero impulse change), detectable with the models' own masses at AUC
  0.63 / 0.58 (paired p 1e-13 / 3e-8) — but the models' pre-collision expectation barely
  registers them (AUC 0.54 / 0.51). §42f's ceiling, not law application.
* **§63 — KE MOVED BY THE REDUCED MASS ALONE READS AT CEILING (within-M slope 1.00, and
  1.00 / 0.99 holding material fixed), BUT THERE IS NO SHARED KE DIRECTION ACROSS
  ROUTES.** Speed-calibrated readout applied where only mu moves: ~0 (source-refit CIs
  contain 0, exclude 1 and 2); mass-calibrated applied where only speed moves: 0.17–0.24.
  The naive cross-family slope of ~1 equals transferring log mu and log v_rel separately
  and adding them (§34b). Open thread: holding material fixed lifts S->M to 0.42 / 0.46,
  plausibly apparent size within material.
* **§64 — §62 REDONE WITHOUT LABELS: WEAK, AND THE DESIGN IS WHY.** Activation-only
  pre->post map on honest clips, off-residual-subspace score: AUC 0.56 / 0.55 (giant 291/497
  paired at k = 20, p 8e-5), +0.02–0.03 over a plain PCA control. Rank-insensitive, and the
  violation moves ALONG the residual directions. Each seed has ONE honest future, so no
  per-scene valid subspace is identifiable without labels. Needs a multi-e-per-seed set.
* **`conservation_set_1000` has three truncated clips** (`elastic_mixed_violating_seed255/
  262/264`, 41–42 frames) marked `_COMPLETE`. Re-render them.

## CURRENT STATE — 2026-08-26

* **§46f — THE MODEL IS SURPRISED BY IMPOSSIBLE ENERGY AND NOT BY IMPOSSIBLE
  MOMENTUM. The strongest result in this document, and nothing is fit.** V-JEPA2's own
  predictor, matched twins, exact sign test, ball-occupied tokens. Energy (`e = k`
  impossible vs `e = 1/k` legal, 20 pairs): **19/20, p = 0.0000** on l1 and mse, 18/20
  on cos. Momentum (honest vs violating, **150** pairs): 84/150, p = 0.083, null in all
  four cells, and the per-tubelet trend runs BACKWARDS, which kills the residual lean.
  n = 150 detects 57.3% pair agreement; energy shows 95%, so the momentum null is not
  power. **§26 predicts exactly this** — `e` is a mass-free ratio of two visible speeds
  so a prior over it is learnable from video, while a momentum violation is visible only
  by relating the deflection SPLIT to the MASSES, the product the representation does not
  form. Behavioural confirmation of the format claim.
* **§46f-i — THE ANCHOR ARM IS WHAT MAKES IT ATTRIBUTABLE.** The `k` / `1/k` pairing is
  symmetric in `e` and asymmetric by exactly a factor `k` in the velocity displacement
  the pixels show (`lambda = (1+e)/2`; measured ratio **1.500 in 20/20 seeds**), and a
  raw-departure detector reproduces §46d's probe result on its own (AUC 0.743, 20/20
  paired). Scoring the `e = 1` anchor as a third future refutes that account: it is the
  least surprising of the three in **0/20** seeds, and the ordering is
  `legal < anchor < impossible`. **Add a third arm to any twin design that can carry one.**
  What is shown is a **DISSIPATION PRIOR** (expected restitution below ~0.83), not
  conservation as a law — separating those needs the k-ladder the pilot lacks.
* **§46 — `energy_voe_pilot` RENDERED AND PROBED (§46d/e).** 20 seeds x 3 clips, all six
  validator checks pass, momentum held to **3.94e-16** while energy is created in 20/20
  impossible clips. The probe on it reads 0.830 / 0.825 against a **mass-free ceiling of
  1.000** — §44 reproduced, not new evidence, exactly as §46c predicted. §34's
  regress-then-norm instrument **loses** here (0.765 / 0.780): `energy_created` is
  one-sided and linear where `momentum_conserved` is a threshold on a norm, and the norm
  destroys what it rescues (§46f-iv). The full 420-clip set is still NOT rendered.
* **§45 — ASKED WHERE THE IDENTITY DOES NOT REACH, THE ANSWER IS THE SAME.** The
  friction sets have no second body, so the floor's recoil is unobservable and §44's
  mass-for-free identity does not exist. Fitting a KE readout on `slide_set_v2` and
  transferring it unchanged to `conservation_set_v1`: **VideoMAE +0.022 (chance),
  V-JEPA2 +0.275 (p < 0.005)** — against transferred-then-composed factors at **+0.291
  / +0.324**. Both controls cross the domain gap, so this is a negative and not a void
  test. The factors travel; the product adds nothing. **§45b also records a trap: two
  read conventions gave SIGNIFICANT gaps pointing OPPOSITE ways, so a CI excluding zero
  is not a robustness check — the sign has to agree too.**
* **§44 — ENERGY IS READ AS ELASTICITY, and elasticity is mass-free.** The binary
  "is this clip losing energy" decodes at AUC **0.946 / 0.983** (balanced accuracy
  0.873 / 0.920) across the full 9-cell pool grid, mask floor 0.500, `embed.out`
  0.62 / 0.60. But the label on the honest half IS `e < 1`, and `e` is a ratio of
  two visible speeds that needs no mass: a kinematic observer fitted through the
  identical pipeline reads **1.000**. Across 8 probe-vs-observer comparisons **not
  one 95% CI excludes zero in the probe's favour**, and six of eight put the probe
  significantly BELOW an observer that also sees the true kinetic energies. Frame-wide
  pools beat object-local everywhere, `.all` is the worst phase in every row, and
  `.pre` ties `.post` because attention is bidirectional (§4).
* **§39 — the factor code is OBJECT-AGNOSTIC, and this is the cleanest positive
  in the project.** A velocity readout fit on ball A **and never shown ball B**
  reads B at R2 **0.943** against B's own probe's 0.920, with the locality control
  failing at **-0.395**; per single token, 0.920 / 0.854. Object identity is WHICH
  TOKENS, not which directions. Both encoders; survives dropping the 24
  stationary-B scenes.
* **§39e — frame-wide pooling computes a SUM, which reframes §16.** `v_A + v_B`
  reads **0.878** frame-wide against 0.389 / 0.152 for the parts and **-0.353** for
  `v_A - v_B`. Frame-wide pools do not lose the objects, they superpose them.
* **§39j — CONCENTRATION and REDUNDANCY are different axes, and this set orders
  them oppositely.** Three principal components recover 0.89-0.91 of
  velocity/direction against a 768-d probe's 0.95, while speed reads ~0 from those
  same three. INLP says the reverse. Direction is both concentrated and redundant;
  speed is neither.
* **§39f-i CUT as replications** of `interpreting_physics.pdf` C.5 / §5.3 / C.4 and
  `cav.pdf`. §39j-bis is the row-by-row audit. Two claims were WITHDRAWN after
  failing their own controls — read §39h's remains and §39j's opening before
  trusting any geometry here.
* **§41 — THE MODEL DOES NOT COMPUTE THE COLLISION.** Patching ball A's
  pre-contact representation with a donor's (which lands A's own readout at 0.796)
  leaves B's post-contact velocity at `|required-observed|/|required|` = **1.002**
  (VideoMAE, identical to a random direction) and **0.990** (V-JEPA2, p = 0.005 but
  a 1% effect). The passive channel is **exactly 0.000000** in all 30 scenes, so
  anything arriving had to be computed. **But the aftermath is VISIBLE, so the
  model never needs to infer it — this shows it does not propagate, not that it
  cannot.** §41d.
* **§43 — TRUNCATION DONE. The model DOES anticipate the outcome**, at 0.584 /
  0.539 against a 0.683 linear-feature oracle and a 0.957 ceiling, holding every
  ingredient of the impulse law pre-contact (velocities 0.94-0.98, masses
  0.92-0.97, contact normal 0.87-0.93). With §41 that reads: the information is
  there and the causal path between the objects is not used.
* **§43g — four claims were withdrawn this session, all the same failure mode**:
  a number from one pool, one split, or a fitted weight vector. Read it before
  trusting anything here.
* **§38 — v1's violation mechanism was the OLD one, and `cos(dv_A, dv_B)`
  separates its classes at AUC 1.000.** Every momentum number in this document
  was measured on that set. `probe_set_v4_legacy_fixed` is v1 regenerated with the
  antiparallel mechanism and the `legacy` arm kept, so mass stays visible: 150
  scenes, 5/6 checks pass, both encoders captured.
* **§38g — §34's headline is now in question.** `momentum_delta` reads R2
  **0.255 / 0.139** on v4 against §34's **0.517** on v1, and the §34a norm route is
  **null in all six cells** (best p = 0.110). Not sufficient to call §34 an
  artifact, but its effect size is mechanism-dependent. **Re-measure §34 on v4
  under its full protocol before relying on it. This is the next job.**
* **§38e — the binary null is diagnosed, and it is not the models.** A linear
  classifier holding the *exact* masses and velocity changes scores **0.490**,
  chance: `Delta-p` is a product and the label a norm on top. Add quadratic terms
  and perfect information reads **0.893**, inside Joseph et al.'s range. Visible
  binaries reach 0.85-0.999 on this same pipeline at the same n. **Neither a bigger
  encoder nor more scenes addresses this.**
* **§38b — the fix opened a new mass-free channel.** `e > 1` (super-elastic) in
  **12/75** violating scenes and 0/75 conserving; VideoMAE `all.post` catches 11 of
  those 12 vs 36 of 63 clean. Bound `delta` so `e <= 1` and re-render (~36 min)
  before anything rests on v4.
* **§38d — object-local pooling does not help.** Frame-wide pools take the top
  three margins over their §20 floors; `B.post` is +0.013, essentially all leak.

---

### Where every experiment stands

| thing | state |
|---|---|
| Code (extraction + probes + dataset + steering + measurement + encoders) | **done**, 27 unit tests pass |
| `momentum_dataset/probe_set` — 150 scenes, v1 `legacy` arm | **done** — all 6 validation checks pass |
| `momentum_dataset/probe_set_v2_latent` — 300 scenes | **done but SUPERSEDED — do not probe, see §11** |
| `momentum_dataset/probe_set_v3_latent` — 300 scenes, `latent_mass` arm | **done**, 300/300, all OPTIX |
| `.venv` / kubric venv / CogVideoX-2b weights | **done** |
| Capture + probes, CogVideoX-2b on v1 | **done** — `outputs/` |
| Capture + probes, VideoMAE-base + V-JEPA2-ViT-L on v1 | **done** — `outputs_{videomae-base,vjepa2-vitl}/` |
| Capture + probes, both encoders on v3 | **done** — `outputs_v3_*/` |
| Capture, both encoders on v2 | done, but of a flawed set — **ignore** |
| Steering (directions, alpha ladder, 90-run sweep) | **done — negative** |
| Generative conservation test (inpainting) | **done — not answerable on this model** |
| Predictor / reconstruction tests, VideoMAE + V-JEPA2 | **done — both routes closed, §15** |
| Object-addressability cross-decode (§16) | **done**, both encoders |
| **Steering in an encoder** | **DONE — POSITIVE, §18.** Both encoders. This was the open question. |
| Activation patching, per-object (§18) | **done** — transfer 0.86 (VideoMAE) / 0.39 (V-JEPA2) |
| Polar vs Cartesian factorisation (§19) | correlational done; **causal test inconclusive by design** |
| Mask-leak control (§20) | **done — a standing control from now on**, revises some §13/§16 numbers |
| Predictor test, rebuilt with object-local pooling (§21) | **done — §15's verdict survives, better measured** |
| Four-velocity concat probe + its mask floor (§23) | **done — no gain over a single pool**, both encoders |
| **Composition test: are products stored? (§26)** | **DONE — NO. Revises §3's headline claim** |
| Pool list completed: `B.pre`, `A.all`, `B.all` (§27) | **done — corrects `velocity_B_pre` 0.74 -> 0.96** |
| Conservation via the Δp vector (§28) | **RESOLVED — POSITIVE (§34).** p = 0.000 against 500-permutation grid-max nulls in three models |
| Momentum-variable diagnostic protocol (§29) | tests 1-3 done; **§34b voids 4 and 5 for the mass ratio**; 6/7 not run and cheap |
| Two-ball friction set (§24, top-down) | **SUPERSEDED and DELETED** — see §25. Generator and validator kept |
| `slide_set_v1` — 300 scenes, cube on a friction floor (§25) | **done**, 300/300; superseded by `slide_set_v2` (banded floor) |
| `slide_set_v2` — 300 scenes, banded floor (§25) | **done**, 300/300, all 6 validation checks pass |
| Capture + probe, both encoders on `slide_set_v2` (§30) | **DONE** — mu decodes 0.93/0.97, and sits AT its kinematic oracle |
| Composition test on friction (§30) | **done** — direct ≈ composed; confirms §26's prediction for a *linear* functional |
| Steering mu, Joseph et al.'s held-out-probe protocol (§30a) | **DONE — passes, incl. the passive control they lack.** Re-derivation NOT established |
| Mask-leak floor on the slide set (§30) | **done — worst in the project**: `cube_size` is 99% mask |
| The INLP 'sawtooth' (§30b) | **artifact of ridge-alpha selection — revises §18b, withdraws a §22 row** |
| Floor-vs-cube causal patching (§31) | **DONE — mu is object-bound**; floor->cube transfer 0.011 vs cube->floor 0.097 |
| `collide_set_v1` — two cubes, shared floor (§31a) | generator + validator done; **render PAUSED at 26/150** |
| **`incline_set_v1` — mu vs acceleration (§32)** | rendered 150/150, all 7 checks green — and **VOID (§32b): mu reads off a single frame at R2 0.88-0.99.** Keep as the negative control |
| **`incline_set_v2` — the §32b fix** | **DONE 2026-08-21, 150/150 in 365 min, all 8 checks PASS.** Mask leak 0.988 -> **0.066**. This is the set to probe |
| Capture/probe on the incline set | **DONE (§35).** mu 0.423 / 0.779; direct vs composed +0.002 / -0.003; mu is 0.997 linear in (a, theta) over the sampled range |
| **Cross-geometry transfer, `mu` (§35c)** | **DONE — NEGATIVE, and this CLOSES the friction line.** A probe reading mu at 0.94 on one floor geometry transfers **none** of it to the other (R2 -0.17 to -0.93, both directions, both encoders) while `size` crosses the same gap at corr 0.83-0.95. The transferred probe tracks the other set's **acceleration** instead |
| §29 tests 1/4/5 on `mu` (§35b) | **VACUOUS — do not run.** cos(w_mu, c1*w_a + c2*w_theta) = 0.997-0.999; §34b in a second domain |
| Taxonomy of physical variables (§33) | written; **level 5 (scene structure) untouched by the field** |
| **`probe_set_v4_legacy_fixed` — leak-free v1, visible mass (§38)** | **DONE 2026-08-23**, 150/150 in 36 min, 5/6 checks pass. `cos` leak gone; **`e>1` leak on 12/75 remains** |
| Capture, both encoders on v4 (§38) | **DONE** — `outputs_v4_*` |
| Joseph et al.'s probe replicated exactly (§38c) | **done — changes nothing**, within noise of ridge. Matched pairs NOT built |
| Object-local pooling on the binary target (§38d) | **done — NEGATIVE**, frame-wide wins; `B.post` is +0.013 over its floor |
| Why the binary AUC is ~0.6 (§38e) | **RESOLVED — the label's algebra.** Exact ingredients score 0.490; +quadratic 0.893 |
| §34a norm route on v4 (§38g) | **DONE — NULL, 6/6 cells.** `momentum_delta` R2 0.517 -> 0.255 |
| **Re-measure §34 on v4, full protocol** | **NOT RUN — the next job.** Captures exist |
| **Factor geometry on v1's conserving 75 (§39)** | **DONE 2026-08-23**, both encoders — shared object-agnostic code, sum-pooling, speed⊥direction, PC anti-alignment |
| §34g's raw per-token capture, finally analysed (§39c-ii/§39f) | **DONE**, both encoders — per-token transfer 0.920 / 0.854, code delocalised ~17x the object |
| **`conservation_set_v1` — the leak-free 2x2 (§42)** | **DONE 2026-08-24. 300/300 clips in 207 min, 0 failures, ALL 9 GATES PASS.** Worst mass-free scalar 0.561 against a 0.053 null sd; joint classifier 0.444-0.514 in all four cells under grouped CV. Not yet captured |
| Per-head capture `block.*.attn_heads` | was 150/150 — **`outputs_heads/` HAS BEEN DELETED**, never analysed, ~110 GPU-min to redo |

### The headline results, in one place


* **READ §38 BEFORE THE MOMENTUM CLAIMS BELOW.** Every one of them was measured on
  `probe_set` v1, whose violating scenes are separable by one mass-free scalar at
  **AUC 1.000**. On the regenerated set the `momentum_delta` effect drops from R2
  0.517 to 0.255 and the norm route goes null.
* Physical variables are strongly decodable. **Conservation violations ARE
  detectable after all (§34 — but see §38g)** — `momentum_delta` decodes with p = 0.000 against
  500-permutation grid-max nulls in three architectures, from a pre-transformer
  baseline of ~0. The earlier "conservation is not decodable" was an artifact of
  the binary label's shape (§34a), not a property of the models. Whether the model
  *applies* the law is still undecided, and §34b shows the mass-ratio route cannot
  decide it even in principle.
* **Friction: THE MODELS HOLD NO GEOMETRY-INDEPENDENT COEFFICIENT (§35c).** A linear
  readout decoding mu at 0.94 within one floor geometry transfers **none** of it to the
  other (R2 -0.167 / -0.553 / -0.663 / -0.925 across both directions and both encoders),
  while `cube_size` crosses the identical domain gap at corr 0.83-0.95 — and the
  transferred probe predicts the target set's **acceleration** (corr +0.51 / +0.79)
  rather than its friction coefficient. This is empirical, not algebraic, so it stands
  independently of the identity trap. **The friction line is closed.**
* **Friction, within one geometry (§35).** mu decodes at 0.423 (VideoMAE) / 0.779
  (V-JEPA2); composition from decoded `a` and decoded `theta` scores 0.425 / 0.776.
  Acceleration decodes at 0.827 / 0.929, theta at 1.000 off the patch embedding in both.
  mu is 0.997 linear in (a, theta) over the sampled theta range, `corr(mu, a) = -0.645`
  — so per §35b the direct-vs-composed tie is an algebraic identity, not a measurement.
* **THE EXPERIMENT DESIGN IS WHERE THE FAILURES LIVE.** Five distinct routes into
  the same false positive have now been found: an algebraic identity in the target
  (§12), a target merely correlated with a visible scalar (§32), a leak through the
  pooling mask (§20), a leak through the rendering (§32a), and a target readable
  from one static frame (§32b). Three of the five were invisible to every validator
  and were found by looking at the clips.
* **PRODUCTS AND QUOTIENTS ARE NOT STORED (§26).** Read mass and velocity from the
  same object-local vector at 0.87 and 0.97, and their product reads at 0.66 —
  below a quadratic oracle. Compose the two decoded factors instead and you get
  0.89. Same result on the v3 mass ratio (0.64 composed vs 0.38 direct) and on
  total momentum. The representation stores factors **separately and linearly**;
  every combination is the readout's arithmetic, not the model's. This is a claim
  about FORMAT, and it is now the best-supported thing in the project.
* **"Probe beats oracle" is meaningless without naming the oracle's BASIS (§26).**
  Mass's apparent excess vanished the moment the oracle was given a quadratic
  basis (0.988 > probe). Momentum's survives, but composition beats the direct
  probe there too.
* **`p = m*v` and `m_A/m_B = |dv_B|/|dv_A|` are identities.** Every probe target
  built on them answers a definition rather than a question about the model.
  This is the single most important thing in this document — see §12.
* **Per-object velocity IS causally steerable in encoders (§18).** Patching one
  scene's A-local representation into another installs the donor's velocity at
  **0.86** of the required change (VideoMAE), with a clean double dissociation
  against radius. This is the project's second real positive and the answer to
  part 2.
* **§16's prediction is REFUTED (§18).** It inferred from "A's tokens decode B's
  velocity at 0.75-0.83" that steering A should move B nearly as much as A.
  Measured leak is **5-14%**, on both encoders, with an intervention that
  demonstrably engages the model. Co-location in a token pool implies nothing
  about shared directions: the two objects' velocity subspaces sit **~75° apart**
  and are separately removable.
* **Single-direction steering does not work, and now we know why (§18).** Velocity
  survives 12 rounds of INLP at R2 0.80, so it is encoded redundantly; a rank-1
  discriminative edit only fools the readout that shares its direction
  (observed/passive = 1.02). The *generative* direction works (1.40).
  Independently confirmed by Joseph et al. — see §22.
* The v3 `latent_mass` redesign **worked as designed** (`mass_A` R2 0.934 -> 0.28)
  and **still found nothing**: the mass ratio decodes at 0.49-0.58, which is
  exactly what a linear readout of the model's own noisy velocity estimates
  produces with no notion of mass anywhere (oracle 0.516). See §13.
* Momentum is **not an independently steerable factor** — `p = m*v` is an
  identity, so there is nothing to disentangle. No steering direction beat its
  own controls. Velocity is the well-posed replacement, and it works.
* The probes were **reading, not predicting**: attention is bidirectional, so
  post-collision quantities decode as well from pre-collision tokens.
* The model **cannot generate** a coherent two-ball continuation here, so
  generation-time physics questions cannot be posed on this checkpoint.

**Prediction remains CLOSED** on all three checkpoints — CogVideoX cannot inpaint
(§7), VideoMAE cannot reconstruct, and V-JEPA2's predictor adds nothing over the
encoding it was computed from. §21 rebuilt that test with object-local pooling
after finding §15's readout was 12x too blunt, and the verdict survived on
stronger evidence.

**Part 2 is answered, and the next questions are §18's follow-ups**: the
joint-patch prediction, a held-out evaluation probe, and video-level measurement
(which needs either a trained decoder or a different model). See "What to do
next".


---

## HOW TO READ THIS FILE

### How to read the results

`momentum_conserved` is binary and is scored by **accuracy** (chance = 0.50),
not R². Everything else is scored by R².

Three baselines print beside every number, and a result means nothing unless it
clears all three:

* `shuf*` — same pipeline on permuted labels, and **selected the same way the
  winner was**: the max over the whole (site x step) grid, not the winning
  cell's own shuffled score. This matters more than it sounds. `best site` is an
  argmax over 279 cells scored on *test*, so it carries a winner's curse, and
  the old per-cell `shuf` did not price it in — it printed 0.42 beside an
  accuracy of 0.68 on plausibility when the grid-max null was really 0.63,
  making a null result look like a 0.18 margin. **Read the margin as
  `best - shuf*`, never `best - 0.5` (or `- 0`).** `p_perm` beside it is the
  share of shuffled cells reaching the winner; treat it as a *lower* bound,
  since the cells are one permutation draw each and are correlated across
  neighbouring blocks and steps.
* `VAE` — the raw autoencoder latent. The paper's central claim is that physics
  is *absent* here (48–53% on plausibility) and constructed inside the
  transformer. If the VAE latent scores as well as a block, nothing was found.
  **For encoders this column is the patch embedding `embed.out` instead** — the
  same role: a linear readout that already works before any block has run means
  the quantity was in the pixels, not built by the transformer.
* `mass_A` / `radius_A` — readable off a single frame. Momentum has to beat
  these or the probe found appearance.

Expected shape if the paper replicates: best blocks in the **middle third**
(15–25 for CogVideoX-2b), signal weak at block 0–1 and at noise level 0.

**A fourth bar, added after §12-§13 and the most important of them: an
*identity* check.** Before trusting any number, write the target as a function of
what a camera can see. If it is an exact function, the probe is measuring
arithmetic on kinematics, and the number to beat is not `shuf*` but the **oracle**
— what a linear readout of those visible quantities scores at the fidelity the
model actually represents them. `momentum_pre` cleared its oracle (0.805 vs
0.695); `log_mass_ratio` did not (0.488/0.584 vs 0.516). Pooling matters here
too: comparing across pools is misleading, since appearance wins in object-local
pools and momentum in frame-wide ones, and *`log_mass_ratio` reads ~0 in every
frame-wide pool while reading 0.75-0.81 in object-local ones on v1* — a summary
that filters to `all.*` will report a null that is not there.

---


---

## PART 1 — METHODOLOGY: the traps every experiment here has hit

### 12. THE IDENTITY TRAP — read this before designing any probe target

Two of this project's three main negatives are the same mistake in different
clothes, and it is worth stating in general form.

**A probe target that is an exact algebraic function of quantities the model can
already see is not a test of the model. It is a test of arithmetic.**

*Case 1, momentum (v1).* `p = m*v`. Measured `R2(p_vector | m_A*v_A, m_B*v_B) =
1.000000`. So "is there momentum information beyond mass and velocity" has the
answer "no" for *every possible model*, by definition.

*Case 2, mass ratio (v3).* This is the subtler one, because the whole point of
the `latent_mass` arm was that mass is **invisible** — and it genuinely is. But
conservation says `m_A*dv_A = -m_B*dv_B`, and taking norms:

```
m_A / m_B = |dv_B| / |dv_A|
```

Verified on the 225 conserving v3 scenes: `median |log(|dv_B|/|dv_A|) -
log(m_A/m_B)| = 2.2e-16`, i.e. exact to machine precision. The right-hand side is
a ratio of two **pixel-visible kinematic** quantities. A model that tracks both
balls before and after contact and forms a quotient has the answer, with no
notion of mass, momentum or conservation anywhere.

The code comment in `generate_scene.py` claiming "a probe that reads it has used
conservation; there is no other route to it" is therefore true only by
definition: the identity *is* conservation. It cannot distinguish "the model
applies conservation" from "the model tracks two velocities and the experimenter
did the division".

Worse, the identity is **linear in the right coordinates**:

```
log(m_A/m_B) = log|dv_B| - log|dv_A|
```

so if the representation encodes *log* deflection magnitudes, a linear probe gets
the ratio for free. Confirmed: an oracle regressing `log_mass_ratio` on the true
`[log|dv_A|, log|dv_B|]` scores **R2 = 1.000**.

**The general rule.** Before adding a target, write it as a function of what a
camera can see. If that function is exact, the probe measures the model's
*kinematics* plus the probe's own ability to compose them — never a physical
concept. The only escape is a target that is **not** determined by the observable
trajectory, and in a two-body elastic collision almost nothing qualifies.

### 2. The winner's curse, and the `shuf*` fix

The summary table used to select the best cell by **test** score over 279 cells
(31 sites x 9 noise levels) while printing the permutation control for *that one
cell*. It printed `shuf = 0.417` beside `acc = 0.683`, implying a margin of 0.27.
The correct null is the max over the same grid on shuffled labels: **0.633**. The
margin collapses to 0.05, under one standard error (SE ~0.065 on 60 scenes).

This is predictable from the design alone. Under H0 each cell is
`Bin(60, 1/2)/60`, sd 0.0645, and the expected max of 279 draws is
`0.5 + 0.0645*2.69 = 0.674`. **Observed best: 0.683.** The result sits exactly
where selection noise puts it.

Fixed in `_best_and_null()` in `scripts/run_experiment.py`; the column is now
`shuf*` with a `p_perm` beside it. The variables results are unaffected (their
nulls sit near 0 against R2 of 0.8+).

**The same scepticism applies to the matched-momentum stratum.** The R2 = 0.685
once quoted for momentum there was a grid max over 279 cells with ~16 test
scenes. Do not cite it.

### 20. THE POOLING-MASK LEAK — a standing control, and it revises some numbers

**Object-local pooling selects which tokens to average using the ground-truth
segmentation.** Transformers carry positional information, so the pooled vector
encodes roughly *where the object is* — and for a target like post-contact
velocity, where the object is over time nearly **is** the answer. The choice of
pooling region is therefore an information channel independent of the tokens'
contents, and it is supplied by ground truth.

**The control: fit a probe on the pooling mask ALONE, no activations.** Whatever
it scores is the floor; object-local numbers must be read against it, not zero.

v1, V-JEPA2, 75 conserved scenes:

| target | A_pre | A_post | B_pre | B_post |
|---|---|---|---|---|
| velocity_A_post | −0.135 | **0.439** | 0.013 | 0.098 |
| velocity_B_post | 0.117 | −0.082 | −0.079 | **0.237** |
| velocity_A_pre | **0.258** | −0.123 | −0.200 | 0.058 |
| radius_A | −0.101 | −0.013 | −0.094 | −0.186 |

v3, both encoders, 225 conserving scenes:

| target | VideoMAE A.pre / A.post | V-JEPA2 A.pre / A.post | B.post (V-JEPA2) |
|---|---|---|---|
| log_mass_ratio | −0.017 / −2.470 | −0.024 / −0.243 | −0.321 |
| delta_v_A | 0.045 / −0.282 | 0.092 / 0.056 | −0.021 |
| delta_v_B | 0.056 / −0.036 | 0.039 / −0.221 | **0.356** |
| radius_A | **0.252** / −0.076 | 0.094 / −0.200 | 0.088 |

*Sanity check built in:* the `all.*` masks are all-ones and identical in every
scene, and score −0.002 to −0.032. If they ever do not, the measurement is broken.

**The pattern is mechanically predictable**, so you can anticipate where it bites:

| target type | leaks? | why |
|---|---|---|
| position/velocity from the *matching* phase pool | **yes, large** (0.24-0.44) | the mask *is* the trajectory |
| size / radius | **yes, moderate** (0.09-0.25) | the mask's *area* is the size |
| relational, needing both objects and both phases | **no** (<=0.003) | no single pool holds the ingredients |

**What this does NOT affect: every steering and patching number in §18.** The mask
is identical between the baseline and intervened runs of the same scene, so
leakage can inflate a readout's absolute R2 but **cannot manufacture a change in
it under intervention**. All of §18 is within-scene differences at fixed mask.

**What it does affect, as absolute values:**

* §13's `radius_A` (0.992 / 0.986) sits on a 0.252 / 0.094 floor, not zero.
* §13's `delta_v_B` (0.704 / 0.683) sits on a 0.356 floor for V-JEPA2.
* §16's cross-decoding matrix — `velocity_A_pre` from `A.pre` at 0.984 sits on a
  0.258 floor. The margin is still large and the key off-diagonal leaks little
  (0.013-0.117), so §16's central claim survives with the numbers restated.
* §13's `log_mass_ratio` (0.488 / 0.584) is **clean** — floors of −2.470 / −0.024,
  because the ratio needs both objects across both phases and a single A-pool mask
  holds neither. It was safe for the same structural reason that made it
  uninteresting.

**Joseph et al. do not have this confound** because they never use ground-truth
masks to pool: they mean-pool (identical operation in every scene), probe
per-patch at fixed coordinates, or use attentive probes whose pooling is learned
from the activations. Our segmentation pooling is forced by the *multi-object*
design — with one ball there is nothing to attribute. **The confound is a direct
consequence of the project's distinctive contribution.**

`fit_attentive_probe` + `PatchRecorder` are the leak-free alternative and are
already implemented, but the attentive probe is **non-linear**, so its number
bounds "present in the patches at all" rather than "linearly accessible" — not
interchangeable with any linear number here.

### 29. How to diagnose whether a momentum variable exists — the protocol

> **VALIDITY CONDITION ADDED BY §34b, and it voids some applications of this
> protocol.** Tests 1 (composition), 4 (subspace containment) and 5 (INLP) are
> **only meaningful when the target is NOT a linear combination of the factor
> targets**. Ridge is linear in Y, so if `Y_t = sum c_i Y_fi` exactly then
> `w_t = sum c_i w_fi` exactly at matched alpha, and all three tests return their
> "confirming" answer by construction. Valid for momentum (`p = m*v`, a product);
> **vacuous for `log_mass_ratio` against the log-magnitudes** (a difference).
> Use the raw or bounded ratio there instead — both are quotients (§34d). Also:
> **pin alpha** across any sequence of fits whose *shape* is being read (§30b).

`p = m*v` is an identity, so "is momentum information present" is trivially yes.
Four distinct claims, in increasing strength, and only the last three are about the
model:

**(a) present** — trivial. **(b) linearly accessible** — the product was computed
somewhere. **(c) a distinct factor** — momentum occupies a direction not reducible
to the mass and velocity subspaces. **(d) used** — the model's computation reads it.

| # | test | status |
|---|---|---|
| 1 | direct vs composed readout | **DONE — §26. The product is not stored** |
| 2 | expressivity-matched oracle (raw / quadratic / exact-product) | **DONE — §26. Mass excess was a basis artifact** |
| 3 | nonlinear probe against a nonlinear oracle | **DONE — §26. RBF lands on the composed number** |
| 4 | subspace containment: project `w_p` onto `span(w_m, w_v)`, measure the residual and the principal angles | **not run**, cheap, no GPU. The most direct test of (c) |
| 5 | INLP: remove the mass and velocity subspaces, re-probe momentum. Control: a random subspace of equal rank (§18 found velocity survives 12 rounds, so survival alone proves nothing) | **not run** |
| 6 | equivalence-class geometry: scenes with equal `p` but different `(m, v)` — the `matched_momentum` stratum is exactly this design — should collapse along a momentum coordinate and spread along mass/velocity ones | **not run**, the conceptual test for (c) |
| 7 | extrapolation: train on light-and-fast, test on heavy-and-slow at the same `p`. A real factor transfers; a memorised product of correlates does not | **not run**, cheapest strong test |
| 8 | steering with matched controls | **done — §5, negative.** No direction beat its controls |
| 9 | double dissociation: move `p` while `m` and `v` readouts stay put (§18's protocol, which succeeded for velocity) | **not run** |
| 10 | behavioural necessity: ablate and degrade a task needing momentum | **BLOCKED** — §7 and §15/§21 close generation and prediction on these checkpoints |

**Run 4, 6 and 7.** They are label-and-capture only and together settle (c). The
prediction from §26 is that `w_p` sits essentially inside `span(w_m, w_v)` and that
extrapolation fails — in which case the honest headline is "mass and velocity are
stored as separate linear factors; momentum is not its own variable."

### 33. A taxonomy of physical variables — and where the field has been

Five levels, and the ones being probed are all at the bottom.

| level | examples | probing it asks | status here |
|---|---|---|---|
| **1 object parameters** | mass, density, size, shape, material, albedo, inertia, stiffness | present, and bound to the right object? | mass/size done; visible ones trivial, latent ones oracle-bound |
| **2 interaction parameters** | friction mu, restitution e, rolling/spinning friction, adhesion, magnetic strength, spring constant, drag | which ENTITY owns it — object, surface, or pair? | mu (§30-§32); e measured at 0.10 uninterpretably. **The live frontier** |
| **3 state variables** | position, velocity, speed, direction, acceleration, spin, momentum, KE, impulse, separation, relative velocity, closing speed, time-to-contact | present and bound | saturated and identity-trapped |
| **4 LAWS** | conservation of momentum/energy/angular momentum, F=ma, `a = mu*g`, restitution law, Newton's third, inertia, solidity, permanence | *not a quantity* — a constraint that must HOLD | conservation null everywhere |
| **5 scene structure** | attached/jointed, connected by spring, supported-by, contained-in, occluding, CAUSED-motion-of, collision graph, grouping | the scene's relational GRAPH | **completely untouched, by us and by the field** |

**Momentum is level 3, not level 4.** `p = mv` is a DEFINITION; what governs is
*conservation*. Keeping those apart is §12's lesson in one line.

**Why the split predicts our results.** Levels 1-3 are quantities — "is this number
present" — and everything visible comes back yes at 0.6-0.98. Level 4 is
categorically different: you cannot probe a law like a scalar, only ask whether the
representation satisfies it or detects violations. And that is where everything
collapses (§4): observables sit at 0.6-0.85 while the RELATIONAL targets
(`momentum_delta`, conservation) sit at zero and chance. **Quantities are
represented; laws are not.**

**The one bridge from a quantity to a law**, and it is the only level-4 methodology
this project has:

> Make a parameter latent such that a law is the ONLY route to recovering it, then
> check whether the model recovers it beyond the kinematic oracle.

§13 did this for conservation (latent mass -> the ratio needs `m_A/m_B =
|dv_B|/|dv_A|`; answer: no). §32 does it for `a = mu*g`. Level 5 has no measurement
at all yet, and is the level a physics engine represents explicitly and a heuristic
does not represent at all.

**Ranked candidates that are novel, impactful and tractable** (all screened against
§12, the linear-proxy trap, §20 and §26):

1. **Time-to-contact** — never probed in a video world model; the canonical quantity
   in biological motion vision; the ONLY predictive target that does not require the
   model to generate. A quotient, so §26 predicts composition beats direct. Needs its
   own render: contact time is pinned at frame 20 +- 0.017 in every existing set, so
   TTC has no variance anywhere on disk.
2. **Hidden constraints** — "are these two linked?" via an unrendered rod or spring.
   Level 5, counterfactual, structurally leak-free, cheap (`pb.createConstraint`).
3. **Contact vs near-miss** — matched pairs, probed PRE-contact. Nonlinear geometric
   (miss distance vs sum of radii). Trivial generator change.
4. **Causal attribution** — did contact cause B's motion, or an unseen impulse with
   matched kinematics? Joseph et al. name force inference as out of scope.
5. **A scene-level constant across objects** — projectile reveals g, slider reveals
   mu*g; is g represented once?
6. **Restitution with latent/visible arms** — latent until impact, so the pre-contact
   read is a theorem-null and the visible arm is the positive control.
7. **Damping alongside Coulomb friction** — `a(v) = mu*g + b*v` separates mechanism
   from net deceleration; trivial to implement and breaks mu == a by a second route.

**Do not** add more targets to the mass/momentum family (§12), and do not build
anything that needs these checkpoints to generate or predict (§7, §15, §21).

#### 33a. Joseph et al. DO probe multi-object — with global mean-pooling

Checked against the paper text. §3.2: "linear probes on **mean-pooled space-time
patches**". App B: "**For all experiments** ... spatiotemporally pooled activations".
CLEVRER (§5.3, App C.3): "per-object probes on each of the objects", "all **48
objects**" — which is 8 colors x 3 shapes x 2 materials, i.e. object *identities*,
confirmed by their groupings by shape and material. And **no mention of masks,
bounding boxes, crops or per-object coordinates anywhere in the paper.**

So the implied method is one probe per object identity reading from the FRAME-WIDE
pooled vector. Two consequences: they genuinely have no mask leak (which is why §20
is ours and not theirs), but **our numbers predict their CLEVRER result should be
weak** — §16 measured frame-wide pooling costing `velocity_A_pre` 0.984 -> 0.548 and
collapsing `delta_v_A` 0.738 -> 0.199. Their Fig 14 values are in an image, not the
text, so this is unchecked. It is question #5 on the list to ask them.

### 22. Joseph et al. — independent corroboration and a method to borrow

`interpreting_physics.pdf` is Joseph et al., *Interpreting Physics in Video World
Models*, arXiv 2602.07050 (FAIR/Meta + Mila, Feb 2026). Same simulator (Kubric),
overlapping model (V-JEPA2-L), several of the same findings by different routes.
This work was done before reading it; the convergence is independent.

| their finding | ours |
|---|---|
| "Physics Emergence Zone" at ~1/3 depth | V-JEPA2 velocity peaks at block 7 of 24 = **29%** |
| steering at layer 8 of V-JEPA2-L | steered at block 7 |
| direction 14-136 dims, speed 16-31 dims (INLP) | velocity subspace ~16-24 dims |
| **1 probe -> >80° error; ~20 probes -> 12°** | **1-D discriminative obs/passive 1.02; multi-dim 1.40** |
| polar factorisation dominates | V-JEPA2 speed axis heading-invariant at 0.86 of ceiling |
| **sawtooth** across orthogonalisations, from sin/cos pairs, unique to direction | ~~our INLP curve is non-monotonic~~ **WITHDRAWN, §30b** — ours tracks ridge-alpha selection exactly. Their App B selects hyperparameters per probe too, so ask before citing theirs |
| smaller VideoMAE-v2 variants fail to show reliable emergence | likely explains our VideoMAE-base vs V-JEPA2 split (§19) |

**Their steering validation, worth adopting.** Steering probes fit on train (70%);
a **separate evaluation probe trained on the test split only** (R2 0.99), so it has
never seen the steering probes or the activations used to build the subspace.
Baseline MAE to target 82.9° -> 11.9° after steering with 20 probes. Their key
control: *as MAE to target falls, MAE to the true label rises correspondingly*,
which rules out "injected the target into a separate subspace" rather than moving
the representation. **We have no equivalent of that check**, and our readouts are
fit on train — weaker. Adopt both.

**What our controls catch that theirs cannot.** A purely passive edit would still
be read by their held-out probe, since it reads the same subspace the vector was
added to; §18a's `probe` condition (obs/passive 1.02) would have scored as a
success under their protocol. Our speed-consistency test is also a step further:
speed is a *nonlinear* function of the edited variable, so a scene-dependent,
sign-flipping response (r = 0.895) cannot come from linear propagation.

**Neither establishes behavioural consequence**, and they say so: "our methods
characterize representational accessibility and *coarse causal influence* rather
than a complete [account]". Representation-level steering has no ground truth
here — you cannot watch the ball move. Closing that needs pixels back in the loop.

**What remains distinct in this project.** Their limitations name our subject as
out of scope: they "do not isolate richer physical computations such as **contact
dynamics, force inference**, or long-horizon interaction." Their toy-ball dataset
is single-ball, straight-line, no collisions; their multi-object work (CLEVRER) is
*decoding*-based. So the open ground is **causal per-object addressability**,
**activation patching between real scenes** (they solve for synthetic target
coordinates instead), **two-body collisions**, and the **identity-trap
methodology**.


---

## PART 2 — MOMENTUM, AND WHETHER IT IS CONSERVED

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
momentum_dataset/plan_conservation_set.py       # physics-only; writes plan.json + probe cache
momentum_dataset/run_conservation_set.py        # resumable renderer, --workers
momentum_dataset/validate_conservation_set.py   # 9 gates, incl. the leak audit
momentum_dataset/conservation_set_v1/plan.json  # 300 clips, donor seed per violating clip

python momentum_dataset/plan_conservation_set.py --n-mixed 120 --n-same 30
KUBRIC_USE_GPU=1 .venv/bin/python momentum_dataset/run_conservation_set.py --workers 6
.venv/bin/python momentum_dataset/validate_conservation_set.py
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

`scripts/joseph_intphys_probe.py`. Their IntPhys probe is **not** `fit_probe`
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

`scripts/why_low_auc.py`, on v4, `all.post`.

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
(`scripts/leak_vs_probe.py`, out-of-fold per-scene predictions, v4):

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
predicted to reach ~0.75. `scripts/norm_route.py`, 200-permutation **grid-max**
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
5. **Lawful errors on v4** (`scripts/lawful_errors.py`). v3 was chosen for it
   because violations are undetectable there — but v3 also hides mass, so the model
   cannot apply the law even if it has it. v4 is the first set with visible mass AND
   no geometric shortcut, which is the setting where that test is meaningful.

#### 38i. Files

```
momentum_dataset/probe_set_v4_legacy_fixed/   150 scenes, 5/6 checks pass
outputs_v4_{videomae-base,vjepa2-vitl}/       150 captures each
scripts/joseph_intphys_probe.py               §38c/§38d — their protocol + §20 floors
scripts/leak_vs_probe.py                      §38f — out-of-fold recall by leak status
scripts/why_low_auc.py                        §38e — ceilings, controls, learning curves
scripts/norm_route.py                         §38g — §34a's two-stage readout
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
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
    --mode capture --model videomae-base --dataset-dir momentum_dataset/probe_set \
    -o outputs_patch_videomae-base \
    --patch-sites "block.1.resid_post,block.5.resid_post,block.8.resid_post,block.11.resid_post"
# same for vjepa2-vitl with blocks 1,7,13,22 -> outputs_patch_vjepa2-vitl
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
Measured with `scripts/mask_only_floor.py` (new, no GPU, rebuilds the pooling
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


### 36. Elastic vs inelastic on the existing sets — measurements

Run 2026-08-21 on existing labels and captures. No new renders. Recorded because
the four-way elastic/inelastic x conserved/violated grid was proposed and these are
the numbers it would be built on.

**Realised restitution vs the sampled label.** `e_real = -(v_rel_post . n) / (v_rel_pre . n)`,
with `n` the line of centres at `collision.actual_frame`, velocities at frames 18
and 22:

| set | scenes | corr(e_label, e_real) | median abs diff |
|---|---|---|---|
| `probe_set` v1, conserving | 75 | +1.0000 | 0.0001 |
| `probe_set` v1, violating | 75 | +0.9999 | 0.0000 |
| v3, conserving | 225 | +0.9999 | 0.0001 |
| v3, violating | 75 | **+0.6333** | **0.6754** |

v1's violation adds a common `dv` to both balls, which cancels in `v_A - v_B`, so
restitution is untouched. v3's scales the two impulse magnitudes, which does not
cancel.

**`cos(dv_A, dv_B)` on violating scenes.** v1: -0.9999 to **+0.9993**. v3: -1.0000
to +0.2404. Conserving is -1.0000 in both.

**Kinetic energy, conserving scenes only.** `KE_post/KE_pre`, frames 18 -> 22:

| | `probe_set` v1 (n=75) | v3 (n=225) |
|---|---|---|
| median | 0.839 | 0.940 |
| range | 0.180 - 1.000 | 0.119 - 1.000 |
| >= 0.98 | 9 (12%) | 70 (31%) |
| 0.90-0.98 | 18 (24%) | 69 (31%) |
| 0.70-0.90 | 20 (27%) | 52 (23%) |
| < 0.70 | 28 (37%) | 34 (15%) |

`KE_loss = 0.5 * mu_red * (1 - e^2) * u_n^2` reproduces the observed loss at
corr **+1.00000**, median relative error 1e-4, in both sets.

**KE ratio is not restitution.** corr(e, KE ratio): v1 conserving **+0.548**, v1
violating **+0.058**; v3 conserving +0.307. On violating scenes the aftermath is
overwritten, so KE ratio runs to **2.63** — energy is created. By impact parameter
`b = |b|/(r_A+r_B)`, v1 conserving median KE ratio: b<0.3 **0.69** (n=23),
0.3-0.6 0.81 (n=31), b>0.6 **0.92** (n=21).

**Probes for restitution and energy, VideoMAE on `probe_set` v1.** Mean held-out R2
over 20 random 60/40 splits, standardised features, fixed alpha 10 (NOT the
`fit_probe` protocol — a separate one-off):

| cell | restitution | KE ratio |
|---|---|---|
| blk 5 all.pre | -0.040 | **0.459** |
| blk 5 all.post | 0.014 | 0.417 |
| blk 9 all.pre | 0.235 | 0.334 |
| blk 11 all.pre | **0.248** | 0.222 |
| blk 9 all.all | -0.112 | 0.025 |

**Retrospective 2x2 on `probe_set`, split at the median restitution 0.53.**
Violation AUC via the §34a route (regress `momentum_delta`, take the norm of the
prediction, score AUC against the conserved/violated label), VideoMAE all.pre
block 9, mean over 40 splits:

| cell | n | violating | AUC |
|---|---|---|---|
| all 150 | 150 | 75 | 0.675 |
| e > 0.53 | 75 | 42 | 0.531 |
| e <= 0.53 | 75 | 33 | 0.579 |

Cell sizes in the existing set: elastic/conserved 33, elastic/violated 42,
inelastic/conserved 42, inelastic/violated 33.

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

### 44. ENERGY — the models read ELASTICITY, which is mass-free, and that accounts for all of it

`scripts/probe_energy.py`, run 2026-08-25 on `conservation_set_v1`, both encoders,
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

* `scripts/probe_energy.py` — the grid, `--mask-floor`, `--transfer`,
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

`scripts/energy_composition.py`, run 2026-08-25, both encoders.

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

* `scripts/energy_composition.py` — `--test A`, `--test B`. numpy only, no GPU.
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

* `momentum_dataset/plan_energy_voe_set.py` — physics-only feasibility, writes `plan.json`
* `momentum_dataset/run_energy_voe_set.py` — resumable render from the manifest
* `momentum_dataset/validate_energy_voe_set.py` — six checks, run it before probing

#### 46f. THE VoE MEASUREMENT, RUN — and the momentum contrast that explains it

Run 2026-08-26. `scripts/voe_energy_twins.py`, V-JEPA2 predictor on the 60-clip
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

`scripts/probe_energy_voe_graded.py`. §34a's fix is "regress the vector, put the
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

* `scripts/voe_energy_twins.py` — the measurement. **~178 s/seed on CPU** (17.1 s per
  20-frame encode, 10.0 s predictor, 31.8 s per 48-frame encode); 59 min for 20 seeds.
* `scripts/probe_energy_voe_graded.py` — §34's instrument, kept for the record that it
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

### 62. THE VALID-COLLISION SUBSPACE FROM PRE-COLLISION CLIPS, AND HOW VIOLATIONS LEAVE IT — `conservation_set_1000`

Run 2026-10-23 on the A100 box, VideoMAEv2-giant (40 blocks) and V-JEPA2-ViT-L (24
blocks). Numbered §62 to stay clear of the register's §47–§61 (`docs/register_gap.md`);
this is the register's §47 (valid-collision subspace on truncation rungs) and §49
(subspace membership, `--fit-honest`) rebuilt and extended on the 1000-clip set.

**The set.** 500 seeds x {honest, violating} twins, e pinned to 1.00 / 0.15, `mixed` /
`same` material: 200/50/200/50 pairs per cell. **997 of 1000 clips are usable — three
violating clips (`elastic_mixed_violating_seed255/262/264`) have only 41–42 of 49 frames
on disk despite carrying `_COMPLETE`**; `load_frames` refuses them. 497 complete pairs.

**Captures.** Full clips (`outputs_cons_<model>/`) and resample-truncation rungs
`--resample-to 10,14,17,19,20,22,25,30,38,48` (`outputs_construnc_<model>_a100/`). Contact
is at frame 20, so **k <= 20 is a genuinely pre-collision clip** — unlike the `.pre`
pools of a full clip, which have attended to the aftermath (§4). Only the `*.all` pools
are exact on resampled rungs. The `_a100` suffix keeps these apart from the partial
`outputs_construnc_*` the other box captured on an RTX 5090 (§58: never mix capture GPUs).
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
python scripts/capture_encoder.py --model <m> --dataset momentum_dataset/conservation_set_1000 \
    --out outputs_cons_<m>/momentum_probe/activations --keep-going
python scripts/capture_encoder.py --model <m> --dataset momentum_dataset/conservation_set_1000 \
    --out outputs_construnc_<m>_a100/activations --resample-to 10,14,17,19,20,22,25,30,38,48 --keep-going
# analysis, CPU only, ~6 min per model at 48 workers
python scripts/momentum_valid_subspace.py --model <m> \
    --full outputs_cons_<m>/momentum_probe/activations --trunc outputs_construnc_<m>_a100/activations
```

* `scripts/momentum_valid_subspace.py` — Parts A and B; `scripts/_fast_cv.py` — the same
  grouped-CV ridge as `stats.cv_predict` with every alpha read off one eigendecomposition
  per split (matches it to ~1e-6; constant features zeroed per §44e).
* `docs/artifacts/momentum_valid_subspace_{videomaev2-giant,vjepa2-vitl}.json`, logs in
  `docs/artifacts/logs_2026-10-23/`.

---

### 63. THE KINETIC-ENERGY READOUT WITH THE REDUCED MASS VARIED — `ke_factor_set`

Run 2026-10-23, same two encoders, all 1000 clips, full-clip captures `outputs_ke_<model>/`.

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
python scripts/ke_reduced_mass.py --model <m> --captures outputs_ke_<m>/momentum_probe/activations
```

* `scripts/ke_reduced_mass.py` — within-family grid, route-calibrated and raw transfer,
  composed-from-factors identity check, split-half ceiling, source-refit bootstrap, depth
  curve, gates, material partial. ~8 min per model at 40 workers, numpy only.
* `docs/artifacts/ke_reduced_mass_{videomaev2-giant,vjepa2-vitl}.json`.

---

### 64. §62 WITHOUT LABELS — a pooled conditional subspace, and why it is not the legal line

Run 2026-10-23/24 on the same captures as §62 (`outputs_cons_<m>`, `outputs_construnc_<m>_a100`),
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
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/momentum_unsup_subspace.py --model <m> \
    --full outputs_cons_<m>/momentum_probe/activations --trunc outputs_construnc_<m>_a100/activations --workers 56
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

## PART 3 — MASS

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
captures for both encoders still exist (`outputs_v2_*`, 192 MB) and are of a
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

## PART 4 — VELOCITY

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
v3, both encoders, 2026-08-13/14. Code: `scripts/steer_velocity.py`,
`steer_multidim.py`, `patch_velocity.py`. Results in `outputs_steer_*/`.

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

**Causal test (`scripts/steer_polar.py`) — INCONCLUSIVE, two design flaws.**

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

Run 2026-08-23. `scripts/geometry_v1.py`, six stages, CPU only, on the existing
`outputs_{videomae-base,vjepa2-vitl}` captures plus §34g's never-analysed raw
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
capture (`outputs_patch_videomae-base`, 1568 tokens x 768, block 5) had never
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
`scripts/geometry_v1.py`; the numbers are in the JSONs.

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
scripts/geometry_v1.py                  six stages; --stage, --moving-only, --alpha
outputs_{videomae-base,vjepa2-vitl}/momentum_probe/
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
hook. The deleted `outputs_heads/` was head outputs, not weights, so it would not
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

Run 2026-08-23. `scripts/causal_collision.py`. §40b-A2 executed.

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
scripts/causal_collision.py                          the experiment
outputs_steer_{videomae-base,vjepa2-vitl}/causal_collision.json
outputs_steer_vjepa2-vitl_pow/causal_collision.json  360-pair powered rerun
```


---


### 43. TRUNCATION — does the model ANTICIPATE the collision, or read it?

Run 2026-08-23/24. `scripts/truncation_curve.py`. §40b-A1 executed, and it is the
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
scripts/truncation_curve.py     the k-ladder capture + probe
scripts/causal_collision.py     §41
outputs_trunc_{videomae-base,vjepa2-vitl}.json
```


---

## PART 5 — FRICTION

### 24. The sliding-friction dataset — a second physical domain

`momentum_dataset/generate_friction_scene.py`, batched by the same
`run_batch.py` (`--script generate_friction_scene.py`). Two balls coast in
separate lanes across a **frictionless** approach region and cross onto **per-lane
friction patches** on frame 20, decelerating at `a = mu*g` from there. The floor
is what carries mu; the balls are rendered identically to each other.

**Why a second domain at all.** Every result in §1-§23 rests on one scenario, and
the project's single positive claim (§3: the probe beats the linear oracle on the
true factors) could be a fact about momentum, about this collision geometry, or
about the model. Friction is the cheapest way to find out — it reuses the camera,
the resolution, the frame count, the renderer settings and the entire capture
path. Three things it adds that the collision set cannot:

* **Deceleration is mass-independent.** `a = mu*g` for any mass, so `radius_A` and
  `mass_A` are genuinely free controls here. In the momentum set they enter the
  target algebraically, which is what made §12's trap unavoidable.
* **The quantity belongs to the SURFACE.** "Which ball is on which patch" is an
  object-to-surface binding question, and it is the multi-object structure this
  project is actually about.
* **A pre-entry read is not an identity.** Before a ball reaches its patch its
  motion carries nothing about the coefficient, so `mu_A` from a `*.pre` pool must
  come from the surface's *appearance*. That makes the two arms a matched pair:
  `visible_friction` is a positive control and `latent_friction` is a null that is
  a theorem about the dataset (§14's shape), not a measurement of the model.

**It does not escape §12, and the section does not pretend otherwise.** `mu = a/g`
and `a` is plainly visible, so a post-entry probe on `mu_A` measures kinematics
plus a division by a constant. Read it against a kinematic oracle the way §13
reads the mass ratio — never against zero.

**Physics traps, all found the expensive way and all fixed.** These are the reason
this took a working afternoon rather than an hour:

* **`useMaximalCoordinates=True` (kubric's sticky-walls fix) and a frozen inertia
  tensor are incompatible, and fail SILENTLY.** Together they produce NaN
  velocities from the first step, which kubric reports as a ball coasting forever
  at exactly its initial speed — a plausible-looking track that passes every eye
  test. `install_sliding_sphere_adder()` re-registers kubric's sphere builder to
  use minimal coordinates; nothing else changes, and there are no walls here for
  the original fix to matter to. Measured after the fix: deceleration
  **0.5999999 against mu*g = 0.600**.
* **A sphere with lateral friction ROLLS.** Measured: it decays to exactly `5/7`
  of its entry speed (1.3 -> 0.9286) and then coasts forever, so mu would control
  a transient duration rather than the deceleration. Rotation is therefore frozen
  with `localInertiaDiagonal = 1e6` — the ball is a puck with a round silhouette,
  which also keeps `video.py`'s circle tracker usable on this set.
* **The inertia override must be applied AFTER `scene += ball`.** kubric's `mass`
  observer fires during the add and `changeDynamics(mass=...)` recomputes the
  inertia tensor from the shape, discarding anything set earlier.
* **Sphere, not box.** A sphere's contact is a point, so the crossing is sharp;
  a box's contact patch straddles the boundary for `2*hx/v`, up to two seconds,
  smearing the entry across a third of the clip and contaminating `POST_FRAME`.
* **g = 2.0, not 9.81.** This world is not metric (14.4 units across the frame).
  At 9.81 an ordinary mu of 0.3 stops a ball in three frames with nothing to see.
  mu keeps its ordinary meaning; only the clock is slower.

**The sampler rejects nothing, and that is the point.** Every draw is feasible by
construction (the frame-margin arithmetic holds across the whole product of the
ranges), so mu cannot pick up a correlation with speed or radius *through
rejection* — which is exactly how the momentum/mass confound entered and forced
the v2 rebuild. Measured over 3000 draws: **0 rejections**, `corr(log mu, speed)
= -0.017`, `corr(log mu, radius) = +0.007`, mu log-uniform to within 0.01 of its
ideal quartiles, `uniform_floor` stratum 29.7% against a requested 30%.

**Validation** is `momentum_dataset/validate_friction.py`, six checks, exit code
gates a render. The sixth is the one that matters and needs the pixels: in the
latent arm the tile seam and the patches must be **invisible**, measured as the
rendered luma step across the boundary in floor-only pixels (segmentation-masked,
frame 0, both lanes). Pilot: mean |step| **0.13 of 255**, mu-driven swing across
the whole mu range **0.59 luma**. Two thresholds in that script scale with `n`
rather than being fixed, because a 6-scene pilot's correlations are noise at
+-0.6 and a fixed bar fails a clean set.

```bash
# 300 scenes, ~140 s/scene at steady state, 5 workers  (pilot: 6/6, all checks pass)
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 300 --output-dir momentum_dataset/friction_set_v1 \
    --script generate_friction_scene.py --arm latent_friction --workers 5

/workspace/kubric/.venv/bin/python momentum_dataset/validate_friction.py \
    --dataset-dir momentum_dataset/friction_set_v1 --out friction_set_v1/_validation.json

# capture needs NO changes: labels carry a `collision` block whose actual_frame is
# the friction-entry frame, which is what the phase split reads.
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir momentum_dataset/friction_set_v1 -o outputs_friction_$M
done

# the first probe. mu from PRE pools is the interesting cell, not the post ones.
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    -o outputs_friction_videomae-base --tag friction \
    --pools "A.pre,A.post,B.pre,B.post,A.pre+B.pre,A.pre+A.post+B.pre+B.post,all.pre,all.post" \
    --targets mu_A,mu_B,log_mu_A,log_mu_ratio,log_mu_scale,decel_A,decel_B,velocity_A_pre,velocity_A_post,radius_A,stop_frame_A
```

**Probe targets** are already wired into `DERIVED` (`mu_A`, `mu_B`, `log_mu_A/B`,
`log_mu_ratio`, `log_mu_scale`, `decel_A/B`, `log_decel_A`, `stop_frame_A/B`,
`patch_albedo_A/B`, `boundary_x`), and every velocity target above works
unchanged because the labels carry `collision.pre/post.velocity_A/B`.

**What to expect, written down before it is run** — this is the point of a
predicted null:

| cell | prediction |
|---|---|
| `mu_A` from `A.post` | decodes; it is `a/g` and `a` is visible. Beat the oracle or it means nothing |
| `mu_A` from `A.pre`, latent arm | **~0. No observer can do better** — nothing before the patch carries mu |
| `mu_A` from `A.pre`, visible arm | decodes, via albedo. The positive control that the pre-pool cell works at all |
| `log_mu_ratio` | needs both lanes; frame-wide pools should lose it, per §16 |
| `radius_A`, `mass_A` | decode, and are now *clean* controls — they cannot enter `a` |

A `visible_friction` render is the other half of that table and has not been run;
it is the same 300 seeds with `--arm visible_friction` into its own directory,
another ~4 GPU-hours. `--violate` exists and renders a patch whose albedo
advertises a coefficient the floor does not have — the one mismatch that needs
appearance and dynamics *bound together* to detect — but no violating scenes have
been rendered, and it is meaningless in the latent arm.

### 25. `slide_set_v2` — one cube, a friction floor, and a SIDE view

> **THE FLAT FLOOR WAS AMBIGUOUS — `slide_set_v1` is superseded by `slide_set_v2`.**
> With a uniform floor, a uniform background and a featureless horizon, the cube
> and its shadow are the ONLY things in frame that move: no parallax, no landmark,
> and no apparent-size change, because the motion is parallel to the image plane.
> A translating cube under a static camera and a static cube under a panning camera
> produce **identical pixels**. Nothing in v1's clips distinguishes friction from a
> camera move, which makes it a poor test of whether a model represents friction.
>
> The fix is a floor built from **bands across the direction of travel**, width 1.0
> world units, alternating between two albedos 9% apart (`FLOOR_BAND_WIDTH`,
> `FLOOR_BAND_CONTRAST`). In the side view they converge toward a vanishing point,
> so the frame carries fixed perspective structure the cube visibly crosses — up to
> six band boundaries per clip against 0.7-6.6 units of travel.
>
> **The pattern is identical in every scene**, so it cannot carry mu; only the arm's
> ramp scales it, and that scales both bands together. Measured: floor luma sd
> **0.027 of 255**, mu-driven swing **0.003**, corr(luma, log mu) = −0.05. Every
> band shares one coefficient and one segmentation id, so pool `B` is still "the
> floor" and nothing downstream changes. Costs ~16% more render time per scene
> (120 extra boxes). A first attempt at 1.5-unit bands and 14% contrast dominated
> the frame; the settings above are visible without competing with the cube.
>
> Everything below describes both sets; only the floor differs.

`momentum_dataset/generate_slide_scene.py`. **300 scenes, rendered, all six checks
pass.** This supersedes §24's two-ball top-down set, whose pixels have been
deleted (the generator is kept — its physics traps are still the record).

A cube slides across a level floor and decelerates at `a = mu * g`, exactly. The
camera is a side view, which nothing else in this project uses.

**The two objects are the cube and the floor**, and that is the point:

    id 1 -> pool "A" -> the cube    the mover; its deceleration is the evidence
    id 2 -> pool "B" -> the floor   the surface; it OWNS mu

So the central probe is a contrast, not a number. `mu` from `A` asks whether the
model reads the coefficient off the motion; `mu` from `B` asks whether it has
attributed the property to the *surface*. In `latent_friction` the floor is one
flat albedo, so **`B` should read nothing** — a prediction about the dataset.
`visible_friction` ramps floor albedo with mu and turns `B` into the positive
control. Only the latent arm is rendered.

**Why a cube and not a ball, measured rather than assumed.** A sphere with
lateral friction *rolls*: launched without spin it decays to exactly `5/7` of its
launch speed (1.5 -> 0.9286 measured) and then coasts forever, so mu would
control a transient duration rather than the deceleration. Two escapes were
tested and both are worse. Freezing rotation with `localInertiaDiagonal = 1e6`
gives perfect Coulomb sliding (0.5999999 against mu*g = 0.600) but **NaNs under
kubric's maximal-coordinates bodies**, reported as a ball coasting forever.
Rolling resistance works and is constant, obeying `a = (5/7) mu_r g / r`
(predicted 0.1143, measured 0.1144) — but that drags radius into the target. A
box slides cleanly, tips only if `mu > half_width/half_height = 1.0`, and keeps
`a = mu*g` mass- AND size-independent, which makes cube size a free control.

**Three tuning rounds, because the first two produced unwatchable clips:**

* `g = 2.0` (inherited from §24) put stopping distance `v^2/2mu g` at **0.18
  world units** for a rough floor — 13 px over the whole clip, a static video.
  `g = 1.0` with speeds 1.0-1.6 and mu 0.15-0.80 gives 47-469 px of travel.
* Camera at distance 24 rendered the cube at ~35 px with two thirds of the frame
  empty floor. At 14 it renders ~50 px.
* Camera height 0.75 was tried and is **worse than 1.1**: the cube's top edge
  intersects the horizon, splitting its silhouette across two backgrounds. At 1.1
  it sits wholly below the horizon against uniform floor.

**A generator bug worth not re-deriving.** kubric's `_BulletClient` disconnects in
its own destructor and PyBullet reissues client id 0 the moment it is free, so
calling `pb.disconnect()` on a rejected trial kills the *next* scene's live
connection — surfacing as `pybullet.error: Not connected to physics server` from
inside `simulator.run()`, which points nowhere near the cause. Scope each trial in
a function and `gc.collect()` before the next connect. §24's generator has the
same latent bug; it survived only because it rarely rejects after building.

**Also fixed, in the shared `write_mp4`:** every clip this project has ever
written had its `moov` atom *after* `mdat`. Players that cannot seek — a browser
handed the clip as a `data:` URI, for one — refuse to play such a file at all.
`-movflags +faststart` is now passed on both the imageio and system-ffmpeg paths,
so all three generators are covered.

**THE BANDED FLOOR COSTS A SYSTEMATIC +1.4% ON THE REALISED COEFFICIENT.** The
label `mu` is what the sampler drew; the motion the renderer actually produces
decelerates at `decel_measured`, and on `slide_set_v2` that is **above `mu*g` in
100% of 300 scenes**:

| mu band | n | median (measured − mu*g)/mu*g |
|---|---|---|
| 0.15-0.25 | 72 | +0.5% |
| 0.25-0.40 | 92 | +1.2% |
| 0.40-0.60 | 83 | +2.2% |
| 0.60-0.80 | 53 | +3.3% |

On the flat floor the same seeds sit at 2.9e-3 median, so the bands are the cause:
the cube crosses tile seams, and high-mu scenes stop sooner (median 16 moving
frames at mu>0.5 against 49 at mu<0.25) so there is less to average over.
`corr(err, n_moving_frames) = -0.774`. It is **not** a fit artifact — dropping the
final partial-stop frame changes the median from 1.40e-2 to 1.43e-2.

Consequences, none fatal but all worth knowing:

* **Probe `mu_effective = decel_measured / g`**, not just the sampled `mu`. Both
  are now targets in `DERIVED`. They correlate at **r = 0.9998**, so rankings
  barely move, but `mu_effective` is the label of what the pixels show.
* The error is **heteroscedastic in mu**, so any residual analysis should expect
  more label noise at the rough end.
* **16 of 300 scenes needed a resample** (15 `decel_mismatch`, 1
  `vertical_motion`) against zero on the flat floor — but
  `corr(log mu, n_rejections) = -0.038`, so acceptance is still not mu-dependent
  and the confound guarantee holds. Verified per scene from
  `sampling_attempts.rejections`; check it again if the geometry changes.

**Validation** (`validate_slide.py`, six checks, exit code gates a render). Full
set, 300 scenes:

| check | `slide_set_v2` (banded) | `slide_set_v1` (flat) |
|---|---|---|
| completeness | 300/300, no `_FAILED` | 300/300 |
| physics | median rel err **1.4e-2**, max 5.0e-2 | 2.9e-3, max 2.9e-2 |
| coverage | mu 0.151-0.797, quartiles within 0.10; 82% come to rest | quartiles within 0.30; 90% |
| confounds | `corr(log mu, speed) = -0.015`, `corr(log mu, size) = +0.011` | +0.078 / +0.053 |
| identifiability | path deviation min **53 px**, median 160 | min 47 px, median 138 |
| appearance leak | luma sd 0.029/255, **swing 0.002** | sd 0.028, swing 0.002 |
| render | 189 min, 5 workers | 133 min |

Zero rejections is the load-bearing one: acceptance never depends on the sampled
parameters, so mu cannot acquire a correlation with speed or size through the back
door — which is exactly how the momentum set's confound entered.

Two of the checks' thresholds **scale with n**, because a bar tuned for 300 scenes
fails a clean 8-scene pilot (the coverage check did precisely that at 0.47 against
a fixed 0.35, where the n-aware bar is 0.71).

```bash
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 300 --output-dir momentum_dataset/slide_set_v1 \
    --script generate_slide_scene.py --arm latent_friction --workers 5   # 133 min

/workspace/kubric/.venv/bin/python momentum_dataset/validate_slide.py \
    --dataset-dir momentum_dataset/slide_set_v1 --out momentum_dataset/slide_set_v1/_validation.json

# capture needs no changes: labels carry a `collision` block whose actual_frame is
# a nominal early/late split at frame 20 (there is no event — the floor is uniform).
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir momentum_dataset/slide_set_v1 -o outputs_slide_$M
done
```

**Predicted, before anything is probed:**

| cell | prediction |
|---|---|
| `mu` from `A.*` (the cube) | decodes; `mu = a/g` and `a` is visible. Beat a kinematic oracle or it means nothing |
| `mu` from `B.*` (the floor), latent arm | **~0. No observer can do better** — the floor is one albedo |
| `mu` from `B.*`, visible arm | decodes off albedo. The positive control, unrendered |
| `size`, `mass` | decode, and are *clean* controls: `a = mu*g` cannot contain them |

> **SCORED IN §30, and two rows were wrong.** `mu` from `A.*` decodes (0.93-0.97)
> and sits *at* its kinematic oracle, so the first row's warning was the right
> one. But: the `B.*` null is **not testable as designed** (attention is global and
> the floor mask is the cube's complement — measured 0.930/0.969 against a 0.796
> mask-only floor), and **`size` is not a clean control** — it is 99% pooling-mask
> area (probe 0.997 against a 0.988 floor). "Cannot enter the target
> algebraically" is a statement about the physics, not about the measurement.

### 30. The friction line, run end to end — probe, oracle, and steering

> **SUPERSEDED IN FRAMING BY §32 — READ THAT FIRST. Everything below is measured
> correctly and named wrongly.** `g = 1.0` on a level floor, so `a = mu*g` makes mu
> and the deceleration the SAME NUMBER: corr = 0.99976, and predicting mu from `a`
> with no fitting gives R2 = 0.995. Every "friction" result here is an ACCELERATION
> result, which Joseph et al. §5.2 already report. The composition finding survives
> (it is about linear functionals vs products, not about the name); the rest must be
> restated. §32 is the set that separates them.

`slide_set_v2`, both encoders, 2026-08-16. Captures in `outputs_slide_*`; probes
in `probes_friction.json`; steering in `outputs_steer_*/steer_friction_*.json`.
This is §25's "not started" experiment, and the predicted-null table it was
written against needs two corrections (below).

**First, the identity, measured rather than assumed.** `g = 1.0` in all 300
scenes, so **mu IS the deceleration numerically** — `mu == decel` to machine
precision, 300/300. And under constant deceleration

```
mu = (s[t1] - s[t2]) * fps          for any two frames while still moving
```

so mu is a **linear functional of the visible speed trace**. That is the whole
key to reading everything below, and it is a different position in §12's taxonomy
from anything else in this project: not a product (§26's `m*v`), not a quotient
(§13's mass ratio), but a *difference*.

**The oracle.** Exact speed trace -> mu at R2 **0.9996**. Degraded to the fidelity
a model actually has, with the readout free to average over all 49 frames:
**0.90 at 80% speed fidelity, 0.97 at 95%**. Only a two-frame difference collapses
(0.34 at 99% fidelity), and the model is not restricted to two frames. So the bar
here is ~0.90-0.97, not zero.

**Decodability.** Read on `all.*` — frame-wide, so the mask is all-ones and
identical in every scene and §20's leak is structurally zero (measured floors
-0.000 to -0.016):

| cell | VideoMAE | V-JEPA2 |
|---|---|---|
| `mu_effective`, `all.all` (whole clip) | **0.934** blk 8 | **0.970** blk 19 |
| `embed.out` baseline | -0.066 | -0.009 |
| grid-max shuffled null | -0.24 | -0.06 |

Nothing at the patch embedding, so the transformer builds it. Depth: 0.61 / 0.55
at block 0, plateau ~0.95 by block 5 (VideoMAE, 42%) and block 8 (V-JEPA2, 33%).
The argmax at blocks 17-21 is noise on a plateau — **do not read it as late
emergence.** This matches Joseph et al. on scalar magnitudes being available early.

**The composition test says nothing is stored.** Decoding the constituents from
the same vector and doing the arithmetic the identity prescribes:

| cell | v0 | t_stop | travel | DIRECT | comp `v0*fps/t` | comp `v0^2/2d` |
|---|---|---|---|---|---|---|
| VideoMAE `all.pre` blk 7 | 0.881 | 0.983 | 0.994 | **0.959** | 0.946 | 0.939 |
| V-JEPA2 `all.pre` blk 17 | 0.934 | 0.987 | 0.994 | **0.977** | 0.966 | 0.969 |
| V-JEPA2 `B.all` blk 17 | 0.878 | 0.991 | 0.993 | 0.957 | **0.966** | 0.960 |

Direct ≈ composed within ±0.03 across all 40 cells. **This is a successful
prediction of §26, on a new physical domain.** §26 found products are not stored
because a linear probe cannot multiply; the complementary prediction is that a
*linear* functional should be directly readable and composition should merely tie.
It is. Sharpened statement of §26's claim:

> The representation stores kinematic factors separately and linearly. **Linear
> functionals of them are directly readable; products and quotients are not.**
> Which side a physical quantity falls on is decided by its algebra, not by
> whether the model "understands" it.

**THE MASK LEAK IS WORSE HERE THAN ANYWHERE ELSE IN THE PROJECT.** Occupancy
convention, `scripts/mask_only_floor.py`:

| target | pool | probe | mask alone | margin |
|---|---|---|---|---|
| `cube_size` | `A.all` | 0.997 | **0.988** | 0.009 |
| `cube_size` | `B.all` (V-JEPA2) | 0.991 | **0.992** | **-0.001** |
| `travel` | `A.all` | 0.996 | **0.974** | 0.023 |
| `mu_effective` | `A.pre` | 0.945 | **0.730** | 0.215 |
| `mu_effective` | `B.all` | 0.930 | **0.796** | 0.133 |

`cube_size` is **not a control here — it is the mask**, whose area is the size;
the probe's margin over it is 0.005 and negative in one cell. §25's table calls
size and mass "clean controls" on the grounds that `a = mu*g` cannot contain them,
which is true about the physics and irrelevant to the measurement. The structural
reason this set is worse than the two-ball sets: **one moving object against a
static background means the segmentation mask is a near-complete description of
the scene's dynamics.** Any future single-object set has this worse.

**§25's predicted null on pool `B` is NOT testable as designed.** It predicted
`mu` from the floor ~0 in the latent arm because the floor is one albedo;
measured 0.930 / 0.969. But (a) attention is global, so floor tokens carry the
cube's motion whatever the floor looks like, and (b) the floor mask is the
*complement* of the cube's and encodes the same trajectory — floor 0.796. The
prediction was about appearance and the measurement cannot isolate it. Answering
it needs the `visible_friction` arm (still unrendered) as the positive control,
and ideally floor pools restricted away from the cube's path.

**`pre`/`post` are a misnomer on this set.** There is no event; `collision.
actual_frame = 20.0` is a compatibility shim so the capture path works unchanged.
`pre` is a fixed crop — tubelets 0-2, frames 3-18 for VideoMAE — applied
identically to every scene. It scores slightly *higher* than the whole clip
(0.959 vs 0.934) because the motion is concentrated early: median `stop_frame`
26.8, and **34% of scenes are fully stopped by frame 21**. The late window still
reads 0.925-0.964 by a different route — a stopped cube's resting position
encodes `travel`. Report `all.all` and call the others `early`/`late`.

> **A RULE, generalising §20.** Any pooling boundary derived from a per-scene
> label is an information channel — spatial or temporal. A segmentation mask
> leaks position; a phase split leaks whatever its boundary correlates with. The
> momentum set is safe by accident (contact is targeted to frame 20 ± 1 in every
> scene, so the split is near-constant). **Splitting this set at each scene's own
> `stop_frame` would be a severe leak: `corr(stop_frame, mu) = -0.873`.** Do not
> "improve" the nominal split into a physical one.

#### 30a. Steering mu under Joseph et al.'s held-out-probe protocol

`scripts/steer_friction.py`. Their App C.12 protocol reproduced exactly — INLP
probe sequence on train only, orthonormal subspace from the probe weights,
min-norm solve for the coordinates that make all K probes read the target while
leaving the complement untouched, and an **evaluation probe fit on TEST
activations only** (R2 0.969 / 0.974, against their 0.99). Edit written as a
uniform shift of every video token at `block.6` (VideoMAE) / `block.8` (V-JEPA2),
read at block 11 / 23, pooled `all.all` — uniform because pooling weights sum to
1, so the pooled mean moves by *exactly* the injected delta, which is what makes
the passive control below exactly computable, and frame-wide to stay clear of the
0.99 mask leak.

Target 0.35 (the set's geometric mean, so the required change flips sign in 57% /
55% of scenes; a fixed high target of 0.70 gives only 5% and is useless for the
consistency check):

| condition | \|delta\| | MAE->target | MAE->truth | obs/passive |
|---|---|---|---|---|
| baseline | — | 0.157 | 0.023 | — |
| k=1 | 0.07 | 0.143 | 0.027 | 2.55 |
| k=4 | 0.32 | 0.086 | 0.076 | 1.94 |
| **k=24** | 0.57 | **0.074** | **0.091** | **2.33** |
| random, matched norm | 0.57 | 0.149 | 0.032 | 0.22 |

**It passes their protocol** — MAE-to-target falls 53% (58% at target 0.70) while
MAE-to-truth quadruples, monotone in K, with a dead random control. **And it
passes the control their protocol lacks**: obs/passive 1.5-2.6, so VideoMAE
*amplifies* the edit rather than transporting it. Contrast §18a, where the
velocity probe direction sat at 1.02 and was pure arithmetic — that condition
would have scored as a success under their protocol, which is the criticism, and
this one does not depend on it.

Perturbation size is small because the edit is minimum-norm inside the subspace
that carries the variable: **1.6%** of the pooled activation norm for VideoMAE
(0.96 against 59.0), 4.1% for V-JEPA2. Compare §5, where CogVideoX needed ~4.6%
to do anything at all.

**V-JEPA2 resists**, as in §18c/§18d: only a 14-16% reduction, and obs/passive
**0.19-0.24** at low K — the network actively attenuates the injected vector,
recovering to ~1.0 only at k=24. Sixteen downstream blocks against VideoMAE's
five.

**mu is high-dimensional.** It survives **30 orthogonalisations at R2 0.47**, and
k=1 buys 8-9% against k=24's 53-58%. This reproduces Joseph et al.'s central
claim — coordinated multi-dimensional intervention is required — on a **scalar
magnitude**, which they never tested and which their own speed results (16-31
dims, no sawtooth) would have predicted to be low-dimensional.

**What is NOT established: re-derivation.** obs/passive > 1 is a *gain*, not a
computation, and a scalar gain leaves every cross-scene correlation unchanged.
The consistency check (does the `stop_frame` readout move as the steered mu
implies, scene by scene, `stop = v0*fps/mu`) measures r = 0.90-0.92 — but its
**passive counterpart measures 0.88-0.95**, excess -0.02 to +0.08. And the test
is under-powered by construction: `corr(mu* - mu, 1/mu* - 1/mu) = -0.917` over
this mu range, so what a linear edit delivers and what the physics implies are 92%
collinear. Worse, mu and `stop_frame` are *both* linear functionals of the same
represented speed trace, so any linear edit moving one moves the other.

**The fix, not yet run:** choose **per-scene targets** `mu*_i` such that
`(mu*_i - mu_i)` and `(1/mu*_i - 1/mu_i)` are uncorrelated across scenes. Two
degrees of freedom, no new capture, only the forward passes. A threshold-valued
consequence ("does the cube stop before frame 48") also breaks the collinearity,
since a threshold is strongly nonlinear.

#### 30b. THE SAWTOOTH IS A HYPERPARAMETER ARTIFACT — this revises §18b and §22

Every up-jump in an INLP curve here is a round where the selected ridge alpha
dropped from 1000 to 100:

```
round   1     2     3     4     5     6     ...  14    15    16    17
R2     .92   .85   .80   .77   .74   .72         .61   .81   .84   .61
alpha  100  1000  1000  1000  1000  1000        1000   100   100  1000
```

Re-running §18b's exact setup (velocity_A_pre, VideoMAE block 6, pool `A.pre`)
reproduces its published numbers — 0.930 at k=2, 0.960 at k=3, 0.889 at k=5,
0.956 at k=6, 0.803 at k=12 — and **every dip lands on an alpha-1000 round, every
recovery on an alpha-100 round.** So §18b's reading ("not instability — it is the
sawtooth Joseph et al. document, the signature of sinusoidal feature pairs") is
wrong, and the instinct it overrode (noise) was right. `fit_readout` selects alpha
on an inner split every round, and the alpha grid is discrete.

This also weakens the §22 corroboration row, which should be withdrawn: our
non-monotonicity is not evidence of their mechanism. **And it is a live question
about their result** — their App B sweeps 20 hyperparameter configurations per
probe selecting on validation, so the same mechanism is available in a pipeline
where the sawtooth carries a load-bearing interpretation (sin/cos pairs unique to
direction). Ask before citing it.

*Practical note:* nothing else in §18b changes. If anything "velocity survives 12
rounds at R2 0.80" is understated, since 0.80 is itself an alpha-1000 dip.

### 31. Is mu bound to the SURFACE or the OBJECT? — causal, and the answer is the object

`scripts/patch_surface.py`, run on the existing `slide_set_v2` captures. §30 read mu
at 0.93/0.97 from the FLOOR's token pool and could conclude nothing: attention is
global, so floor tokens carry the cube's motion whatever the floor looks like, and
the floor mask is the cube mask's complement, so it encodes the same trajectory
(mask-only floor 0.796). Decodability cannot answer this. Patching can.

**The design's one elegant property.** Patch the FLOOR's tokens from a donor scene
with a different mu, then read mu from the CUBE's tokens — and vice versa. The
patched tokens and the read tokens are DISJOINT pools, so the read pool's pooled
vector cannot move by arithmetic. The passive channel is not argued away, it is
*computed*: the read pool shifts by `(sum of its pooling weights over the patched
tokens) * delta`, and for floor->cube that weight sum comes out **exactly 0.000**.
Contrast §18a, where edit and readout shared tokens and a purely passive effect
scored obs/passive 1.02 and looked like success.

Passive-corrected transfer, bootstrap 95% CIs:

| condition | tokens | transfer | 95% CI | r(dmu required) | per token |
|---|---|---|---|---|---|
| **VideoMAE** floor->cube | 973 | 0.011 | [+0.003, +0.019] | 0.38 | 1.1e-05 |
| floor_sub->cube (count-matched) | 11 | 0.000 | [0.000, 0.000] | 0.29 | 2.0e-05 |
| rand->cube (matched norm) | 973 | 0.005 | [-0.001, +0.011] | 0.22 | 5.1e-06 |
| **cube->floor** | **11** | **0.097** | **[+0.060, +0.133]** | **0.67** | **8.8e-03** |
| **V-JEPA2** floor->cube | 3734 | 0.011 | [+0.005, +0.018] | 0.47 | 3.0e-06 |
| **cube->floor** | **50** | **0.048** | **[+0.021, +0.072]** | **0.49** | **9.6e-04** |

**The asymmetry is the result.** Replacing the ENTIRE floor's representation moves
the cube's mu readout ~1% of the required change, with a CI overlapping the random
control. Replacing the cube's 11-50 tokens moves the floor's readout 5-10%. Per
token the cube is **300-800x more effective**. So the floor pool's high
decodability is INHERITED, not intrinsic: information flows cube -> floor.

**What it does not establish.** In the latent arm the floor is one flat albedo and
*cannot* carry mu, so this shows where a dynamically-inferred mu lives, not whether
the model could bind an advertised material property to a surface. That needs the
`visible_friction` arm, still unrendered, where the sharp cell is mu from floor
tokens at frame 0.

*Geometric note:* at a 14x14 token grid the cube spans ~16-19 patches and NOT ONE is
pure cube — every one also contains floor. "Cube-only" tokens do not exist, so the
cube side is defined by DOMINANCE while the floor side stays strict. That asymmetry
is why floor->cube is the zero-passive direction.

#### 31a. `collide_set_v1` — two cubes on a shared floor, built and PAUSED at 21/150

`generate_collide_friction_scene.py` + `validate_collide_friction.py`. One shared mu,
two equal cubes, A (rear, fast) catching B (front, slow) at frame 14.5. Three things
it can ask that one cube cannot: is mu represented once or per-object; does a mu
readout fit PRE-collision still work POST; and the §26 composition test four times
over the same scalar. **Superseded in priority by §32 and stopped**, but the
generator is validated and the 21 rendered scenes are on disk and resumable.

Traps found the expensive way, all fixed and all worth not rediscovering:

* **Head-on geometry does not work.** Friction eats most of the speed before contact
  and restitution eats the rest: post-collision velocities came out 0.00-0.25, so
  there is no observable deceleration in exactly the phase the transfer question
  needs. Same-direction catch-up fixes it, and has a bonus — both cubes decelerate
  identically, so their RELATIVE motion is frictionless and the contact-time solve
  is exact with no quadratic.
* **The collision pitches the cubes** 9-33 degrees, and the tilt scales with the
  impulse, which depends on mu — so the silhouette would become a second channel
  carrying the target. §24's `localInertiaDiagonal = 1e6` reproduces §25's NaN
  warning here (the cubes never collide, reported as `no_collision`). **Angular
  damping is the fix**: drift drops to 0.00-0.02 deg with linear dynamics untouched.
* **Contact at frame 14.5, not 14.** `tubelet_phase_split` raises if a temporal cell
  holds frames from both sides of contact, and V-JEPA2's cell 6 is {13, 14}.
  Targeting 14 with +-0.13 frame of solver jitter puts contact at 13.93 in a third
  of scenes -> capture raises PER SCENE after the model is loaded. Target a cell's
  INTERIOR.
* **`segmentation_%04d.png`, not `seg_%04d.png`.** `video.load_segmentation` globs
  exactly the former and a mismatch does not degrade gracefully.

### 32. THE INCLINE SET — separating mu from acceleration. §30 WAS AN ACCELERATION RESULT

> **READ THIS BEFORE CITING ANYTHING IN §30 OR §30a.**

`g = 1.0` on a level floor, so `a = mu*g` means **mu and the deceleration are the
same number**. Measured on `slide_set_v2`: corr(mu, decel) = **0.99976**, and
predicting mu from the deceleration with NO fitting at all gives **R2 = 0.995**.
There is no scene in that set where a probe could tell them apart.

So every friction claim in §30/§30a is a claim about acceleration:

| §30 said | it actually showed |
|---|---|
| "friction decodable at 0.93/0.97" | the cube's **deceleration** is decodable, at its kinematic oracle |
| "mu steers under Joseph et al.'s protocol" | the **acceleration** variable steers |
| "mu is object-bound, not surface-bound" (§31) | **acceleration** is object-bound — nearly tautological once named |
| direct ~ composed (§26's format claim) | **survives** — that claim is about linear functionals vs products, not about the name |

And it is not even novel under the honest name: Joseph et al. §5.2 already report
acceleration decodable with high R2 from early layers.

**THE GENERAL LESSON, and it is broader than §12.** The identity trap said: beware a
target that is an exact algebraic function of the visible trajectory. This adds:
**beware a target that is merely LINEARLY CORRELATED with a directly visible
quantity, even when it is not an identity.** Measure `corr(target, nearest visible
scalar)` at DESIGN time, not after. That check now lives in `validate_incline.py`.

**The fix.** An incline breaks the identity, because theta enters differently:

```
a = g*(sin th - mu*cos th)        so     mu = (sin th - a/g) / cos th
```

theta is *visible* (the floor's orientation in the image) and drawn independently of
mu. Measured over 80k draws **corr(mu, a) = -0.65**, realised -0.52 to -0.67 on
rendered scenes, so **a linear readout of acceleration alone tops out at R2 ~ 0.27-0.45
on mu.** Anything above that is not reading acceleration.

**And it makes §26/§30's format claim FALSIFIABLE on the same variable.** On the flat
floor mu was a linear functional of the speed trace and direct ~ composed. Here mu is
a QUOTIENT, so the claim predicts the direct probe should now FALL SHORT of composing
decoded `a` with decoded `theta`. Opposite outcome, same variable, different geometry.

`generate_incline_scene.py` + `validate_incline.py`. Cube released from rest on a
tilted banded slope; floor and cube rotated by theta about y, gravity vertical.
theta in (28, 42) deg (a cube tips beyond 45), mu in (0.15, 0.40) log-uniform, g = 0.7,
contact-free so the phase split is a FIXED frame. Rejection-free over 80k draws.
Physics exact: `a` matches `g(sin th - mu cos th)` to a median 0.46%, and mu recovered
from (a, theta) matches the sampled mu to under 1%.

#### 32a. Three rendering bugs, and what they teach

**1. The camera at height 0 renders a BLACK WEDGE.** With `CAMERA_HEIGHT = 0.0` the
camera sits exactly IN the slope plane, so every surface point is edge-on: the slope
covered 50% of frame at **luma 0.6 of 255**. At 5.0 the surface is lit but fills 89%
of frame and the slope/background boundary — the ONLY cue for theta — leaves the
image. **1.5 keeps both**: lit bands and a diagonal horizon.

**2. Six validation checks passed on that black render.** The appearance-leak check
measures the mu-driven SWING of floor luma, which was correctly ~0 — on a floor that
was black. It answered its own question honestly while the premise underneath was
broken. `check_visibility` now asserts the slope covers >15% of frame, is lit above
luma 20, shows within-frame luma sd >1.0 (the bands resolve), and the cube is present
in every frame.

**3. THE FRAME-0 LEAK — the worst of the three, and invisible to every check.**
Centring each trajectory on the origin (`s0 = -travel/2`) makes the start position a
deterministic function of travel, hence of `a`. Measured: corr(s0, a) = **-0.9998**,
`a` from start position alone **R2 = 0.9987**, and since theta is visible in frame 0
too, **mu was recoverable FROM THE FIRST FRAME ALONE at R2 = 1.0000.** A probe could
have scored perfectly with the model representing no dynamics whatsoever.

> This is the identity trap arriving through the **RENDERING** rather than the
> physics — a third distinct route into the same failure. The start position is now
> pinned (`START_S`), spread 0.0000, and `check_frame0_leak` reconstructs `a` from the
> start position then mu from (a, theta) and fails the set above R2 0.25. Post-fix it
> reads **-0.35**.

**It was found by LOOKING AT THE CLIPS**, not by any validator. Publish a contact
sheet of every new set and look at it before probing.

### 32b. THE SINGLE-FRAME LEAK — `incline_set_v1` is void, and the fix for §32a caused it

Found 2026-08-20 by watching the clips and noticing the cube sits in different
places. **`incline_set_v1` rendered 150/150 with all seven checks green and is
still not probeable.** Nothing has been captured on it, so nothing published moves.

**The mechanism.** §32a pinned `START_S` because centring each path made frame 0
give mu away at R2 = 1.0000. Pinning it fixed frame 0 and broke every other frame:
with `s0` a constant every scene shares and the cube released from rest,

```
s(k) - s0 = 0.5 * a * t_k^2          exactly
```

so the cube's position in ANY frame IS the acceleration, theta is visible in the
same frame, and `mu = (sin th - a/g)/cos th`. Measured on the render, held out,
60/40 split seed 0:

| route to mu | R2 |
|---|---|
| cube mask centroid (x, y), frame 0 | **−0.010** — the check that exists, and it passes |
| cube mask centroid, frame 21 | **+0.877** |
| centroid + mask area, frame 21 | **+0.988** |
| world position + theta, any frame >= 6 | +0.996 – 0.997 |
| *acceleration alone — the bar the set was built to impose* | +0.459 |

Two numbers from one frame beat the entire point of the set. §20's mask-only floor
would sit near 0.99 here, the worst in the project.

**The frame-0 centroid does move** — 11 px in x, 14 px in y — but that spread is
theta (r = +0.95, −0.98), which is independent of mu (r = −0.07, +0.02). The check
was right about the frame it looked at and blind to the other 48.

> **RULE, and it is the third route into the identity trap.** §12 said: beware a
> target that is an exact function of the visible trajectory. §32 added: beware one
> merely *correlated* with a visible scalar. This adds: **beware a target that is a
> function of a single frame's static configuration**, and check EVERY frame, in
> PIXEL space. Three of the four traps found in this project arrived through the
> rendering or the sampler, not the physics.

**The fix, implemented in `generate_incline_scene.py` (uncommitted).** Add nuisance
draws to the initial condition until a single frame is not enough:

| unknowns in the initial condition | what one snapshot gives you |
|---|---|
| v1: none (`s0` fixed, from rest) | `a` — solved |
| + start jitter | 1 equation, 2 unknowns — needs 2 frames, a *velocity* |
| + launch speed `v0` | 1 equation, 3 unknowns — needs 3 frames, an *acceleration* |

Constants: `START_JITTER = 2.5`, `LAUNCH_SPEED_RANGE = (0.0, 0.60)`,
`GRAVITY 0.7 -> 0.45`, `CAMERA_DISTANCE 14 -> 18`, and the start anchored to
`-runway(theta, h) + jitter` rather than to a constant. g and the camera are pure
frame budget: v0 and the jitter consume up to 6.2 world units of runway. The start
may be anchored to theta and h because both are drawn independently of mu — what
v1 got wrong was anchoring to `travel`, which is `a`.

Measured through the real sampler, 80k draws, **zero rejections** (a rejection that
depends on travel depends on mu), and 20 subsamples of 150:

| | 1 frame | 2 frames | 3 frames |
|---|---|---|---|
| as rendered | 0.997 | 0.997 | 0.997 |
| **post-fix** | **0.077 mean / 0.29 max** | 0.46 | **0.997** |

`corr(mu, a)` is preserved at −0.61 to −0.74 across draws, so the dissociation §32
exists for survives. Cost: the smallest cube renders ~28 px across instead of ~36.

**`validate_incline.py` gained two checks** and lost `check_frame0_leak`:
`check_single_frame_leak` sweeps every frame in world coordinates (cheap, no
pixels) and `check_mask_leak` runs the same attack on the segmentation masks in
pixel space, which is what the model sees. Both gate at R2 < 0.25 over 20 repeated
60/40 splits — one split at n = 150 is too noisy to gate on. Both were confirmed to
**fail `incline_set_v1`** (0.997 and 0.975) and pass five independent post-fix draws
(0.008–0.066). `check_single_frame_leak` also asserts the 3-frame route stays above
0.8, so a set cannot pass by being unreadable.

#### 32b-render. `incline_set_v2` — rendered, and the fix holds on real pixels

150/150 in 365 min (2.4 min/scene, 2.3 GB), zero failures, **all eight checks pass**
— the six original plus the two new leak checks. Measured on the actual
segmentation masks, not on simulated trajectories:

| mu from the cube's mask alone | `incline_set_v1` | `incline_set_v2` |
|---|---|---|
| worst frame | **+0.988** (f21) | **+0.066** (f48) |
| frame 0 | -0.010 | -0.065 |
| worst single frame, world coords + theta | +0.997 | **+0.145** (worst split 0.227) |
| three frames + theta | 0.997 | **0.997** — the acceleration signal is intact |

And the dissociation the set exists for is unchanged: `corr(mu, a) = -0.645`, so
acceleration alone still caps at R2 = 0.416 on mu. `corr(log mu, start position)`
+0.007 and `corr(log mu, launch speed)` -0.062, so neither nuisance draw picked up
a relationship with the target. Two rejections in 150, uncorrelated with mu.

Two things the camera move cost, both within bars: the slope now covers 58.6-61.5%
of frame (was 61.7-65.6%) and the smallest per-frame cube is **874 px** (was 1465).
The cube is the object being probed, so if object-local decodability comes back
weak on this set, that 40% area reduction is the first thing to check.

**`incline_set_v1` is kept, not deleted.** It is the only set in the project with a
known, quantified single-frame leak, which makes it the negative control for
`check_mask_leak` — the check was verified by failing v1 at 0.975 before it was
trusted to pass v2 at 0.066.

**Re-render to a NEW directory** — `run_batch.py` resumes by `_COMPLETE` marker and
would skip all 150 existing scenes:

```bash
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 150 --output-dir momentum_dataset/incline_set_v2 \
    --script generate_incline_scene.py --arm latent_friction
python3 momentum_dataset/validate_incline.py --dataset-dir momentum_dataset/incline_set_v2
```

About five GPU-hours. `incline_set_v1`'s pixels are 2.2 GB and are worth keeping
only as the negative control for the mask-leak check.


---

### 35. The friction probe on `incline_set_v2` — numbers and method

Run 2026-08-21, both encoders, 150 scenes.

**Commands.**

```bash
# capture: one forward pass per encoder, ~6 min each, 150/150 no failures
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir momentum_dataset/incline_set_v2 -o outputs_incline_$M
done

# probe: seconds, no GPU
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --model $M -o outputs_incline_$M --dataset-dir momentum_dataset/incline_set_v2 \
    --targets mu,log_mu,theta,accel_measured,cube_size \
    --pools A.pre,A.all,B.all,all.pre,all.post,all.all --tag incline
```

Results: `outputs_incline_{videomae-base,vjepa2-vitl}/momentum_probe/probes_incline.json`
(**390 and 500 fits** — an earlier revision said 300. The V-JEPA2 run is also
**missing `all.post` and `log_mu`**, so the two grids are not matched; keep that in
mind alongside the cross-model caveat at the end). Protocol is `fit_probe` unchanged
— n=150, 60/40 scene split at `split_seed 0` (90 train / 60 test), standardised
features, ridge alpha from `(1e-2 ... 1e4)` chosen on an inner 75% carve-out of
train, `shuf*` = grid-max on shuffled labels.

#### 35a. `pre`/`post` ARE MISNOMERS ON THIS SET — read them as `early`/`late`

**There is no collision on the incline set.** A cube is released and slides; nothing
happens at any particular frame. `labels.json` carries `collision.actual_frame = 20.0`
with `nominal: true`, a compatibility shim so the capture path runs unchanged. This is
the same misnomer §30 flagged on the slide set, and it applies identically here.

What the temporal split actually is, verified across all 150 scenes of both captures:

| | VideoMAE | V-JEPA2 |
|---|---|---|
| frames sampled | 16, indices `[3, 6, 9, ... 48]` | 48, indices `[1 ... 48]` |
| temporal cells | 8 tubelets | 24 tubelets |
| `.pre` = **early** | cells **0-2** | cells **0-9** |
| `.post` = **late** | cells **3-7** | cells **10-23** |
| `.all` | cells 0-7 | cells 0-23 |
| **distinct splits across 150 scenes** | **1** | **1** |

So `.pre` is a fixed crop of roughly the first third of the clip, at the same cell
indices in every scene, aligned to no physical event.

**Keep the pool STRINGS as they are** — `A.pre`, `all.pre` are what `--pools` takes and
what the capture stores, so renaming them in the commands above would break every
reproduction. Rename them only when *reading*: `A.pre` is "the cube's tokens, early
window"; `all.pre` is "every token, early window".

Three consequences, and the first is the one that matters:

* **The temporal split is leak-free, by luck rather than design.** §30's rule is that a
  pooling boundary derived from a per-scene label is an information channel. Here the
  boundary is a *constant*, so it cannot carry `mu`. Contrast splitting at each scene's
  own `stop_frame`, which on the slide set would have leaked badly
  (`corr(stop_frame, mu) = -0.873`). The nominal shim does the right thing.
* **The SPATIAL split is per-scene and is where §20 still bites.** `A` is the cube
  (seg id 1), `B` the slope (seg id 2), `all` an all-ones mask. Quote every `A.*` number
  against the cube-mask floor of **0.066** — small, which is precisely what the §32b
  rebuild bought (v1 read 0.988 there).
* **Do NOT read `early` > `late` as anything temporal.** Both pools crop the same clip
  and attention is bidirectional, so an "early" token has already attended to the late
  frames — §4's point. The ordering says where information is *pooled from*, not when it
  becomes available. `early` wins here for a mundane reason: the cube is released from
  rest, so the early window holds the cleanest acceleration signal, while by the late
  window many scenes have already travelled most of their distance.

Nothing in the `mu` conclusion rests on the `early`/`late` contrast. The headline holds
at `all.all` too, where the mask is all-ones and the crop is the whole clip.

**Best cell over the grid.** (`.pre` = early window, `.post` = late; see §35a.)

| model | target | pool | site | R2 | shuf* | alpha |
|---|---|---|---|---|---|---|
| VideoMAE | `mu` | A.pre | block.7 | **0.423** | -1.180 | 10 |
| VideoMAE | `accel_measured` | A.pre | block.7 | 0.827 | -1.251 | 10 |
| VideoMAE | `theta` | all.all | block.0 | 1.000 | -1.894 | 0.01 |
| VideoMAE | `cube_size` | A.all | block.11 | 0.990 | -1.024 | 0.01 |
| V-JEPA2 | `mu` | all.pre | block.21 | **0.779** | -0.469 | 10 |
| V-JEPA2 | `accel_measured` | all.pre | block.21 | 0.929 | -0.499 | 1 |
| V-JEPA2 | `theta` | B.all | embed.out | 1.000 | -0.915 | 1 |
| V-JEPA2 | `cube_size` | A.all | block.10 | 0.992 | -0.862 | 0.01 |

`embed.out` at the same target/pool (VideoMAE / V-JEPA2): `mu` -0.002 / -0.068,
`accel_measured` 0.344 / 0.596, `theta` 1.000 / 1.000, `cube_size` 0.468 / 0.832.

`mu` by pool — VideoMAE: A.pre (cube/early) 0.423, A.all (cube/whole clip) 0.330,
all.post (frame-wide/late) 0.307, all.pre (frame-wide/early) 0.137, all.all 0.108,
B.all (slope/whole clip) 0.088. V-JEPA2: all.pre 0.779, all.all 0.709, A.pre 0.654,
B.all 0.651, A.all 0.617. **V-JEPA2 has no `all.post` row because that pool was not
probed in its run**, not because it scored badly.

**Composition test.** `scripts/compose_incline.py <model>`. For each (pool, block) it
calls `fit_probe` three times on the same feature matrix — targets `mu`,
`accel_measured`, `theta` — refits ridge at the alpha `fit_probe` selected, takes
held-out predictions on the same 60 test scenes, and forms
`mu_hat = (sin(theta_hat) - a_hat/g) / cos(theta_hat)` with `g = 0.45`. Scored by R2
against true `mu` on those 60.

| model | pool | block | mu direct | mu composed | a | theta |
|---|---|---|---|---|---|---|
| VideoMAE | A.pre | 7 | 0.423 | 0.425 | 0.827 | 0.989 |
| VideoMAE | A.all | 10 | 0.330 | 0.315 | 0.774 | 0.992 |
| VideoMAE | all.pre | 10 | 0.137 | 0.116 | 0.703 | 0.999 |
| VideoMAE | all.all | 8 | 0.108 | 0.053 | 0.700 | 0.999 |
| V-JEPA2 | all.pre | 21 | 0.779 | 0.776 | 0.929 | 0.999 |
| V-JEPA2 | all.all | 19 | 0.709 | 0.698 | 0.901 | 0.999 |
| V-JEPA2 | A.pre | 21 | 0.645 | 0.666 | 0.886 | 0.993 |
| V-JEPA2 | A.all | 21 | 0.612 | 0.616 | 0.873 | 0.995 |

Best direct anywhere vs best composed anywhere: VideoMAE **0.423 / 0.425**,
V-JEPA2 **0.779 / 0.776**.

**Linear approximation of `mu`.** OLS on the 150 true labels, in-sample R2, no probe:

| `mu` regressed on | R2 |
|---|---|
| `a`, `theta` | 0.9972 |
| `a`, `sin(theta)`, `cos(theta)` | 0.9975 |
| exact `tan(theta) - a/(g cos(theta))` | 0.9974 |
| `a` alone | 0.4160 |

Ranges over the set: `tan(theta)` 0.532-0.899, `1/(g cos(theta))` 2.52-2.99
(mean 2.74), `a` 0.059-0.245, sd(a) 0.0405, sd(mu) 0.0715.

**Error propagation.** `d(mu) = -d(a)/(g cos(theta))`. Residual sd of `a` taken as
`sqrt((1 - R2(a)) * Var(a))`, times the mean amplification 2.74, over sd(mu).
Assumes exact theta and homoscedastic error.

| model | R2(a) | implied R2(mu) | observed direct |
|---|---|---|---|
| VideoMAE | 0.827 | 0.584 | 0.423 |
| V-JEPA2 | 0.929 | 0.829 | 0.779 |

**Set-level numbers.** `corr(mu, a) = -0.645`, so a linear readout of acceleration
alone caps at R2 0.416. Cube-mask leak floor for `mu`: 0.066. All eight validator
checks pass (`momentum_dataset/incline_set_v2/_validation.json`).

**Caveat on the cross-model comparison.** V-JEPA2's best `mu` cell is frame-wide
(`all.pre` = every token, early window) and VideoMAE's is object-local
(`A.pre` = cube tokens, early window), and the camera move to distance 18 cut the
smallest per-frame cube from 1465 px to 874 px. The two grids also cover different
pool lists (see the fit counts above).

#### 35b. §29's tests 1/4/5 ARE VACUOUS FOR `mu` — §34b in a second domain

**Do not run INLP, subspace containment, or the composition test on `mu` against
(`a`, `theta`) on this set.** They confirm by construction, exactly as §34b found for
`log_mass_ratio` against the log-deflections.

OLS on the 150 true labels: `mu = -0.0207 - 2.6454*a + 1.1564*theta` at **R2 0.9972**.
So `mu` is a *linear combination* of two other probed targets, and ridge is linear in
Y, so the fitted weights inherit the relation. Measured at `A.pre` / block.7:

| ridge alpha | \|\|w_mu - (c1*w_a + c2*w_theta)\|\| / \|\|w_mu\|\| | cos(w_mu, combination) |
|---|---|---|
| 1 | 0.072 | **0.9974** |
| 10 | 0.068 | 0.9977 |
| 100 | 0.058 | 0.9983 |
| 1000 | 0.049 | 0.9988 |

Not the 1e-15 of §34b's exact identity — `mu` is affine to 0.997, not exactly — but
close enough that removing the (a, theta) subspace must destroy `mu`, projections must
show containment, and composition must tie, whatever the model contains. §35's own
direct-vs-composed rows (0.423/0.425 and 0.779/0.776) are that fact, not a measurement.

**This is why §32's falsifiable prediction never materialised.** The set was built so
`mu` would be a QUOTIENT and composition would beat the direct probe. Over
theta in (28, 42) deg, `1/(g cos theta)` spans only 2.52-2.99, so the quotient is
affine and a linear probe performs the composition itself. The test has no teeth at
this theta range; widening theta is what would give it any.

#### 35c. CROSS-GEOMETRY TRANSFER — the models hold NO geometry-independent `mu`

`scripts/transfer_geometry.py`. No GPU, no render, ~2 min on existing captures. This
is the one test of "is `mu` a variable" that **escapes §35b**, because it compares the
same coefficient across two datasets where it is a *different function* of the visible
kinematics:

```
slide_set_v2    flat,   mu = a/g  (g=1.0)           corr(mu, a) = +0.9998
incline_set_v2  tilted, mu = (sin th - a/g)/cos th  corr(mu, a) = -0.645
```

Pool `all.all` (all-ones mask, so no mask-geometry difference), restricted to the
overlapping `mu` range [0.151, 0.399] (slide 164 / incline 148 scenes), **each domain
standardised by its own moments** and the target z-scored per domain, so only the
direction has to survive. Ridge alpha picked on a within-source holdout.

| model | direction | readout | within-source | transfer R2 | transfer corr |
|---|---|---|---|---|---|
| VideoMAE | incline -> slide | `mu` | 0.127 | **-0.663** | +0.541 |
| VideoMAE | incline -> slide | **`size` (control)** | 0.982 | **+0.829** | **+0.939** |
| VideoMAE | slide -> incline | `mu` | 0.937 | **-0.167** | +0.072 |
| VideoMAE | slide -> incline | **`size` (control)** | 0.995 | +0.172 | +0.440 |
| V-JEPA2 | incline -> slide | `mu` | 0.622 | **-0.925** | -0.072 |
| V-JEPA2 | incline -> slide | **`size` (control)** | 0.982 | **+0.874** | **+0.948** |
| V-JEPA2 | slide -> incline | `mu` | 0.944 | **-0.553** | -0.176 |
| V-JEPA2 | slide -> incline | **`size` (control)** | 0.990 | +0.449 | +0.826 |

**A readout decoding `mu` at 0.94 within one floor geometry transfers none of it to the
other**, in both directions and both encoders, while cube size crosses the same domain
gap (camera 14 vs 18, flat vs tilted, different band pattern) at corr 0.83-0.95. The
control is strongest in the incline -> slide direction, which is also where the `mu`
negative is largest — so the failure is not generic domain shift.

**The diagnostic makes it concrete.** Score the `mu` probe against the target set's
ACCELERATION instead of its `mu`:

| model | slide -> incline | `mu` -> `mu` | `mu` -> `accel` |
|---|---|---|---|
| VideoMAE | corr | +0.072 | **+0.513** |
| V-JEPA2 | corr | -0.176 | **+0.793** |

The slide-trained "friction" probe predicts the incline set's **acceleration**, not its
coefficient. On the flat floor `mu` IS `a/g`, so that probe was an acceleration readout
all along — **§32's reframe, demonstrated across datasets rather than inferred from
`corr(mu, decel) = 0.9998`.** Built-in consistency check: in the incline -> slide
direction `mu -> mu` and `mu -> accel` come out numerically identical (-0.663 vs -0.666;
-0.925 vs -0.926), which they must, since on the flat floor they are the same number at
g = 1.0. If they ever differ, the pipeline is broken.

**Two caveats, both recorded rather than argued away.**

* **The per-domain standardisation is load-bearing.** Without it *everything* including
  the control transfers at large negative R2 (-13 to -49), because the raw render gap
  dominates. The reported numbers are the repaired version, and the repair is only
  legitimate because the control passes after it.
* **The largest |corr| anywhere on the layer grid** for incline -> slide `mu` is
  **+0.771 / +0.731** at early blocks (2 and 11), not ~0. But those layers read `mu`
  near zero within the incline set itself, and the same cells give an equal
  `mu -> accel` correlation (+0.769 / +0.729), so the natural reading is a shared
  low-level kinematic direction. Do not quote it as a positive without a permutation
  null on the best-over-grid selection.

**The claim this supports**, and it is empirical rather than algebraic, so it stands
beside the identity-trap argument rather than depending on it:

> A linear readout that decodes `mu` at 0.94 within one floor geometry transfers none
> of it to the other, while a visible control crosses the same gap at 0.83-0.95, and
> the transferred probe tracks the other set's acceleration instead. **These models
> hold no geometry-independent coefficient of friction.**



---

## PART 6 — WHAT THE MODELS CAN AND CANNOT DO

### 9. Encoders replicate the diffusion result exactly (v1)

`models/encoders.py` adds VideoMAE-base (12 blocks, 768d) and V-JEPA2-ViT-L (24
blocks, 1024d). An encoder answers the same question with two fewer axes: no
timestep, no inverted trajectory, one forward pass on the real pixels. Capture is
**0.24 s/scene of model time** against the diffusion path's 36 s — 300 scenes in
under 10 minutes.

Everything above `models/` is reused unchanged, so this is a genuine replication
rather than a reimplementation. Best over the (site x pool) grid on v1:

| target | CogVideoX-2b | VideoMAE | V-JEPA2 |
|---|---|---|---|
| `momentum_magnitude_pre` | 0.849 | 0.808 | 0.781 |
| `momentum_pre` | 0.805 | 0.825 | 0.706 |
| `momentum_post` | 0.819 | 0.845 | 0.789 |
| `kinetic_energy_pre` | 0.820 | 0.833 | 0.788 |
| `momentum_conserved` (acc) | 0.683 / null 0.633 | 0.633 / null 0.633 | 0.550 / null 0.550 |

**The encoders agree with the diffusion model on every point**, including the
conservation null. So none of the negatives are peculiar to diffusion, to
inversion, or to CogVideoX. That is worth a lot: it means the structural
arguments in §12 are doing the work, not a quirk of one checkpoint.

### 15. The prediction route is CLOSED on both encoders

> **Revisited in §21.** Everything below was measured through *mean-pooled*
> readouts, which lose ~12x of the signal on object-local quantities — `radius_A`
> reads 0.881 object-local against 0.070 frame-wide at the same block. §21 rebuilt
> the test with object-local pooling plus a mask-leak control (§20). **The verdict
> here survives**, on better evidence, and gains a sharper form: the predictor's
> output does not preserve object identity at all. Read §21 before quoting any
> number in this section.

§14 argued that prediction is the only formulation escaping the identity trap:
conservation is identifiable only when the aftermath is a free variable the model
chooses. Both encoders were tested. **Neither can support it**, for two different
reasons, and both are properties of the checkpoints rather than of the design.

**VideoMAE — the better-posed setup, and it produces nothing.**

`VideoMAEForPreTraining` is mechanically ideal: it is a true MAE, so its encoder
*only sees unmasked patches*. Masking the post-contact tubelets is therefore a
genuine prediction test — the encoder never sees the aftermath. (This is strictly
better than V-JEPA2, see below.) It has a 4-layer pixel decoder, so its output
could in principle be measured with `video.py`'s already-validated tracker.

It reconstructs nothing on this domain. Loss is per-patch-normalised MSE, so
**1.00 = predicting the patch mean = no signal**:

| mask pattern | loss |
|---|---|
| temporal, post-contact (the test) | 0.998 |
| tube 62.5% (the pattern it was trained on) | 0.954 |
| tube 90% (its pretraining ratio) | 0.981 |
| random 62.5% | 0.958 |

It fails equally in its *own* training regime, so this is not temporal
extrapolation being hard. Restricting to patches that actually contain a sphere
does not rescue it, and neither does the easiest possible variant:

| | R2 |
|---|---|
| all masked patches | −0.002 |
| **patches containing a sphere** (5.3% of masked) | **−0.009** |
| background-only patches | −0.002 |
| mask a single interior tubelet, context on both sides | −0.001 |

(Means over 5 scenes. `scene_00000` alone gives −0.0026 overall and −0.0116 on
sphere patches, which is what the §17 snippet prints — do not read the small
difference as a discrepancy.)

Target construction was validated against HF's own loss (1.0017 vs 0.9977, 0.4%
apart), so the metric is not the problem. This is the §7 wall again: Kubric
renders of two small spheres on flat grey are far outside a Kinetics-pretrained
prior. **Do not spend time here.** It also kills the most attractive steering
design — steer the encoder, decode to pixels, measure with the 0.35%-error
tracker.

**V-JEPA2 — the predictor runs, and adds nothing over its own encoder.**

Two implementation facts worth knowing before touching this:

* HF's `VJEPA2Model` runs the **encoder on the full clip** and only then splits
  `context_mask` / `target_mask`. So the default path's "context" tokens have
  already attended to the future, and using it as a prediction test would be
  §4's contamination all over again. The honest version encodes the pre-contact
  frames *only* and calls `m.predictor(...)` directly with `context_mask =
  arange(2560)` and `target_mask = arange(2560, 6144)`; the predictor addresses
  positions beyond the encoded range fine.
* How large that contamination is, measured: encoding 20 frames gives tokens only
  `cos = 0.74` similar to the first 2560 tokens of a 48-frame encode. A
  "pre-contact" token in a full-clip encode is substantially **not** what the
  model computes from pre-contact frames alone.

Run honestly (encode frames 1-20, predict all 14 post-contact tubelets), the
predictor is weak but non-trivial:

| | R2 | cosine |
|---|---|---|
| V-JEPA2 predictor | 0.058 | 0.651 |
| persistence (copy last pre-contact tubelet) | −0.346 | 0.531 |
| predict the mean | 0.000 | — |

**But the control kills it.** Compare what is decodable from the predictor's
output against what is decodable from the *pre-only encoder output it was
computed from* (150 scenes, mean-pooled, same split):

| target | pre-only encoder (control) | predictor output | delta |
|---|---|---|---|
| `velocity_A_post` | 0.469 | 0.471 | **+0.002** |
| `velocity_B_post` | 0.230 | 0.332 | +0.101 |
| `momentum_post` | 0.273 | 0.256 | −0.017 |
| `momentum_magnitude_post` | 0.066 | 0.046 | −0.020 |
| `ke_ratio` | 0.024 | 0.036 | +0.012 |
| `radius_A` | 0.963 | 0.951 | −0.012 |

**The predictor is a pass-through.** Whatever post-collision physics is readable
from its output was already linearly readable from the encoding of the
pre-contact frames. It is not making a physical commitment beyond its input.

**And the readout cannot be made independent.** A probe fit on *true* post-contact
latents scores ~0 when applied to predicted latents (0.024, −0.037, 0.020 for
`velocity_A_pre`, `velocity_A_post`, `velocity_B_post`) — the predicted latents
live in a different subspace. So any readout must be fit *on predicted latents
against true labels*, which trains it to emit true velocities and launders the
model's prediction toward the ground truth. **A conservation test built on such a
readout is circular by construction.**

**Conclusion.** The escape route proposed in §14 does not exist on these
checkpoints. Prediction-time physics is unanswerable on CogVideoX-2b (§7,
cannot inpaint), on VideoMAE (cannot reconstruct), and on V-JEPA2 (predictor
adds nothing, readout cannot be made non-circular). Reviving it needs a
*different, generatively competent* model on this domain — not more work on
these three.

### 21. The predictor test rebuilt — §15's verdict survives, better measured

§15 concluded prediction is closed. That conclusion stands, but its evidence was
measured through a readout discarding most of the signal. Rebuilt in
`scripts/predict_pre_only.py` and `predict_object_local.py`.

**Setup, and why each choice is forced.** V-JEPA2 (the only checkpoint with a
predictor); **v1, not v3** (conservation-based prediction needs the mass ratio,
which `latent_mass` makes invisible — the test would be unanswerable by design);
encode **frames 1-20 only** and call `m.predictor(...)` directly, since HF's
default path encodes the full clip and only then splits masks; **conserved scenes
only** (75 of 150, 45 train / 30 test), since a violating aftermath was edited
after the fact.

**Do not score total momentum.** Measured: on conserved scenes `p_post = p_pre` to
a median relative difference of **1.3e-16**. "Predicting total post-collision
momentum by conservation" is copying the pre-collision total — the identity trap
again. The informative target is the **split**: individual post-collision
velocities, which need mass ratio, restitution and contact geometry together.

**The oracle ceiling.** A perfect physicist seeing everything except restitution
reaches **0.970** on `velocity_A_post` — restitution is *not* a serious bottleneck,
contrary to first suspicion. Copying the pre-collision velocity scores −0.064.

**Mean pooling was the wrong instrument, by 12x.** The first rebuild reproduced
§15's number (encoder 0.463 vs 0.469 reported) but `radius_A` came out at 0.050
and `velocity_A_pre` at 0.187 — absurd for trivially visible properties. Two
hypotheses were proposed and **both were wrong**: out-of-distribution clip length
(refuted — `radius_A` gets *worse* with more frames, 0.196 at 16 -> 0.027 at 48)
and last-layer degradation (minor — 0.881 last block vs 0.922 best). The cause is
pooling: on the full-clip capture at the last block, `radius_A` reads **0.070**
frame-wide against **0.881** object-local.

With object-local pooling the sanity rows recover — `radius_A` 0.050 -> **0.885**,
`velocity_A_pre` 0.187 -> **0.877** — so the pre-only encoding was never degraded.

**And then the mask control (§20) decides it.** Proper variance decomposition,
activations fit against the target *after* removing what the mask alone predicts:

| readout | mask only | acts only | **acts \| mask** |
|---|---|---|---|
| velocity_A_post — **encoder** `A.pre` | −0.135 | 0.434 | **0.303** |
| velocity_A_post — **predictor** `A.post` | 0.439 | 0.729 | **−0.076** |
| velocity_A_pre — encoder `A.pre` | 0.258 | 0.877 | 0.630 |
| radius_A — encoder `A.pre` | −0.101 | 0.885 | 0.785 |
| radius_A — predictor `A.post` | −0.013 | 0.070 | 0.054 |

**The predictor's 0.729 is entirely its pooling address.** Its unique contribution
is zero; the pre-contact encoding it was computed from contributes 0.303. Note
`mask+acts` in that script's output is broken (45 train scenes, 3584 concatenated
features, single shared ridge penalty) — ignore that column. `acts | mask` is
biased slightly *low* because the mask probe is fit on train, so −0.076 is
consistent with zero rather than meaningfully negative.

**The leak-free evidence is `radius_A`**, which no mask predicts: it drops from
**0.928** in the true encoding to **0.070** in the predictor's output. Radius is
static — a faithful continuation must preserve it.

**And in latent space, split by content** (20 scenes, `render_prediction.py`):

| | cos |
|---|---|
| background tokens | **0.630 ±0.009** |
| shuffled-pairing chance | 0.523 ±0.008 |
| **object tokens** | **0.465 ±0.045** |

Objects worse than background in **20/20** scenes, worse than **chance** in 18/20.
Objects are only 3.7% of tokens, so any aggregate cosine is 96% a measurement of
"does grey stay grey" — the naive 0.615 mean was exactly that. **A methodological
warning: visual inspection of an error map is not evidence at a 3.7% object
fraction.** The map looks like unstructured noise; quantified, it is not.

**Conclusion, stronger than §15's.** Three independent measurements — pixel space
(VideoMAE reconstructs at loss 0.997, indistinguishable from the patch mean),
latent space (below chance on object tokens), and probe space (`acts | mask`
−0.076, radius 0.928 -> 0.054) — agree that **the predictor's output is not a
representation of the future scene**. It does not preserve object identity and
carries no aftermath information beyond where you point the readout.

**Rendering.** `scripts/render_prediction.py` writes the VideoMAE reconstruction
as PNG strips and mp4. V-JEPA2 **cannot be rendered at all** — the checkpoint has
only `encoder` and `predictor`, no pixel head, by JEPA design. When rendering
VideoMAE, note the decoder is trained with `norm_pix_loss` and predicts *per-patch
normalised* pixels, so the **true** per-patch mean/std must be restored to see
anything — which injects ground truth and makes the render strictly flattering.
The faint blobs that appear to track the objects come from that, not the model.

### 8. Captured but not analysed

`outputs_heads/` **no longer exists** — it was deleted in a disk cleanup (the same
one that removed 225 of v2's rendered scenes), and nothing here depends on it.
It held a complete 150-scene capture at `block.*.attn_heads` — the
**input** to `attn1.to_out.0`, the only site where head identity survives, since
`to_out` mixes the 30 heads (30 x 64 = 1920). Head `j` is exactly slice
`[64j : 64(j+1)]`.

The motivation was inference-time-intervention style per-head steering: a single
1920-dim direction spread over all heads puts most of its norm in heads that do
not carry the variable, which buys damage rather than control. Per-head activation
std varies **4.3x** at block 15, so the heads are genuinely heterogeneous.

It was left unanalysed after finding (3) — no head can localise a factor that is
not independently represented. If someone picks this up, **retarget it at
velocity**, which is the well-posed quantity, rather than at momentum.


---

## PART 7 — WHAT TO DO NEXT

### What to do next, given all of the above

> **Updated 2026-08-21 — RESUME HERE. The pod was stopped after this.**
>
> **The friction line is finished.** `incline_set_v2` rendered (150/150, all eight
> checks pass), both encoders captured, probes run — numbers in §35. mu 0.423
> (VideoMAE) / 0.779 (V-JEPA2), direct vs composed +0.002 / -0.003, and mu is 0.997
> linear in (a, theta) over the sampled range so §32's quotient test did not
> materialise.
>
> **Open, cheapest first:**
> 1. **§37's reproduction gap.** `momentum_delta` reads 0.517 in §34 and 0.205-0.261
>    under a different fitting protocol at the same cell. No GPU. Settle it before
>    any of it is written up.
> 2. **Widen theta** to make mu a real quotient (15-70 deg needs a shape that does
>    not topple — a sphere with rolling friction, or a low puck). New generator.
> 3. **The elastic/inelastic grid** (§36 holds the measurements it would rest on).
>    ~600 scenes for 150 per cell, ~25 GPU-hours at this set's 2.4 min/scene.
> 4. `collide_set_v1` is still paused at 26/150.
>
> **Not backed up.** Everything since commit `87e9fad` is uncommitted, including
> today's generator fix, the validator, `scripts/compose_incline.py`, this file and
> `docs/`. `/workspace` is a RunPod network volume so it survives a pod *stop*; a
> pod *delete* would take it.
>
> **0. [DONE 2026-08-21] `incline_set_v1` was VOID.**
> §32b: mu reads off a single frame's cube position at R2 0.88–0.99, against the
> 0.459 the set exists to impose. The generator and validator are fixed and audited
> but **not re-rendered** — that is the next GPU job, to `incline_set_v2`. Do not
> capture on v1.
>
> **1. [SUPERSEDED] Finish `incline_set_v1`.** It reported paused at 36/150;
> `run_batch.py` resumes by seed and re-renders only what is missing:
>
> ```bash
> KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
>     --num-scenes 150 --output-dir momentum_dataset/incline_set_v1 \
>     --script generate_incline_scene.py --arm latent_friction --workers 5
> /workspace/kubric/.venv/bin/python momentum_dataset/validate_incline.py \
>     --dataset-dir momentum_dataset/incline_set_v1 \
>     --out momentum_dataset/incline_set_v1/_validation.json
> ```
>
> ~169 s/scene wall-clock at 5 workers, so ~5 h for the remaining 114. Render is
> NOT sample-bound (64 -> 32 spp bought only 17%), so do not try to shorten it that way.
>
> **2. Capture, then READ `theta` FIRST.** It is a precondition, not a result: if the
> slope angle does not decode near ceiling, the composition test has no second
> ingredient and the set is void whatever mu does.
>
> ```bash
> for M in videomae-base vjepa2-vitl; do
>   .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
>       --mode capture --model $M \
>       --dataset-dir momentum_dataset/incline_set_v1 -o outputs_incline_$M
> done
> .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
>     -o outputs_incline_videomae-base --tag incline \
>     --pools "A.pre,A.post,A.all,B.pre,B.post,B.all,all.pre,all.post,all.all" \
>     --targets mu,log_mu,mu_from_kinematics,theta,sin_theta,cos_theta,accel_measured,cube_size,speed_final
> ```
>
> **3. The prediction to test.** mu here is a QUOTIENT, so §26/§30 predict the DIRECT
> probe on mu should now fall short of composing decoded `a` with decoded `theta` —
> the opposite of the flat floor, on the same variable. And any mu result must clear
> **R2 ~ 0.27-0.45**, which is what a linear readout of acceleration alone affords.
> Report every object-local number against `mask_only_floor.py`.
>
> **4. Publish a contact sheet and LOOK AT IT** before probing. Two of this session's
> three rendering bugs were invisible to every validator and obvious in one glance at
> the clips.
>
> Lower priority but cheap: resume `collide_set_v1` (21/150) for the shared-vs-per-object
> question; render the `visible_friction` arm, without which §31's surface-attribution
> null is unfalsifiable; §29's tests 4/6/7.

> **Updated 2026-08-16.** The three cheapest high-value items, all label-and-capture
> only, no GPU:
>
> 1. **§29 tests 4, 6, 7** — subspace containment, equal-momentum equivalence
>    classes, extrapolation across the m-v plane. Together they settle whether
>    momentum is a distinct factor. §26 predicts they come out negative, which
>    would make "factors stored separately, combinations composed by the readout"
>    the project's headline.
> 2. **§28's cross-validated AUC** with a grid-max permutation null, to close the
>    one genuinely open question about conservation.
> 3. **Re-run every published grid with all nine pools** (§27). Six-pool summaries
>    understate B's numbers and miss `A.all`/`B.all`, which win 6 of 12 rows.
>
> And on the friction line: capture + probe `slide_set_v2` (10 min per encoder,
> §25 has the commands and the predicted nulls), then render the
> `visible_friction` arm — without it, the latent arm's expected null on pool `B`
> is unfalsifiable.

The project has two questions. Keep them separate — they are in very different
states.

**Part 1 — do models internally represent physical properties in multi-object
video?** Substantially answered, with one positive result and one sharp
structural limit.

* *Positive:* momentum magnitude decodes at 0.78-0.85 **and beats the linear
  oracle on the true factors** (0.805 vs 0.695, §3). The network has genuinely
  recombined mass and velocity nonlinearly. This is the one surviving positive
  claim in the whole project and it should be the headline of any writeup.
* *Positive:* **object addressability** (§16) — objects are selectively but not
  exclusively represented, replicated across two architectures. Directly about
  multi-object video, identity-trap-free, already measured.
* *Structural limit:* observable per-object quantities sit at 0.6-0.85 while
  *relational* ones (`momentum_delta`, conservation) sit at zero and chance. **The
  models encode what they see, not the relation between the parts.**
* *Do not* add probe targets to this dataset family (§12), and *do not* render
  the `visible_mass` arm — §13's oracle already gives a tighter bound than that
  arm would, for free.

**Part 2 — can physical properties be controllably steered in video encoders?**
**ANSWERED, positively, in §18.** Per-object velocity is causally addressable:
patching installs a donor's velocity at 0.86 of the required change with a clean
double dissociation against radius, and leaves the *other* object's
representation 86-93% intact on both encoders. The §16 prediction that motivated
the experiment is refuted, and the ~75° subspace separation explains why.

The headline for a writeup is now two positives plus a method:

* **Objects are separately addressable** (§18) — causal, not decoding, replicated
  across two architectures, and it corrects an inference (decodability implies
  coupling) that the field makes casually.
* **Momentum beats its linear oracle** (§3) — 0.805 vs 0.695.
* **The identity trap** (§12) and **the mask-leak control** (§20) are transferable
  methodology; the second is not present in Joseph et al. and is forced by any
  multi-object design.

**The next experiments, in rough priority order:**

1. **The joint-patch prediction (§18d).** If cross-object redundancy is an
   error-correcting channel, patching A *and* B together should transfer far more
   than A alone, and should explain why V-JEPA2 transfers 0.39 against VideoMAE's
   0.86. Cheap — no new capture, no new rendering.
2. **Adopt Joseph et al.'s evaluation protocol (§22).** A held-out probe trained
   on the *test* split only, plus their "MAE-to-target down, MAE-to-truth up"
   check. Both are cheap and both close gaps our controls do not.
3. **Polar vs Cartesian, done properly (§19).** Subspace intervention with
   per-scene target coordinates. `inlp_subspace()` already builds the basis. Also
   reproduces their headline number on a collision dataset.
4. **`videomaev2-base`** — registered and cached. Tests whether the VideoMAE-base
   vs V-JEPA2 factorisation split is capacity rather than architecture.
5. **Video-level measurement.** The honest limitation of everything in §18 is that
   it is **representation-level steering, not video-level steering** — there is no
   ground truth, and you cannot show a steered clip. Two routes: train a decoder
   from frozen encoder latents to Kubric pixels on the existing scenes (design the
   equivalent of the mask-only control in from the start, or it will hallucinate
   plausible scenes regardless of the latent), or move to a generatively competent
   model. §21 confirms neither existing checkpoint can render.
6. **Friction / drag** as the first latent *material* property. Not in this
   dataset at all — `gravity=(0,0,0)` and all friction terms are zero — so it
   needs a new arm. Linear damping is the cheap version: it keeps the
   floating-sphere geometry and the no-rotation guarantee while adding a
   per-object material parameter with sustained observable effect. Unlike momentum
   it is a genuine independent degree of freedom, which is exactly what makes it
   steerable; it is identity-trapped for *probing* (`b = -d(ln|v|)/dt`), so the
   oracle bar applies there.

**Do not:** add probe targets to the mass/momentum family (§12); render the
`visible_mass` arm (§13's oracle bounds it for free); or read any object-local
number without its mask-only floor (§20).

**Lower priority: the per-head analysis** was never done and its capture has since
been deleted (§8), so it costs ~110 GPU-min to re-capture first. Retarget it at
velocity rather than momentum. It is now a natural follow-up to §18: since the
residual stream keeps the two objects ~75° apart, the question is whether
individual heads are *more* object-selective still.

---


---

## PART 8 — REFERENCE

### Version control state — READ BEFORE ANY `git` OPERATION

> **2026-08-21.** New this session and uncommitted: the `generate_incline_scene.py`
> fix (§32b), the two new checks in `validate_incline.py`, `scripts/compose_incline.py`,
> `momentum_dataset/incline_set_v2/` (2.3 GB), `outputs_incline_videomae-base` (31 MB)
> and `outputs_incline_vjepa2-vitl` (69 MB), this file's restructure into parts, and
> `docs/` — artifact sources plus `CLAUDE.md.pre-restructure.bak`. Scratch under
> `/tmp` was NOT preserved.

**Everything since commit `87e9fad` is uncommitted, and none of it is backed up.**
That includes all of §9-§22 and the code they rest on.

| | |
|---|---|
| modified, tracked | `CLAUDE.md`, `generate_scene.py`, `run_batch.py`, `run_experiment.py`, `setup.sh`, `config.py`, `interp/__init__.py`, `models/__init__.py`, `models/cogvideox.py`, `runner.py`, `video.py`, `KUBRIC_INSTALL.md` |
| untracked, **code** | `physics_interp/models/encoders.py` — the entire encoder line, §9 |
| untracked, **code** | `scripts/steer_velocity.py`, `steer_multidim.py`, `patch_velocity.py`, `steer_polar.py`, `predict_pre_only.py`, `predict_object_local.py`, `render_prediction.py`, `mask_only_floor.py` — all of §18-§21 and §23 |
| untracked, **code** | `momentum_dataset/generate_friction_scene.py`, `validate_friction.py` — the two-ball line, §24 (its rendered data was deleted) |
| untracked, **code** | `momentum_dataset/generate_slide_scene.py`, `validate_slide.py` — the slide line, §25 |
| untracked, data | `probe_set_v2_latent/`, `probe_set_v3_latent/`, `slide_set_v1/`, `slide_set_v2/`, `slide_pilot/`, `outputs_v2_*`, `outputs_v3_*`, `outputs_videomae-base/`, `outputs_vjepa2-vitl/`, `outputs_steer_*`, `outputs_predict_*`, `outputs_render/` |
| untracked, reference | `cav.pdf`, `interpreting_physics.pdf` |

`encoders.py` being untracked is the dangerous one: it is ~390 lines of load-
bearing code with no copy anywhere, and `.gitignore` will not save it. A commit
costs nothing and would protect every result in §9-§22. **The probe result JSONs
(`*/momentum_probe/probes_*.json`, `outputs_steer_*/*.json`) are the only record
of several experiments and are small enough to commit outright.**

### This box

| | |
|---|---|
| GPU | RTX 5090, 32 GiB (Blackwell, sm_120) — old box was a 4090 |
| CPU | 120 cores |
| repo venv | Python 3.12.3 |
| kubric venv | Python 3.11.15 (bpy 4.3 wheels are 3.11-only) |
| `HF_HOME` | **set** to `/workspace/.cache/huggingface/` — this matters, see below |

### Environment — the traps

**Two separate Pythons, and they cannot be merged.**

| | path | why |
|---|---|---|
| Kubric / rendering | `/workspace/kubric/.venv/bin/python` | needs `numpy==1.26.4` for bpy 4.3's C ABI |
| Probing / CogVideoX | `/workspace/physics_interp_video_diffusion/.venv/bin/python` | pins `numpy==2.2.6` |

Never run the dataset scripts with the repo venv or vice versa. Rebuilding the
Kubric side on a new machine: `momentum_dataset/KUBRIC_INSTALL.md` (pins,
bootstrap, GPU-device trap) plus `momentum_dataset/kubric_requirements_locked.txt`.
Kubric itself is not vendored here — it is a public repo pinned to commit
`61f2422`, re-cloned by those notes.

**Rendering dies with `libXrender.so.1: cannot open shared object file`.** The
apt libraries bpy dlopens have disappeared from an instance once already. Fix:

```bash
apt-get update -qq && apt-get install -y libxrender1 libxi6 libxxf86vm1 \
    libxfixes3 libxkbcommon0 libsm6 libice6 libgl1 libglx0 libegl1 libgomp1
```

**Docker does not work here** (no `CAP_SYS_ADMIN`), which is why Kubric is a
native install rather than the upstream `kubricdockerhub/kubruntu` path.

**Render with `--workers 3`, not 5.** At 5 workers 17 of 150 scenes were killed
with SIGKILL. System RAM was not the constraint (251 GB, 231 free), so it was
GPU-side contention, made worse by `setup.sh` running concurrently. `run_batch.py`
is resumable — re-running the same command regenerates only what is missing.

**Do not trust Kubric's `Using the following GPU Device(s)` log line.** It reads
stale flags and prints a GPU name even when rendering on CPU. Trust
`scene_constants.render_device` in `labels.json`.

**The first GPU render on a new box is ~4x slower than every one after it — do
not benchmark on it.** Blender 4.3 predates Blackwell, so Cycles has no
precompiled OptiX kernels for sm_120 and the driver PTX-JITs them on first use.
Measured here: the very first scene took 509s, the *identical* scene re-rendered
502s faster at **127s** (2.6 s/frame, matching the 4090's documented 2.4
s/frame). The compiled kernels are cached under `~/.cache/cycles`, so the cost is
paid once per machine. Extrapolating from that first scene predicts ~21h for the
set instead of the real ~1.9h, and invites a hunt for a hardware problem that
does not exist. OPTIX itself works fine on the 5090.

**On this box, CPU rendering is deceptively competitive** — 120 cores put a
CPU-only scene at 436s against the GPU's 127s. That is close enough that a
CPU-rendered set will not obviously look broken, which is exactly why the
`KUBRIC_USE_GPU=1` omission is easy to miss. Check `render_device`, not the wall
clock.

**`setup.sh` exits 127 with `HF_HOME=...: No such file or directory` when
`HF_HOME` is set in the environment.** Fixed here, but understand it before
touching that block. The prefetch used a bare

```bash
${CACHE_DIR:+HF_HOME="$CACHE_DIR"} "$VPY" - "$MODEL_REPO"
```

and bash decides which words are assignment prefixes at **parse** time, before
expansion — so the expanded `HF_HOME=/...` became the *command name*. It only
fires when `CACHE_DIR` is non-empty, i.e. when `HF_HOME` is set or `--cache-dir`
is passed, which is why the old box never hit it and this one did immediately.
The fix routes it through `env`, which takes `VAR=value` as arguments. Any new
conditional env var in that script needs the same treatment.

**`validate.py` needs matplotlib, which only the *kubric* venv has.** It writes
`validation_plots.png`. The repo venv has no matplotlib at all, so run it as
`/workspace/kubric/.venv/bin/python momentum_dataset/validate.py ...`.

**`setup.sh` looks hung during the CUDA wheels; it usually is not** — and on this
box it is simply fast (150–190 MB/s, minutes end to end). The slow case is a
property of the host: pip fetches serially, and some hosts cap a single
connection around 0.5–1.0 MB/s, which is where the old box's 10m09s for
`nvidia_cudnn_cu12` came from. Before concluding the network is bad, measure
*aggregate* throughput rather than pip's one stream. To confirm progress rather
than guess, read the partial file:

```bash
p=$(pgrep -f 'pip install'); ls -l /proc/$p/fd | grep pip-unpack
```

The weights are the opposite case: `hf_transfer` parallelises, so 12.83 GiB lands
near aggregate rate regardless.

Disk to provision: ~10 GB venv, ~13 GB HF cache, **~7.4 GB of datasets** (`probe_set` 2.2 GB + `probe_set_v2_latent` 1.1 GB + `probe_set_v3_latent` 4.1 GB) and ~1.7 GB of captures. Put the venv
and the HF cache on separate mounts if you can, so they do not compete. On this
box `HF_HOME` is `/workspace/.cache/huggingface/`.

### Layout

```
physics_interp/
  models/base.py       ModelAdapter ABC + the schedule algebra (alpha_sigma,
                       split_prediction) that inversion rests on
  models/cogvideox.py  CogVideoX adapter: site map, encode/embed/denoise
  models/encoders.py   VideoMAE v1/v2 + V-JEPA2 adapters. Same hooks, same
                       pooling, same probes; the five diffusion-only primitives
                       raise instead of pretending. tubelet_phase_split() checks
                       mechanically that no temporal cell straddles contact.
  interp/hooks.py      hook substrate — step/CFG-branch aware taps. No opinion
                       about what to record.
  interp/__init__.py   linear probes (ridge, closed form, numpy — no sklearn),
                       label extraction, the DERIVED target table,
                       fit_attentive_probe (attention-pooled probe over raw
                       patch tokens), ActivationSteer (the write side of the
                       hook substrate), build_directions (the seven
                       steering-direction estimators)
  runner.py            generate() for prompts; invert()/capture_scene() for
                       existing videos; encode_scene() for the encoder path (one
                       forward pass, no inversion); resample() for the forward
                       pass, with steering; inpaint() to regenerate chosen latent
                       frames and ask the model to do physics; ActivationRecorder
                       and PatchRecorder
  video.py             frame/mask IO + pixel->latent-cell mapping (and
                       tubelet_frame_to_cell for encoders), plus the measurement
                       half: track_spheres/measure_momentum recover physics from
                       pixels alone, for scoring generated clips
momentum_dataset/
  generate_scene.py    one scene from one seed; --arm selects legacy /
                       latent_mass / visible_mass (see §10)
  generate_friction_scene.py  §24 — two balls slide onto per-lane FLOOR friction
                       patches at frame 20; --arm selects latent_friction /
                       visible_friction. Shares the momentum generator's camera,
                       renderer config and output layout
  run_batch.py         N scenes, parallel, resumable; --script picks the generator
  validate.py          correctness + coverage + confound report
  validate_friction.py the same for the two-ball friction set (§24)
  generate_slide_scene.py  §25 — ONE cube sliding on a level floor whose mu varies,
                       SIDE view. The floor is the second object and carries mu
  validate_slide.py    six checks for the slide set, including the floor-luma
                       appearance-leak check that gates the latent arm (§25)
  generate_collide_friction_scene.py  §31a — TWO equal cubes, same-direction catch-up
                       on a shared friction floor. Contact at frame 14.5 (a tubelet
                       INTERIOR), rotation damped not frozen
  validate_collide_friction.py  six checks incl. both cubes visible in every frame
  generate_incline_scene.py  §32 — one cube released from rest on a TILTED banded
                       slope. The set that separates mu from acceleration. Start
                       position PINNED — see derive() for why that is load-bearing
  validate_incline.py  eight checks, incl. DISSOCIATION (corr(mu, a) must stay clear
                       of -1), frame-0 leak, and visibility (the slope must be LIT)
  probe_set/           150 scenes, v1 `legacy` arm — mass VISIBLE
  probe_set_v2_latent/ 300 scenes, SUPERSEDED, do not probe (§11). Only the 75
                       violating scenes still have pixels.
  probe_set_v3_latent/ 300 scenes, `latent_mass` arm — mass INVISIBLE. THE
                       dataset for anything about mass or conservation.
  slide_set_v1/        300 scenes, cube on a friction floor, SIDE view (§25) —
                       mu INVISIBLE, recoverable only from the deceleration
  collide_set_v1/      21 of 150, two cubes on a shared friction floor (§31a). PAUSED
  incline_set_v1/      36 of 150, cube on a tilted slope (§32). PAUSED, all checks
                       green on the 6-scene pilot. `run_batch.py` resumes by seed
  incline_pilot/       6 scenes, the render that validated §32
  slide_pilot/         8 scenes, the render that validated the above
                       (the two-ball `friction_set_v1/` and `friction_pilot/` were
                        deleted once §25 superseded them; the generator that makes
                        them is still here and still works)
  pilot/               50 OLD scenes — see warning below
  KUBRIC_INSTALL.md    rebuilding the renderer on a fresh machine
  kubric_requirements_locked.txt   frozen off the box that rendered probe_set/
configs/momentum_probe.yaml   the probing config
configs/red_ball_slope.yaml   unrelated text-to-video generation example
scripts/
  run_experiment.py    capture / probe CLI, the only wired-in sweep
  steer_velocity.py    §18a — 1-D steering, alpha ladder, block sweep, and the
                       passive-propagation control that made it readable.
                       Also holds SceneRunner / fit_readout / load_captures,
                       which the other steering scripts import.
  steer_multidim.py    §18c — generative vs discriminative directions, and the
                       scene-dependent speed-consistency test
  patch_velocity.py    §18d — activation patching between real scenes, plus
                       inlp_subspace() used by §19's proposed fix
  steer_polar.py       §19 — polar vs Cartesian causal test (inconclusive)
  predict_pre_only.py  §21 — pre-contact-only prediction, mean-pooled
  predict_object_local.py  §21 — the same with object-local pooling and the
                       mask-only leak control (§20)
  render_prediction.py §21 — VideoMAE reconstruction to PNG/mp4; V-JEPA2
                       latent-space map, since it has no decoder
  mask_only_floor.py   §20/§23 — the pooling-mask leak floor for an arbitrary
                       pool spec, normalised and occupancy conventions. No model
                       load: the geometry comes from the capture metadata
  patch_surface.py     §31 — floor-vs-cube causal patching. The patched and read
                       pools are DISJOINT, so the passive channel is computed and
                       comes out exactly 0.000 rather than being argued away
  analyse_collide_friction.py  §31a — where / shared? / stored? for a two-cube set
  steer_friction.py    §30a — Joseph et al.'s held-out-probe steering protocol
                       (App C.12) plus the passive-propagation control it lacks,
                       and the stop_frame/travel consistency check. Frame-wide
                       edit, so no mask is involved anywhere
```

Steering / prediction output directories, all untracked:

| dir | what |
|---|---|
| `outputs_steer_videomae-base/`, `outputs_steer_vjepa2-vitl/` | §18-§19 run records, small JSON |
| `outputs_predict_vjepa2-vitl/` | §21 |
| `outputs_render/` | §21 renders |

Capture output directories, one per (model, dataset) pair. Each holds
`momentum_probe/activations/scene_*.npz` plus `probes_<tag>.json`:

| dir | model | dataset | size |
|---|---|---|---|
| `outputs/` | CogVideoX-2b | v1 | 1.3 GB |
| `outputs_heads/` | CogVideoX-2b, per-head | v1 | **deleted** — was 1.3 GB, never analysed |
| `outputs_videomae-base/`, `outputs_vjepa2-vitl/` | encoders | v1 | 33 / 71 MB |
| `outputs_v2_*` | encoders | v2 | 192 MB — **flawed set, ignore** |
| `outputs_v3_videomae-base/`, `outputs_v3_vjepa2-vitl/` | encoders | v3 | 58 / 134 MB |
| `outputs_slide_videomae-base/`, `outputs_slide_vjepa2-vitl/` | encoders | `slide_set_v2` | 58 / 134 MB |
| `outputs_patch_videomae-base/` | VideoMAE, **raw tokens** at blocks 1/5/8/11 | v1 | 830 MB |
| `outputs_patch_vjepa2-vitl/` | V-JEPA2, **raw tokens** at blocks 1/7/13/22 | v1 | 6.6 GB |

---

### Reproducing what has been run

Everything below is **already done on this box**; the commands are kept so the
sequence is reproducible elsewhere. Probing is free to re-run (no GPU, seconds).

```bash
cd /workspace/physics_interp_video_diffusion

# 0. dataset present — expect 150 and 150
ls -d momentum_dataset/probe_set/scene_*/ | wc -l
ls momentum_dataset/probe_set/scene_*/_COMPLETE | wc -l

# 1. environment (resumable; re-run is safe)
./scripts/setup.sh
.venv/bin/python tests/test_hooks.py                      # expect "27 passed"

# 2. capture, residual stream (~90 min, 36 s/scene, resumable)
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode capture

# 2b. capture, per-head attention (~110 min). Separate output dir: the site
#     pattern changes what is tapped, not how much it costs.
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
    --mode capture --sites "block.*.attn_heads" -o outputs_heads

# 3. the two probe experiments (no GPU, seconds each)
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_conserved --tag plausibility

.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_pre,momentum_post,momentum_magnitude_pre,kinetic_energy_pre,ke_ratio,velocity_A_pre,velocity_B_pre,impulse,restitution,mass_A,radius_A,speed_A_pre \
    --tag variables

# 4. consistency check on the decorrelated stratum (41 scenes — see the caveat
#    in the results section; this is NOT a clean number)
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_pre,momentum_magnitude_pre,mass_A,radius_A,velocity_A_pre \
    --where stratum=matched_momentum --tag variables_matched
```

Results land in `outputs/momentum_probe/probes_<tag>.json`, with a summary table
printed to stdout.

**The v3 / encoder line (§9-§14).** The dataset is 300 scenes at ~13 min render
per 5 scenes with 3 workers; the full set cost ~4 h. Capture is ~10 min per
encoder for all 300, and probing is seconds.

```bash
# v3 dataset. --arm and --violate-frac are BOTH non-default and both matter.
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 300 --output-dir momentum_dataset/probe_set_v3_latent \
    --seed-offset 0 --violate-frac 0.25 --workers 3 --arm latent_mass

# capture, one output dir per model (~10 min each, 300 scenes)
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir momentum_dataset/probe_set_v3_latent -o outputs_v3_$M
done

# probes. NOTE --where momentum_conserved=true on the ratio run: in a violating
# scene the label is unrecoverable by anyone and behaves as noise (§14).
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    -o outputs_v3_vjepa2-vitl --tag ratio_conserving --where momentum_conserved=true \
    --targets log_mass_ratio,log_density_scale,log_speed_change_A,log_speed_change_B,delta_v_A,delta_v_B,radius_A

.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    -o outputs_v3_vjepa2-vitl --tag plausibility --targets momentum_conserved

# §23  the four-velocity concat probe. '+' concatenates pools on the FEATURE
#      axis; --pools overrides the config's list. Seconds, no GPU.
for D in outputs_v3_videomae-base outputs_v3_vjepa2-vitl; do
  .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
      -o $D --tag ratio_concat --where momentum_conserved=true \
      --pools "A.pre+A.post+B.pre+B.post,A.pre+B.pre,A.post+B.post,A.pre,A.post,B.pre,B.post,all.pre,all.post" \
      --targets log_mass_ratio,log_density_scale,delta_v_A,delta_v_B,log_speed_change_A,log_speed_change_B,velocity_A_pre,velocity_B_pre,radius_A
  # its mask-only floor (§20's control, any pool spec, no model load)
  .venv/bin/python scripts/mask_only_floor.py --capture-dir $D \
      --dataset-dir momentum_dataset/probe_set_v3_latent --where momentum_conserved=true \
      --pools "A.pre+A.post+B.pre+B.post,A.pre+B.pre,A.pre,A.post,B.pre,B.post" \
      --targets log_mass_ratio,delta_v_A,delta_v_B,velocity_A_pre,velocity_B_pre,radius_A \
      --out $D/momentum_probe/mask_floor_concat.json
done
```

Traps in that sequence, all cheap to hit and expensive to debug:

* **`--arm latent_mass` defaults to `legacy`.** Omit it and you silently
  re-render v1 physics into a v3-named directory.
* **`-o` is required on `--mode probe`.** Without it the probe reads `outputs/`
  (CogVideoX on v1) whatever `--model` said, and the numbers look plausible.
* **Delete a `_FAILED` scene directory before re-rendering it.** `run_batch.py`
  resumes on the absence of `_COMPLETE`, so it *will* retry a failed scene — but
  the directory still holds the partial `frames/` and `segmentation/` from the
  killed attempt. The five scenes finished on 2026-08-13 were moved aside first.
  Verify afterwards with `ls scene_*/frames | wc -l` per scene, expecting 49.

**Before re-rendering anything into an existing set, prove the code still
reproduces it.** `generate_scene.py` was edited *after* v3 was rendered, so
finishing v3's five missing scenes risked mixing two physics versions in one
directory. The check is cheap (`--no-render`, ~7 s/scene) and was decisive:

```bash
S=/tmp/dryrun && mkdir -p $S
for i in 0 3 100 278; do idx=$(printf "%05d" $i); V=""
  [ $((i % 4)) -eq 3 ] && V="--violate"      # violate_frac 0.25 -> every 4th scene
  /workspace/kubric/.venv/bin/python momentum_dataset/generate_scene.py \
      --seed $i --output-dir $S/scene_$idx --scene-id scene_$idx \
      --arm latent_mass --no-render $V &
done; wait
```

Then diff each `labels.json` against the rendered one, flattening nested keys.
Result on 2026-08-13: **0 differing fields, max absolute numeric difference 0**
across all four scenes including a violating one and a late index — so the
post-render edit had not touched physics or sampling, and finishing the set was
safe. Do this every time; it is the same discipline that validated the v1
regeneration after the migration.

**The steering / patching line (§18-§21).** All of this reads the *existing*
captures for its directions and readouts, so only the intervention passes need a
GPU. Runtimes are for this box: VideoMAE ~0.5 s per forward pass including I/O,
V-JEPA2 ~1.2 s. Loading 49 PNGs off the network mount costs more than the forward
pass it feeds, which is why `SceneRunner` loads each scene once and runs all
conditions against it.

```bash
# §18a  1-D steering. Pilot picks the block from the depth curve, then sweeps
#       alpha; the sweep varies the steered block. ~8 min and ~10 min.
.venv/bin/python scripts/steer_velocity.py --mode pilot --n-scenes 30 --alphas 1,2,4,8,16
.venv/bin/python scripts/steer_velocity.py --mode sweep --n-scenes 120 \
    --blocks 1,6,9 --alphas 8 --directions probe,shuffled,random

# §18c  generative vs discriminative directions (~5 min VideoMAE, ~12 min V-JEPA2)
.venv/bin/python scripts/steer_multidim.py --n-scenes 60 --alphas 4,8
.venv/bin/python scripts/steer_multidim.py --model vjepa2-vitl \
    --capture-dir outputs_v3_vjepa2-vitl --block 7 --read-block block.23.resid_post \
    --n-scenes 40 --alphas 8

# §18d  activation patching (~2 min VideoMAE, ~3 min V-JEPA2)
.venv/bin/python scripts/patch_velocity.py --n-scenes 60 --rounds 8
.venv/bin/python scripts/patch_velocity.py --model vjepa2-vitl \
    --capture-dir outputs_v3_vjepa2-vitl --block 7 --read-block block.23.resid_post \
    --n-scenes 40 --rounds 8

# §19  polar vs Cartesian (inconclusive — see the two design flaws first)
.venv/bin/python scripts/steer_polar.py --block 6 --n-scenes 40

# §21  prediction, on v1 because v3 makes mass invisible. Object-local pooling
#      and the mask-only control are what make this readable (~3.5 min).
.venv/bin/python scripts/predict_object_local.py

# §21  renders. VideoMAE only; V-JEPA2 has no decoder.
.venv/bin/python scripts/render_prediction.py --scene scene_00000
```

Traps specific to this line:

* **`--model` and `--capture-dir` must agree.** Every one of these scripts fits its
  directions and readouts from the capture directory and then runs the *loaded*
  model. Mismatch them and it will run happily and produce numbers that mean
  nothing.
* **Cast to float64 before any norm.** Activations are stored fp16; V-JEPA2's reach
  `|x| ~ 264`, so a sum of squares overflows and `np.linalg.norm` returns `inf`,
  reporting a perturbation of exactly 0.000 (§18).
* **`--block` defaults are per-model** — 6 for VideoMAE (12 blocks), 7 for V-JEPA2
  (24 blocks), both chosen from the depth curve, not by hand. Pass
  `--read-block block.23.resid_post` for V-JEPA2 or it reads the wrong site.
* **Report every object-local number against its mask-only floor** (§20). The
  control is free and it caught a confound three other rounds of controls missed.

### 6. The measurement harness, and its floors

Steering claims are only as good as the measurement, so this was built first.
`physics_interp/video.py` recovers physics from **pixels alone** — no masks, no
labels — because a generated clip has neither.

| quantity | floor |
|---|---|
| tracker vs simulator labels, 150 scenes | median **0.35%** momentum error, p90 2.2% |
| invert -> resample round trip (`max_timestep=599`) | median **0.53%**, pixel MAE 1.1 (VAE floor 0.82) |
| camera calibration | predicted 50.0 px/world, measured **50.03**, residual **0.035 px** |

Three tracker traps, all fixed and all worth not rediscovering:

* **Shadows.** The threshold blob includes each sphere's shadow, whose offset
  drifts with position, biasing speed by a consistent **1.6%**. Corrected by
  `SPEED_CALIBRATION`, not a cleverer detector — tightening the threshold makes
  it *worse* (the darker core is the shaded side, which slides), and both
  saturation-plus-variance masking and Hough circles were measured and rejected.
  It cancels in a paired comparison anyway.
* **Identity swaps across the collision.** The spheres merge into one blob at
  contact and emerge having swapped sides, so nearest-neighbour continuity
  resumes on the wrong ball — silently, since both velocities come out wrong by
  the same amount. Re-anchoring on *appearance* took this from ~2% of scenes to
  ~0.7%. The residue is irreducible (same-material pairs of near-equal radius)
  and self-limiting: a swap changes the total by `(m_A - m_B)(v_A - v_B)`.
* **The fit residual is not a swap detector** (r = 0.20). A swap sits exactly at
  the tracking gap, so each fit window stays perfectly straight.

Paired design is what makes the floors usable: both arms share a scene, an
inverted latent and an integrator, so the multiplicative bias cancels and only
the 0.005 scatter survives.

### 17. Reproducing the one-off analyses (§13, §15, §16)

None of these are wired into `scripts/run_experiment.py` — they are analyses, not
sweeps, and adding four modes to the CLI for things run once would be worse than
recording them here. They are kept verbatim because each one took longer to get
*right* than to write, and every one of them changed a conclusion.

**A trap that costs ten minutes every time:** `fit_probe` requires a 2-D `Y`.
Passing a 1-D target raises `IndexError: tuple index out of range` from deep
inside the binary-target check, which looks nothing like the actual cause.
Always `Y = Y[:, None]` for scalar targets.

**§13 — the kinematic oracle.** The number that reframed the mass-ratio result.

```python
import json, glob, sys; sys.path.insert(0, '.')
import numpy as np
from physics_interp.interp import fit_probe
rng = np.random.default_rng(0)

L = []
for f in sorted(glob.glob('momentum_dataset/probe_set_v3_latent/scene_*/labels.json')):
    d = json.load(open(f))
    if not d['momentum_conserved']:          # violating labels are unrecoverable, §14
        continue
    c = d['collision']
    dvA = np.array(c['post']['velocity_A'][:2]) - np.array(c['pre']['velocity_A'][:2])
    dvB = np.array(c['post']['velocity_B'][:2]) - np.array(c['pre']['velocity_B'][:2])
    L.append((np.linalg.norm(dvA), np.linalg.norm(dvB), np.log(d['mass_ratio'])))

a = np.array([x[0] for x in L]); b = np.array([x[1] for x in L])
y = np.array([x[2] for x in L])[:, None]                     # 2-D — see the trap above
A = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
P = dict(alphas=A, train_fraction=0.6, split_seed=0, standardize=True)

print('raw magnitudes ', fit_probe(np.c_[a, b], y, **P)['r2'])            # 0.716
print('log magnitudes ', fit_probe(np.c_[np.log(a), np.log(b)], y, **P)['r2'])  # 1.000
for fid in [0.90, 0.80, 0.71, 0.60]:      # degrade to the models' own delta_v fidelity
    an = a + rng.normal(0, np.sqrt((1 - fid) / fid * a.var()), a.shape)
    bn = b + rng.normal(0, np.sqrt((1 - fid) / fid * b.var()), b.shape)
    print(fid, fit_probe(np.c_[an, bn], y, **P)['r2'])        # 0.71 -> 0.516
```

**§15a — VideoMAE reconstruction.** The target construction must match HF's
internal one or the number is meaningless; check it against `out.loss` every time
(they agreed to 0.4% here).

```python
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from transformers import VideoMAEForPreTraining
from transformers.image_utils import IMAGENET_DEFAULT_MEAN as MU, IMAGENET_DEFAULT_STD as SD
from physics_interp.models.encoders import VideoMAEAdapter
from physics_interp.video import load_frames, load_segmentation
dev = 'cuda:0'; T, S, P, TS = 8, 196, 16, 2       # 8 tubelets, 14x14 patches, patch 16, tubelet 2
m = VideoMAEForPreTraining.from_pretrained('MCG-NJU/videomae-base', dtype=torch.float32).eval().to(dev)
ad = VideoMAEAdapter(pipe=m.videomae, repo='x', device=torch.device(dev), dtype=torch.float32)

def targets(x):                                    # replicates HF's norm_pix_loss target
    mu = torch.as_tensor(MU, device=x.device)[None, None, :, None, None]
    sd = torch.as_tensor(SD, device=x.device)[None, None, :, None, None]
    f = x * sd + mu; B, t, c, h, w = f.shape
    f = (f.view(B, t // TS, TS, c, h // P, P, w // P, P)
           .permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
           .view(B, t // TS * (h // P) * (w // P), TS * P * P * c))
    return (f - f.mean(-1, keepdim=True)) / (f.var(-1, unbiased=True, keepdim=True).sqrt() + 1e-6)

d = 'momentum_dataset/probe_set_v3_latent/scene_00000'
fr, seg = load_frames(d), load_segmentation(d)
idx = ad.frame_indices(49)
x = ad.preprocess(fr[idx]).to(dev)
mask = torch.zeros(1, T * S, dtype=torch.bool, device=dev); mask[:, 3 * S:] = True   # cells 3-7 = post
with torch.no_grad():
    o = m(pixel_values=x, bool_masked_pos=mask)
tg = targets(x)[mask].reshape(1, -1, TS * P * P * 3)
print('HF loss', float(o.loss), 'my MSE', float(((o.logits - tg) ** 2).mean()))   # must agree
print('R2', float(1 - ((o.logits - tg) ** 2).sum() / (tg ** 2).sum()))            # ~ -0.002

# restrict to patches that actually contain a sphere
sm = ad.preprocess_masks(seg[idx])
occ = torch.as_tensor((sm > 0).astype(np.float32))[None]
occ = occ.view(1, T, TS, 14, P, 14, P).amax(dim=(2, 4, 6)).reshape(1, -1).bool().to(dev)
om = occ[mask].reshape(1, -1)
print('sphere patches R2', float(1 - ((o.logits[om] - tg[om]) ** 2).sum() / (tg[om] ** 2).sum()))
```

**§15b — V-JEPA2 prediction, run honestly.** The default `VJEPA2Model` path
encodes the **full clip** and only then splits context/target, so it is not a
prediction test at all. Encode the pre-contact frames only and call the predictor
directly.

```python
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from transformers import AutoModel
from physics_interp.models.encoders import VJEPA2Adapter
from physics_interp.video import load_frames
dev = 'cuda:0'; NPRE, NTOK = 2560, 6144        # 10 pre-contact tubelets x 256, of 24 x 256
m = AutoModel.from_pretrained('facebook/vjepa2-vitl-fpc64-256', dtype=torch.float32).eval().to(dev)
ad = VJEPA2Adapter(pipe=m, repo='x', device=torch.device(dev), dtype=torch.float32)
ctx = [torch.arange(NPRE, device=dev)[None]]
tgt = [torch.arange(NPRE, NTOK, device=dev)[None]]

fr = load_frames('momentum_dataset/probe_set_v3_latent/scene_00000')
with torch.no_grad():
    hp = m(pixel_values_videos=ad.preprocess(fr[list(range(1, 21))]).to(dev),
           skip_predictor=True).last_hidden_state                     # PRE-ONLY encoding
    pred = m.predictor(encoder_hidden_states=hp, context_mask=ctx,
                       target_mask=tgt).last_hidden_state             # predicted post latents
    true = m(pixel_values_videos=ad.preprocess(fr[list(range(1, 49))]).to(dev),
             skip_predictor=True).last_hidden_state[:, NPRE:]         # target (full-clip encode)
```

Mean-pool `hp`, `pred` and `true` over tokens for ~150 scenes, then fit probes
from each to the labels. **The `hp` control is the whole experiment** — without
it the predictor looks like it carries physics, and with it the physics turns out
to have been in the pre-contact encoding all along.

**§16 — the cross-decoding matrix.** Reads existing captures only, no GPU. Takes
a few minutes for V-JEPA2 (25 sites x 1024-d); use a 3-alpha grid for a quick look.

```python
import json, glob, sys; sys.path.insert(0, '.')
import numpy as np
from physics_interp.interp import fit_probe, extract_label

def load(root):
    acts, labs, pools = {}, [], None
    for f in sorted(glob.glob(f'{root}/momentum_probe/activations/scene_*.npz')):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(bytes(z['meta']).decode())
            if not acts:
                pools = meta['pools']; acts = {k[4:]: [] for k in z if k.startswith('act.')}
            for s in acts: acts[s].append(z['act.' + s])
            labs.append(meta['labels'])
    return {s: np.stack(v) for s, v in acts.items()}, labs, pools

A, labs, pools = load('outputs_v3_vjepa2-vitl')
pi = {p: i for i, p in enumerate(pools)}
for t in ['velocity_A_pre', 'velocity_B_pre', 'delta_v_A', 'delta_v_B']:
    Y = np.stack([extract_label(l, t) for l in labs])
    for p in ['A.pre', 'B.pre', 'A.post', 'B.post', 'all.pre', 'all.post']:
        best = max(fit_probe(arr[:, 0, pi[p], :], Y, alphas=[0.1, 1, 10, 100, 1e3, 1e4],
                             train_fraction=0.6, split_seed=0, standardize=True)['r2']
                   for s, arr in A.items() if s != 'embed.out')
        print(t, p, round(best, 3))
```

Note `arr[:, 0, pi[p], :]` — the capture axes are `[scene, step, pool, dim]`, and
for an encoder there is exactly one step, so the second index is always 0.

### Moving to another machine

The transfer unit is **this directory**, not the git repo. `.gitignore` excludes
`momentum_dataset/*/scene_*/` and `probe_set/` is untracked, so a `git clone`
arrives with the code but **no rendered scenes**. This is not hypothetical — the
2026-08-07 migration to this box was done by `git clone` and cost the whole
dataset, which had to be re-rendered from seeds (~1.9 GPU-hours).

```bash
# preferred: copy the directory, skipping what is rebuildable or dead
rsync -a --info=progress2 \
    --exclude '.venv/' --exclude 'momentum_dataset/pilot/' \
    --exclude 'momentum_dataset/probe_set_v2_latent/' --exclude 'outputs_v2_*' \
    --exclude '__pycache__/' --exclude 'outputs/' \
    /workspace/physics_interp_video_diffusion/ NEW_HOST:/workspace/physics_interp_video_diffusion/
```

Dropping v2 is not a judgement call — it is superseded and its captures are of a
set with a geometry leak (§11).

Excluding `outputs/` is a judgement call: the CogVideoX residual-stream capture
is **1.3 GB** and costs ~90 GPU-minutes to rebuild. It is a pure function of the
dataset plus the config, so re-capturing is safe — but if the new box has the
disk, copying it saves those GPU-hours and is usually the better trade. The
**encoder captures are cheap to redo** (~10 min each for 300 scenes) and small
(33–135 MB), so let them come along either way. The probe results in
`*/momentum_probe/probes_*.json` are tiny and should **always** come along —
several of them are the only record of experiments described in §9-§14.

**~9.1 GB** now (excluding `.venv/` and `.git/`), across three datasets:

| dir | scenes | size | keep? |
|---|---|---|---|
| `probe_set/` | 150, v1 `legacy` | 2.2 GB | yes — the only visible-mass set |
| `probe_set_v2_latent/` | 75 of 300 still have pixels | 1.1 GB | **no** — superseded, §11 |
| `probe_set_v3_latent/` | 300, `latent_mass` | 4.1 GB | yes — the current set |

Per scene it is ~14 MB (98 PNGs, `labels.json`, `video.mp4`; the RGB frames are
~214 KB each at 720x480 and are essentially all of it). Earlier revisions of this
file said ~1.6 GB / ~11 MB per scene; the figures above are measured. **All three
are untracked and single-copy — nothing is backing any of them up.** If you would
rather go through git, the
dataset needs its own path — either force-add it (`git add -f
momentum_dataset/probe_set`, 2.2 GB of PNGs in history, and GitHub starts warning
past 1 GB), put it on Git LFS, or park a tarball in object storage and fetch it
on the new box. Re-rendering from seeds is the last resort: `run_batch.py` is
deterministic and resumable, but it costs ~1.9 GPU-hours and needs the whole
Kubric install standing up first.

On arrival, verify before trusting it:

```bash
ls -d momentum_dataset/probe_set/scene_*/ | wc -l          # 150
ls momentum_dataset/probe_set/scene_*/_COMPLETE | wc -l     # 150
```

### What happened on the migration (read this before trusting a future move)

The old box was replaced and this one was populated by **`git clone` — which lost
the entire dataset**, exactly as the warning below predicts. `probe_set/` arrived
with `manifest.json` and `_validation/` (both tracked) and **zero rendered
scenes**. It was regenerated from seeds here rather than recovered.

The regeneration is trustworthy: per-scene labels were diffed against the tracked
`_validation/validation_report.json` and match **bit-exactly** — max absolute
difference 0 across every numeric field in all 150 scenes, zero non-numeric
mismatches. Class balance, stratum split and confounds all reproduce (75/75,
109 wide / 41 matched, +0.61 overall / +0.20 matched).

**`probe_set/` is once again untracked and single-copy.** It is 2.2 GB and cost
~1.9 GPU-hours. Nothing is backing it up. Snapshot it somewhere before relying on
this box.

Everything needed to resume still lives under this directory — nothing outside it
is required except a GPU, Python 3.10+, and network. The Kubric renderer is
described by `momentum_dataset/KUBRIC_INSTALL.md` +
`kubric_requirements_locked.txt` (all 100 packages pinned exactly; the rebuild
here reproduced them without drift), and no source file references an absolute
path — `/workspace` appears nowhere in `*.py`, `*.yaml`, or `*.sh`. Safe to leave
behind on a move: `.venv/` (~10 GB, machine-specific) and
`momentum_dataset/pilot/` (538 MB, stale — see the warning at the bottom).

### Regenerating probe_set instead of copying it

Viable, because scene `i` is a pure function of seed `seed_offset + i`
(`rng = np.random.default_rng([seed, attempt])`) and `generate_scene.py` is
committed. The **labels reproduce exactly** — verified 2026-08-07 against the
tracked validation report: max absolute difference **0** across every numeric
field in all 150 scenes. Rendered pixels may differ by a hair across
GPU/denoiser versions, which does not matter for probing. Requires the whole
Kubric install standing up first (`KUBRIC_INSTALL.md`) and costs ~1.9 GPU-hours.

**Use exactly this command. The env var and two of the flags are not defaults,
and all three matter:**

```bash
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 150 --output-dir momentum_dataset/probe_set \
    --seed-offset 0 --violate-frac 0.5 --workers 3
```

* `KUBRIC_USE_GPU=1` — **the single most expensive omission.** Without it
  `configure_gpu()` short-circuits and the entire set renders on **CPU**:
  `labels.json` records `render_device: "cpu (KUBRIC_USE_GPU not set)"` and a
  scene takes 436s instead of ~135s. Earlier revisions of this file omitted it
  from this command. Always read `scene_constants.render_device` back afterwards
  — on this box all 150 say `OPTIX: NVIDIA GeForce RTX 5090`.
* `--violate-frac 0.5` — **defaults to 0.0.** Omit it and every scene conserves
  momentum, so `momentum_conserved` has one class and the entire plausibility
  experiment is dead. The recorded set is 75 conserved / 75 violated.
* `--workers 3` — defaults to 4. At 5, seventeen scenes were SIGKILLed.
* `--seed-offset 0` — the recorded set's offset. Changing it changes every scene.

**Budget ~1.9h at `--workers 3`** (measured here: 150 scenes in 112.7 min, 45
s/scene throughput, ~135s wall per scene — within noise of the old 4090 box's
manifest, 17 scenes in 870s).

**Check the labels before committing GPU-hours.** `--no-render` produces physics
and labels only, ~7s/scene, and is enough to verify every reproduction target in
the table below. Send it to a throwaway directory, **never to `probe_set/`** — it
writes `_COMPLETE` markers, and `run_batch.py` skips anything marked complete, so
a `--no-render` pass into the real directory would permanently suppress the
actual render of those scenes:

```bash
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 150 --output-dir /tmp/labels_dryrun \
    --seed-offset 0 --violate-frac 0.5 --workers 3 --no-render
```

Then validate and diff against what git already carries (note the *kubric*
interpreter — `validate.py` needs matplotlib, which the repo venv lacks):

```bash
/workspace/kubric/.venv/bin/python momentum_dataset/validate.py \
    --dataset-dir momentum_dataset/probe_set
```

This **overwrites** `probe_set/_validation/`, which is the tracked reference you
are diffing against. Copy it aside first if you want to compare before and after.

`probe_set/manifest.json` and `probe_set/_validation/validation_report.json` are
tracked, so they survive a clone and are the reference to check a regenerated set
against. Targets to reproduce:

| quantity | recorded value |
|---|---|
| scenes | 150, all with `_COMPLETE` |
| `momentum_conserved` | 75 True / 75 False |
| `stratum` | 109 `wide` / 41 `matched_momentum` |
| `corr(log abs p_total, log mass_A)` | +0.61 overall, +0.20 matched |

If the class balance or the stratum split comes out different, the flags were
wrong — stop and fix them before spending GPU-hours on capture.

---

### Design decisions worth not re-deriving

#### Inversion is DDIM inversion, not the paper's flow-matching Euler

The paper studies flow-matching models and integrates a velocity field backwards.
CogVideoX-2b is a DDIM model, so the literal update does not transfer. Both are
expressed as `x_t = alpha*x0 + sigma*eps` (`models/base.py:alpha_sigma`), which
makes the step a re-projection onto the next noise level — the paper's explicit
approximation, and *exact* for rectified flow. `prediction_type` is read off the
scheduler at runtime because it genuinely differs between checkpoints of the same
architecture; do not hardcode it.

Inversion runs **unguided** (`guidance_scale: 1.0`) — a CFG trajectory is not the
one being inverted — and with a **constant empty prompt**, so no scene
information enters through the text stream.

`num_steps` is trajectory fidelity and is not the knob to economise on: the paper
measures probe accuracy collapsing 0.82 → 0.57 going from 100 to 20 steps, while
reconstruction still looks fine. Currently **50** (on the plateau, half the cost).
Raise to 100 for final numbers; never go below 40. To cut cost, lower
`num_captures` instead — that is storage only.

#### Full inversion does not round-trip — truncate at `max_timestep` to edit

**Probing and editing need different inversion depths, and using the probing
depth to edit produces flat grey.** CogVideoX-2b is trained with
`rescale_betas_zero_snr`, so `alpha(999)` is exactly 0 and the terminal latent
is nothing but the model's own predicted noise. Inverting from a *clean* latent
there is no noise to find, so the early steps under-predict it and the
trajectory never catches up: the terminal latent comes out at **std 0.53**
against the 1.0 a sampler expects. Resampling from that is out of distribution.

Measured on `scene_00000` — invert, resample, re-measure momentum from decoded
pixels:

| `max_timestep` | pixel MAE (of 255) | momentum error |
|---|---|---|
| 199 | 1.26 | 0.21% |
| 399 | 1.34 | 0.23% |
| **599** | **1.41** | **0.23%** |
| 799 | 3.90 | tracking fails |
| 999 (`None`) | **58.97** | tracking fails |

At or below 599 this sits on the VAE's own round-trip floor of 0.82. Set
`inversion.max_timestep = 599` for anything that decodes; leave it `None` for
capture.

**This is not evidence against the recorded activations.** The oracle check —
add known noise at a level, ask the model to recover it — returns `eps` at std
1.00 and cosine 0.99+ at every level from t=199 to t=999, so `alpha_sigma` and
`split_prediction` are correct and the drift is a property of inverting from
clean data. It does mean the captured states at the *top* noise levels sit on a
drifted trajectory, which is consistent with the probe results: the winning
cells cluster at noise 0.00–0.63 and almost never at 1.00.

#### The pre/post collision split is exact, and that is not luck

CogVideoX's causal VAE encodes frame 0 alone, then groups of 4. Contact is
measured at frame 20.03–20.07, i.e. just *after* rendered frame 20, which is the
last frame of latent cell 5. So **latent frames 0–5 are purely pre-collision and
6–12 purely post**, with no straddling cell. `video.py:pixel_frame_to_latent`.
If `TARGET_CONTACT_FRAME` is ever changed, re-check this.

#### Pooling is by segmentation mask, not frame-wide mean

One site at one step is 67 MB unpooled. Pooling happens inside the hook, over
regions {A, B, all} × phases {pre, post, all} = 9 pools, ~3 MB/scene. A frame-wide
mean would drown a small sphere; a two-body momentum question needs the two
bodies addressable separately.

#### The momentum/mass confound is the whole reason the dataset was rebuilt

Momentum is mass × velocity, so if mass has the wider range the two are nearly
the same variable. The **original 50-scene pilot had `corr(|p|, mass_A) = 0.97`**
— a probe "decoding momentum" there could have been reading apparent size and
colour. Fixes, all in `generate_scene.py`:

* momentum sampled **before** speed; speed derived as `p/m`
* a **matched-momentum stratum** (~27% of scenes) where `|p_A|` comes from a band
  independent of mass
* sphere B usually starts moving, so total momentum is a vector sum equal to
  neither ball's
* continuous radii/densities/restitution instead of 3-level grids

Measured on the 150-scene set: `corr(log|p_total|, log mass_A)` = **+0.61**
overall, **+0.20** in the matched stratum.

**The stratum is only 41 scenes** — too few to fit a 1920-dim probe alone (a
60/40 split leaves ~16 test scenes). Report full-set numbers with `mass_A` beside
them as the bar to clear; treat the stratum as a consistency check. If momentum
looks decodable, the clean follow-up is a second render weighted toward that
stratum (raise `CONTROL_STRATUM_FRAC`).

#### Conservation needs both classes

Bullet conserves momentum by construction, so every honest scene has the same
answer and "is momentum conserved" is unprobeable. Violating scenes are
*authored*: the collision is simulated and validated honestly, then post-contact
velocities are overwritten and the spheres re-keyframed, starting at the first
frame after contact. The injected momentum is split by mass so both balls take
the same velocity change — equal-and-opposite would conserve momentum and defeat
the point, and this way neither trajectory looks individually absurd, so the
class difference lives in the *relationship* between them.

`--violate-frac 0.5` interleaves the classes deterministically by index.

#### `momentum_dataset/pilot/` is stale — do not probe it

50 scenes from the old sampler: 3 radius levels, B always at rest, contact always
at the frame centre, no violations, `corr(|p|, mass_A) = 0.97`. Valid physics,
fine for exercising the pipeline, **not identifiable for probing**.

---

### Physics facts baked into the labels

* Gravity is zero and both sphere centres sit at the same height, so contact
  normals are horizontal and motion is exactly planar. No constraint, no solver
  hack — the geometry makes it exact.
* Frictionless, so `angular_velocity` is identically zero in every scene. Do not
  use it as a probe target.
* Two Bullet defaults would silently corrupt this and are fixed in
  `build_and_simulate`: linear/angular damping (0.04 → 0) and
  `splitImpulsePenetrationThreshold` (-0.04 → 0, which otherwise *adds energy*).
* Restitution is a **pair** coefficient; each sphere carries `sqrt(e)`. Read
  `restitution`, not `restitution_object`.
* In a violating scene, `momentum_rel_error` is the realised violation;
  `momentum_rel_error_simulated` is the solver's pre-edit figure (~1e-16).

---

### Conventions

* Probe fitting is deliberately linear and reports a permutation control; the
  ridge penalty is selected on a validation split carved out of *train*, never
  test. Do not "improve" a probe by selecting on test.
* Splits are **by scene**. Two pooled vectors from one clip share everything
  except which tokens they averaged.
* Adding a probe target = adding one entry to `DERIVED` in `interp/__init__.py`.
  No re-capture needed; the full `labels.json` is stored in every `.npz`.
* Adding a model = one adapter subclass + a registry entry. Nothing above
  `models/` addresses a concrete module path.
* `--mode probe` never touches a GPU and re-reads captures, so asking a new
  question is cheap. Use `--targets` / `--where` / `--tag` rather than writing
  new config files.
* The user asked for **no unnecessary new files** — refactor existing ones.
