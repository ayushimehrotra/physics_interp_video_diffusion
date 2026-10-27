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

This file holds the overview, the methodology every experiment depends on, what the models
can and cannot do, what to do next, and the reference material. **The findings for each
physical quantity live in their own file** under `docs/physics/`:

| file | what is in it |
|---|---|
| [`docs/physics/momentum.md`](docs/physics/momentum.md) | momentum and conservation, plus the mass and velocity lines (`p = m v`) |
| [`docs/physics/kinetic_energy.md`](docs/physics/kinetic_energy.md) | energy probes, `energy_voe_v1`, the `ke_factor_set` reduced-mass readout |
| [`docs/physics/elasticity.md`](docs/physics/elasticity.md) | elastic vs inelastic collisions, restitution |
| [`docs/physics/friction.md`](docs/physics/friction.md) | the sliding, incline and friction-floor sets (line closed) |

Section numbers are unchanged from when they were written, so a `§N` anywhere in the docs
or the code still resolves through the index below.

| part (here) | what is in it |
|---|---|
| **Methodology** | the traps every experiment here has hit. Read §12 before designing any probe target |
| **Capability** | what the models can do at all: prediction, reconstruction, generation |
| **Next** | the resume block. Start here after reading CURRENT STATE |
| **Reference** | environment, layout, reproduction commands, design decisions |

### Section index

| § | lives in |
|---|---|
| 1 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 2 | here |
| 3 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 4 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 5 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 6 | here |
| 7 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 8 | here |
| 9 | here |
| 10 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 11 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 12 | here |
| 13 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 14 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 15 | here |
| 16 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 17 | here |
| 18 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 19 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 20 | here |
| 21 | here |
| 22 | here |
| 23 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 24 | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 25 | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 26 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 27 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 28 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 29 | here |
| 30 | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 31 | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 32 | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 32b | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 33 | here |
| 34 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 35 | [`docs/physics/friction.md`](docs/physics/friction.md) |
| 36 | [`docs/physics/elasticity.md`](docs/physics/elasticity.md) |
| 37 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 38 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 39 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 40 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 41 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 42 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 43 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 44 | [`docs/physics/kinetic_energy.md`](docs/physics/kinetic_energy.md) |
| 45 | [`docs/physics/kinetic_energy.md`](docs/physics/kinetic_energy.md) |
| 46 | [`docs/physics/kinetic_energy.md`](docs/physics/kinetic_energy.md) |
| 62 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 63 | [`docs/physics/kinetic_energy.md`](docs/physics/kinetic_energy.md) |
| 64 | [`docs/physics/momentum.md`](docs/physics/momentum.md) |
| 47–61 | the register (other GPU box) — not in this notebook; see [`docs/register_gap.md`](docs/register_gap.md) |

Nearly every obvious next step has already been tried. Read the relevant file before
proposing new work — the ones that failed mostly failed for *structural* reasons that no
amount of engineering fixes.

## RESUME — 2026-10-27 (RTX 4090 pod, stopped mid-capture)

* **CogVideoX-2b single-step-noise capture on `conservation_set_1000` is PAUSED at
  684/1000** (residuals + region-to-region attention, ~35 s/clip, ~3 h left). After a
  restart run `bash /workspace/envs/rebuild_envs.sh` (venvs, git dir, HF cache), then
  re-run the same command; it skips finished clips:
  ```bash
  HF_HOME=/root/hf_cache HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python \
      scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode capture \
      --model cogvideox-2b --inversion-method noise --record-attention \
      --dataset-dir datasets/conservation_set_1000 -o outputs/consnoise_cogvideox-2b --keep-going
  ```
  Three clips fail by design (the truncated seed255/262/264). Then probe |Δp|:
  `scripts/momentum/delta_p_norm_probe.py --outputs outputs/consnoise_cogvideox-2b`.
* Branch `refactor-layout` is **not pushed**; the git dir is restored from
  `/workspace/envs/git-dirs.tar`, so re-tar it before every stop.

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
| `datasets/probe_set` — 150 scenes, v1 `legacy` arm | **done** — all 6 validation checks pass |
| `datasets/probe_set_v2_latent` — 300 scenes | **done but SUPERSEDED — do not probe, see §11** |
| `datasets/probe_set_v3_latent` — 300 scenes, `latent_mass` arm | **done**, 300/300, all OPTIX |
| `.venv` / kubric venv / CogVideoX-2b weights | **done** |
| Capture + probes, CogVideoX-2b on v1 | **done** — `outputs/` |
| Capture + probes, VideoMAE-base + V-JEPA2-ViT-L on v1 | **done** — `outputs/{videomae-base,vjepa2-vitl}/` |
| Capture + probes, both encoders on v3 | **done** — `outputs/v3_*/` |
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
| Capture, both encoders on v4 (§38) | **DONE** — `outputs/v4_*` |
| Joseph et al.'s probe replicated exactly (§38c) | **done — changes nothing**, within noise of ridge. Matched pairs NOT built |
| Object-local pooling on the binary target (§38d) | **done — NEGATIVE**, frame-wide wins; `B.post` is +0.013 over its floor |
| Why the binary AUC is ~0.6 (§38e) | **RESOLVED — the label's algebra.** Exact ingredients score 0.490; +quadratic 0.893 |
| §34a norm route on v4 (§38g) | **DONE — NULL, 6/6 cells.** `momentum_delta` R2 0.517 -> 0.255 |
| **Re-measure §34 on v4, full protocol** | **NOT RUN — the next job.** Captures exist |
| **Factor geometry on v1's conserving 75 (§39)** | **DONE 2026-08-23**, both encoders — shared object-agnostic code, sum-pooling, speed⊥direction, PC anti-alignment |
| §34g's raw per-token capture, finally analysed (§39c-ii/§39f) | **DONE**, both encoders — per-token transfer 0.920 / 0.854, code delocalised ~17x the object |
| **`conservation_set_v1` — the leak-free 2x2 (§42)** | **DONE 2026-08-24. 300/300 clips in 207 min, 0 failures, ALL 9 GATES PASS.** Worst mass-free scalar 0.561 against a 0.053 null sd; joint classifier 0.444-0.514 in all four cells under grouped CV. Not yet captured |
| Per-head capture `block.*.attn_heads` | was 150/150 — **`outputs/heads/` HAS BEEN DELETED**, never analysed, ~110 GPU-min to redo |

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

Fixed in `_best_and_null()` in `scripts/pipeline/run_experiment.py`; the column is now
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
`old_experiments/scripts/predict_pre_only.py` and `predict_object_local.py`.

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

**Rendering.** `old_experiments/scripts/render_prediction.py` writes the VideoMAE reconstruction
as PNG strips and mp4. V-JEPA2 **cannot be rendered at all** — the checkpoint has
only `encoder` and `predictor`, no pixel head, by JEPA design. When rendering
VideoMAE, note the decoder is trained with `norm_pix_loss` and predicts *per-patch
normalised* pixels, so the **true** per-patch mean/std must be restored to see
anything — which injects ground truth and makes the render strictly flattering.
The faint blobs that appear to track the objects come from that, not the model.

### 8. Captured but not analysed

`outputs/heads/` **no longer exists** — it was deleted in a disk cleanup (the same
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
> today's generator fix, the validator, `old_experiments/scripts/compose_incline.py`, this file and
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
> KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
>     --num-scenes 150 --output-dir datasets/incline_set_v1 \
>     --script generate_incline_scene.py --arm latent_friction --workers 5
> /workspace/kubric/.venv/bin/python datasets/validate_incline.py \
>     --dataset-dir datasets/incline_set_v1 \
>     --out datasets/incline_set_v1/_validation.json
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
>   .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
>       --mode capture --model $M \
>       --dataset-dir datasets/incline_set_v1 -o outputs/incline_$M
> done
> .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
>     -o outputs/incline_videomae-base --tag incline \
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
> fix (§32b), the two new checks in `validate_incline.py`, `old_experiments/scripts/compose_incline.py`,
> `datasets/incline_set_v2/` (2.3 GB), `outputs/incline_videomae-base` (31 MB)
> and `outputs/incline_vjepa2-vitl` (69 MB), this file's restructure into parts, and
> `docs/` — artifact sources plus `CLAUDE.md.pre-restructure.bak`. Scratch under
> `/tmp` was NOT preserved.

**Everything since commit `87e9fad` is uncommitted, and none of it is backed up.**
That includes all of §9-§22 and the code they rest on.

| | |
|---|---|
| modified, tracked | `CLAUDE.md`, `generate_scene.py`, `run_batch.py`, `run_experiment.py`, `setup.sh`, `config.py`, `interp/__init__.py`, `models/__init__.py`, `models/cogvideox.py`, `runner.py`, `video.py`, `KUBRIC_INSTALL.md` |
| untracked, **code** | `physics_interp/models/encoders.py` — the entire encoder line, §9 |
| untracked, **code** | `scripts/momentum/steer_velocity.py`, `steer_multidim.py`, `patch_velocity.py`, `steer_polar.py`, `predict_pre_only.py`, `predict_object_local.py`, `render_prediction.py`, `mask_only_floor.py` — all of §18-§21 and §23 |
| untracked, **code** | `datasets/generate_friction_scene.py`, `validate_friction.py` — the two-ball line, §24 (its rendered data was deleted) |
| untracked, **code** | `datasets/generate_slide_scene.py`, `validate_slide.py` — the slide line, §25 |
| untracked, data | `probe_set_v2_latent/`, `probe_set_v3_latent/`, `slide_set_v1/`, `slide_set_v2/`, `slide_pilot/`, `outputs/v2_*`, `outputs/v3_*`, `outputs/videomae-base/`, `outputs/vjepa2-vitl/`, `outputs/steer_*`, `outputs/predict_*`, `outputs/render/` |
| untracked, reference | `cav.pdf`, `interpreting_physics.pdf` |

`encoders.py` being untracked is the dangerous one: it is ~390 lines of load-
bearing code with no copy anywhere, and `.gitignore` will not save it. A commit
costs nothing and would protect every result in §9-§22. **The probe result JSONs
(`*/momentum_probe/probes_*.json`, `outputs/steer_*/*.json`) are the only record
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
Kubric side on a new machine: `datasets/KUBRIC_INSTALL.md` (pins,
bootstrap, GPU-device trap) plus `datasets/kubric_requirements_locked.txt`.
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
`/workspace/kubric/.venv/bin/python datasets/validate.py ...`.

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

Reorganised 2026-10-27. Commands elsewhere in the docs use these paths.

```
CLAUDE.md            this file: overview, methodology, capability, next steps, reference
docs/physics/        the findings, one file per physical quantity (see the section index)
docs/                DATASETS.md, EXPERIMENTS.md, RESULTS.md, protocol notes;
  artifacts/         every result JSON and run log the docs quote (left as written)
figures/             paper figures (PDF/PNG) and their captions
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
datasets/            Kubric scene generators, planners, batch runners, validators
  generate_*.py      one generator per set family (generate_scene.py is the shared core)
  plan_*.py, run_*.py, validate_*.py   per-set planning, rendering and checks
  conservation_set_1000/   1000 two-ball clips, honest/violating twins (§62); 997 usable
  ke_factor_set/     1000 clips, 200 groups x 5 KE levels, families M and S (§63)
  pilot/, probe_set/ validation records only; the clips are gone
  KUBRIC_INSTALL.md, kubric_requirements_locked.txt   rebuilding the renderer
scripts/             every script imports `_momentum_path` (one copy per folder),
                     which puts the repo root and all script folders on sys.path
  pipeline/          capture_encoder.py, run_experiment.py/.sh, setup.sh
  common/            _fast_cv.py, mask_only_floor.py (§20 leak control)
  momentum/          conservation probes, valid-collision subspace (§62, §64),
                     truncation, velocity steering, register re-implementations
  kinetic_energy/    probe_energy.py (shared loaders), energy VoE, ke_reduced_mass.py (§63)
  elasticity/        elasticity_manifold.py, probe_elasticity_depth.py
  figures/           fig_*.py, one per paper figure
outputs/             encoder captures, <tag>_<model>/ (untracked, reproducible)
  cons_<model>/      full clips of conservation_set_1000
  construnc_<model>/ truncation rungs k10..k48 — PARTIAL, RTX 5090 box; never mix with
                     another GPU's captures (§58)
  ke_<model>/        ke_factor_set
old_experiments/     superseded work; MANIFEST.txt says what each entry was and why
  scripts/           30 scripts whose every input is archived (friction line, probe_set
                     v1–v4, steering/patching, prediction, v1 truncation, energy pilot)
configs/             momentum_probe.yaml (probing), red_ball_slope.yaml (t2v example)
```

### Reproducing what has been run

Everything below is **already done on this box**; the commands are kept so the
sequence is reproducible elsewhere. Probing is free to re-run (no GPU, seconds).

```bash
cd /workspace/physics_interp_video_diffusion

# 0. dataset present — expect 150 and 150
ls -d datasets/probe_set/scene_*/ | wc -l
ls datasets/probe_set/scene_*/_COMPLETE | wc -l

# 1. environment (resumable; re-run is safe)
./scripts/pipeline/setup.sh
.venv/bin/python tests/test_hooks.py                      # expect "27 passed"

# 2. capture, residual stream (~90 min, 36 s/scene, resumable)
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode capture

# 2b. capture, per-head attention (~110 min). Separate output dir: the site
#     pattern changes what is tapped, not how much it costs.
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
    --mode capture --sites "block.*.attn_heads" -o outputs/heads

# 3. the two probe experiments (no GPU, seconds each)
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_conserved --tag plausibility

.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_pre,momentum_post,momentum_magnitude_pre,kinetic_energy_pre,ke_ratio,velocity_A_pre,velocity_B_pre,impulse,restitution,mass_A,radius_A,speed_A_pre \
    --tag variables

# 4. consistency check on the decorrelated stratum (41 scenes — see the caveat
#    in the results section; this is NOT a clean number)
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
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
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 300 --output-dir datasets/probe_set_v3_latent \
    --seed-offset 0 --violate-frac 0.25 --workers 3 --arm latent_mass

# capture, one output dir per model (~10 min each, 300 scenes)
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir datasets/probe_set_v3_latent -o outputs/v3_$M
done

# probes. NOTE --where momentum_conserved=true on the ratio run: in a violating
# scene the label is unrecoverable by anyone and behaves as noise (§14).
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
    -o outputs/v3_vjepa2-vitl --tag ratio_conserving --where momentum_conserved=true \
    --targets log_mass_ratio,log_density_scale,log_speed_change_A,log_speed_change_B,delta_v_A,delta_v_B,radius_A

.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
    -o outputs/v3_vjepa2-vitl --tag plausibility --targets momentum_conserved

# §23  the four-velocity concat probe. '+' concatenates pools on the FEATURE
#      axis; --pools overrides the config's list. Seconds, no GPU.
for D in outputs/v3_videomae-base outputs/v3_vjepa2-vitl; do
  .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
      -o $D --tag ratio_concat --where momentum_conserved=true \
      --pools "A.pre+A.post+B.pre+B.post,A.pre+B.pre,A.post+B.post,A.pre,A.post,B.pre,B.post,all.pre,all.post" \
      --targets log_mass_ratio,log_density_scale,delta_v_A,delta_v_B,log_speed_change_A,log_speed_change_B,velocity_A_pre,velocity_B_pre,radius_A
  # its mask-only floor (§20's control, any pool spec, no model load)
  .venv/bin/python scripts/common/mask_only_floor.py --capture-dir $D \
      --dataset-dir datasets/probe_set_v3_latent --where momentum_conserved=true \
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
  /workspace/kubric/.venv/bin/python datasets/generate_scene.py \
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
.venv/bin/python scripts/momentum/steer_velocity.py --mode pilot --n-scenes 30 --alphas 1,2,4,8,16
.venv/bin/python scripts/momentum/steer_velocity.py --mode sweep --n-scenes 120 \
    --blocks 1,6,9 --alphas 8 --directions probe,shuffled,random

# §18c  generative vs discriminative directions (~5 min VideoMAE, ~12 min V-JEPA2)
.venv/bin/python old_experiments/scripts/steer_multidim.py --n-scenes 60 --alphas 4,8
.venv/bin/python old_experiments/scripts/steer_multidim.py --model vjepa2-vitl \
    --capture-dir outputs/v3_vjepa2-vitl --block 7 --read-block block.23.resid_post \
    --n-scenes 40 --alphas 8

# §18d  activation patching (~2 min VideoMAE, ~3 min V-JEPA2)
.venv/bin/python old_experiments/scripts/patch_velocity.py --n-scenes 60 --rounds 8
.venv/bin/python old_experiments/scripts/patch_velocity.py --model vjepa2-vitl \
    --capture-dir outputs/v3_vjepa2-vitl --block 7 --read-block block.23.resid_post \
    --n-scenes 40 --rounds 8

# §19  polar vs Cartesian (inconclusive — see the two design flaws first)
.venv/bin/python old_experiments/scripts/steer_polar.py --block 6 --n-scenes 40

# §21  prediction, on v1 because v3 makes mass invisible. Object-local pooling
#      and the mask-only control are what make this readable (~3.5 min).
.venv/bin/python old_experiments/scripts/predict_object_local.py

# §21  renders. VideoMAE only; V-JEPA2 has no decoder.
.venv/bin/python old_experiments/scripts/render_prediction.py --scene scene_00000
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

None of these are wired into `scripts/pipeline/run_experiment.py` — they are analyses, not
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
for f in sorted(glob.glob('datasets/probe_set_v3_latent/scene_*/labels.json')):
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

d = 'datasets/probe_set_v3_latent/scene_00000'
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

fr = load_frames('datasets/probe_set_v3_latent/scene_00000')
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

A, labs, pools = load('outputs/v3_vjepa2-vitl')
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
`datasets/*/scene_*/` and `probe_set/` is untracked, so a `git clone`
arrives with the code but **no rendered scenes**. This is not hypothetical — the
2026-08-07 migration to this box was done by `git clone` and cost the whole
dataset, which had to be re-rendered from seeds (~1.9 GPU-hours).

```bash
# preferred: copy the directory, skipping what is rebuildable or dead
rsync -a --info=progress2 \
    --exclude '.venv/' --exclude 'datasets/pilot/' \
    --exclude 'datasets/probe_set_v2_latent/' --exclude 'outputs/v2_*' \
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
datasets/probe_set`, 2.2 GB of PNGs in history, and GitHub starts warning
past 1 GB), put it on Git LFS, or park a tarball in object storage and fetch it
on the new box. Re-rendering from seeds is the last resort: `run_batch.py` is
deterministic and resumable, but it costs ~1.9 GPU-hours and needs the whole
Kubric install standing up first.

On arrival, verify before trusting it:

```bash
ls -d datasets/probe_set/scene_*/ | wc -l          # 150
ls datasets/probe_set/scene_*/_COMPLETE | wc -l     # 150
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
described by `datasets/KUBRIC_INSTALL.md` +
`kubric_requirements_locked.txt` (all 100 packages pinned exactly; the rebuild
here reproduced them without drift), and no source file references an absolute
path — `/workspace` appears nowhere in `*.py`, `*.yaml`, or `*.sh`. Safe to leave
behind on a move: `.venv/` (~10 GB, machine-specific) and
`datasets/pilot/` (538 MB, stale — see the warning at the bottom).

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
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 150 --output-dir datasets/probe_set \
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
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 150 --output-dir /tmp/labels_dryrun \
    --seed-offset 0 --violate-frac 0.5 --workers 3 --no-render
```

Then validate and diff against what git already carries (note the *kubric*
interpreter — `validate.py` needs matplotlib, which the repo venv lacks):

```bash
/workspace/kubric/.venv/bin/python datasets/validate.py \
    --dataset-dir datasets/probe_set
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

#### `datasets/pilot/` is stale — do not probe it

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
