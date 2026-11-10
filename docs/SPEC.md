# Benchmark Spec v2: Physical Laws Inside Video Models

Working name: **PhysLaw-Internal**. Implementation spec for Claude Code. Supersedes v1 (`physlaw_benchmark_spec.md`, 2026-11-04).

---

## How to apply this spec to the existing repo

**Do not rebuild anything that is already done.** v2 keeps every implemented module, decision and deviation recorded in `docs/STATUS.md` (2026-11-05 to 2026-11-09). It makes those decisions official, then adds new scene families, new models, revised L1 and L2 scoring, and new controls.

Rules:
1. Milestones marked **DONE** in Section 10 stay as they are. Re-run them only where Section 10 says so (for example, when F1 grows or a new score is added).
2. Every change from v1 is listed in Section 0.2 with the section that defines it.
3. Where v2 and `docs/STATUS.md` disagree on something already built, keep the built behavior and flag the disagreement in `docs/STATUS.md` for the user.
4. Do not stop the running CogVideoX1.5-5B-I2V pass-1 job to apply this spec.

---

## 0. Goal

Build a benchmark that tests whether video models obey physical laws **in their internal representations**, and compare that with whether their **generated outputs** obey the same laws.

Hypothesis under test: models encode per-object physical *states*, while the *laws relating those states* hold weakly or not at all in what the models compute internally.

The benchmark must be able to return either answer. A trained positive control (Section 4.4) shows the tests are passable; an untrained negative control shows they cannot be passed from pixel layout alone.

### 0.1 Deliverables

1. A data generator producing labeled synthetic videos with matched twin clips, for ten scene families.
2. A model-adapter layer exposing activations, forecasts, surprise and generation for every model.
3. Five evaluation levels (L0 to L4) plus a trained-detection companion to L2 (L2b).
4. Output-level evaluation in two tracks (A: scene continuation, B: self-consistency).
5. A results pipeline: per-model JSON, a leaderboard and standard figures.

### 0.2 Changes from v1

| Area | Change | Section |
|---|---|---|
| Rendering | 48 fps master, 289 frames, contact frames 112 to 128 (was 24 fps, 81 frames) | 2.1 |
| Twins | One simulation per group; closed-form legal outcomes; KE-reducing drift sign for momentum violations; `energy_physical` flag | 2.2 |
| F1 size | Target restored to 400 groups (currently 64); confirm with user | 2.3, 11 |
| F1 laws | Center-of-mass velocity and angular momentum added (no new rendering) | 2.3 |
| F5 | Merged: drop pairs move to F7, slide pairs move to F6 | 2.3 |
| New families | F7 free fall / projectiles / bounces; F8 ramp energy exchange; F9 pendulum (optional); F10 rolling (optional) | 2.3 |
| Models | Grouped by research question; added Wan2.1-1.3B, Wan2.1-14B, V-JEPA 2 ViT-g, two physics post-training pairs (PhysRVG, VideoREPA), Cosmos-Predict2.5 for track A, Wan2.1-I2V-14B for track B | 4 |
| Controls | Second positive control (simulator-fine-tuned DiT); positive control moved onto the critical path | 4.4 |
| L0 | Noised-input-latent baseline for DiTs; one-twin-per-group refit; matched velocity protocol check | 5.1 |
| L1 | Momentum score demoted (degenerate); outcome score and new law-family score are primary; three probe-transfer options; held-out space selection; leak-free pooling primary | 5.2 |
| L2 | Motion-adjusted protocol is now the primary analysis; pixel baselines required | 5.3 |
| L2b | New: trained violation detection, reported beside zero-fit surprise | 5.4 |
| L4 | Output readout for DiTs (the steering experiment); third direction control | 5.6 |
| Outputs | Two tracks (A continuation, B self-consistency); colour-tracker readout replaces SAM 2 / CoTracker; strict and loose gates | 6 |

---

## 1. Repository layout

Keep the existing layout. New modules:

```
physlaw/
  data/
    families/
      f7_freefall.py      # free fall, projectiles, bounces, drop pairs
      f8_ramp.py          # ramp energy exchange (shares scene code with F6)
      f9_pendulum.py      # optional
      f10_rolling.py      # optional
  models/
    wan21.py              # Wan2.1-T2V-1.3B and 14B
    wan22_ti2v.py         # Wan2.2-TI2V-5B and PhysRVG (same architecture)
    cogvideox5b.py        # CogVideoX-5B and VideoREPA-5B (LoRA adapter)
    cosmos_predict.py     # Cosmos-Predict2.5 Video2World (track A)
    positive_control.py   # PC1 small predictive transformer
  eval/
    l1_law_family.py      # law-family score
    l2b_detection.py      # trained violation detection
    laws/                 # one module per law: residual, legal family, null
      momentum.py
      energy.py
      com.py
      angular_momentum.py
      gravity.py
      axis_independence.py
      bounce.py
      ramp_energy.py
      galilean.py
      mass_independence.py
      pendulum.py
      rolling.py
```

Every stage stays a separate resumable CLI command reading from and writing to disk.

---

## 2. Data

### 2.1 Rendering defaults (as implemented)

- Simulator: Kubric (PyBullet physics, Blender Cycles rendering), with the existing fast post-processing (`physlaw/data/fast_post.py`), Blender thread cap of 8, and persistent render data.
- Master render: **48 fps, 289 frames (6.0 s), 832 × 480**. Each adapter subsamples to its native rate by exact stride (8, 16 or 24 fps) and resizes by whole-frame squash (letterbox available).
- Event frame (contact, branch or release, depending on family): sampled in frames 112 to 128 and stored per clip.
- Per clip: `video.mp4`, `masks.npz`, `labels.json` (per-object per-frame position, velocity, angular velocity, mass, radius, density; event frame; contact normal and point where relevant; scene parameters; camera intrinsics and extrinsics; seed; twin group ID; family; split; `physlaw.*` flags).
- Splits by hash of twin group ID (about 70/15/15), stable as sets grow.

### 2.2 Twin construction (as implemented, generalized)

A **twin group** is a set of clips that are byte-identical up to the event frame and differ only afterwards.

General rules for every family:
1. **One simulation per group.** Legal and violating outcomes after the event frame are computed in closed form from the simulated pre-event state, then re-keyframed exactly at the event time. This guarantees identical pre-event frames.
2. **Each violating twin breaks exactly one law.** Where a violation could also break a second law (for example, a momentum violation that also creates energy), choose the edit sign that avoids it, reject draws that still break it, and record a flag (for example `physlaw.energy_physical`) so L2 can stratify.
3. **At least two legal twins per group where the family allows it.** Two legal outcomes with different amounts of motion give L2 its legal-vs-legal control.
4. **Violation magnitude is recorded** per twin so L2 can test dose-response.

### 2.3 Scene families

Priority: **P0** required for the main paper, **P1** strongly recommended, **P2** optional.

#### F1. Two-body collisions (P0, built)

Two spheres on a frictionless plane, zero gravity, radius and density sampled independently.

| Twin | Construction | Laws broken |
|---|---|---|
| `conserve_e1` | Closed-form elastic outcome | None |
| `conserve_e015` | Closed-form outcome at e = 0.15 | None |
| `mom_violate_e1` | Common drift of both balls, sign chosen to lower \|v_cm\| | Momentum (energy too in about 1/3 of groups; flagged) |
| `mom_violate_e015` | Same edit on the e = 0.15 outcome; energy-creating draws rejected | Momentum |
| `energy_create` | Both Δv scaled by λ = (1 + 1.5)/(1 + e_sim) | Energy |

Laws scored on F1: linear momentum, no energy creation, restitution, **center-of-mass velocity** (constant through contact; new, no new data), **angular momentum about a fixed point** (Σ m r × v, conserved in frictionless sphere collisions; new, no new data).

Size: **400 groups target** (currently 64; see Section 11).

#### F2. Restitution sweep (P1)

Same scene as F1; e ∈ {0.0, 0.1, …, 1.0}; identical pre-contact frames. 100 groups × 11 = 1,100 clips. Used for L1 law-family scoring and L4.

#### F3. Hidden mass (P1)

As F1 legal twins, with **identical material, texture and radius** for both balls. Mass ratio m_A/m_B ∈ {0.33, 0.5, 1, 2, 3} through density only. 300 groups × 5. Used for L1 and L4.

#### F4. Galilean boosts (P0)

Take F1 legal groups (e = 1 and e = 0.15). Add a common velocity u to both initial velocities, u ∈ {0, small, large}, directions sampled; enlarge the floor so objects stay in frame. Restitution and center-of-mass-frame scattering angle are invariant. 200 seeds × 2 e-values × 3 boosts = 1,200 clips. **No other benchmark tests this.**

#### F6. Friction and slides (P1)

- **Friction geometry:** one cube on a flat banded floor (μ = a/g) or an incline (μ = tanθ − a/(g cosθ)); θ and μ sampled independently. 300 flat + 300 incline. Used for L3 transfer.
- **Mass-independent slides** (from v1 F5): two cubes of different density, same μ, same initial speed. Legal: equal deceleration. Violating twin: heavier cube decelerates 1.3× faster. 200 pairs.

#### F7. Free fall, projectiles and bounces (P0, new)

The most-tested law across the literature (Principia, Morpheus, LikePhys, RigidBench, Kneifl et al.).

Gravity on, standard g. Sub-families, each with a branch frame in 112 to 128 (mid-flight):

| Sub-family | Legal twin(s) | Violating twins | Laws |
|---|---|---|---|
| **Projectile** | Closed-form parabola | (a) gravity scaled ×1.5 after branch; (b) horizontal velocity changes ×0.6 at branch (axis coupling) | Constant g; axis independence |
| **Drop pairs** (from v1 F5) | Two objects of identical size and look, different density, released together; equal fall | Heavier object's acceleration ×1.3 after branch | Mass independence |
| **Floor bounce** | Rebound at e_floor = 0.8 and e_floor = 0.4 (two legal twins) | Rebound height ratio 1.3 (energy created) | Bounce height ratio = e²; no energy creation |

Sizes: projectile 300 groups, drop pairs 200 groups, bounce 200 groups. Projectile launch angle and speed sampled so the full arc stays in frame.

#### F8. Ramp energy exchange (P0, new)

Frictionless sliding block (cube, so no rotational energy) released from rest at height h. Ramp built from static rotated boxes; shares scene code with F6.

| Twin | Construction | Law |
|---|---|---|
| Legal, slope A | Straight ramp at angle θ₁ | Speed at bottom = √(2gh) |
| Legal, slope B | Straight ramp at θ₂ ≠ θ₁, same h; identical until the branch at the top of the ramp | Same final speed, different travel time |
| Legal, two-segment path | Ramp with a kink, same h | Same final speed |
| Violating, energy | Bottom speed × 1.25 | Energy created |
| Violating, path dependence | Bottom speed scaled with slope (√(sinθ/sinθ₁)) | Speed depends on path |

Pre-branch identity: the release frame and a short hold on a shared flat ledge form the identical segment; branch where the ledge meets the ramp. 250 groups.

#### F9. Pendulum (P2, new)

Rigid rod and bob on a hinge. **First check that Kubric's PyBullet backend exposes a hinge constraint; if it does not, skip F9 and record that in STATUS.** Legal: small-angle period 2π√(L/g), independent of bob mass and amplitude. Violating: period scaled with mass (heavier bob ×0.8 period). 150 groups.

#### F10. Rolling (P2, new)

Solid vs hollow cylinder released together down the same incline. Legal: solid arrives first (rotational inertia). Violating: hollow arrives first, or both arrive together. 150 groups. Gate on L0: if angular velocity is not decodable (R² < 0.3) in V-JEPA 2, report F10 as L0-only.

#### Nuisance variants (all families)

For 20% of seeds per family, render an extra variant with randomized camera pose, lighting and floor/object textures, physics identical. Tagged `nuisance=1`.

### 2.4 Data acceptance tests

Existing F1 tests stay. For each new family add:
- Twins byte-identical in every frame before the event frame.
- Legal twins satisfy their law to closed-form precision (relative error < 1e-6 in simulator units).
- Each violating twin breaks its target law by the specified factor and breaks no other scored law beyond tolerance (or carries a flag).
- F8: legal twins' bottom speeds equal √(2gh) within 1e-6; slope twins share h exactly.
- F7 drop pairs and F3: per-object mean RGB inside masks matches within tolerance.
- F4: restitution and CoM-frame angle match across boosts to 1e-6.
- Labels validate against the JSON schema.

---

## 3. Token projection and pooling (as implemented)

`tokens/project.py` maps each object's mask to each model's token grid (patch size, tubelet, resize). Pools (mean over tokens):

- **ROI per object:** disk at the projected centre, radius 1.5× projected object radius.
- **Pair ROI:** union of both ROIs.
- **Frame-wide:** all tokens.

Temporal windows: `pre`, `post`, `all`.

Leakage rule: an ROI placed at an object's **true post-event location** leaks the outcome into a forecast readout. For L1, the leak-free **frame-wide** pool is primary; the ROI pool is reported as secondary with the leak noted.

Mask-only floors (probes on the pooling mask itself) are reported for every ROI result.

---

## 4. Models

### 4.1 Adapter interface (as implemented)

```python
class ModelAdapter:
    def encode(self, video, noise_level=None) -> dict[int, Tensor]
    def forecast(self, video, context_end_frame) -> dict[int, Tensor]
    def surprise(self, video, context_end_frame, object_masks) -> dict[str, float]
    def generate(self, video, context_end_frame, seed) -> Video   # generators only
```

Implementation rules carried over from STATUS:
- **Encoders: encode context frames alone.** Never use `transformers`' `VJEPA2Model.forward(context_mask=...)` for context, which encodes the whole clip and leaks the future.
- **DiTs: context latents come from VAE-encoding the context frames alone** (the VAE's temporal GroupNorm otherwise leaks post-event frames).
- **DiT encode():** noise to the timestep matching each SNR in {−20, −8, 0, +8} dB, one conditional pass.
- **DiT surprise():** the model's own training loss on post-event latents over the SNR timesteps × 2 noise draws (identical across twins), with clean context and with noisy context.
- **Two-pass caching for diffusion models:** `--no-forecast` caches encode + surprise; a later pass adds forecasts and generations without recomputing.
- Weights hash-verified; `HF_HUB_OFFLINE=1` for runs.

### 4.2 Models, grouped by the question each answers

Status: **DONE**, **RUNNING**, **ADD**. Priority as in Section 2.3.

#### Group 1. Does the training objective matter? (encoders)

| Model | Objective | Priority | Status | Notes |
|---|---|---|---|---|
| V-JEPA 2 ViT-L | Predictive latent | P0 | DONE | |
| V-JEPA 2 ViT-g | Predictive latent, larger | P1 | ADD | Scale within objective (Joseph et al. used L, H, g) |
| VideoMAE-v2-giant (pre-training checkpoint with decoder) | Pixel reconstruction | P0 | DONE | |
| V-JEPA 2-AC | Action-conditioned predictor | P2 | ADD | Encoder is the frozen V-JEPA 2 encoder; only the predictor differs. Passive scenes need a zero or null action, which is off-distribution. Run only L1/L2 with that caveat stated |

#### Group 2. Do results replicate on the literature's own models? (DiTs)

| Model | Used by | Priority | Status |
|---|---|---|---|
| CogVideoX-2b | Esmati, LikePhys, Seeking Physics | P0 | DONE |
| CogVideoX1.5-5B-I2V | Kneifl | P1 | RUNNING (pass 1) |
| **Wan2.1-T2V-1.3B** | Esmati (best model), LikePhys | **P0** | ADD |
| Wan2.1-T2V-14B | Kneifl (main model), LikePhys | P1 | ADD |
| Open-Sora 2.0, LTX-Video, HunyuanVideo | Kneifl, Esmati, LikePhys | P2 | ADD if time allows |

#### Group 3. Does physics post-training change internals? (pairs)

| Base | Physics-trained | Training signal | Priority |
|---|---|---|---|
| Wan2.2-TI2V-5B | PhysRVG (weights: HappyP4nda/PhysRVG) | RL reward for rigid-body collisions | P0 |
| CogVideoX-5B | VideoREPA-5B (released LoRA adapter, github.com/aHapBean/VideoREPA) | Aligns DiT token relations to VideoMAE-v2 | P0 |

Run L0, L1, L2, L2b on both members of each pair with identical settings. VideoREPA's teacher (VideoMAE-v2) is at chance on L2 in this benchmark; report whether the student gains law sensitivity the teacher lacks. Both pairs are text-to-video or TI2V: skip output track A for them.

Follow-up (not in this spec's milestones): a VideoREPA variant aligned to V-JEPA 2 instead of VideoMAE-v2, using VideoREPA's public training code.

#### Group 4. Output track A (scene continuation)

| Model | Priority | Notes |
|---|---|---|
| Cosmos-Predict2.5 Video2World | P0 | Natively multi-frame conditioned; the remaining route to track A after CogVideoX latent replacement, held-frame SDEdit and reconstruction guidance all failed |
| LTX-Video (multi-frame conditioning) | P1 | Backup if Cosmos fails the gate |

#### Group 5. Output track B (self-consistency from frame 0)

| Model | Priority | Status |
|---|---|---|
| Wan2.2-I2V-A14B | P0 | Started (22 clips). Finish 20 groups × 4 seeds |
| Wan2.1-I2V-14B | P1 | ADD |

#### Group 6. Controls (Section 4.4)

| Control | Priority | Status |
|---|---|---|
| Untrained V-JEPA 2 (same architecture, random init) | P0 | DONE |
| PC1: small predictive transformer trained on the render distribution | **P0, critical path** | ADD |
| PC2: a DiT fine-tuned on simulator data | P1 | ADD |

### 4.3 Checkpoint pinning

Pin exact repo IDs, revisions and sha256 for every model in its config, as done for VideoMAE-v2.

### 4.4 Positive controls

**PC1. Small predictive transformer (critical path).**
- Architecture: V-JEPA-style encoder + predictor, ViT-S scale, same input format as V-JEPA 2.
- Training data: a **separate training render set** (never the benchmark's evaluation twins): legal clips only from F1, F2, F3, F4, F7 and F8, at least 3,000 clips per family. Lower resolution (256 × 256) and lower samples per pixel are allowed for this set.
- Objective: masked latent prediction, with future-frame masking so the predictor learns to forecast past the event frame.
- Report training curves and in-distribution prediction error.
- Pass criterion for "the test is passable": PC1 must reach L1 S_family ≥ 0.5 and L2 adjusted win rate ≥ 0.7 on F1 momentum contrasts. If PC1 fails, the test is too hard or broken: stop and report before claiming that any large model fails.

**PC2. Simulator-fine-tuned DiT.**
Fine-tune Wan2.2-TI2V-5B (or CogVideoX-2b if memory requires) on the PC1 training render set, following RigidBench's recipe (first-frame pinning; full fine-tune or rank-32 LoRA). Shows whether a full-scale generator can pass when trained on the right data.

---

## 5. Evaluation levels

### Shared protocol (as implemented)

- Ridge probes, penalty by nested CV on training folds.
- Cross-validation grouped by twin group ID; 12 fold seeds; with fewer than 100 groups, 5-fold × 4 seeds cross-fitting is allowed and must be stated.
- Permutation null: 500 shuffled refits, max over the searched grid (layers, pools, noise levels), corrected p-values.
- Kinematic observers fit through the identical pipeline (`obs_vel`, `obs_speed`, `obs_quad`, `obs_normal`).
- **Selection rule (new):** whenever a score is reported for a best layer, pool or noise level, select it on held-out groups and report the score on different held-out groups. Report the number of variants searched.

### 5.1 L0. States

Targets per object: position, velocity, speed, direction, log mass, radius, angular velocity (F9, F10); pre and post event.

Baselines, all required:
- Mask-only floor for every ROI result.
- Untrained control.
- **Noised-input-latent baseline for DiTs** (probe on the model's own noised VAE latent, same pooling; Kneifl et al., Esmati et al.).
- **One-twin-per-group refit** for post-event targets when the number of groups is under 200 (pre-event targets repeat across twins, post-event ones do not; STATUS milestone 3).

**Velocity protocol check (new):** for Wan2.2-TI2V-5B, report velocity decodability under (a) this benchmark's protocol (ridge, ROI) and (b) RigidBench's protocol (unregularized linear probe on actor-mask-averaged tokens, Adam, 20 epochs). RigidBench reports weak velocity in Wan 2.2; this benchmark finds strong velocity in CogVideoX-2b. Every momentum test depends on velocity, so the disagreement needs a matched comparison.

### 5.2 L1. Forecast law consistency (revised)

Give the model only pre-event frames; take `forecast()` activations for post-event token-frames; decode predicted states; score them against laws.

#### Probes for decoding forecasts (report all three)

| Option | Training data for the probe | Role |
|---|---|---|
| **P-real-mm** (as implemented) | `encode()` of real post-event frames; forecast features moment-matched to real features (per-dimension mean and sd from training groups, no labels) | Current default |
| **P-pred-real** (new, preferred) | The model's own forecast-space outputs for **masked real** post-event frames with the full clip as context (V-JEPA 2: predictor outputs for masked real future tokens; VideoMAE: decoder hidden states with real future visible elsewhere) | Same feature space as forecasts, independent of the context-only forecast |
| **P-forecast** (as implemented) | Forecast features themselves, cross-fitted | Information ceiling only; never used to score laws |

#### Scores

1. **Outcome score E** (as implemented; primary): E = ‖v̂′ − v′‖ / ‖v′_null − v′‖, where v′_null is the no-event null. S_outcome = 1 − E. 1 = perfect, 0 = no better than the null.
2. **Law-family score** (new; primary): separates knowing the **law** from knowing a **hidden parameter**.
   - Each law defines a legal family of outcomes given the pre-event state, parameterized by quantities not visible before the event.
   - d(x) = min over the family of the normalized distance from x to a family member.
   - S_family = 1 − (d_forecast − d_floor) / (d_null − d_floor), where d_floor is the probe floor (d of probes applied to real legal frames) and d_null is d of the no-event null. Clip to [−1, 1]. Also report the argmin parameter (for example ê).

| Family | Legal family | Hidden parameter | Null |
|---|---|---|---|
| F1, F2, F3 | Momentum-conserving outcomes v′(e) for e ∈ [0, 1] given masses, pre-velocities and contact normal | e | Balls pass through (equivalent to e = −1, outside the family) |
| F7 projectile, drop | Parabola with standard g (unique) | None | Constant velocity after branch (no gravity) |
| F7 bounce | Rebound with e_floor ∈ [0, 1] | e_floor | No bounce (continues through floor) |
| F8 | Bottom speed √(2gh) (unique) | None | Speed held at ramp-entry value |
| F9 | Period 2π√(L/g) (unique) | None | Frozen bob |

3. **Momentum residual S_momentum** (reported, **not primary**): a forecast that ignores the collision conserves momentum exactly in F1, so this score cannot show understanding on its own. Report it next to S_outcome with that caveat.
4. **Energy residual**: report only when decoded forecast speeds are within 20% of the real spread; otherwise mark "not scorable" (shrunk forecasts never create energy).
5. CoM and angular momentum residuals (F1): same caveat as momentum; report beside S_family.

Pooling: frame-wide (leak-free) primary; ROI at the true post location secondary, labeled as leaking location.

For DiTs, L1 is run per (SNR, layer) space; select the space on held-out groups (selection rule above).

### 5.3 L2. Violation sensitivity, zero-fit (revised)

For each twin group, compute `surprise()` on the post-event segment for every twin, with identical context, masks and noise draws.

**Primary analysis: motion-adjusted** (as implemented).
1. On legal twin pairs only, regress Δlog surprise on Δlog covariates: post-event speed, copy-last-frame pixel error, constant-velocity pixel error. Report R² of the calibration.
2. For violating vs legal contrasts, compute the adjusted surprise difference using leave-one-group-out calibration.
3. Report adjusted paired win rate, bootstrap CI, Wilcoxon p, Cliff's delta, under all covariate sets.
4. The legal-vs-legal contrast after adjustment must sit at chance; if it does not, the adjustment is insufficient and the contrast is reported as inconclusive.

Also report unadjusted win rates and the pixel baselines (copy-last-frame, constant-velocity) for every contrast.

Contrasts per family:
- F1: each violation vs its matching legal twin; `energy_create` vs `conserve_e015`; legal-vs-legal control. Stratify momentum contrasts on `energy_physical`.
- F7: gravity change, axis coupling, heavier-falls-faster, energy-creating bounce, each vs legal; legal-vs-legal (two e_floor values).
- F8: energy and path-dependence violations vs legal; legal-vs-legal (two slopes).
- F6 slides, F9, F10: each violation vs legal.

Distances for encoders: L1, MSE and cosine, object region and whole frame. Dose-response: Spearman correlation of adjusted surprise with violation magnitude.

### 5.4 L2b. Trained violation detection (new)

Same twins as L2. Train a logistic probe on post-event activations (ROI and frame pools) to classify violating vs legal, grouped CV, permutation null.

Report beside L2 for every contrast, and compute an **encoded-but-unused index**: L2b AUC − 0.5 minus (L2 adjusted win rate − 0.5). A large positive value means the information is decodable while the model's own prediction ignores it.

### 5.5 L3. Invariance (as v1)

- **Galilean (F4):** train restitution and CoM-angle probes on u = 0 clips; evaluate on boosted clips. Invariance error = |decoded(boosted) − decoded(base)| / sd(target), compared with within-condition error. Repeat L2 on boosted twins.
- **Mass independence (F7 drop pairs, F6 slides):** from `forecast()`, decode fall time or deceleration per object; report the slope against log mass (law predicts zero) and compare with the probe floor.
- **Transfer (F6 and nuisance):** flat → incline and the reverse for μ, acceleration and size (control); clean → nuisance for F1 and F7.

### 5.6 L4. Causal propagation (revised)

On F3 hidden-mass clips: fit a mass-ratio direction from pre-event ROI activations; add α · direction to object A's ROI tokens in pre-event token-frames, α ∈ {−2, −1, 0, 1, 2} activation-sd units; rerun `forecast()`; decode post-event velocities.

- Metric: correlation between the predicted change in Δv_B′ (from momentum conservation at the implied mass ratio) and the observed change; specificity on non-target quantities.
- Controls: rank-matched random subspace, variance-matched principal components, matched random direction.
- **Output readout (DiTs, new):** after the intervention, also `generate()` and track the result with the track B readout. Report whether the generated outcome shifts in the direction momentum conservation predicts. This is the steering experiment.

---

## 6. Output-level evaluation (revised)

### 6.1 Readout (as implemented)

Colour tracker initialized from the real context, with:
- blob splitting only for elongation ≥ 1.35 and area ≤ 2× the two balls;
- shadow folding (blobs closer than 0.8 × (r_A + r_B) merge into the larger);
- rejections: `size_change` (> 1.5× context area), `extra_objects`, scene cut (background change of the first 12 generated frames vs context median > 0.25), jitter, short window, still approaching, no collision.

Validated per family on real legal and violating clips before use. F1 (16 fps): 312/320 measurable strict, 318/320 loose; momentum residual 0.03 to 0.04 on legal twins. **Each new family needs its own readout validation on real clips** (F7: parabola fit for g ratio and axis independence; bounce height ratio; F8: bottom speed ratio).

### 6.2 Track A: scene continuation

Generate from the real pre-event context; score with the same laws as L1 using the readout. Run a feasibility gate first (20 clips); if fewer than 50% are measurable, report the gate result and skip extra seeds. Generators: Group 4.

### 6.3 Track B: self-consistency from frame 0

Generate from frame 0 with the minimal prompt; score whether the generated event obeys each law given its own generated pre-event state. Report strict and loose gates. Generators: Group 5.

### 6.4 External output benchmarks (optional)

If Principia or Morpheus evaluation code is available, score Group 2 and Group 3 models on the phenomena that overlap F1, F7 and F8.

**Key analysis:** correlation across models, and within each post-training pair, between internal scores (L1 S_family, L2 adjusted win rate, L2b) and output scores.

---

## 7. Statistics

- Bootstrap 95% CIs over twin groups.
- Holm correction across contrasts within each family and level.
- Every "best" variant follows the held-out selection rule (Section 5).
- Report the number of groups, test groups per fold and variants searched for every number.

---

## 8. Results format

Per model: `results/<model>/<family>/<level>.{json,md}` with config hash, checkpoint, metrics, CIs, nulls, and the layers, pools and noise levels used.

Leaderboard columns: L0 (mean R²), L1 S_family per law, L2 adjusted win per contrast, L2b AUC, encoded-but-unused index, L3 invariance errors, L4 propagation correlation, output scores (tracks A and B).

Figures:
1. L0 vs L1 per model: states decodable, laws weak.
2. L2 adjusted vs unadjusted: the motion confound.
3. L2 vs L2b: encoded but unused.
4. Galilean invariance error vs boost size.
5. Internal vs output adherence scatter.
6. Post-training pairs: base vs physics-trained deltas on every level.
7. Objective comparison: predictive (V-JEPA 2) vs reconstruction (VideoMAE-v2) vs denoising (DiTs) across laws.
8. Positive control vs large models on the same axes.

---

## 9. Engineering requirements

As in v1, plus:
- Diffusion models run in two passes (Section 4.1).
- Each new family has its own data config and acceptance tests before any model runs on it.
- Every stage records the git hash, config hash and seeds.
- `docs/STATUS.md` is updated after each milestone with results and any deviation from this spec.

---

## 10. Milestones

### Done (keep)

| # | Milestone |
|---|---|
| 1 | F1 generator, twins, acceptance tests; 64 groups rendered |
| 2 | V-JEPA 2 adapter, token projection, overlays, activation caching |
| 3 | L0 + L2 on V-JEPA 2 and untrained control (64 groups) |
| 4 | L1 on V-JEPA 2 (64 groups) |
| 6 | VideoMAE-v2-giant: L0 to L2 |
| 7 (part) | CogVideoX-2b: L0, L1 (preliminary), L2, output gate; track B readout validated; Wan2.2-A14B 22 track B clips |

### Next, in order

| # | Milestone | Depends on |
|---|---|---|
| 12 | Render F1 to 400 groups (if confirmed, Section 11); re-run milestones 3, 4, 6 and CogVideoX-2b L0/L2 on the full set | — |
| 13 | L1 revision: law-family score, P-pred-real probes, held-out space selection; re-run L1 for V-JEPA 2, VideoMAE-v2, CogVideoX-2b | — |
| 14 | L2b on all finished models | — |
| 15 | F1 add-on laws (CoM, angular momentum) in L1 and outputs | 13 |
| 16 | **PC1 training render set + PC1 training + PC1 through L0 to L2b** | 12, 13 |
| 17 | Wan2.1-T2V-1.3B adapter; L0, L1, L2, L2b on F1 | 13 |
| 18 | F7 generator, tests, readout validation; all finished models on F7 | 16 |
| 19 | F4 generator; L3 Galilean on all finished models | 12 |
| 20 | Post-training pairs: Wan2.2-TI2V-5B + PhysRVG; CogVideoX-5B + VideoREPA-5B; L0 to L2b on F1 and F7 | 13, 18 |
| 21 | Cosmos-Predict2.5 adapter; track A gate on F1 | — |
| 22 | Finish Wan2.2-A14B track B (20 groups × 4 seeds); add Wan2.1-I2V-14B | — |
| 23 | F8 generator; all finished models on F8 | 16 |
| 24 | F2, F3; L1 law-family on F2/F3; L4 on V-JEPA 2 and one DiT, with output readout | 13 |
| 25 | PC2 simulator fine-tune; L0 to L2b | 16 |
| 26 | CogVideoX1.5-5B-I2V results (after its passes finish); Wan2.1-T2V-14B; V-JEPA 2 ViT-g | — |
| 27 | F6 (friction, slides), L3 transfer and mass independence | — |
| 28 | Aggregation, leaderboard, figures | all |
| 29 | Optional: F9, F10, V-JEPA 2-AC, Open-Sora 2.0, LTX-Video, HunyuanVideo, external output benchmarks | — |

---

## 11. Open decisions for the user

- [ ] **F1 size.** STATUS records a 2026-11-09 decision to keep F1 at 64 groups. This spec recommends 400 because CIs at ~9 test groups per fold are wide and PC1 and L2b need more data. Confirm 400, or pick a number (minimum recommended: 200).
- [ ] PC1 pass criterion values (S_family ≥ 0.5, adjusted L2 win ≥ 0.7): accept or adjust.
- [ ] PC2 base model: Wan2.2-TI2V-5B or CogVideoX-2b.
- [ ] F9 and F10: include or drop.
- [ ] Target venue and deadline, which sets how far down the milestone list to go.
