# Benchmark Spec: Physical Laws Inside Video Models

Working name: **PhysLaw-Internal**. Implementation spec for Claude Code.

## 0. Goal

Build a benchmark that tests whether video models obey physical laws **in their internal representations**, and compare that with whether their **generated outputs** obey the same laws.

Hypothesis under test: models encode per-object physical *states* (position, velocity, size), but the *laws relating those states* (momentum conservation, energy bounds, Galilean invariance, mass independence) do not hold in what the models compute internally.

The benchmark must be able to return either answer. A positive control (Section 4.3) shows the tests can be passed; a negative control shows they can't be passed by pixel layout alone.

### Deliverables

1. A data generator producing labeled synthetic videos with matched "twin" clips.
2. A model-adapter layer that extracts activations, model forecasts and surprise scores from each model.
3. Five evaluation levels (L0 to L4) with fixed metrics and baselines.
4. An output-level evaluation of generated videos using the same laws.
5. A results pipeline: per-model JSON results, a leaderboard table and standard figures.

---

## 1. Repository layout

```
physlaw/
  configs/               # YAML configs (Hydra or OmegaConf); no hardcoded paths
    data/                # one config per scene family
    models/              # one config per model adapter
    eval/                # one config per level
  physlaw/
    data/
      generate.py        # Kubric scene generation
      twins.py           # twin construction (violations, boosts)
      labels.py          # per-frame physical labels
      nuisance.py        # camera, lighting, texture randomization
    models/
      base.py            # ModelAdapter interface
      vjepa2.py
      videomae.py
      cogvideox.py
      wan.py
      untrained.py       # negative control
      positive_control.py
    tokens/
      project.py         # object masks -> token-grid indices per model
      pool.py            # ROI and frame-wide pooling
    probes/
      ridge.py           # ridge probes, grouped CV, nulls
      observer.py        # kinematic-observer baselines
    eval/
      l0_states.py
      l1_forecast_laws.py
      l2_surprise.py
      l3_invariance.py
      l4_causal.py
      outputs.py         # output-level law checks on generated video
    track/
      track.py           # SAM 2 / CoTracker tracking for generated videos
    analysis/
      aggregate.py
      figures.py
  scripts/               # CLI entry points, one per stage
  tests/
  README.md
```

All stages are separate CLI commands that read and write to disk, so any stage can be rerun alone.

---

## 2. Data

### 2.1 Rendering defaults

- Simulator: Kubric (PyBullet physics, Blender rendering). If an existing Kubric collision generator is present in this repo, reuse its twin-editing logic.
- Master render: 81 frames at 24 fps, 832×480. Each model adapter resamples frames and resizes to its native input.
- Contact frame for all collision clips: sampled in frames 36 to 44 and stored per clip.
- Store per clip:
  - `video.mp4`
  - `masks.npz`: per-object instance masks, every frame
  - `labels.json`: per-object per-frame position, velocity, mass, radius, density; contact frame; contact normal; scene params; camera intrinsics and extrinsics; seed; twin group ID; family; split
- Splits are assigned by **twin group ID**, so all twins of one scene land in the same split. Default: 70% train, 15% val, 15% test.

### 2.2 Scene families

**F1. Two-body collisions (core).** Two spheres on a frictionless plane, zero gravity. Radius and density sampled independently.

Each scene seed produces a twin group whose members are **bit-identical up to the contact frame** and differ only afterwards:

| Twin | Construction | Physical status |
|---|---|---|
| `conserve_e1` | simulator, e = 1.0 | legal, energy conserved |
| `conserve_e015` | simulator, e = 0.15 | legal, energy lost |
| `mom_violate_e1` | keep \|Δv_A\| + \|Δv_B\| fixed, change the split between objects | momentum violated |
| `mom_violate_e015` | same edit on the e = 0.15 run | momentum violated |
| `energy_create` | scale both Δv by λ = (1 + e_target)/(1 + e_sim), e_target = 1.5 | momentum conserved, energy created (impossible) |

Twin construction must hold these identities, checked by unit tests:
- Frictionless contact: |Δv_A| + |Δv_B| = (1 + e)·u_n and |Δv_B|/|Δv_A| = m_A/m_B, where u_n is the approach speed along the contact normal n.
- Momentum-violating twins keep the sum and change the ratio, so restitution and contact geometry are unchanged.
- Energy-creating twins scale both Δv by λ, so m_A·Δv_A + m_B·Δv_B = 0 still holds exactly.

Size: 400 seeds × 5 twins = 2,000 clips.

**F2. Restitution families.** Same scene, e ∈ {0.0, 0.1, …, 1.0} (11 levels), identical pre-contact segments. 100 seeds × 11 = 1,100 clips. Legal outcomes lie on a 1-D line; used for L1 and L4.

**F3. Hidden mass.** As F1 (legal twins only), but both spheres use **identical material and texture** and equal radii. Mass ratio m_A/m_B ∈ {0.33, 0.5, 1, 2, 3} via density only, so mass is visible only through collision dynamics. 300 seeds × 5 ratios = 1,500 clips.

**F4. Galilean boosts.** Take 200 F1 seeds (legal e = 1 and e = 0.15). Add a common velocity u to both objects' initial velocities, u ∈ {0, small, large} with directions sampled. Enlarge the floor so objects stay in frame. Restitution and the center-of-mass-frame scattering angle are invariant across boosts. 200 seeds × 2 e-values × 3 boosts = 1,200 clips.

**F5. Mass independence.**
- `drop`: two objects of identical size and appearance but different density, released together from the same height under gravity. Fall time must be equal. 200 pairs.
- `slide`: two cubes of different density on the same floor with the same friction coefficient μ and the same initial speed. Deceleration must be equal. 200 pairs.

**F6. Friction geometry.** One cube on a flat banded floor (μ = a/g) and on inclines (μ = tanθ − a/(g·cosθ)), θ and μ sampled independently. 300 flat + 300 incline. Used for L3 transfer.

**Nuisance variants.** For 20% of seeds in every family, render an extra variant with randomized camera pose, lighting and floor/object textures (`nuisance.py`), keeping physics identical. Tagged `nuisance=1`.

### 2.3 Data acceptance tests (`tests/test_data.py`)

- Twins are byte-identical in every frame before the contact frame.
- Conserving twins: momentum residual < 1e-6 (simulator units). Violating twins: residual above a set threshold.
- `energy_create`: KE_post/KE_pre > 1 for every clip.
- Hidden-mass pairs: per-object mean RGB inside masks differs by less than a small tolerance.
- Galilean boosts: restitution and CoM-frame angle match across boosts to 1e-6.
- Every label file validates against a JSON schema.

---

## 3. Token projection and pooling

`tokens/project.py` maps each object's instance mask to the token grid of each model using that model's patch size, tubelet size and resize/crop, giving per-object token indices per token-frame.

Pools, all mean-pooled over tokens:
- **ROI (per object):** disk centered on the projected object center, radius 1.5× projected object radius (following Kneifl et al.).
- **Pair ROI:** union of both objects' ROIs.
- **Frame-wide:** all tokens.

Temporal windows: `pre` (token-frames fully before contact), `post` (fully after), `all`.

Acceptance test: overlay projected ROI tokens on frames for 20 random clips per model and save images for manual inspection.

---

## 4. Models

### 4.1 Adapter interface (`models/base.py`)

```python
class ModelAdapter:
    name: str
    layers: list[int]          # subset of blocks to cache (about 12, evenly spaced)
    token_grid: TokenGridSpec  # patch, tubelet, resize/crop for token projection

    def encode(self, video, noise_level=None) -> dict[int, Tensor]:
        """Activations per cached layer, shape [T_tok, H_tok, W_tok, D].
        For diffusion models, noise_level is an SNR in dB; the clip is
        VAE-encoded, forward-noised and passed once through the DiT."""

    def forecast(self, video, context_end_frame) -> dict[int, Tensor]:
        """The model's own internal prediction of the future, given frames
        before context_end_frame. Returns activations for future token-frames.
        - V-JEPA 2: predictor outputs for masked future tokens.
        - VideoMAE: decoder hidden states for masked future tubelets.
        - DiTs: activations captured while generating the continuation
          conditioned on the context frames, at fixed sampling steps."""

    def surprise(self, video, context_end_frame, object_masks) -> float:
        """Zero-fit surprise on the post-context segment, restricted to
        object regions.
        - V-JEPA 2: predictor error (L1, MSE and cosine all reported).
        - VideoMAE: masked reconstruction loss, identical mask draws across twins.
        - DiTs: denoising loss on future latents with clean context latents,
          averaged over K noise draws and a fixed timestep set."""

    def generate(self, video, context_end_frame, seed) -> Video:
        """Generators only: pixel continuation for output-level evaluation."""
```

### 4.2 Model list

| Priority | Model | Role |
|---|---|---|
| P0 | V-JEPA 2 ViT-L + predictor | encoder with native forecasting |
| P0 | VideoMAE-v2 (base; giant if compute allows) | reconstruction encoder |
| P0 | CogVideoX (2B T2V and 1.5-5B I2V) | diffusion generator |
| P0 | Untrained ViT matching V-JEPA 2 ViT-L | negative control |
| P0 | Positive control (4.3) | shows tests are passable |
| P1 | Wan 2.2 TI2V-5B | diffusion generator |
| P1 | PhysRVG (Wan 2.2 TI2V-5B physics RL fine-tune) | physics post-training comparison |
| P2 | Wan 2.1 I2V-14B, Cosmos, VideoREPA (if checkpoints exist) | coverage |

Pin exact checkpoint IDs in each model config. Cache activations as float16 on disk, one file per (model, clip, layer, noise level).

### 4.3 Positive control

A small transformer trained from scratch on the F1 to F5 **train split** with a latent-prediction objective (V-JEPA-style), at the same input format as V-JEPA 2. It sees the same rendered videos, so if it passes L1 to L3 while large models fail, the failure belongs to the large models. Report its training curve and in-distribution scores. Optional second positive control: CogVideoX fine-tuned on the train split.

---

## 5. Evaluation levels

Shared probe protocol (`probes/ridge.py`):
- Ridge regression from a pooled activation vector to a target; penalty chosen by nested CV on training folds.
- Grouped K-fold by twin group ID (K = 5), repeated over 12 fold seeds; report mean ± sd.
- Continuous targets scored by out-of-fold R²; binary targets by AUC.
- Permutation null: refit on shuffled labels (500 draws), take the max over the searched grid of layers, pools and noise levels, and report corrected p-values.

Kinematic observers (`probes/observer.py`) are fitted through the identical pipeline, from ground-truth visible quantities:
- `obs_vel`: pre-contact velocity components of both objects
- `obs_speed`: plus speeds
- `obs_quad`: quadratic features of the above
- `obs_normal`: velocities projected on the contact normal

### L0. States (sanity floor)

Targets per object: velocity, speed, direction, log mass, radius; pre and post contact.
Pools: ROI per object, frame-wide. All cached layers and noise levels.
Report: best-layer R² per target with observer, untrained-model and permutation baselines.
**Expected:** high, matching prior work. L0 also produces the probes used by L1 and L3.

### L1. Forecast law consistency (core test)

1. Fit post-contact state probes (v_A′, v_B′, per object, ROI pool) on `encode()` activations of **real** post-contact frames from the training split.
2. On test clips, give the model only pre-contact frames and take `forecast()` activations for the post-contact token-frames.
3. Apply the step-1 probes to the forecast activations to get decoded predicted states.
4. Compute law residuals using true masses and true pre-contact velocities:
   - momentum: r_p = |m_A·v̂_A′ + m_B·v̂_B′ − p_pre| / |p_pre|
   - energy: r_E = max(0, KE′ − KE_pre) / KE_pre (energy must not increase)
   - restitution (F2): |ê − e_true|, with ê from decoded relative normal velocities

Calibration, to avoid the circularity of accurate probes satisfying laws automatically:
- **Probe floor:** the same residuals from probes applied to `encode()` of the **real** post-contact frames. This measures probe noise alone.
- **Null forecast:** residuals if both objects continue unchanged through contact (no collision).
- **Law adherence score:** S = 1 − (r_forecast − r_floor) / (r_null − r_floor), clipped to [−1, 1]. S = 1 means the forecast obeys the law as well as real video does under the same probes. S = 0 means no better than ignoring the collision.

Report S per law, per model, per family (F1 legal twins, F2, F3), with bootstrap 95% CIs over twin groups.

### L2. Violation sensitivity (zero-fit)

For each twin group, compute `surprise()` on the post-contact segment for every twin, using identical context, masks and noise draws across twins.

Contrasts:
- `mom_violate_*` vs matching `conserve_*`
- `energy_create` vs `conserve_e015` (impossible vs legal energy change)
- `conserve_e1` vs `conserve_e015` as a legal-vs-legal control; this difference should be small relative to the violation contrasts

Metrics: paired win rate (violating more surprising), Wilcoxon signed-rank p, Cliff's delta, and the Spearman correlation between surprise and violation magnitude. Report all three surprise distances for V-JEPA 2.

### L3. Invariance

- **Galilean (F4):** train restitution and CoM-angle probes on `u = 0` clips; evaluate on boosted clips. Invariance error = |decoded(boosted) − decoded(base)| / sd(target). Compare against the same probes' within-condition error. Also run L2 surprise on boosted twins and check that violation detection does not depend on u.
- **Mass independence (F5):** from `forecast()` given pre-release frames, decode fall time (drop) or deceleration (slide) for each object. Report the slope of the decoded quantity against log mass; the law predicts zero. Compare with the slope on real-frame encodings (probe floor).
- **Transfer (F6 and nuisance):** train μ, acceleration and size probes on flat; test on incline, and the reverse. Transfer R² and Pearson r for each target, with size as the control. Repeat for clean → nuisance splits in F1.

### L4. Causal propagation (P1)

On F3 hidden-mass clips: fit a mass-ratio direction from pre-contact ROI activations. Add α·direction to object A's ROI tokens in pre-contact token-frames, then rerun `forecast()`. Decode post-contact velocities and compare the observed change with the change momentum conservation predicts for the implied new mass ratio.

Metrics: correlation between predicted and observed Δv_B′ across α ∈ {−2, −1, 0, 1, 2} (in units of the activation sd along the direction); specificity as change in non-target quantities.

---

## 6. Output-level evaluation

For each generator, call `generate()` from pre-contact frames for F1 legal twins, F2, F3 and F5 (4 seeds per clip).

1. Track objects with SAM 2 (masks) and CoTracker (points).
2. Estimate post-contact velocities from tracks.
3. Compute the same residuals as L1 (momentum, energy, restitution, mass independence), normalized the same way using the null forecast.
4. Drop clips where tracking fails quality checks and report the drop rate.

Optional: if Principia or PhysWorldBench evaluation code is available, score the same models on them.

**Key analysis:** correlation across models (and across post-trained vs base pairs) between internal law adherence (L1 S, L2 win rate) and output adherence.

---

## 6b. Output track B: self-consistency (added 2026-11-06, decided by the user)

Section 6 (now "track A: continuation") needs a generator that genuinely continues the real pre-contact frames,
and scores its outcome against the TRUE masses and pre-contact velocities. Image- and text-to-video models cannot
take a velocity from one frame, and training-free conditioning (context replacement, held-frame SDEdit,
reconstruction guidance) did not make CogVideoX-2b / 1.5-5B-I2V continue F1 reliably (docs/STATUS.md, milestone 7).
Track B opens the output level to any image-to-video generator:

1. Input: frame 0 of each F1 group (identical across twins) + the minimal prompt
   (`physlaw.models.cogvideox.minimal_prompt`: names the two balls, "The two balls collide and then separate.").
   Official settings only. The model invents the whole clip, including the approach (Physics-IQ's I2V protocol).
2. Readout: track both balls; contact; velocities in short windows on both sides of contact from the GENERATED clip.
3. Laws, relative to the clip's OWN pre-contact motion: momentum (true masses: size and material are visible in F1),
   energy not created (restitution e <= 1 along the contact normal estimated from the velocity changes), straight
   paths, opposite impulses. No comparison with the true outcome: the generated approach is not the real one.
4. Quality gate and drop rate as in §6 (cuts, size changes, extra objects, lost tracks, jitter -> dropped and counted).
   **Loose gate (2026-11-06, user decision; `readout_self(..., gate="loose")`, `track_b_score.py --gate loose`):**
   only MEASUREMENT failures drop a clip (size change, extra objects, lost tracks, no collision, never separates,
   < 5 frames on a side, velocity standard error > 7% of the scene's top speed). Law breaks (impulses not opposite,
   still approaching after contact, curved paths, a ball accelerating on its own) are kept and listed per clip as
   `violations`, so a model that collides badly scores as violating, not as unmeasurable. The strict gate (any
   failed check drops) stays available for comparison.

Track B tests whether a model's collisions obey the laws; track A tests whether it predicts the right outcome. Both
feed the internal-vs-output analysis. Track A needs a native multi-frame model (e.g. Cosmos Video2World, Wan VACE)
or a CogVideoX fine-tuned for continuation.

---

## 7. Results format

Per model, write `results/<model>/<level>.json` with config hash, checkpoint ID, metric values, CIs, null distributions and the layers, pools and noise levels used.

`analysis/aggregate.py` builds:
- A leaderboard: rows = models, columns = L0 (mean), L1 S per law, L2 win rates, L3 invariance errors, output adherence.
- Figures:
  1. L0 vs L1 per model: states decodable, laws failing.
  2. L2 paired surprise distributions per contrast.
  3. Galilean invariance error vs boost size.
  4. Internal vs output adherence scatter.
  5. Base vs physics post-trained model deltas.

---

## 8. Engineering requirements

- Python 3.11, PyTorch, Hugging Face `transformers` and `diffusers`. Pin versions in `pyproject.toml`.
- Runs on a single 80 GB GPU per job; target RunPod or Lambda.
- Every stage resumable and idempotent: skip outputs that exist unless `--force`.
- Global seeds and per-clip seeds recorded in outputs.
- Activation cache layout documented in README with a disk estimate per model.
- Logging to stdout plus a JSONL run log; no notebooks required to reproduce results.
- Unit tests for data invariants, token projection and probe pipeline on synthetic linear data (probe must recover a known direction).

---

## 9. Milestones (implement in order)

1. Data generator for F1 with twin construction and all F1 acceptance tests passing.
2. V-JEPA 2 adapter (`encode`, `forecast`, `surprise`) and token projection with overlays.
3. L0 and L2 on V-JEPA 2 + untrained control. Checkpoint: reproduce the earlier finding that V-JEPA 2 is more surprised by energy-creating than energy-losing collisions.
4. L1 on V-JEPA 2, including probe floor and null forecast.
5. Positive control trained and run through L0 to L2.
6. VideoMAE-v2 adapter; L0 to L2.
7. CogVideoX adapter (`encode` with noise levels, `surprise`, `forecast`, `generate`); L0 to L2; output-level evaluation.
8. F2 to F6 data; L1 on F2/F3; L3.
9. Wan 2.2 and PhysRVG adapters; full suite.
10. L4 on V-JEPA 2 and CogVideoX.
11. Aggregation, leaderboard and figures.

---

## 10. Open decisions (fill in before running)

- [ ] Target venue and deadline, which sets how many P1/P2 models are in scope.
- [x] DiT `forecast()` mode: video-to-video continuation from pre-contact latents, or I2V from the last pre-contact frame. Default: video-to-video if the model supports it. **Resolved 2026-11-06:** two output tracks, §6 (A, continuation, native multi-frame models only) and §6b (B, self-consistency, I2V from frame 0).
- [ ] DiT noise grid for `encode()`: default SNR {−20, −8, 0, +8} dB.
- [ ] Positive control architecture size and training budget.
- [ ] Whether the benchmark data release includes cached activations or only videos and labels.
