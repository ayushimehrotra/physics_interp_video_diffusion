# Friction

Split out of `CLAUDE.md` on 2026-10-27, verbatim apart from paths updated to the new layout. Section numbers are unchanged; the index in
`CLAUDE.md` says which file each § lives in. The friction line is closed (see
`old_experiments/MANIFEST.txt`); this is its full record.

---

### 24. The sliding-friction dataset — a second physical domain

`datasets/generate_friction_scene.py`, batched by the same
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

**Validation** is `datasets/validate_friction.py`, six checks, exit code
gates a render. The sixth is the one that matters and needs the pixels: in the
latent arm the tile seam and the patches must be **invisible**, measured as the
rendered luma step across the boundary in floor-only pixels (segmentation-masked,
frame 0, both lanes). Pilot: mean |step| **0.13 of 255**, mu-driven swing across
the whole mu range **0.59 luma**. Two thresholds in that script scale with `n`
rather than being fixed, because a 6-scene pilot's correlations are noise at
+-0.6 and a fixed bar fails a clean set.

```bash
# 300 scenes, ~140 s/scene at steady state, 5 workers  (pilot: 6/6, all checks pass)
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 300 --output-dir datasets/friction_set_v1 \
    --script generate_friction_scene.py --arm latent_friction --workers 5

/workspace/kubric/.venv/bin/python datasets/validate_friction.py \
    --dataset-dir datasets/friction_set_v1 --out friction_set_v1/_validation.json

# capture needs NO changes: labels carry a `collision` block whose actual_frame is
# the friction-entry frame, which is what the phase split reads.
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir datasets/friction_set_v1 -o outputs/friction_$M
done

# the first probe. mu from PRE pools is the interesting cell, not the post ones.
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
    -o outputs/friction_videomae-base --tag friction \
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

`datasets/generate_slide_scene.py`. **300 scenes, rendered, all six checks
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
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 300 --output-dir datasets/slide_set_v1 \
    --script generate_slide_scene.py --arm latent_friction --workers 5   # 133 min

/workspace/kubric/.venv/bin/python datasets/validate_slide.py \
    --dataset-dir datasets/slide_set_v1 --out datasets/slide_set_v1/_validation.json

# capture needs no changes: labels carry a `collision` block whose actual_frame is
# a nominal early/late split at frame 20 (there is no event — the floor is uniform).
for M in videomae-base vjepa2-vitl; do
  .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir datasets/slide_set_v1 -o outputs/slide_$M
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

`slide_set_v2`, both encoders, 2026-08-16. Captures in `outputs/slide_*`; probes
in `probes_friction.json`; steering in `outputs/steer_*/steer_friction_*.json`.
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
convention, `scripts/common/mask_only_floor.py`:

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

`old_experiments/scripts/steer_friction.py`. Their App C.12 protocol reproduced exactly — INLP
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

`old_experiments/scripts/patch_surface.py`, run on the existing `slide_set_v2` captures. §30 read mu
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
KUBRIC_USE_GPU=1 /workspace/kubric/.venv/bin/python datasets/run_batch.py \
    --num-scenes 150 --output-dir datasets/incline_set_v2 \
    --script generate_incline_scene.py --arm latent_friction
python3 datasets/validate_incline.py --dataset-dir datasets/incline_set_v2
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
  .venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml \
      --mode capture --model $M \
      --dataset-dir datasets/incline_set_v2 -o outputs/incline_$M
done

# probe: seconds, no GPU
.venv/bin/python scripts/pipeline/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --model $M -o outputs/incline_$M --dataset-dir datasets/incline_set_v2 \
    --targets mu,log_mu,theta,accel_measured,cube_size \
    --pools A.pre,A.all,B.all,all.pre,all.post,all.all --tag incline
```

Results: `outputs/incline_{videomae-base,vjepa2-vitl}/momentum_probe/probes_incline.json`
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

**Composition test.** `old_experiments/scripts/compose_incline.py <model>`. For each (pool, block) it
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
checks pass (`datasets/incline_set_v2/_validation.json`).

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

`old_experiments/scripts/transfer_geometry.py`. No GPU, no render, ~2 min on existing captures. This
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
