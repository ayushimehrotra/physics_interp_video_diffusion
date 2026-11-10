# Two-body collision dataset (Kubric) — momentum probing

Frictionless two-sphere collisions on a plane, shot straight down, 49 frames at 8 fps,
720x480. Every scene puts first contact on **frame 20** so activations can be aligned
across clips, and ships exact per-frame physics labels from the simulator.

```
generate_scene.py    one scene from one seed          (runs inside the container)
run_batch.py         N scenes, parallel, resumable    (runs inside the container)
validate.py          correctness + coverage report    (needs only labels.json; runs anywhere)
contact_sheet.py     visual spot-check grid           (needs only the PNG frames; runs anywhere)
```

## Sampling for probe identifiability

The dataset is built to make a *linear probe result interpretable*, which is a
stronger requirement than being physically correct. Four things follow from it, and
they are the main difference from the original pilot:

**Momentum is sampled before speed.** Momentum is mass x velocity, so drawing speed
directly makes momentum a near-deterministic function of mass whenever mass has the
wider range. The pilot had `corr(|p|, mass_A) = 0.97` — a probe "decoding momentum"
there could have been reading apparent size and colour and nothing else. Here a
target momentum is drawn first and speed is derived as `p / m`.

**A matched-momentum stratum.** Decorrelating a product needs both factors to span
comparable log ranges, and speed is capped by how far a ball can travel while staying
in frame, so sampling order alone is not enough. ~35% of scenes therefore draw `|p_A|`
from a fixed band that does not depend on mass at all. Measured over 400 scenes:

| subset | `corr(log \|p_total\|, log mass_A)` | `corr(log \|p_A\|, log mass_A)` |
|---|---|---|
| pilot (old) | +0.97 | +0.97 |
| all scenes | +0.59 | +0.87 |
| **matched-momentum stratum** | **+0.11** | **+0.29** |
| wide stratum | +0.63 | +0.90 |

Fit and report on the matched-momentum stratum; use the rest for coverage. Every
scene records its `stratum` so the split is a filter, not a re-run.

**Sphere B usually starts moving too** (`B_MOVING_PROB = 0.7`). With B always at rest
the system's total momentum is exactly A's momentum, so "the momentum of the system"
and "the momentum of the one moving ball" are the same variable and a probe cannot
distinguish them. Once both move, total momentum is a vector sum equal to neither.

**Half the scenes break momentum conservation** when generated with `--violate-frac
0.5`. Conservation holds identically in every honest scene, which makes "is momentum
conserved" a constant and therefore unprobeable — a classifier needs both classes.
See below for how the violation is produced.

Continuous radii, densities and restitution replace the pilot's 3-level grids, so
derived quantities form a continuum rather than a handful of clusters that a probe
could fit as a lookup table.

---

## Environment

### The Docker path you asked for

```bash
cd path/to/physics_interp_video_diffusion/datasets

# one scene
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$(pwd):/kubric" \
  --env KUBRIC_USE_GPU=1 \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 generate_scene.py --seed 0 --output-dir pilot/scene_00000

# the whole pilot (the batch runner spawns its own subprocesses inside the container)
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$(pwd):/kubric" \
  --env KUBRIC_USE_GPU=1 \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 run_batch.py --num-scenes 50 --output-dir pilot --workers 5
```

**Check the labels before paying for the pixels.** `--no-render` runs physics and
writes `labels.json` only, at roughly a second per scene instead of two minutes, so
the coverage and confound structure of a proposed sampler can be measured on
hundreds of scenes in a few minutes. `validate.py` works on the result:

```bash
python3 run_batch.py --num-scenes 400 --output-dir /tmp/coverage \
    --workers 12 --no-render --violate-frac 0.5
python3 validate.py --dataset-dir /tmp/coverage      # prints the confound table
```

A full two-class set for probing, rendered:

```bash
python3 run_batch.py --num-scenes 1000 --output-dir full \
    --workers 5 --violate-frac 0.5
```

Drop `--gpus all` and `--env KUBRIC_USE_GPU=1` to render on CPU (~2.5x slower here, and
it does not need nvidia-container-toolkit). `--user` keeps the outputs owned by you
instead of root. The image's WORKDIR is `/kubric`, which is why the volume mounts there.

Validation and contact sheets do not need the container — they read JSON and PNGs:

```bash
python3 validate.py --dataset-dir pilot
python3 contact_sheet.py --dataset-dir pilot --num-scenes 5 --seed 0
```

### What actually produced this pilot, and why

**Docker cannot run in this environment**, so the pilot was generated on a native
Kubric install. This is not a preference — the container has no `CAP_SYS_ADMIN` and no
user namespaces, so `runc` cannot create a container and the daemon cannot even
bind-mount a layer:

```
failed to register "bridge" driver: iptables ... Permission denied
failed to extract layer ... operation not permitted: mount ... /var/lib/docker/...
```

`dockerd` does start with `--iptables=false --bridge=none`, and `docker pull` works, but
every `docker run` dies at rootfs setup. Nothing in the scripts depends on that
decision: they use only Kubric's public API and are written to run unmodified in
kubruntu.

The native install used for the pilot is Blender 4.3 via `bpy==4.3.0` with Kubric at
commit `61f2422` (the "Blender v4_3 fixes" commit), in a venv at `<kubric>/.venv` and
documented in that clone's `INSTALL_NOTES.md`. It lives outside this repo because
Kubric needs `numpy==1.26.4` for bpy's C ABI while this repo pins `numpy==2.2.6` — the
two cannot share a venv. To reproduce the pilot exactly as it was generated:

```bash
KUBRIC_USE_GPU=1 <kubric>/.venv/bin/python run_batch.py \
    --num-scenes 50 --output-dir pilot --workers 5
```

**One thing to check before scaling:** kubruntu ships an older Kubric and Blender
(2.93/3.x). The physics is Bullet either way and the labels should be identical, but
the *rendering* may differ slightly (Principled BSDF changed between Blender 3 and 4).
Generate the whole dataset in one environment; do not mix pilot and full-set renders.
Two small compatibility hedges are already in the code: `render(return_layers=...)` and
the mp4 writer both fall back if the older stack lacks them.

---

## Design decisions

### Constraining the spheres to the plane

**No constraint, no hack in the solver — the geometry makes planar motion exact.**
Gravity is set to `(0,0,0)` and **both sphere centres sit at the same height**
(`CENTER_HEIGHT = 0.65`), so the line of centres at contact is always horizontal, the
collision impulse has no vertical component, and nothing ever leaves the plane. Zero
friction means no torque, so the spheres also never spin.

The alternative — resting the spheres on the floor under gravity — is wrong here: with
different radii, the line of centres would tilt, the collision would kick a sphere
upward, and floor contacts would inject vertical impulses that corrupt exactly the
momentum conservation this dataset is built to check.

**The cost, and it is visible in the contact sheet.** Every centre sits at the same
height but radii differ, so spheres smaller than the maximum float above the floor —
by `0.65 - r`, i.e. up to 0.35 world units at the smallest radius. The floor is a
visual backdrop only; the spheres never touch it.

With the light tilted 24.6° off vertical, each sphere's shadow is displaced by
`CENTER_HEIGHT x tan(24.6°) = 0.30` world units ≈ **15 px** at the current camera
distance of 20. A sphere genuinely resting on the plane would still cast an offset
shadow (`r x tan θ`), so the *artifact* is only the excess: about **8 px at r=0.30**
falling to **2 px at r=0.60**. In practice the small spheres read as hovering a little
above the ground, and the shadow is a dark blob beside the ball rather than tucked
under it.

Three ways out, if that matters for the diffusion model — all one-line changes:

| change | effect |
|---|---|
| light to `position=(2,-2,20)` (~8° tilt) | shadow offset 15 px → 5 px; shading gets flatter |
| sample one radius per scene (`radius_b = radius_a`) | removes the artifact entirely; costs the mass-ratio range that comes from radius |
| drop the floor | no shadow at all, no floating cue, but also no ground plane |

This was left as-is for the pilot rather than pre-empted, because it trades against
shading legibility and against parameter coverage — your call.

### The overhead camera and the 2D pixel labels

`pixel` in the labels is the camera projection of the simulator's 3D centre. The camera
looks straight down (`quaternion = (1,0,0,0)`, which avoids the degenerate `look_at`
up-vector when the view direction is exactly -Z), so there is **no depth ambiguity that
breaks the labels**: with all centres coplanar, the world-to-pixel map in that plane is
an exact scale factor, and pixel coordinates are one-to-one with world x/y.

There is one honest caveat, and `validate.py` measures it rather than assuming it away.
Under a *perspective* camera an off-axis sphere projects to an ellipse whose centroid is
displaced slightly outward from the projected centre, and spheres of different radii sit
at slightly different distances from the lens. So the projected centre and the
segmentation-mask centroid do not coincide exactly. The validation report prints the
measured distribution of that gap in pixels (median/p95/max) — it is a couple of pixels,
not a broken label. If you want them to coincide identically, switch to
`kb.OrthographicCamera`, at the cost of looking less like real footage.

### The 720x480 aspect ratio and the scene layout

The visible world region at the sphere plane is 14.4 x 9.6 units (camera 20 units
above the plane, 50 mm lens, 36 mm sensor). The short axis is the binding constraint
because direction is sampled over the full 360°.

The camera was pulled back from 14 to 20 specifically to buy speed range. A ball must
travel `speed x 2.5 s` before contact and still start fully inside the frame, so the
visible half-height is what caps the top speed — and speed range is exactly what
decorrelates momentum from mass. The cost is that spheres are smaller on screen: at
r=0.60 the radius is ~35 px (was ~43), which is a little over 2 latent cells at
CogVideoX's 16-px-per-cell grid.

Consequences, all deliberate:

* Contact is jittered within `CONTACT_JITTER_RADIUS = 1.2` of the frame centre, so
  collision *location* varies even though collision *time* is fixed at frame 20. In
  the pilot this was 0.0 and every collision happened at the same pixel, which meant
  contact position was a constant a probe could not be tested on.
* `speed_A` is sampled in [0.35, 1.55] units/s — a 4.4x range, against 2x in the
  pilot. It is derived from the sampled momentum rather than drawn directly (see
  above). The floor is a legibility constraint, not a physical one: below ~0.35 a
  ball moves under 2 px/frame and the video barely encodes its velocity, so a
  momentum probe would fail there for reasons that say nothing about the model.
* **Nothing is clipped at frame 0** — that is a hard check, enforced before simulating
  and re-verified by `validate.py`. But **after the collision, B can leave the frame**,
  and sometimes does: an elastic hit from a much heavier A sends a light B off at up to
  ~2x A's speed. Rather than distort the sampling to prevent it, every frame carries an
  `in_frame` flag per object and the validation report gives the in-frame fraction. The
  3D labels stay exact when an object is outside the image.

### Physics settings that had to be corrected

Two Bullet defaults silently corrupt this dataset, and both are fixed in
`build_and_simulate`:

1. **Linear/angular damping defaults to 0.04.** Kubric never touches it. A would decelerate
   on the way in — arriving late and bleeding momentum before the collision even happens.
   Set to zero.
2. **`splitImpulsePenetrationThreshold` defaults to -0.04.** Contacts shallower than that
   get penetration recovery as an ERP velocity bias applied to the *real* velocity, which
   adds energy. Measured effect before the fix: realised restitution came out at
   `e + 0.08` for every level, and elastic collisions **gained** up to 5% kinetic energy.
   Setting the threshold to 0 routes recovery through the split-impulse pass, which fixes
   position without touching velocity. After the fix, realised restitution matches target
   to 4 decimals and `e = 1.0` conserves energy exactly.

**Restitution is a pair coefficient.** Bullet multiplies the two objects' restitution
values, so each sphere carries `sqrt(e)` and the pair yields `e`. Don't read
`restitution_object` in the labels as "the restitution of the collision" — that is
`restitution`, and `collision.restitution_measured` is what the simulator actually
delivered.

### Timing: how contact lands on frame 20

A moves in a straight line at constant velocity until it touches B (nothing else is in
the scene and gravity is zero), so the initial separation that puts first contact at
`t = 20/8 s` is exact, not iterative: place B so that `|p_A(t_c) - p_B| = r_A + r_B`
with the contact normal's component along A's velocity positive (the *entering* root),
then start A a distance `speed x t_c` upstream.

The simulator still gets the final say. Every scene is simulated **without rendering**
first; if first contact is not within ±1 frame of the target, or any other check fails,
the parameters are resampled and re-simulated — before a single pixel is rendered. The
accepted parameters are then rebuilt with the renderer attached and re-simulated. All
rejected draws and their reasons are recorded in `labels.json` under `sampling_attempts`.

Observed contact frames land at 20.03–20.07, i.e. consistently ~0.05 frames late — one
or two 240 Hz substeps of contact-detection lag, far inside tolerance.

### GPU rendering (a trap worth knowing about)

`KUBRIC_USE_GPU=1` alone does **not** get you GPU rendering. It sets `cycles.device =
"GPU"`, but `Blender.__init__` calls `bpy.ops.wm.read_factory_settings()`, which resets
`compute_device_type` to `NONE` — so Cycles falls back to CPU. Worse, the
`Using the following GPU Device(s)` line Kubric logs at that moment reads stale
per-device flags and **prints a GPU name even when rendering on the CPU**. Do not trust
that log line; `labels.json` records `scene_constants.render_device`, which is set after
the renderer exists and reflects what was really used.

`configure_gpu()` therefore runs after renderer construction, and also switches the
denoiser from OpenImageDenoise (CPU) to OptiX: 6.0 s/frame → 2.4 s/frame. At that point
sample count barely matters (2.38 s/frame at 64 spp vs 2.30 at 16), because the
remaining cost is Kubric's second "AuxOutputs" view layer and its EXR round-trip, not
sampling — so samples stay at 64.

---

## The sliding-friction scenario (`generate_friction_scene.py`)

A second scenario lives in this directory, sharing this one's camera, resolution,
frame count, renderer settings, output layout and batch runner. Two balls coast
across a **frictionless** approach region and cross onto **per-lane friction
patches** on frame 20, decelerating at `a = mu*g` from there — so mu is a property
of the FLOOR, and the entry frame plays the part the collision frame plays here.

```bash
KUBRIC_USE_GPU=1 python3 run_batch.py --num-scenes 300 \
    --output-dir friction_set_v1 --script generate_friction_scene.py \
    --arm latent_friction --workers 5
python3 validate_friction.py --dataset-dir friction_set_v1
```

Arms mirror this dataset's `latent_mass` / `visible_mass`: `latent_friction`
paints every floor tile one albedo so the patches are invisible, `visible_friction`
ramps a patch's albedo with its mu. Deceleration is mass-independent (`a = mu*g`
for any mass), so radius and mass are free controls rather than confounds.

Four physics facts that took measurement to establish, all documented at length in
`../CLAUDE.md` §24:

* kubric's `useMaximalCoordinates=True` and a frozen inertia tensor together
  produce **NaN velocities reported as a ball coasting forever** — hence
  `install_sliding_sphere_adder()`, which rebuilds spheres in minimal coordinates.
* a sphere with lateral friction **rolls**, decaying to exactly `5/7` of its entry
  speed and then coasting; rotation is frozen so it slides at `mu*g` instead.
* the inertia override must be applied **after** `scene += ball`, because kubric's
  mass observer recomputes the tensor from the shape.
* gravity is 2.0, not 9.81 — this world is not metric, and at 9.81 an ordinary mu
  stops a ball in three frames.

Its `labels.json` carries a `collision` block whose `actual_frame` is the friction
entry, so every downstream tool that splits pre/post on that key works unchanged.

## Output layout

```
pilot/
  manifest.json
  scene_00000/
    frames/frame_0000.png … frame_0048.png     RGB, 720x480
    segmentation/segmentation_0000.png …       uint8, pixel values 0=floor 1=A 2=B
    video.mp4                                  h264, 8 fps, CRF 17
    labels.json
    _COMPLETE
  _validation/
    validation_report.json
    validation_plots.png
    contact_sheet.png
```

Segmentation masks store raw instance ids (0/1/2), so they look almost black in a
viewer — that is intentional, the values are meant to be read, not looked at.

## labels.json

| key | contents |
|---|---|
| `sampled_parameters` | every sampled value + `seed`; enough to reproduce the scene |
| `objects.A/.B` | radius, material, **density**, volume, mass, friction, per-object restitution, segmentation id, colour |
| `mass_ratio`, `restitution` | scene-level derived values |
| `stratum` | `"matched_momentum"` or `"wide"` — which sampling regime produced the scene |
| `b_moving` | whether sphere B started with a velocity |
| `violates_momentum`, `momentum_conserved` | the plausibility class, as a pair of complementary booleans so either polarity is a direct probe target |
| `per_frame[i].A/.B` | `position` (3D), `pixel` (2D), `in_frame`, `velocity`, `angular_velocity`, `momentum`, `speed`, `kinetic_energy`, `segmentation_pixel_count`, `segmentation_centroid_pixel` |
| `per_frame[i]` | `total_momentum`, `total_kinetic_energy` |
| `collision` | target/actual/rounded contact frame + discrepancy, contact point (world **and** pixel), contact normal (solver **and** analytic), impulse vector and magnitude, peak normal force, `restitution_measured`, pre/post velocities+momenta for both objects, pre/post total momentum and KE, momentum abs/rel error, KE lost, KE ratio |
| `scene_constants` | camera, resolution, frame rate, gravity, all level sets, `render_device`, samples |
| `sampling_attempts` | which attempt was accepted and why each earlier draw was rejected |

Pre/post velocities are read at frames 18 and 22, both far enough from contact that a
±1 frame timing error cannot contaminate them; the generator asserts velocities are
piecewise constant on [0,18] and [22,48], which is what proves exactly one collision
happened and nothing else touched the spheres.

**`angular_velocity` is identically zero in every scene.** Frictionless spheres cannot
spin up. It is logged because you asked for it, not because it varies.

In a violating scene `momentum_rel_error` is the *realised* error after the aftermath
was edited, i.e. the size of the violation. `momentum_rel_error_simulated` keeps the
solver's own figure from before the edit, which stays at ~1e-16 and confirms the
underlying collision was simulated correctly.

### Breaking conservation

Bullet conserves momentum by construction, so a violating scene cannot be simulated —
it has to be authored. The collision is simulated honestly and validated, and only
then are the post-contact velocities overwritten and the spheres re-keyframed to fly
along the edited ones, starting at the first frame rendered after contact. Everything
before contact and every visual property is untouched, so the two classes differ only
in what happens *after* the balls touch.

The injected momentum is split between the balls in proportion to mass, giving both
the same velocity change. Equal-and-opposite kicks would conserve momentum and defeat
the point; equal-and-same-direction kicks break the sum while leaving each ball's own
trajectory unremarkable — nothing stops dead or reverses implausibly. The difference
between classes lives in the *relationship* between the two trajectories, which is
what makes it a test of whether the model represents the conserved quantity rather
than a test of whether it can spot one weird-looking ball.

`VIOLATION_SCALE_RANGE = (0.35, 0.75)` sets `|Δp| / |p_pre|`, either sign. Below ~0.2
the violation starts to be within the range of what a viewer would read as an
inelastic collision, which would blur the classes.

## Tunables

All at the top of `generate_scene.py`: `TARGET_CONTACT_FRAME`, `CONTACT_TOLERANCE`,
`RADIUS_RANGE`, `MATERIALS` (colour + density), `RESTITUTION_RANGE`, `SPEED_RANGE`,
`B_MOVING_PROB`, `CONTROL_STRATUM_FRAC`, `CONTROL_MOMENTUM_RANGE`,
`VIOLATION_SCALE_RANGE`, `IMPACT_PARAM_MAX_FRAC`, `CONTACT_JITTER_RADIUS`,
`CAMERA_DISTANCE`, `SAMPLES_PER_PIXEL`, `PRE_FRAME`/`POST_FRAME`.

## The pilot in `pilot/` predates all of this

Those 50 scenes were generated with the old sampler: 3 radius levels, 3 restitution
levels, B always at rest, contact always at the frame centre, no violations, and
`corr(|p|, mass_A) = 0.97`. They are still valid physics and fine for exercising the
capture path end to end, but a probe fit on them is not identifiable. Regenerate
before drawing conclusions.
