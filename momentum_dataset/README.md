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

---

## Environment

### The Docker path you asked for

```bash
cd path/to/physics_interp_video_diffusion/momentum_dataset

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
height but radii differ, so smaller spheres float above the floor — by 0.05 (r=0.60),
0.20 (r=0.45), or 0.35 (r=0.30) world units. The floor is a visual backdrop only; the
spheres never touch it.

With the light tilted 24.6° off vertical, each sphere's shadow is displaced by
`CENTER_HEIGHT x tan(24.6°) = 0.30` world units ≈ **21 px**. A sphere genuinely resting
on the plane would still cast an offset shadow (`r x tan θ`), so the *artifact* is only
the excess: **11 px for r=0.30, 6 px for r=0.45, 2 px for r=0.60**. In practice the
small spheres read as hovering a little above the ground, and the shadow is a dark blob
beside the ball rather than tucked under it.

Three ways out, if that matters for the diffusion model — all one-line changes:

| change | effect |
|---|---|
| light to `position=(2,-2,20)` (~8° tilt) | shadow offset 21 px → 7 px; shading gets flatter |
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

The visible world region at the sphere plane is 10.08 x 6.72 units (camera 14 units
above the plane, 50 mm lens, 36 mm sensor). The short axis is the binding constraint
because direction is sampled over the full 360°.

Consequences, all deliberate:

* Contact is placed at the **frame centre** (`CONTACT_JITTER_RADIUS = 0.0`), which
  maximises how long both spheres stay in frame given that they fly apart in arbitrary
  directions. This does mean collision *location* is constant, mirroring the constant
  collision *time* you asked for. Raise `CONTACT_JITTER_RADIUS` to decorrelate them.
* `speed_A` is sampled in [0.45, 0.90] units/s. The upper end is set by the short axis:
  A must start `speed x 2.5 s` upstream of the centre and still be fully in frame at
  frame 0.
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

## Tunables

All at the top of `generate_scene.py`: `TARGET_CONTACT_FRAME`, `CONTACT_TOLERANCE`,
`RADII`, `MATERIALS` (colour + density), `RESTITUTIONS`, `SPEED_RANGE`,
`IMPACT_PARAM_MAX_FRAC`, `CONTACT_JITTER_RADIUS`, `CAMERA_DISTANCE`,
`SAMPLES_PER_PIXEL`, `PRE_FRAME`/`POST_FRAME`.
