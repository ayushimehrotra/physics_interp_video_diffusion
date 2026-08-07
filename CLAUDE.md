# physics_interp_video_diffusion

Probing whether **CogVideoX-2b internally represents momentum and momentum
conservation**. Videos come from a Kubric/Bullet simulator with exact physics
labels; the model's internal activations are recovered along an inverted latent
trajectory; linear probes are fit from activations to labels.

Method follows `diffusion_interp.pdf` in this directory ("The Invisible Hand of
Physics", Esmati/Nath et al.). Section 3.1 is reverse sampling, Section 4.4 is
the quantitative-variable probing this project extends to momentum.

---

## STATE AS OF HANDOFF (2026-08-07 03:20) — MIGRATING TO A NEW GPU BOX

**Blocked on:** the environment, twice over. `scripts/setup.sh` was abandoned
mid-run on the old instance and that box is being replaced. Nothing about the
code or the dataset is blocking.

| thing | state |
|---|---|
| Code (extraction + probes + dataset) | **done**, 27 unit tests pass |
| `momentum_dataset/probe_set` — 150 scenes | **done**, all 6 validation checks pass |
| `.venv` (torch/diffusers) | **INCOMPLETE** — `import torch` fails; rebuild on the new box |
| CogVideoX-2b weights | **not downloaded** — 12.83 GiB, safetensors only |
| Activation capture | **not started** |
| Probe experiments | **not started** |

### What this directory now carries

Everything needed to resume lives under this directory. Nothing outside it is
required except a GPU, Python 3.10+, and network. Specifically:

* the Kubric renderer install is no longer an external file — see
  `momentum_dataset/KUBRIC_INSTALL.md` + `kubric_requirements_locked.txt`
* no source file references an absolute path; `/workspace` appears nowhere in
  `*.py`, `*.yaml`, or `*.sh`

**Transfer this directory as a directory (rsync/tar/volume snapshot), not via
`git clone`.** `.gitignore` excludes `momentum_dataset/*/scene_*/`, and
`probe_set/` is untracked, so a clone silently arrives with **zero rendered
scenes** — 1.6 GB and several GPU-hours of rendering, gone. If you must go
through git, see "Moving to another machine" below.

Safe to leave behind: `.venv/` (10 GB when complete, machine-specific),
`momentum_dataset/pilot/` (538 MB, stale — see the warning at the bottom).
Dropping both takes the transfer from ~2.1 GB to ~1.6 GB.

### On the old box, setup.sh was slow for a specific reason

Not the instance's bandwidth. Measured there: **20.1 MB/s aggregate** across 8
parallel streams, 4.5 MB/s single-stream to `download.pytorch.org` — but pip
fetches serially, and the host serving the `nvidia-*` CUDA wheels gave ~0.5–1.0
MB/s per connection (`nvidia_cudnn_cu12`, 706 MB, took 10m09s). So the CUDA wheel
download is single-connection-bound and slow wherever you run it; the 12.83 GiB
of weights afterwards is *not*, because `hf_transfer` parallelises and lands near
the aggregate rate. Budget roughly **1h for wheels, ~15 min for weights.**

If the wheels are the bottleneck again, the escape hatch is a venv with
`--system-site-packages` reusing a working system torch, installing only
diffusers/transformers/accelerate — that skips ~10 GB of CUDA wheels. torch ≥2.1
suffices (SDPA is the only hard requirement).

### Pick up here

```bash
cd /workspace/physics_interp_video_diffusion

# 0. confirm the dataset actually made the trip — expect 150
ls -d momentum_dataset/probe_set/scene_*/ | wc -l
ls momentum_dataset/probe_set/scene_*/_COMPLETE | wc -l

# 1. build the environment (resumable; re-run is safe)
./scripts/setup.sh

# 2. sanity check before spending 2 hours
.venv/bin/python tests/test_hooks.py                      # expect "27 passed"
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml \
    --mode capture --limit 1                              # one scene end to end

# 3. capture all 150 scenes (~2h on the 4090; resumable, skips finished scenes)
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode capture

# 4. the two experiments the user asked for (no GPU, seconds each)
#    (a) can a probe tell physically plausible from implausible?
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_conserved --tag plausibility

#    (b) are the physical variables decodable?
.venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe \
    --targets momentum_pre,momentum_post,momentum_magnitude_pre,kinetic_energy_pre,ke_ratio,velocity_A_pre,velocity_B_pre,impulse,restitution,mass_A,radius_A,speed_A_pre \
    --tag variables
```

Results land in `outputs/momentum_probe/probes_<tag>.json`, with a summary table
printed to stdout.

### How to read the results

`momentum_conserved` is binary and is scored by **accuracy** (chance = 0.50),
not R². Everything else is scored by R².

Three baselines print beside every number, and a result means nothing unless it
clears all three:

* `shuf` — same pipeline on permuted labels. With 1920 dims and 150 scenes a
  ridge probe can look good on noise.
* `VAE` — the raw autoencoder latent. The paper's central claim is that physics
  is *absent* here (48–53% on plausibility) and constructed inside the
  transformer. If the VAE latent scores as well as a block, nothing was found.
* `mass_A` / `radius_A` — readable off a single frame. Momentum has to beat
  these or the probe found appearance.

Expected shape if the paper replicates: best blocks in the **middle third**
(15–25 for CogVideoX-2b), signal weak at block 0–1 and at noise level 0.

---

## Layout

```
physics_interp/
  models/base.py       ModelAdapter ABC + the schedule algebra (alpha_sigma,
                       split_prediction) that inversion rests on
  models/cogvideox.py  CogVideoX adapter: site map, encode/embed/denoise
  interp/hooks.py      hook substrate — step/CFG-branch aware taps. No opinion
                       about what to record.
  interp/__init__.py   linear probes (ridge, closed form, numpy — no sklearn),
                       label extraction, the DERIVED target table
  runner.py            generate() for prompts; invert()/capture_scene() for
                       existing videos; ActivationRecorder pooling
  video.py             frame/mask IO + pixel->latent-cell mapping
momentum_dataset/
  generate_scene.py    one scene from one seed
  run_batch.py         N scenes, parallel, resumable
  validate.py          correctness + coverage + confound report
  probe_set/           150 scenes, THE dataset to use
  pilot/               50 OLD scenes — see warning below
  KUBRIC_INSTALL.md    rebuilding the renderer on a fresh machine
  kubric_requirements_locked.txt   frozen off the box that rendered probe_set/
configs/momentum_probe.yaml   the probing config
configs/red_ball_slope.yaml   unrelated text-to-video generation example
```

---

## Environment — the traps

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

**`setup.sh` looks hung during the CUDA wheels; it usually is not.** pip fetches
serially, and the host serving the `nvidia-*` wheels caps a single connection
around 0.5–1.0 MB/s — `nvidia_cudnn_cu12` alone (706 MB) took 10m09s. Before
concluding the network is bad, measure *aggregate* throughput rather than pip's
one stream; on the old box that was 20.1 MB/s across 8 parallel streams while
pip crawled. To confirm progress rather than guess, read the partial file:

```bash
p=$(pgrep -f 'pip install'); ls -l /proc/$p/fd | grep pip-unpack
```

The weights are the opposite case: `hf_transfer` parallelises, so 12.83 GiB lands
near aggregate rate (~15 min), not near pip's. Rough budget: **1h wheels, 15 min
weights.** If the wheels are intolerable, use a venv with
`--system-site-packages` over a working system torch and install only
diffusers/transformers/accelerate — skips ~10 GB of CUDA wheels. Any torch ≥2.1
works (SDPA is the only hard floor).

Disk to provision: ~10 GB venv, ~13 GB HF cache, ~1.6 GB dataset. Put the venv
and the HF cache on separate mounts if you can, so they do not compete.

## Moving to another machine

The transfer unit is **this directory**, not the git repo. `.gitignore` excludes
`momentum_dataset/*/scene_*/` and `probe_set/` is untracked, so a `git clone`
arrives with the code but **no rendered scenes**.

```bash
# preferred: copy the directory, skipping what is rebuildable
rsync -a --info=progress2 \
    --exclude '.venv/' --exclude 'momentum_dataset/pilot/' \
    --exclude '__pycache__/' --exclude 'outputs/' \
    /workspace/physics_interp_video_diffusion/ NEW_HOST:/workspace/physics_interp_video_diffusion/
```

~1.6 GB, dominated by `probe_set/` (150 scenes x ~11 MB: 98 PNGs, `labels.json`,
`video.mp4`). If you would rather go through git, the dataset needs its own path
— either force-add it (`git add -f momentum_dataset/probe_set`, ~1.6 GB of PNGs
in history, and GitHub starts warning past 1 GB), put it on Git LFS, or park a
tarball in object storage and fetch it on the new box. Re-rendering from seeds is
the last resort: `run_batch.py` is deterministic and resumable, but it costs
several GPU-hours and needs the whole Kubric install standing up first.

On arrival, verify before trusting it:

```bash
ls -d momentum_dataset/probe_set/scene_*/ | wc -l          # 150
ls momentum_dataset/probe_set/scene_*/_COMPLETE | wc -l     # 150
```

## Regenerating probe_set instead of copying it

Viable, because scene `i` is a pure function of seed `seed_offset + i`
(`rng = np.random.default_rng([seed, attempt])`) and `generate_scene.py` is
committed. The **labels reproduce exactly**; rendered pixels may differ by a hair
across GPU/denoiser versions, which does not matter for probing. Requires the
whole Kubric install standing up first (`KUBRIC_INSTALL.md`) and costs several
GPU-hours.

**Use exactly this command. Two flags are not defaults and both matter:**

```bash
/workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py \
    --num-scenes 150 --output-dir momentum_dataset/probe_set \
    --seed-offset 0 --violate-frac 0.5 --workers 3
```

* `--violate-frac 0.5` — **defaults to 0.0.** Omit it and every scene conserves
  momentum, so `momentum_conserved` has one class and the entire plausibility
  experiment is dead. The recorded set is 75 conserved / 75 violated.
* `--workers 3` — defaults to 4. At 5, seventeen scenes were SIGKILLed.
* `--seed-offset 0` — the recorded set's offset. Changing it changes every scene.

Then validate and diff against what git already carries:

```bash
python momentum_dataset/validate.py --dataset-dir momentum_dataset/probe_set
```

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

## Design decisions worth not re-deriving

### Inversion is DDIM inversion, not the paper's flow-matching Euler

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

### The pre/post collision split is exact, and that is not luck

CogVideoX's causal VAE encodes frame 0 alone, then groups of 4. Contact is
measured at frame 20.03–20.07, i.e. just *after* rendered frame 20, which is the
last frame of latent cell 5. So **latent frames 0–5 are purely pre-collision and
6–12 purely post**, with no straddling cell. `video.py:pixel_frame_to_latent`.
If `TARGET_CONTACT_FRAME` is ever changed, re-check this.

### Pooling is by segmentation mask, not frame-wide mean

One site at one step is 67 MB unpooled. Pooling happens inside the hook, over
regions {A, B, all} × phases {pre, post, all} = 9 pools, ~3 MB/scene. A frame-wide
mean would drown a small sphere; a two-body momentum question needs the two
bodies addressable separately.

### The momentum/mass confound is the whole reason the dataset was rebuilt

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

### Conservation needs both classes

Bullet conserves momentum by construction, so every honest scene has the same
answer and "is momentum conserved" is unprobeable. Violating scenes are
*authored*: the collision is simulated and validated honestly, then post-contact
velocities are overwritten and the spheres re-keyframed, starting at the first
frame after contact. The injected momentum is split by mass so both balls take
the same velocity change — equal-and-opposite would conserve momentum and defeat
the point, and this way neither trajectory looks individually absurd, so the
class difference lives in the *relationship* between them.

`--violate-frac 0.5` interleaves the classes deterministically by index.

### `momentum_dataset/pilot/` is stale — do not probe it

50 scenes from the old sampler: 3 radius levels, B always at rest, contact always
at the frame centre, no violations, `corr(|p|, mass_A) = 0.97`. Valid physics,
fine for exercising the pipeline, **not identifiable for probing**.

---

## Physics facts baked into the labels

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

## Conventions

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
