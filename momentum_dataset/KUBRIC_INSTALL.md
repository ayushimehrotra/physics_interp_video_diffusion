# Kubric — native install notes

Rebuilding the renderer from scratch on a new machine. **Only needed to render
scenes.** `probe_set/` is already rendered and validated, and capture/probe read
it off the filesystem — so if you are only running the probing experiments, skip
this entire file and go to `scripts/setup.sh`.

Kubric upstream only ships a Docker path (`kubricdockerhub/kubruntu`). Docker
needs `CAP_SYS_ADMIN`, which the GPU instances used here do not grant, so this is
a native install into a dedicated venv.

Companion file: `kubric_requirements_locked.txt` — the exact working set, frozen
off the machine that rendered `probe_set/`. It pins Kubric itself to commit
`61f2422c84bab75006df33c6989e0b483db3ccfe` ("Blender v4_3 fixes", #355).

## Bootstrap

Python **3.11** specifically — bpy 4.3 wheels are built for 3.11 only.

    # 1. shared libs bpy dlopens (these have vanished from an instance mid-project
    #    before; the symptom is `libXrender.so.1: cannot open shared object file`)
    apt-get update -qq && apt-get install -y \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
        libgl1 libglx0 libegl1 libsm6 libice6 libgomp1 python3.11-venv

    # 2. clone at the pinned commit
    git clone https://github.com/google-research/kubric.git /workspace/kubric
    cd /workspace/kubric
    git checkout 61f2422c84bab75006df33c6989e0b483db3ccfe

    # 3. venv + the frozen set. The -e git+... line in the lock file refers to
    #    kubric itself; drop it and install from the working tree instead.
    python3.11 -m venv .venv
    grep -v '^-e git+' \
        /workspace/physics_interp_video_diffusion/momentum_dataset/kubric_requirements_locked.txt \
        > /tmp/kubric_reqs.txt
    .venv/bin/pip install -r /tmp/kubric_reqs.txt
    .venv/bin/pip install --no-deps -e .        # --no-deps: see below

    # 4. verify
    KUBRIC_USE_GPU=1 .venv/bin/python examples/helloworld.py

Then render with the repo's own driver, not Kubric's examples:

    cd /workspace/physics_interp_video_diffusion
    /workspace/kubric/.venv/bin/python momentum_dataset/run_batch.py --help

`--no-deps` on the editable install is deliberate: Kubric's `requirements.txt`
drags in `apache-beam[gcp]` and `google-cloud-storage`, which only
`kubric/sunds/kubric_builder.py` (the TFDS/Beam dataset builder) imports. The
simulate-and-render path never touches them. Warnings about them missing are
expected and harmless.

## Why these versions

| Package | Version | Reason |
| --- | --- | --- |
| `bpy` | 4.3.0 | Blender as a Python module. The pinned Kubric commit is the "Blender v4_3 fixes" commit, so 4.3 is the matching ABI. Wheels exist for Python 3.11 only. |
| `numpy` | **1.26.4** | bpy 4.3 is compiled against the NumPy 1.x C ABI. With NumPy 2 installed, `import bpy` fails with `_ARRAY_API not found`. Do not upgrade. |
| `tensorflow` | 2.19.1 | Kubric imports `tf` in `file_io`, `asset_source`, `renderer/blender`, `simulator/pybullet` (mostly for `tf.io.gfile`). Needs `protobuf<6`. |
| `tensorflow-datasets` | 4.9.7 | 4.9.10 pulls `tensorflow-metadata` 1.21, whose protobuf gencode is 6.31 — incompatible with TF 2.19's protobuf 5.29. Pinned together with `tensorflow-metadata==1.16.1`. |
| `pybullet` | 3.2.7 | Rigid-body simulator. |

## This venv cannot be merged with the probing venv

| | interpreter | pin |
|---|---|---|
| Kubric / rendering | `/workspace/kubric/.venv/bin/python` | `numpy==1.26.4` for bpy's C ABI |
| Probing / CogVideoX | `physics_interp_video_diffusion/.venv/bin/python` | `numpy==2.2.6`, torch 2.8/cu128 |

Never run the dataset scripts with the repo venv or vice versa. The two stages
hand off over the filesystem: Kubric writes `scene_*/`, the probing side reads it.

## GPU rendering — configure AFTER the renderer is constructed

`KUBRIC_USE_GPU=1` sets `cycles.device = "GPU"`, but Kubric never sets Blender's
`compute_device_type`, which is `NONE` in a fresh bpy session — so Cycles falls
back to CPU.

Setting it beforehand does not help, and neither does saving it into
`userpref.blend`: `Blender.__init__` calls `bpy.ops.wm.read_factory_settings()`,
which resets `compute_device_type` to `NONE`. **Do not trust Kubric's log line**

    INFO:kubric.renderer.blender:Using the following GPU Device(s): ['NVIDIA GeForce RTX 4090', ...]

— it is emitted right after that reset and reads stale per-device `use` flags, so
it prints a GPU name even when the render runs entirely on the CPU. Verify with
`nvidia-smi` during a render, or read `scene_constants.render_device` out of the
scene's `labels.json`.

The working sequence is: construct `Blender(...)` first, then set the device, then
render. Also switch the denoiser to OptiX — the default OpenImageDenoise runs on
the CPU and costs more than the render (6.0 s/frame vs 2.4 s/frame at 720x480):

    renderer = Blender(scene, scratch_dir, ...)
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == "OPTIX")
    renderer.blender_scene.cycles.device = "GPU"
    renderer.blender_scene.cycles.denoiser = "OPTIX"

`generate_scene.py::configure_gpu()` implements this and records the result in
each scene's `labels.json`. It tries `OPTIX` first and falls back to `CUDA`, so a
GPU whose OptiX path is unavailable still renders on the GPU.

**`KUBRIC_USE_GPU=1` must actually be exported when you render.** `configure_gpu()`
returns early unless it is set to one of `1/true/t/yes`, and the run silently
proceeds on CPU with `render_device: "cpu (KUBRIC_USE_GPU not set)"`. The
regeneration command in `../CLAUDE.md` carries it; if you assemble your own, do
not drop it.

**First render on a new machine pays a one-time OptiX JIT.** Blender 4.3 predates
Blackwell, so there are no precompiled sm_120 OptiX kernels and the driver JITs
them from PTX on first use, caching the result in `~/.cache/cycles`. On the RTX
5090 the first scene took 509s and the identical scene then took 127s. Verified
working: `render_device: OPTIX: NVIDIA GeForce RTX 5090`. Benchmark on the second
scene, never the first.

## Render with `--workers 3`, not 5

At 5 workers, 17 of 150 scenes were killed with SIGKILL. System RAM was not the
constraint (251 GB, 231 free), so it was GPU-side contention — made worse by
`scripts/setup.sh` downloading concurrently. `run_batch.py` is resumable:
re-running the same command regenerates only what is missing.

## Upstream bug you will hit in the examples

`examples/bouncing_balls.py:39` passes `width=`/`height=` to
`parser.set_defaults()`, but `kubric/utils.py` replaced those flags with a single
`--resolution`, so the example asserts on startup. One-line fix:

    -parser.set_defaults(frame_end=24, frame_rate=12, width=128, height=128)
    +parser.set_defaults(frame_end=24, frame_rate=12, resolution="128x128")

Nothing in this repo imports that example — it only matters if you use it to
smoke-test the install.

## Known-benign noise

- TF logs `Unable to register cuFFT/cuDNN/cuBLAS factory ...` on import.
- Blender prints `Error: Not freed memory blocks: 1, total unfreed memory
  0.001335 MB` at exit, and `Unable to initialise audio`. Neither affects output;
  exit code is 0.
