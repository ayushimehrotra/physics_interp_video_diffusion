"""The runner: one loaded model, many runs.

``VideoDiffusionRunner`` owns a loaded :class:`ModelAdapter` and a persistent
:class:`HookManager`. It has two modes, and they answer different questions:

**Generation** (:meth:`generate`, :meth:`run`) — sample a video from a prompt.
This is the mode the config grid and :class:`RunSpec` describe.

**Inversion** (:meth:`invert`, :meth:`capture_scene`) — take a video that
already exists and recover the latent trajectory the model would have traversed
to produce it, recording internal activations along the way. This is the mode
probing needs, because a probe has to be trained against *known* ground truth,
and the only videos with known ground truth are ones a simulator produced. A
generated video has no labels.

Interactive use::

    runner = VideoDiffusionRunner.from_config(load_config("configs/momentum_probe.yaml"))
    print(runner.sites[:5])                     # discover tap points
    rec = runner.capture_scene("momentum_dataset/pilot/scene_00000")
    rec["activations"]["block.20.resid_post"].shape   # [steps, pools, dim]

Hooked use — the hook layer is the only extension point the runner knows about,
and it deliberately knows nothing about what a hook does::

    log = []
    runner.hooks.add("block.20.resid_post",
                     lambda x, ctx: log.append((ctx.step, x.float().std().item())))
    runner.generate(...)
"""

from __future__ import annotations

import importlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import ExperimentConfig, GenerationConfig, HookConfig, InversionConfig, ModelConfig
from .experiment import DONE_MARKER, RunSpec
from .interp.hooks import HookManager
from .models import ModelAdapter, load_model
from .video import (
    load_frames,
    load_segmentation,
    masks_to_latent_grid,
    pixel_frame_to_latent,
    save_frames,
    save_video,
    write_json,
)

__all__ = [
    "GenerationResult",
    "InversionResult",
    "ActivationRecorder",
    "VideoDiffusionRunner",
]


@dataclass
class GenerationResult:
    """One generated video plus everything needed to reproduce or audit it."""

    frames: list[Any]
    prompt: str
    seed: int
    generation: GenerationConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    latents: torch.Tensor | None = None

    def save(self, directory: str | Path) -> Path:
        """Write video/frames/latents/metadata, then the completion marker.

        The marker is written last, so an interrupted run leaves a directory
        without one and is re-attempted on the next launch rather than being
        silently treated as finished.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        g = self.generation
        if g.save_video:
            save_video(self.frames, directory / "video.mp4", fps=g.fps)
        if g.save_frames:
            save_frames(self.frames, directory / "frames")
        if g.save_latents and self.latents is not None:
            torch.save(self.latents.cpu(), directory / "latents.pt")
        write_json(self.metadata, directory / "metadata.json")
        (directory / DONE_MARKER).write_text(self.metadata.get("run_id", "") + "\n")
        return directory


@dataclass
class InversionResult:
    """A recovered latent trajectory plus whatever was recorded along it."""

    noise: torch.Tensor
    """The endpoint ``Z_0``: the noise the model would have started from."""

    clean_latent: torch.Tensor | None = None
    """The starting ``Z_1``, kept for the VAE-latent control probe."""

    timesteps: list[float] = field(default_factory=list)
    """Noise levels visited, ascending (clean -> noise)."""

    captured_steps: list[int] = field(default_factory=list)
    """Inversion step indices at which activations were recorded."""

    activations: dict[str, np.ndarray] = field(default_factory=dict)
    """site -> ``float16 [captured_steps, pools, dim]``."""

    pools: list[str] = field(default_factory=list)
    """Names of the pooling regions, indexing axis 1 of every activation array."""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def noise_levels(self) -> list[float]:
        """Captured steps as a fraction of the trajectory, in ``[0, 1]``.

        This is the paper's y-axis: 0 is clean data, 1 is pure noise.
        """
        n = max(len(self.timesteps) - 1, 1)
        return [s / n for s in self.captured_steps]


class ActivationRecorder:
    """Pools each tapped site down to a handful of vectors and stores them.

    Storing raw activations is not an option: one site at one step is
    ``17550 x 1920`` at fp16, i.e. 67 MB, and the probing grid wants 30 blocks
    across many steps and thousands of scenes. Pooling has to happen inside the
    hook, before anything reaches host memory.

    *What* to pool over is the interesting choice. A plain mean over all video
    tokens throws away exactly the structure the momentum question needs — a
    two-body collision has two bodies, and their momenta are separate
    quantities that sum. So pooling weights are supplied per named region
    (typically one per object, from the simulator's own segmentation masks,
    crossed with pre/post-collision phases), and the reduction is a single
    ``[pools, tokens] @ [tokens, dim]`` matmul on-device.
    """

    def __init__(self, weights: torch.Tensor, names: Sequence[str], branch: str = "cond"):
        if weights.shape[0] != len(names):
            raise ValueError(f"{weights.shape[0]} weight rows for {len(names)} names")
        self.weights = weights
        self.names = list(names)
        self.branch = branch
        self.store: dict[str, dict[int, np.ndarray]] = {}

    def __call__(self, x: torch.Tensor, ctx) -> None:
        rows = ctx.rows(self.branch) or [0]
        vid = x[rows[0], ctx.layout.video_slice(), :]           # [tokens, dim]

        w = self.weights.to(device=vid.device, dtype=torch.float32)
        pooled = w @ vid.float()                                # [pools, dim]

        self.store.setdefault(ctx.site.name, {})[ctx.step] = (
            pooled.to(torch.float16).cpu().numpy()
        )
        return None  # observe only; never perturb the trajectory being measured

    def collect(self, steps: Sequence[int]) -> dict[str, np.ndarray]:
        """Stack the per-step records into ``site -> [steps, pools, dim]``."""
        out = {}
        for site, by_step in self.store.items():
            missing = [s for s in steps if s not in by_step]
            if missing:
                raise RuntimeError(
                    f"site {site!r} was not recorded at step(s) {missing}. "
                    "The capture step filter and the inversion loop disagree."
                )
            out[site] = np.stack([by_step[s] for s in steps])
        return out


def build_pool_weights(
    layout,
    regions: dict[str, np.ndarray],
    phases: dict[str, tuple[int, int]],
) -> tuple[torch.Tensor, list[str]]:
    """Cross ``regions`` with ``phases`` into normalised token-pooling weights.

    ``regions`` maps a name to a ``[frames, height, width]`` occupancy array on
    the latent grid; ``phases`` maps a name to a half-open latent-frame range.
    Returns ``([n_pools, num_video_tokens] float32, names)`` with each row
    summing to 1, so a pooled vector is a weighted mean and is comparable
    across scenes where an object covers different numbers of cells.

    An empty region/phase intersection (an object that has left the frame, for
    instance) yields an all-zero row rather than a division by zero: the pooled
    vector is then exactly zero, which downstream code can detect and drop
    instead of silently probing against NaNs.
    """
    rows, names = [], []
    for r_name, occupancy in regions.items():
        occ = np.asarray(occupancy, dtype=np.float32)
        if occ.shape != (layout.frames, layout.height, layout.width):
            raise ValueError(
                f"region {r_name!r} has shape {occ.shape}, expected "
                f"{(layout.frames, layout.height, layout.width)}"
            )
        for p_name, (start, end) in phases.items():
            masked = np.zeros_like(occ)
            masked[start:end] = occ[start:end]
            total = masked.sum()
            rows.append(masked.reshape(-1) / total if total > 0 else masked.reshape(-1))
            names.append(f"{r_name}.{p_name}")
    return torch.from_numpy(np.stack(rows)), names


class VideoDiffusionRunner:
    """A loaded video diffusion model, ready to sample from and to hook."""

    def __init__(
        self,
        adapter: ModelAdapter,
        generation: GenerationConfig | None = None,
        *,
        experiment: ExperimentConfig | None = None,
    ):
        self.adapter = adapter
        self.generation = generation or GenerationConfig()
        self.experiment = experiment

        g = self.generation
        self.layout = adapter.latent_layout(g.num_frames, g.height, g.width)
        self.hooks = HookManager(
            adapter,
            self.layout,
            guidance_scale=g.guidance_scale,
            num_steps=g.num_inference_steps,
        )
        self.hooks.attach()

    # ---- construction ----------------------------------------------------- #

    @classmethod
    def from_config(
        cls, cfg: ExperimentConfig, device: str | torch.device = "cuda"
    ) -> "VideoDiffusionRunner":
        adapter = load_model(cfg.model, device)
        runner = cls(adapter, cfg.generation, experiment=cfg)
        runner.install_hooks(cfg.hooks)
        return runner

    @classmethod
    def from_model(
        cls,
        model_id: str = "cogvideox-2b",
        device: str | torch.device = "cuda",
        **generation_kwargs,
    ) -> "VideoDiffusionRunner":
        """Convenience constructor for notebooks: no YAML required."""
        adapter = load_model(ModelConfig(id=model_id), device)
        return cls(adapter, GenerationConfig(**generation_kwargs))

    # ---- discovery -------------------------------------------------------- #

    @property
    def sites(self) -> list[str]:
        """Every canonical tap point this model exposes."""
        return self.adapter.site_names(self.layout)

    def describe(self) -> dict[str, Any]:
        return {
            **self.adapter.describe(),
            "layout": self.layout.__dict__,
            "num_sites": len(self.sites),
        }

    # ---- hooks ------------------------------------------------------------ #

    def install_hooks(self, hook_cfgs: Sequence[HookConfig]) -> list[Any]:
        """Resolve declarative ``hooks:`` entries from a config into live hooks.

        ``factory`` is ``"package.module:callable"``; the callable is invoked
        with ``**kwargs`` and must return ``fn(x, ctx) -> Tensor | None``.
        Factories are imported lazily so a config referencing a collaborator's
        analysis module does not make this package depend on it.
        """
        handles = []
        for hc in hook_cfgs:
            module_name, _, attr = hc.factory.partition(":")
            if not attr:
                module_name, _, attr = hc.factory.rpartition(".")
            try:
                factory = getattr(importlib.import_module(module_name), attr)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"could not resolve hook factory {hc.factory!r}: {e}"
                ) from e
            handles.append(
                self.hooks.add(
                    hc.sites,
                    factory(**hc.kwargs),
                    steps=hc.steps,
                    name=hc.id or hc.factory,
                )
            )
        return handles

    # ---- generation ------------------------------------------------------- #

    def _generator(self, seed: int) -> torch.Generator:
        """Seeded RNG on CPU, deliberately.

        diffusers draws the initial latents on the generator's device. A CPU
        generator therefore gives byte-identical starting noise on any GPU model
        — which is what makes a sweep run on someone else's cards comparable to
        one run here.
        """
        return torch.Generator(device="cpu").manual_seed(int(seed))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int = 0,
        *,
        generation: GenerationConfig | None = None,
        extra_metadata: dict[str, Any] | None = None,
        **pipe_kwargs,
    ) -> GenerationResult:
        """Sample one video."""
        g = generation or self.generation
        if generation is not None:
            self._retune(g)

        keep_latents = g.save_latents
        stash: dict[str, Any] = {}
        if keep_latents and "callback_on_step_end" not in pipe_kwargs:
            def _cb(pipe, i, t, kwargs):
                stash["latents"] = kwargs["latents"]
                return kwargs

            pipe_kwargs["callback_on_step_end"] = _cb
            pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        self.hooks.reset()
        start = time.perf_counter()
        out = self.adapter.generate(
            prompt=prompt, generation=g, generator=self._generator(seed), **pipe_kwargs
        )
        elapsed = time.perf_counter() - start

        frames = out.frames[0]
        self._check_step_count(g)

        metadata = {
            "prompt": prompt,
            "seed": int(seed),
            "generation": g.__dict__,
            "model": self.adapter.describe(),
            "layout": self.layout.__dict__,
            "timesteps": self.hooks.timesteps_seen,
            "denoiser_calls": self.hooks.call_idx,
            "num_frames_returned": len(frames),
            "elapsed_seconds": round(elapsed, 2),
            "env": _env_info(),
            "hooks": [
                {"name": h.name, "sites": sorted(h.sites), "calls": h.calls}
                for h in self.hooks._hooks
            ],
        }
        metadata.update(extra_metadata or {})

        return GenerationResult(
            frames=frames,
            prompt=prompt,
            seed=int(seed),
            generation=g,
            metadata=metadata,
            latents=stash.get("latents"),
        )

    def _retune(self, g: GenerationConfig) -> None:
        """Re-derive the token layout when generation geometry changes.

        Resolution and frame count change the latent grid, which every hook's
        spatial reasoning depends on — so this must not be silently skipped.
        """
        layout = self.adapter.latent_layout(g.num_frames, g.height, g.width)
        if layout != self.layout:
            # The adapter caches site maps per layout, so nothing to invalidate.
            self.layout = layout
            self.hooks.layout = layout
            self.hooks._checked_layout.clear()
            self.hooks._refresh()
        self.hooks.guidance_scale = g.guidance_scale
        self.hooks.num_steps = g.num_inference_steps

    def _check_step_count(self, g: GenerationConfig) -> None:
        """Verify the hook step counter tracked the sampler.

        If a pipeline ever runs the denoiser a different number of times per
        step than the adapter declares, every step-indexed measurement would be
        quietly misaligned. Cheap to check, expensive to discover later.
        """
        expected_calls = g.num_inference_steps * self.adapter.calls_per_step
        if self.hooks.call_idx != expected_calls:
            print(
                f"[runner] warning: denoiser was called {self.hooks.call_idx} times "
                f"but {expected_calls} were expected for "
                f"{g.num_inference_steps} steps x calls_per_step="
                f"{self.adapter.calls_per_step}. Step indices on captured "
                f"activations may be misaligned; check "
                f"{type(self.adapter).__name__}.calls_per_step.",
                flush=True,
            )

    # ---- inversion -------------------------------------------------------- #

    @torch.no_grad()
    def invert(
        self,
        video: torch.Tensor | np.ndarray,
        *,
        inversion: InversionConfig | None = None,
        recorder: ActivationRecorder | None = None,
        sites: str | Sequence[str] = "block.*.resid_post",
    ) -> InversionResult:
        """Recover the latent trajectory of an existing video, recording as it goes.

        The model gives no way to ask "what internal state would you associate
        with *this* clip?" — it only ever produces trajectories starting from
        noise it drew itself. The trick is to run the sampler backwards: encode
        the clip to a clean latent and integrate the learned field from data
        toward noise, which visits an approximation of the states the forward
        pass would have gone through.

        Each step evaluates the denoiser at the noise level the latent is
        *currently* at and re-projects onto the next level::

            (x0, eps) <- split(model(z, t_cur), z, alpha_cur, sigma_cur)
            z         <- alpha_next * x0 + sigma_next * eps

        The exact inverse would need ``z`` on both sides and an iterative solve
        per step. Evaluating at the known endpoint instead makes it explicit:
        one network call per step, same cost as generation.

        Two settings matter more than they look:

        * ``num_steps``. Reconstruction quality and probe signal come apart
          sharply — a coarse inversion still reconstructs the clip but visits
          different intermediate states, and decodability collapses. Do not
          economise here; the reconstruction looking fine is not evidence the
          trajectory is right.
        * ``guidance_scale = 1.0`` (the default). Classifier-free guidance is
          not a gradient of anything, so a CFG trajectory is not invertible in
          the sense used here. Inversion runs unguided, which also means the
          batch is a single row and ``ctx.rows("cond")`` is trivially [0].
        """
        cfg = inversion or InversionConfig()
        device = self.adapter.device

        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        if video.dtype == torch.uint8:
            video = video.float() / 255.0

        scheduler = self.adapter.scheduler
        scheduler.set_timesteps(cfg.num_steps, device=device)
        # Sampling walks noise -> data; inversion walks data -> noise.
        ladder = list(reversed(scheduler.timesteps.tolist()))

        capture_at = self._capture_steps(cfg, len(ladder))
        handle = None
        if recorder is not None:
            handle = self.hooks.add(
                sites, recorder, steps=capture_at, name="activation_recorder"
            )

        prev_guidance = self.hooks.guidance_scale
        try:
            self.hooks.reset()
            # Inversion runs one unguided row, so the batch axis is not the
            # [uncond, cond] pair a generation config implies. Without this the
            # branch labels would only come out right via branch_labels'
            # "unexpected layout" fallback — correct by accident, and silently
            # wrong the moment anything batches inversions together.
            self.hooks.guidance_scale = cfg.guidance_scale
            embeds = self.adapter.embed_prompt(cfg.prompt)
            # The latent state is carried in fp32 even though the model runs in
            # fp16. Over 100 sequential steps the rounding error of an fp16
            # accumulator is not negligible, and it lands on exactly the
            # intermediate states being measured.
            z = self.adapter.encode_video(video).float()
            clean_latent = z.clone()

            start = time.perf_counter()
            alpha_cur, sigma_cur = self.adapter.alpha_sigma(ladder[0])
            t_cur = ladder[0]

            for step, t_next in enumerate(ladder):
                alpha_next, sigma_next = self.adapter.alpha_sigma(t_next)
                ts = torch.tensor(t_cur, device=device)

                self.hooks.step_override = step
                out = self.adapter.denoise(z, ts, embeds)
                x0, eps = self.adapter.split_prediction(
                    out.float(), z, alpha_cur, sigma_cur
                )
                z_next = alpha_next * x0 + sigma_next * eps

                if cfg.solver == "heun" and step + 1 < len(ladder):
                    # Second-order correction: re-evaluate at the point Euler
                    # landed on and average the two estimates. A more accurate
                    # integrator tracks the model's continuous dynamics more
                    # closely, which is what the recorded states are supposed
                    # to represent. Marked as a non-capture call so the
                    # recorder does not overwrite this step's activations.
                    self.hooks.step_override = -1
                    out2 = self.adapter.denoise(
                        z_next, torch.tensor(t_next, device=device), embeds
                    )
                    x0_2, eps_2 = self.adapter.split_prediction(
                        out2.float(), z_next, alpha_next, sigma_next
                    )
                    z_next = alpha_next * 0.5 * (x0 + x0_2) + sigma_next * 0.5 * (eps + eps_2)

                z = z_next
                t_cur, alpha_cur, sigma_cur = t_next, alpha_next, sigma_next

            elapsed = time.perf_counter() - start
        finally:
            self.hooks.step_override = None
            self.hooks.guidance_scale = prev_guidance
            if handle is not None:
                self.hooks.remove(handle)

        activations, pools = {}, []
        if recorder is not None:
            activations = recorder.collect(capture_at)
            pools = recorder.names

        return InversionResult(
            noise=z,
            clean_latent=clean_latent,
            timesteps=[float(t) for t in ladder],
            captured_steps=capture_at,
            activations=activations,
            pools=pools,
            metadata={
                "num_steps": cfg.num_steps,
                "solver": cfg.solver,
                "prompt": cfg.prompt,
                "prediction_type": getattr(
                    scheduler.config, "prediction_type", "epsilon"
                ),
                "scheduler": type(scheduler).__name__,
                "elapsed_seconds": round(elapsed, 2),
                "denoiser_calls": self.hooks.call_idx,
                "layout": self.layout.__dict__,
            },
        )

    def _capture_steps(self, cfg: InversionConfig, num_steps: int) -> list[int]:
        """Which inversion steps to record at.

        Capture is decoupled from integration on purpose. Trajectory fidelity
        wants many steps; storage wants few recordings. Running 100 steps and
        recording at 9 of them costs the same compute as recording at all 100
        and a tenth of the disk.
        """
        if cfg.capture_steps is not None:
            bad = [s for s in cfg.capture_steps if not 0 <= s < num_steps]
            if bad:
                raise ValueError(f"capture step(s) {bad} outside [0, {num_steps})")
            return sorted(cfg.capture_steps)
        n = max(1, int(cfg.num_captures))
        if n == 1:
            return [num_steps // 2]
        return sorted({round(i * (num_steps - 1) / (n - 1)) for i in range(n)})

    @torch.no_grad()
    def resample(self, noise: torch.Tensor, *, inversion: InversionConfig | None = None):
        """Integrate forward from recovered noise back to a video.

        The reconstruction check: if this does not return the clip that was
        inverted, the trajectory is not one the model would have taken and
        nothing recorded along it means what it is supposed to mean. Uses the
        same explicit scheme as :meth:`invert`, run the other way.
        """
        cfg = inversion or InversionConfig()
        device = self.adapter.device
        scheduler = self.adapter.scheduler
        scheduler.set_timesteps(cfg.num_steps, device=device)
        ladder = scheduler.timesteps.tolist()          # noise -> data

        embeds = self.adapter.embed_prompt(cfg.prompt)
        z = noise.float().to(device)

        for i, t in enumerate(ladder):
            alpha, sigma = self.adapter.alpha_sigma(t)
            out = self.adapter.denoise(z, torch.tensor(t, device=device), embeds)
            x0, eps = self.adapter.split_prediction(out.float(), z, alpha, sigma)
            nxt = ladder[i + 1] if i + 1 < len(ladder) else None
            a_n, s_n = self.adapter.alpha_sigma(nxt) if nxt is not None else (1.0, 0.0)
            z = a_n * x0 + s_n * eps

        return self.adapter.decode_latents(z)

    # ---- dataset capture --------------------------------------------------- #

    def capture_scene(
        self,
        scene_dir: str | Path,
        *,
        inversion: InversionConfig | None = None,
        sites: str | Sequence[str] = "block.*.resid_post",
    ) -> dict[str, Any]:
        """Invert one simulator scene and return pooled activations + its labels.

        The pooling regions come from the scene's own segmentation masks, so a
        probe can be trained on "the tokens where sphere A actually is" rather
        than on a frame-wide average in which a small sphere is a rounding
        error. Phases are split at the collision frame recorded in
        ``labels.json``, which is what makes "momentum before" and "momentum
        after" separately addressable — and therefore makes *conservation*, as
        opposed to momentum, something a probe can be asked about.
        """
        cfg = inversion or InversionConfig()
        scene_dir = Path(scene_dir)
        labels = json.loads((scene_dir / "labels.json").read_text())

        frames = load_frames(scene_dir, num_frames=self.generation.num_frames)
        regions: dict[str, np.ndarray] = {}
        if cfg.use_segmentation:
            seg = load_segmentation(scene_dir)
            occ = masks_to_latent_grid(seg, self.layout, ids=(1, 2))
            regions = {"A": occ[0], "B": occ[1]}
        regions["all"] = np.ones(
            (self.layout.frames, self.layout.height, self.layout.width), np.float32
        )

        contact = int(labels["collision"]["actual_frame_rounded"])
        contact_latent = pixel_frame_to_latent(contact)
        phases = {
            "pre": (0, contact_latent + 1),
            "post": (contact_latent + 1, self.layout.frames),
            "all": (0, self.layout.frames),
        }

        weights, names = build_pool_weights(self.layout, regions, phases)
        recorder = ActivationRecorder(weights, names)
        result = self.invert(frames, inversion=cfg, recorder=recorder, sites=sites)

        activations = dict(result.activations)
        activations["vae.latent"] = self._pool_clean_latent(result.clean_latent, weights)

        return {
            "scene_id": labels.get("scene_id", scene_dir.name),
            "activations": activations,
            "pools": result.pools,
            "captured_steps": result.captured_steps,
            "noise_levels": result.noise_levels,
            "phases": {k: list(v) for k, v in phases.items()},
            "contact_latent_frame": contact_latent,
            "labels": labels,
            "metadata": result.metadata,
        }

    def _pool_clean_latent(self, latent: torch.Tensor, weights: torch.Tensor) -> np.ndarray:
        """Pool the raw VAE latent with the same weights, as a control probe.

        This is the comparison that makes a positive probing result mean
        something. The VAE is a reconstruction autoencoder with no physical
        objective, so if momentum were already linearly decodable from its
        latents, then finding it inside the transformer would say nothing about
        the transformer. Probing this should land at chance; a transformer
        block well above it is the actual claim.

        Returned with a leading length-1 step axis so it stacks and indexes
        exactly like a real site's ``[steps, pools, dim]`` record.
        """
        if latent is None:
            raise ValueError("no clean latent on the inversion result")

        _, frames, channels, h, w = latent.shape
        patch = h // self.layout.height
        # Same 2x2 spatial patching the embedder applies, so a latent "token"
        # covers the same pixels as a transformer token and the pooling weights
        # transfer unchanged.
        tokens = (
            latent[0]
            .reshape(frames, channels, self.layout.height, patch, self.layout.width, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(self.layout.num_video_tokens, channels * patch * patch)
        )
        pooled = weights.to(tokens.device, torch.float32) @ tokens.float()
        return pooled.to(torch.float16).cpu().numpy()[None]

    # ---- spec execution --------------------------------------------------- #

    def run(self, spec: RunSpec, root: str | Path | None = None) -> Path:
        """Execute one :class:`RunSpec` and write its output directory."""
        out_dir = spec.output_dir(root)
        result = self.generate(
            spec.prompt.text,
            spec.seed,
            generation=spec.config.generation,
            extra_metadata={
                "run_id": spec.run_id,
                "experiment": spec.experiment,
                "prompt_id": spec.prompt.id,
                "tags": spec.prompt.tags,
                "notes": spec.config.notes,
                "signature": spec.signature,
            },
        )
        return result.save(out_dir)

    # ---- teardown --------------------------------------------------------- #

    def close(self) -> None:
        self.hooks.detach()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _env_info() -> dict[str, Any]:
    """Version/hardware fingerprint, so a surprising result can be traced back."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    try:
        import diffusers

        info["diffusers"] = diffusers.__version__
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
        info["cuda"] = torch.version.cuda
    return info
