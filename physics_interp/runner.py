"""The runner: one loaded model, many runs.

``VideoDiffusionRunner`` owns a loaded :class:`ModelAdapter` and a persistent
:class:`HookManager`, and turns :class:`RunSpec`\\ s into output directories. It
is built to be *held onto*: loading CogVideoX takes far longer than sampling a
short clip, so a sweep loads once and then calls :meth:`run` in a loop.

Interactive use::

    runner = VideoDiffusionRunner.from_config(load_config("configs/red_ball.yaml"))
    print(runner.sites[:5])                     # discover tap points
    result = runner.generate("a red ball rolls down a slope", seed=0)
    result.save("outputs/scratch")

Hooked use — the hook layer is the only extension point the runner knows about,
and it deliberately knows nothing about what a hook does::

    log = []
    runner.hooks.add("block.20.resid_post",
                     lambda x, ctx: log.append((ctx.step, x.float().std().item())))
    runner.generate(...)
"""

from __future__ import annotations

import importlib
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import ExperimentConfig, GenerationConfig, HookConfig, ModelConfig
from .experiment import DONE_MARKER, RunSpec
from .interp.hooks import HookManager
from .models import ModelAdapter, load_model
from .video import save_frames, save_video, write_json

__all__ = ["GenerationResult", "VideoDiffusionRunner"]


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
