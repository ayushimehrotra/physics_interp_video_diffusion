"""Experiment configuration.

Everything that determines the numerical result of a run lives in these
dataclasses. They round-trip to/from plain dicts (and therefore YAML/JSON), and
every concrete run gets a deterministic ``run_id`` derived from a hash of its
config. That makes mass-running resumable: if the output directory for a run_id
already exists, the run can be skipped.

Conventions
-----------
* Add new fields with defaults so old YAML files keep parsing.
* Unknown keys raise, so typos in a config are caught before a 10-minute job.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ModelConfig",
    "GenerationConfig",
    "PromptConfig",
    "InversionConfig",
    "ProbeConfig",
    "HookConfig",
    "ExperimentConfig",
    "load_config",
    "stable_hash",
]


def _from_dict(cls, data: dict[str, Any]):
    """Build a dataclass from a dict, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"valid keys: {sorted(known)}"
        )
    return cls(**data)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Which checkpoint to load and how to fit it on the device."""

    # Key in physics_interp.models.REGISTRY (e.g. "cogvideox-2b"), which pins
    # the HF repo, the adapter class, and the checkpoint's native dtype.
    id: str = "cogvideox-2b"
    # Override the registry's HF repo (e.g. to point at a fine-tune).
    repo: str | None = None
    # "auto" uses the dtype the checkpoint was trained in. CogVideoX-2b is fp16;
    # running it in bf16 measurably degrades output, so don't change this
    # casually.
    dtype: str = "auto"
    # Memory knobs, in increasing order of slowdown. On an 80GB card none are
    # needed for the 2b model; on a 24GB card enable vae_tiling + cpu_offload.
    vae_tiling: bool = True
    vae_slicing: bool = True
    cpu_offload: bool = False  # sequential module offload; big slowdown
    model_cpu_offload: bool = False  # whole-module offload; modest slowdown
    # Cache location for HF weights. None -> default HF_HOME.
    cache_dir: str | None = None
    # Compile the denoiser. Off by default: torch.compile fuses modules and
    # makes hook site paths unreliable, which breaks interp work.
    compile: bool = False


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


@dataclass
class GenerationConfig:
    """Sampling parameters. These are the axes a physics experiment varies."""

    num_frames: int = 49
    height: int = 480
    width: int = 720
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    negative_prompt: str | None = None
    fps: int = 8
    # Written alongside the video for inspection.
    save_video: bool = True
    save_frames: bool = False
    # Keep the final latent so you can re-decode or diff runs without resampling.
    save_latents: bool = False


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


@dataclass
class PromptConfig:
    """A prompt plus a stable id used in output paths and analysis tables."""

    id: str
    text: str
    # Free-form labels for the analysis side: {"direction": "left_to_right",
    # "object": "ball", "physics": "rolling"}. Never affects generation.
    tags: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# inversion / probing
# --------------------------------------------------------------------------- #


@dataclass
class InversionConfig:
    """How to recover a latent trajectory for an existing video.

    These are the knobs that decide whether recorded activations mean anything,
    so the defaults are deliberately expensive rather than fast.
    """

    # Integration steps from clean latent to noise. Probe signal degrades much
    # faster than reconstruction quality as this drops: a coarse trajectory
    # still rebuilds the video but stops passing through the states the model
    # would actually have visited. 100 is the safe default; below ~40 the
    # recorded states stop being worth probing.
    num_steps: int = 100
    # "euler" (1 network call/step) or "heun" (2 calls/step, 2nd order). Heun
    # measurably recovers signal at low step counts; at num_steps=100 Euler is
    # usually enough and half the cost.
    solver: str = "euler"
    # Text conditioning during inversion. Held constant across the dataset on
    # purpose: a per-scene prompt would inject scene information through the
    # text stream, and a probe reading that is not reading the video.
    prompt: str = ""
    # Inversion runs unguided. CFG is not the gradient of any potential, so a
    # guided trajectory is not the one being inverted. Exposed to make the
    # choice visible, not because it should be changed.
    guidance_scale: float = 1.0
    # How many points along the trajectory to record at, spread evenly.
    # Integration cost is set by num_steps; this only sets storage.
    num_captures: int = 9
    # Explicit step indices, overriding num_captures when set.
    capture_steps: list[int] | None = None
    # Pool activations over per-object regions from the simulator's masks. With
    # this off, only the frame-wide "all" region is recorded.
    use_segmentation: bool = True


@dataclass
class ProbeConfig:
    """How to fit and score linear probes on captured activations."""

    # Label paths into labels.json, dotted, with the probe target derived by
    # `physics_interp.probe.extract_label`. Vector-valued targets are fit
    # component-wise and scored jointly.
    targets: list[str] = field(
        default_factory=lambda: [
            "collision.pre.total_momentum",
            "collision.post.total_momentum",
            "momentum_conserved",
        ]
    )
    # Which pooled regions to use as probe input. Each is fit separately.
    pools: list[str] = field(default_factory=lambda: ["all.pre", "all.post", "A.pre", "B.post"])
    # Ridge penalties swept per fit; the best is chosen on a validation split
    # carved out of the training half, never on the test half.
    alphas: list[float] = field(
        default_factory=lambda: [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
    )
    # Fraction of scenes used for training. The paper's protocol is 60/40.
    train_fraction: float = 0.6
    # Seed for the train/test split, so a rerun scores the same partition.
    split_seed: int = 0
    # Standardise features before fitting. With 1920 dims and few hundred
    # scenes this matters more than the penalty value itself.
    standardize: bool = True
    # Restrict the fit to scenes whose labels match, e.g.
    # {"stratum": "matched_momentum"}. Momentum and mass are only decorrelated
    # inside that stratum, so it is the subset where a momentum probe can be
    # told apart from a mass probe. Empty means all scenes.
    where: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# hooks (extension point)
# --------------------------------------------------------------------------- #


@dataclass
class HookConfig:
    """Declarative reference to a hook factory, resolved at run time.

    This is the seam for the interpretability layer. ``factory`` is a dotted
    path to a callable returning ``fn(x, ctx) -> Tensor | None`` (see
    ``physics_interp.interp.hooks``), and ``kwargs`` is passed to it. Nothing in
    the runner knows what any particular hook does, so new analysis and steering
    primitives are added by writing a factory — not by changing this file::

        hooks:
          - factory: my_project.probes:track_object_position
            sites: "block.*.resid_post"
            steps: {every: 5}
            kwargs: {axis: x}
    """

    factory: str
    sites: str | list[str] = "block.*.resid_post"
    steps: Any = "all"
    kwargs: dict[str, Any] = field(default_factory=dict)
    id: str = ""


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentConfig:
    """The full description of an experiment: a grid of runs plus how to run them.

    The executed set of runs is the cartesian product ``prompts x seeds``, which
    is what gets sharded across GPUs.
    """

    name: str = "unnamed"
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompts: list[PromptConfig] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    hooks: list[HookConfig] = field(default_factory=list)
    # Probing side. Unused by the generation grid, and vice versa — a config
    # normally drives one or the other.
    inversion: InversionConfig = field(default_factory=InversionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    # Directory of simulator scenes to invert (each with labels.json).
    dataset_dir: str = ""
    # Which sites to record during inversion.
    sites: str | list[str] = "block.*.resid_post"
    # Skip runs whose output directory already has a completion marker.
    resume: bool = True
    # Free-form notes carried into every run's metadata.json.
    notes: str = ""

    # ---- (de)serialisation ------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        data = dict(data or {})
        # "experiment:" is an optional wrapper so configs read nicely.
        if "experiment" in data and isinstance(data["experiment"], dict):
            wrapper = data.pop("experiment")
            data = {**wrapper, **data}

        model = _from_dict(ModelConfig, data.pop("model", {}))
        generation = _from_dict(GenerationConfig, data.pop("generation", {}))
        prompts = [_from_dict(PromptConfig, p) for p in data.pop("prompts", []) or []]
        hooks = [_from_dict(HookConfig, h) for h in data.pop("hooks", []) or []]
        inversion = _from_dict(InversionConfig, data.pop("inversion", {}))
        probe = _from_dict(ProbeConfig, data.pop("probe", {}))

        cfg = _from_dict(cls, data)
        cfg.model = model
        cfg.generation = generation
        cfg.prompts = prompts
        cfg.hooks = hooks
        cfg.inversion = inversion
        cfg.probe = probe
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        # A config drives generation (prompts x seeds) or probing (a dataset of
        # existing videos), so neither side is required on its own — but one of
        # them has to be there or the config does nothing at all.
        if not self.prompts and not self.dataset_dir:
            raise ValueError(
                "config has neither `prompts` (generation) nor `dataset_dir` (probing)"
            )
        ids = [p.id for p in self.prompts]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate prompt ids: {ids}")
        if self.prompts and not self.seeds:
            raise ValueError("config has prompts but no seeds")
        if self.inversion.solver not in ("euler", "heun"):
            raise ValueError(
                f"unknown solver {self.inversion.solver!r}; use 'euler' or 'heun'"
            )
        if not 0.0 < self.probe.train_fraction < 1.0:
            raise ValueError("probe.train_fraction must be strictly between 0 and 1")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML (or JSON) file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text())
    cfg = ExperimentConfig.from_dict(data)
    if cfg.name == "unnamed":
        cfg.name = path.stem
    return cfg


def stable_hash(obj: Any, length: int = 10) -> str:
    """Deterministic short hash of any JSON-able object.

    Used for run_ids. Stable across processes and machines (unlike hash()).
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:length]
