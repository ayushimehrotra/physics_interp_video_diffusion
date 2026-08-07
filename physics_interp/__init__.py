"""physics_interp — a hookable video diffusion pipeline.

Layers, bottom to top. Each depends only on the ones below it, which is what
keeps adding a model or an analysis from becoming a refactor:

    models/     adapters. Map a concrete diffusers pipeline onto canonical site
                names ("block.12.resid_post") and a TokenLayout that says where
                each token sits in the latent video grid. Also the three
                primitives inversion needs: encode, embed, denoise.
    interp/     the hook substrate (no opinion about what to record) plus the
                linear-probe readout that scores what was recorded.
    video       frame IO in both directions, and the pixel-mask -> latent-cell
                mapping that lets a probe pool over "where the object is".
    runner      one loaded model, many runs. Generates videos from prompts, and
                inverts existing videos into recorded latent trajectories.
    experiment  expands a config into a grid of independent, resumable runs.
    distributed shards that grid across GPUs, replica-per-GPU, no collectives.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .config import (
    ExperimentConfig,
    GenerationConfig,
    HookConfig,
    InversionConfig,
    ModelConfig,
    ProbeConfig,
    PromptConfig,
    load_config,
)
from .distributed import DistInfo, get_dist_info, shard
from .experiment import RunSpec, expand_grid, pending
from .interp import fit_probe, extract_label, probe_grid
from .interp.hooks import HookContext, HookManager, StepFilter
from .models import REGISTRY, ModelAdapter, SiteSpec, TokenLayout, load_model, register
from .runner import (
    ActivationRecorder,
    GenerationResult,
    InversionResult,
    VideoDiffusionRunner,
)

__all__ = [
    "ExperimentConfig",
    "GenerationConfig",
    "HookConfig",
    "InversionConfig",
    "ModelConfig",
    "ProbeConfig",
    "PromptConfig",
    "load_config",
    "DistInfo",
    "get_dist_info",
    "shard",
    "RunSpec",
    "expand_grid",
    "pending",
    "extract_label",
    "fit_probe",
    "probe_grid",
    "HookContext",
    "HookManager",
    "StepFilter",
    "REGISTRY",
    "ModelAdapter",
    "SiteSpec",
    "TokenLayout",
    "load_model",
    "register",
    "ActivationRecorder",
    "GenerationResult",
    "InversionResult",
    "VideoDiffusionRunner",
]
