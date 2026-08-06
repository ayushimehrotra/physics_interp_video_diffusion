"""physics_interp — a hookable video diffusion pipeline.

Layers, bottom to top. Each depends only on the ones below it, which is what
keeps adding a model or an analysis from becoming a refactor:

    models/     adapters. Map a concrete diffusers pipeline onto canonical site
                names ("block.12.resid_post") and a TokenLayout that says where
                each token sits in the latent video grid.
    interp/     the hook substrate. Denoising-step and CFG-branch aware taps on
                any site. No opinion about what to record or edit.
    runner      one loaded model, many runs. Turns RunSpecs into directories.
    experiment  expands a config into a grid of independent, resumable runs.
    distributed shards that grid across GPUs, replica-per-GPU, no collectives.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    GenerationConfig,
    HookConfig,
    ModelConfig,
    PromptConfig,
    load_config,
)
from .distributed import DistInfo, get_dist_info, shard
from .experiment import RunSpec, expand_grid, pending
from .interp.hooks import HookContext, HookManager, StepFilter
from .models import REGISTRY, ModelAdapter, SiteSpec, TokenLayout, load_model, register
from .runner import GenerationResult, VideoDiffusionRunner

__all__ = [
    "ExperimentConfig",
    "GenerationConfig",
    "HookConfig",
    "ModelConfig",
    "PromptConfig",
    "load_config",
    "DistInfo",
    "get_dist_info",
    "shard",
    "RunSpec",
    "expand_grid",
    "pending",
    "HookContext",
    "HookManager",
    "StepFilter",
    "REGISTRY",
    "ModelAdapter",
    "SiteSpec",
    "TokenLayout",
    "load_model",
    "register",
    "GenerationResult",
    "VideoDiffusionRunner",
]
