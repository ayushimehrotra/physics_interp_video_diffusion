"""Model registry.

Adding a model is: write an adapter subclass, register it here. Nothing above
this package changes — the runner, the hook layer, and every experiment config
address models by registry key and sites by canonical name.

    from physics_interp.models import load_model, REGISTRY
    adapter = load_model(cfg.model, device)
"""

from __future__ import annotations

import torch

from .base import ModelAdapter, SiteSpec, TokenLayout
from .cogvideox import CogVideoXAdapter

__all__ = [
    "REGISTRY",
    "register",
    "load_model",
    "ModelAdapter",
    "SiteSpec",
    "TokenLayout",
    "CogVideoX5BAdapter",
]


class CogVideoX5BAdapter(CogVideoXAdapter):
    """CogVideoX-5b: same architecture, 42 blocks at dim 3072, and RoPE.

    Trained in bf16 — running it in fp16 produces artifacts, which is the
    opposite of the 2b checkpoint. That asymmetry is exactly why dtype lives in
    the registry rather than in a config default.
    """

    key = "cogvideox-5b"
    default_repo = "zai-org/CogVideoX-5b"
    native_dtype = "bfloat16"


REGISTRY: dict[str, type[ModelAdapter]] = {
    CogVideoXAdapter.key: CogVideoXAdapter,
    CogVideoX5BAdapter.key: CogVideoX5BAdapter,
}


def register(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Decorator to add an adapter to the registry."""
    if not cls.key:
        raise ValueError(f"{cls.__name__} must set a non-empty `key`")
    REGISTRY[cls.key] = cls
    return cls


def get_adapter_cls(key: str) -> type[ModelAdapter]:
    if key not in REGISTRY:
        raise KeyError(f"unknown model id {key!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[key]


def load_model(cfg, device: str | torch.device = "cuda") -> ModelAdapter:
    """Load the model named by ``cfg.id`` onto ``device``."""
    return get_adapter_cls(cfg.id).load(cfg, torch.device(device))
