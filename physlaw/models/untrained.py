"""Negative control (spec §4.2): the V-JEPA 2 ViT-L architecture with random weights.

Same input geometry and the same encode() as vjepa2.py. Weights come from the checkpoint's
config with ``torch.manual_seed(init_seed)``, so the control is reproducible. Its predictor is
random too, so its surprise() is reported only as a sanity floor.
"""
from __future__ import annotations

from physlaw.models.vjepa2 import VJEPA2Adapter


class UntrainedVJEPA2Adapter(VJEPA2Adapter):

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "UntrainedVJEPA2Adapter":
        import torch
        from transformers import AutoConfig, VJEPA2Model
        src, kw = (cfg["local_path"], {}) if cfg.get("local_path") else (cfg["hf_id"], {"revision": cfg.get("revision")})
        conf = AutoConfig.from_pretrained(src, **kw)
        conf._attn_implementation = "sdpa"
        torch.manual_seed(int(cfg.get("init_seed", 0)))
        model = VJEPA2Model(conf).to(getattr(torch, cfg.get("dtype", "float32"))).to(device).eval()
        return cls(cfg, model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device)
