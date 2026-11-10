"""Model config -> adapter class."""
from __future__ import annotations


def load_adapter(cfg, device: str = "cuda"):
    kind = cfg["kind"]
    if kind == "encoder" and "vjepa2" in cfg["hf_id"]:
        from physlaw.models.vjepa2 import VJEPA2Adapter
        return VJEPA2Adapter.load(cfg, device)
    if kind == "encoder" and cfg.get("arch") == "videomae2_pretrain":
        from physlaw.models.videomae2 import VideoMAEv2Adapter
        return VideoMAEv2Adapter.load(cfg, device)
    if kind == "diffusion" and "CogVideoX" in cfg["hf_id"] and not cfg["token_grid"].get("patch_t"):
        from physlaw.models.cogvideox import CogVideoXAdapter
        return CogVideoXAdapter.load(cfg, device)
    if kind == "diffusion" and "CogVideoX1.5" in cfg["hf_id"] and "I2V" in cfg["hf_id"]:
        from physlaw.models.cogvideox import CogVideoX15I2VAdapter
        return CogVideoX15I2VAdapter.load(cfg, device)
    if kind == "positive_control":
        from physlaw.models.positive_control import PC1Adapter
        return PC1Adapter.load(cfg, device)
    if kind == "encoder_untrained":
        from physlaw.models.untrained import UntrainedVJEPA2Adapter
        return UntrainedVJEPA2Adapter.load(cfg, device)
    raise NotImplementedError(f"no adapter yet for {cfg['name']} (kind={kind}); see docs/STATUS.md")
