"""Model config -> adapter class, and weight pinning (spec §4.3).

Dispatch: ``cfg.adapter`` ("module:Class" under physlaw.models) if set, else by ``kind`` (controls), else by the
base model's exact ``hf_id``. Stage 07 track B uses ``cfg.track_b_adapter`` instead when set (``role="track_b"``).
Fine-tunes (VideoREPA, PhysRVG) keep their base's hf_id and add weights on top.

Pinning: every config pins ``hf_id`` + ``revision`` and ``sha256: {file: hex}`` for its weight files (paths relative to
``local_path``; LoRA / DiT override files under their own blocks, as videomae_v2_giant.yaml's ``checkpoint``).
``verify_weights`` checks them before a model is loaded. A ``<file>.sha256`` sidecar newer than the file stands in
for re-hashing (written by this check); delete it to force a re-hash.

    python -m physlaw.models.registry configs/models/X.yaml [--hash]   # verify; --hash prints a sha256 block
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

ADAPTERS_BY_KIND = {
    "encoder_untrained": "untrained:UntrainedVJEPA2Adapter",
    "positive_control": "positive_control:PC1Adapter",
}
ADAPTERS_BY_HF_ID = {
    "facebook/vjepa2-vitl-fpc64-256": "vjepa2:VJEPA2Adapter",
    "facebook/vjepa2-vitg-fpc64-256": "vjepa2:VJEPA2Adapter",
    "OpenGVLab/VideoMAE2": "videomae2:VideoMAEv2Adapter",
    "OpenGVLab/VideoMAEv2-Base": "videomae2:VideoMAEv2Adapter",
    "zai-org/CogVideoX-2b": "cogvideox:CogVideoXAdapter",
    "zai-org/CogVideoX-5b": "cogvideox:CogVideoXAdapter",
    "zai-org/CogVideoX1.5-5B-I2V": "cogvideox:CogVideoX15I2VAdapter",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "wan:WanAdapter",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "wan:WanAdapter",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "wan:WanAdapter",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "wan:WanI2VAdapter",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "wan:WanI2VAdapter",
    "nvidia/Cosmos-Predict2.5-2B": "cosmos_predict:CosmosPredictAdapter",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers": "wan_vace:WanVACEAdapter",
    "Wan-AI/Wan2.1-VACE-14B-diffusers": "wan_vace:WanVACEAdapter",
    "Lightricks/LTX-Video-0.9.8-13B-distilled": "ltx:LTXConditionAdapter",
}


def adapter_class(cfg, role: str | None = None):
    ref = ((cfg.get("track_b_adapter") if role == "track_b" else None) or cfg.get("adapter")
           or ADAPTERS_BY_KIND.get(cfg.get("kind")) or ADAPTERS_BY_HF_ID.get(cfg.get("hf_id")))
    if ref is None:
        raise KeyError(f"no adapter for {cfg['name']} (kind={cfg.get('kind')}, hf_id={cfg.get('hf_id')}): set "
                       "`adapter: module:Class` in its config or add the hf_id to ADAPTERS_BY_HF_ID")
    mod, cls = ref.split(":")
    return getattr(importlib.import_module(f"physlaw.models.{mod}"), cls)


def load_adapter(cfg, device: str = "cuda", verify: bool = True, role: str | None = None, need: str | None = None):
    """Load the config's adapter. ``need``: a capability the caller uses ("internals": encode / forecast / surprise;
    "generator": generate(); physlaw.models.base.Adapter), checked on the class before any weights are touched."""
    cls = adapter_class(cfg, role)
    if need is not None and not getattr(cls, need):
        raise ValueError(f"{cfg['name']}: {cls.__name__} has no {need} (physlaw.models.base.Adapter capabilities: "
                         f"internals={cls.internals}, generator={cls.generator})")
    if verify:
        verify_weights(cfg)
    return cls.load(cfg, device)


# --------------------------------------------------------------------------- pinning

def sha256_file(path: Path, use_sidecar: bool = True) -> str:
    """sha256 of a file, streamed. With ``use_sidecar`` a ``<file>.sha256`` newer than the file is trusted, and a
    fresh hash is written there."""
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    if use_sidecar and side.exists() and side.stat().st_mtime >= path.stat().st_mtime:
        return side.read_text().split()[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if use_sidecar:
        try:
            side.write_text(f"{digest}  {path.name}\n")
        except OSError:
            pass
    return digest


def pinned_files(cfg) -> dict[Path, str]:
    """{absolute path: pinned sha256} over the config's ``sha256`` map and its weight blocks (``checkpoint``,
    ``dit_weights``, ``lora``: each with ``path`` and ``sha256``). Files that are not on disk are skipped (models
    loaded from the Hub are verified by the Hub's revision pin)."""
    out = {}
    root = cfg.get("local_path")
    for rel, digest in (cfg.get("sha256") or {}).items():
        if root:
            out[Path(root) / rel] = digest
    te = cfg.get("text_encoder") or {}
    for rel, digest in (te.get("sha256") or {}).items():
        if te.get("path"):
            out[Path(te["path"]) / rel] = digest
    for blk in ("checkpoint", "dit_weights", "lora"):
        b = cfg.get(blk) or {}
        if b.get("sha256") and b.get("path"):
            out[Path(b["path"])] = b["sha256"]
    return {p: d for p, d in out.items() if p.exists()}


def verify_weights(cfg) -> int:
    """Raise if any pinned weight file on disk does not match its sha256. Returns the number checked."""
    bad = []
    files = pinned_files(cfg)
    for p, want in files.items():
        got = sha256_file(p)
        if got != want:
            bad.append(f"{p}: {got} != pinned {want}")
    if bad:
        raise RuntimeError("weight files do not match their pins:\n  " + "\n  ".join(bad))
    return len(files)


def weight_files(root: Path) -> list[Path]:
    """Weight files under a local snapshot (what the ``sha256`` map pins)."""
    exts = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix in exts and ".cache" not in p.parts)


def main():
    import argparse
    from physlaw.config import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--hash", action="store_true", help="hash every weight file under local_path; print a sha256 block")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.hash:
        root = Path(cfg["local_path"])
        print("sha256:")
        for p in weight_files(root):
            print(f"  {p.relative_to(root)}: {sha256_file(p)}")
        return
    print(f"{cfg['name']}: {verify_weights(cfg)} pinned files verified")


if __name__ == "__main__":
    main()
