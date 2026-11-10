"""Fast replacement for Kubric's Blender.postprocess, run in a process pool.

Kubric's postprocess decodes every layer of every EXR (flow, depth, normals, UV,
object coordinates, cryptomatte) and reads PNGs with pure-Python pypng, all in
the render process AFTER all frames are rendered: ~1.0-1.2 s/frame of CPU while
the GPU idles (measured on the H100; rendering itself is ~0.95 s/frame).

Here each frame is decoded as soon as Blender writes it, in worker processes,
reading only what PhysLaw stores:
  * RGB from the PNG (PIL, C decoder)    -- identical to Kubric's "rgba"[..., :3]
  * segmentation from CryptoObject00.r   -- identical to Kubric's process_segementation,
    which also uses only the first cryptomatte rank, mapping hashes -> segmentation_id.

No bpy import: this module is loaded by spawned pool workers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def decode_frame(exr_path: str, png_path: str, hash_to_id: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    import Imath
    import OpenEXR
    from PIL import Image

    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    exr = OpenEXR.InputFile(str(exr_path))
    win = exr.header()["dataWindow"]
    w, h = win.max.x - win.min.x + 1, win.max.y - win.min.y + 1
    raw = exr.channel("CryptoObject00.r", Imath.PixelType(Imath.PixelType.FLOAT))
    exr.close()
    hashes = np.frombuffer(raw, dtype=np.float32).view(np.uint32).reshape(h, w)
    seg = np.zeros((h, w), np.uint8)
    for hsh, sid in hash_to_id.items():
        seg[hashes == np.uint32(hsh)] = sid
    Path(exr_path).unlink(missing_ok=True)       # keep scratch small; frames are consumed once
    Path(png_path).unlink(missing_ok=True)
    return rgb, seg


def hash_table(scene) -> dict[int, int]:
    """Cryptomatte hash -> segmentation id, exactly as kubric's replace_cryptomatte_hashes_by_asset_index."""
    from kubric.renderer.blender_utils import mm3hash
    table = {}
    for idx, asset in enumerate(scene.assets, start=1):
        sid = asset.segmentation_id if getattr(asset, "segmentation_id", None) is not None else idx
        table[int(mm3hash(asset.uid))] = int(sid)
    return table


def minimal_passes(renderer) -> None:
    """Disable render passes PhysLaw never reads (flow, UV, normals, depth). Measured: RGB changes
    by <= 1/255 (denoiser noise), segmentation identical, EXR 5.8 -> 4.0 MB/frame."""
    sc = renderer.blender_scene
    sc.view_layers[0].use_pass_z = False
    aux = sc.view_layers["AuxOutputs"]
    aux.use_pass_vector = False
    aux.use_pass_uv = False
    aux.use_pass_normal = False


def render_frames(renderer, scene, frames: list[int], scratch: Path, pool) -> tuple[np.ndarray, np.ndarray]:
    """Render ``frames`` and decode them concurrently in ``pool``. Returns (rgb [T,H,W,3], seg [T,H,W])."""
    import bpy
    scratch = Path(scratch)
    renderer.set_exr_output_path(scratch / "exr" / "frame_")
    table = hash_table(scene)
    futs = []
    for f in frames:
        bpy.context.scene.frame_set(f)
        png = scratch / "images" / f"frame_{f:04d}.png"
        bpy.context.scene.render.filepath = str(png)
        bpy.ops.render.render(animation=False, write_still=True)
        exr = scratch / "exr" / f"frame_{f:04d}.exr"
        if not exr.exists():
            raise RuntimeError(f"Blender wrote no EXR for frame {f} at {exr}")
        futs.append(pool.submit(decode_frame, str(exr), str(png), table))
    out = [fu.result() for fu in futs]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
