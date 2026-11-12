#!/usr/bin/env python3
"""Stage 3 — spec §3 acceptance: overlay projected ROI tokens on frames for N random clips.

    /root/envs/physics_venv/bin/python scripts/03_token_overlays.py \\
        --model configs/models/vjepa2_vitl.yaml --data data/f1 --n 20

CPU only (no model is loaded). For each clip: the model's input frames are fitted to its
input size; for 4 token-frames (2 pre, 2 post contact) the first frame is drawn with A's ROI
tokens tinted red, B's blue, and a white tick on straddling token-frames. Output:
results/<model>/overlays/<group>_<twin>.png. Inspect by eye: every tint should sit on its ball.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config, resolve_path  # noqa: E402
from physlaw.data.io import list_clips, load_clip  # noqa: E402
from physlaw.models.base import TokenGridSpec, fit_frames, sample_frames  # noqa: E402
from physlaw.tokens.project import project_masks, roi_object_ids, token_frame_phase  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.model)
    from physlaw.models.base import token_spec
    spec = token_spec(cfg)
    clips = list_clips(resolve_path(a.data))
    random.Random(a.seed).shuffle(clips)
    out = REPO / "results" / cfg.name / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    for path in clips[: a.n]:
        c = load_clip(path)
        sc, pl = c.labels["scene_constants"], c.pl
        src = cfg["official"] if "official" in cfg else cfg["input"]
        fr = sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=int(src["fps"]),
                           n_frames=int(src["num_frames"]), contact_frame=pl["first_post_contact_frame"],
                           center_on_contact=bool(cfg.get("input", {}).get("center_on_contact", False)))
        obj = project_masks(c.masks, fr, spec, roi_object_ids(c.labels))
        ph = token_frame_phase(fr, spec, pl["first_post_contact_frame"])
        pre = [j for j, p in enumerate(ph) if p == "pre"]
        post = [j for j, p in enumerate(ph) if p == "post"]
        picks = [pre[0], pre[-1], post[0], post[-1]]
        tiles = []
        for j in picks:
            img = fit_frames(c.frames[[fr[j * spec.tubelet]]], spec.input_hw, spec.fit)[0].astype(float)
            for name, col in (("A", (255, 0, 0)), ("B", (0, 80, 255))):
                m = np.kron(obj[name].roi[j], np.ones((spec.patch, spec.patch))).astype(bool)
                img[m] = 0.55 * img[m] + 0.45 * np.array(col)
            img = np.pad(img.astype(np.uint8), ((2, 2), (2, 2), (0, 0)), constant_values=255 if ph[j] == "straddle" else 0)
            tiles.append(img)
        iio.imwrite(out / f"{pl['twin_group_id']}_{pl['twin']}.png", np.concatenate(tiles, axis=1))
    print(f"wrote {min(a.n, len(clips))} overlays to {out}")


if __name__ == "__main__":
    main()
