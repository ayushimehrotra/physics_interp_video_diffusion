#!/usr/bin/env python
"""Render what the models actually predict for the collision aftermath.

Two very different pictures, because the two architectures predict in different
spaces.

**VideoMAE — an actual video.** `VideoMAEForPreTraining` has a 4-layer pixel
decoder, so its prediction can be rendered. §15 measured the loss at 0.998 where
1.00 = predicting the patch mean, so we already know it reconstructs nothing;
this shows what that looks like. Note the decoder is trained with
`norm_pix_loss`, so it predicts *per-patch normalised* pixels and cannot recover
absolute intensity. To render at all, the true per-patch mean and std have to be
put back — which is a free gift of information the model never produced, and
makes the output strictly *better* than what the model knows. If it still looks
like nothing with that help, that is decisive rather than merely suggestive.

**V-JEPA2 — no video exists.** The checkpoint has exactly two modules, `encoder`
and `predictor`, and no pixel head of any kind: a Joint-Embedding Predictive
Architecture predicts in representation space by design. So the honest picture is
a map, not a movie: cosine similarity between each predicted token and its true
counterpart, laid out over the spatial grid for each post-contact tubelet, with a
shuffled-pairing baseline for what chance looks like.

    .venv/bin/python old_experiments/scripts/render_prediction.py --scene scene_00000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402

from physics_interp.video import load_frames, save_video          # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tile(frames, path, scale=2):
    """Save a strip of frames side by side as one PNG."""
    h, w, _ = frames[0].shape
    strip = np.concatenate(list(frames), axis=1)
    img = Image.fromarray(strip.astype(np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)
    return path


def render_videomae(scene_dir: Path, out: Path, dev="cuda:0"):
    from transformers import VideoMAEForPreTraining
    from physics_interp.models.encoders import VideoMAEAdapter

    T, S, P, TS = 8, 196, 16, 2                      # tubelets, patches, patch, tubelet
    m = VideoMAEForPreTraining.from_pretrained(
        "MCG-NJU/videomae-base", dtype=torch.float32).eval().to(dev)
    ad = VideoMAEAdapter(pipe=m.videomae, repo="x", device=torch.device(dev),
                         dtype=torch.float32)
    fr = load_frames(scene_dir, num_frames=49)
    idx = ad.frame_indices(len(fr))
    x = ad.preprocess(fr[idx]).to(dev)               # [1, 16, 3, 224, 224]

    mask = torch.zeros(1, T * S, dtype=torch.bool, device=dev)
    mask[:, 3 * S:] = True                           # tubelets 3-7 = post-contact
    with torch.no_grad():
        o = m(pixel_values=x, bool_masked_pos=mask)

    mu = torch.as_tensor(IMAGENET_MEAN, device=dev)[None, None, :, None, None]
    sd = torch.as_tensor(IMAGENET_STD, device=dev)[None, None, :, None, None]
    f = x * sd + mu                                  # back to [0,1] pixels
    B, t, c, h, w = f.shape
    patches = (f.view(B, t // TS, TS, c, h // P, P, w // P, P)
                .permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                .view(B, t // TS * (h // P) * (w // P), TS * P * P * c))
    pm = patches.mean(-1, keepdim=True)
    ps = patches.var(-1, unbiased=True, keepdim=True).sqrt() + 1e-6

    # The decoder emits normalised patches; put the TRUE per-patch statistics
    # back so there is something to look at. This hands the model information it
    # never predicted, so the render flatters it.
    recon = patches.clone()
    recon[mask] = (o.logits.reshape(-1, TS * P * P * c) * ps[mask] + pm[mask])

    def unpatch(pt):
        v = pt.view(B, t // TS, h // P, w // P, TS, P, P, c)
        v = v.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
        return v.view(B, t, c, h, w)

    true_v = (unpatch(patches)[0].clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy()
    rec_v = (unpatch(recon)[0].clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy()

    out.mkdir(parents=True, exist_ok=True)
    keep = list(range(6, 16))                        # the masked (post-contact) half
    tile(true_v[keep], out / "videomae_true.png")
    tile(rec_v[keep], out / "videomae_reconstruction.png")
    tile(np.abs(true_v[keep] - rec_v[keep]), out / "videomae_error.png")

    # Videos. `rec_v` holds the TRUE patches for the unmasked tubelets (0-2) and
    # the model's guess for the masked ones (3-7), so playing it straight through
    # shows the handover: real frames, then the prediction takes over at frame 6.
    def mp4(arr, name, fps=3):
        save_video([Image.fromarray(f.astype(np.uint8)) for f in arr], out / name, fps=fps)

    mp4(true_v, "videomae_true.mp4")
    mp4(rec_v, "videomae_reconstruction.mp4")
    # Stacked, so the two play in sync and can be compared frame by frame.
    band = np.full((6, true_v.shape[2], 3), 90, dtype=np.float64)
    stack = np.stack([np.concatenate([t, band, r], axis=0)
                      for t, r in zip(true_v, rec_v)])
    mp4(stack, "videomae_stacked.mp4")
    print(f"  VideoMAE loss {float(o.loss):.4f}  (1.00 = predicting the patch mean)")
    print(f"  wrote videomae_true.png / videomae_reconstruction.png / _error.png / .mp4")


def render_vjepa2(scene_dir: Path, out: Path, n_pre=20, dev="cuda:0"):
    from transformers import AutoModel
    from physics_interp.models.encoders import VJEPA2Adapter

    repo = "facebook/vjepa2-vitl-fpc64-256"
    m = AutoModel.from_pretrained(repo, dtype=torch.float32).eval().to(dev)
    ad = VJEPA2Adapter(pipe=m, repo=repo, device=torch.device(dev), dtype=torch.float32)
    side, per = 16, 256
    n_pre_tok, n_tok = (n_pre // 2) * per, 24 * per
    ctx = [torch.arange(n_pre_tok, device=dev)[None]]
    tgt = [torch.arange(n_pre_tok, n_tok, device=dev)[None]]

    fr = load_frames(scene_dir, num_frames=49)
    with torch.no_grad():
        hp = m(pixel_values_videos=ad.preprocess(fr[list(range(1, n_pre + 1))]).to(dev),
               skip_predictor=True).last_hidden_state
        pred = m.predictor(encoder_hidden_states=hp, context_mask=ctx,
                           target_mask=tgt).last_hidden_state[0]
        true = m(pixel_values_videos=ad.preprocess(fr[list(range(1, 49))]).to(dev),
                 skip_predictor=True).last_hidden_state[0, n_pre_tok:]

    p = torch.nn.functional.normalize(pred.float(), dim=-1)
    t_ = torch.nn.functional.normalize(true.float(), dim=-1)
    cos = (p * t_).sum(-1).cpu().numpy()                       # [3584]
    # Chance: same tokens paired with a different token's target.
    perm = torch.randperm(t_.shape[0])
    chance = (p * t_[perm]).sum(-1).cpu().numpy()

    cells = cos.reshape(-1, side, side)
    print(f"  cos(predicted, true): mean {cos.mean():+.3f}  median {np.median(cos):+.3f}")
    print(f"  shuffled-pairing baseline: mean {chance.mean():+.3f}")
    print(f"  per-tubelet mean: {np.round(cells.mean(axis=(1,2)), 3).tolist()}")

    lo, hi = float(np.percentile(cos, 2)), float(np.percentile(cos, 98))
    norm = np.clip((cells - lo) / max(hi - lo, 1e-6), 0, 1)
    # Blue = predicted well, red = predicted badly. No matplotlib in this venv.
    rgb = np.stack([1 - norm, np.zeros_like(norm) + 0.15, norm], axis=-1)
    out.mkdir(parents=True, exist_ok=True)
    tile((rgb * 255).astype(np.uint8), out / "vjepa2_predicted_vs_true_cos.png", scale=8)
    print("  wrote vjepa2_predicted_vs_true_cos.png "
          "(one 16x16 tile per post-contact tubelet; blue = accurate, red = not)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="scene_00000")
    ap.add_argument("--dataset-dir", default="datasets/probe_set")
    ap.add_argument("-o", "--out", default="outputs/render")
    args = ap.parse_args()
    d = Path(args.dataset_dir) / args.scene
    out = Path(args.out) / args.scene
    lab = json.loads((d / "labels.json").read_text())
    print(f"{args.scene}: contact at frame {lab['collision']['actual_frame']:.1f}, "
          f"conserved={lab['momentum_conserved']}")
    print("VideoMAE (has a pixel decoder):")
    render_videomae(d, out)
    print("V-JEPA2 (no decoder exists - latent-space map instead):")
    render_vjepa2(d, out)
    print(f"\nall outputs under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
