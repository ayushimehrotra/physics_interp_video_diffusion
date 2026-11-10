"""VideoMAE v2-giant adapter (spec §4.1/4.2, milestone 6): encode / forecast / surprise.

Checkpoint: OpenGVLab's PRE-TRAINING checkpoint ``mae-g/vit_g_hybrid_pt_1200e.pth`` (HF repo
OpenGVLab/VideoMAE2, file pinned by sha256 in the model config). It is the only public
VideoMAE v2 weight file that keeps the pixel decoder: the model repos ``OpenGVLab/VideoMAEv2-*``
are encoder-only. Loaded with ``torch.load(weights_only=True)``: tensors only, no pickled code.

Architecture (re-implemented here from OpenGVLab/VideoMAEv2 ``models/modeling_pretrain.py`` and
``modeling_finetune.py``, ``pretrain_videomae_giant_patch14_224``; state-dict names match and
the load is strict): encoder ViT-g/14 (1408-d, 40 blocks, 16 heads, MLP 6144, qkv bias on q and
v, LayerNorm eps 1e-6, fixed sinusoidal positions, no layer scale); decoder 4 blocks of 512-d,
8 heads, head -> 1176 = 2 x 14 x 14 x 3 per-patch-normalised pixels.

Input. 16 frames at 8 fps (stride 6 of the 48 fps master; pre-training sampled every 4th frame of
~30 fps video), placed so the first post-contact frame is input frame 8: token-frames 0-3 are
entirely pre-contact and 4-7 post-contact. Squashed to 224 x 224, ImageNet mean / std.

encode():   encoder block outputs (configured layers) + key 40 = encoder final norm (all tokens
            visible) + FINAL = the decoder's last normalised hidden state with every token
            visible, i.e. the space forecast() returns. FINAL is 512-d, the encoder 1408-d.
forecast(): the encoder sees ONLY the context tokens (token-frames < k / 2); every future token is
            a mask token; the decoder's normalised hidden states at those positions (FINAL) and
            its pixel predictions are returned. No future frame enters, so twins share it exactly.
surprise(): ``fut_*`` (primary) the pre-training loss, MSE between the decoder's predictions and
            the per-patch-normalised true pixels, on the masked future; over all future tokens and
            over ``object_masks``. ``tube_*`` (secondary, spec "identical mask draws") the same loss
            under the pre-training mask, random 90% TUBES over the whole clip, ``n_mask_draws``
            draws, identical across twins, scored on masked post-contact tokens. ``*_wmse_*``: the same
            errors in raw pixels (prediction de-normalised with the TRUE patch's mean / sd). On F1 most
            patches are flat floor (patch sd ~0.007 of full range), and normalising turns their render
            noise into unit-variance noise, so ``*_mse_*`` sit near 1 and are noise-dominated;
            ``*_wmse_*`` weight patches by their real contrast (balls, edges, shadows). Tube masks keep
            10% of positions visible at ALL times, so the future leaks; report it apart.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from physlaw.models.base import ModelAdapter, TokenGridSpec, fit_frames, sample_frames, spatial_mask_groups

FINAL = -1          # forecast space: decoder hidden (512-d)
ENC_NORM = 40       # encoder final LayerNorm output (1408-d)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _build(torch):
    nn, F = torch.nn, torch.nn.functional

    class Attention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.h = heads
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, N, C = x.shape
            b = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
            qkv = F.linear(x, self.qkv.weight, b).reshape(B, N, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])   # scale = head_dim ** -0.5, as the original
            return self.proj(y.transpose(1, 2).reshape(B, N, C))

    class Mlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self, dim, heads, mlp_ratio):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attention(dim, heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class PatchEmbed(nn.Module):
        def __init__(self, dim, patch, tubelet):
            super().__init__()
            self.proj = nn.Conv3d(3, dim, kernel_size=(tubelet, patch, patch), stride=(tubelet, patch, patch))

        def forward(self, x):                      # [B, 3, T, H, W] -> [B, N, D], N ordered (t, h, w)
            return self.proj(x).flatten(2).transpose(1, 2)

    class Encoder(nn.Module):
        def __init__(self, dim, depth, heads, mlp_ratio, patch, tubelet):
            super().__init__()
            self.patch_embed = PatchEmbed(dim, patch, tubelet)
            self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    class Decoder(nn.Module):
        def __init__(self, dim, depth, heads, mlp_ratio, out):
            super().__init__()
            self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.head = nn.Linear(dim, out)

    class VideoMAEv2Pretrain(nn.Module):
        def __init__(self, enc_dim=1408, enc_depth=40, enc_heads=16, dec_dim=512, dec_depth=4, dec_heads=8,
                     mlp_ratio=48 / 11, patch=14, tubelet=2):
            super().__init__()
            self.encoder = Encoder(enc_dim, enc_depth, enc_heads, mlp_ratio, patch, tubelet)
            self.decoder = Decoder(dec_dim, dec_depth, dec_heads, mlp_ratio, 3 * tubelet * patch * patch)
            self.encoder_to_decoder = nn.Linear(enc_dim, dec_dim, bias=False)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

    return VideoMAEv2Pretrain


def sinusoid_table(n: int, d: int) -> np.ndarray:
    """Exactly OpenGVLab's get_sinusoid_encoding_table."""
    pos = np.arange(n)[:, None] / np.power(10000, 2 * (np.arange(d) // 2) / d)[None]
    pos[:, 0::2], pos[:, 1::2] = np.sin(pos[:, 0::2]), np.cos(pos[:, 1::2])
    return pos.astype(np.float32)


class VideoMAEv2Adapter(ModelAdapter):

    def __init__(self, cfg, model, device):
        super().__init__(cfg)
        import torch
        self.torch, self.model, self.device = torch, model, device
        R = int(cfg["input"]["resolution"])
        tg = cfg["token_grid"]
        self.token_grid = TokenGridSpec(input_hw=(R, R), patch=int(tg["patch"]), tubelet=int(tg["tubelet"]),
                                        fit=cfg["input"].get("fit", "squash"))
        self.n_frames, self.fps = int(cfg["input"]["num_frames"]), int(cfg["input"]["fps"])
        gh, gw = self.token_grid.grid_hw
        n = (self.n_frames // self.token_grid.tubelet) * gh * gw
        self.pos_enc = torch.from_numpy(sinusoid_table(n, model.encoder.norm.normalized_shape[0]))[None].to(device)
        self.pos_dec = torch.from_numpy(sinusoid_table(n, model.mask_token.shape[-1]))[None].to(device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1, 1)
        s = cfg.get("surprise", {})
        self.tube_ratio, self.tube_draws = float(s.get("mask_ratio", 0.9)), int(s.get("n_mask_draws", 8))
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "VideoMAEv2Adapter":
        import torch
        ck = Path(cfg["checkpoint"]["path"])
        want = cfg["checkpoint"].get("sha256")
        if want:
            side = ck.with_suffix(ck.suffix + ".sha256")
            got = side.read_text().split()[0] if side.exists() else hashlib.sha256(ck.read_bytes()).hexdigest()
            if got != want:
                raise RuntimeError(f"{ck}: sha256 {got} != pinned {want}")
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        model = _build(torch)()
        model.load_state_dict(sd, strict=True)
        model = model.to(getattr(torch, cfg.get("dtype", "float32"))).to(device).eval()
        return cls(cfg, model, device)

    # ------------------------------------------------------------------ geometry / input

    def input_frames(self, labels: dict) -> list[int]:
        return sample_frames(master_fps=labels["scene_constants"]["frame_rate"], n_master=labels["scene_constants"]["num_frames"],
                             fps=self.fps, n_frames=self.n_frames,
                             contact_frame=labels["physlaw"]["first_post_contact_frame"], center_on_contact=True)

    def context_frames(self, labels: dict) -> int:
        fr = self.input_frames(labels)
        k = sum(f < labels["physlaw"]["first_post_contact_frame"] for f in fr)
        return k - k % self.token_grid.tubelet

    def to_tensor(self, frames_uint8: np.ndarray):
        """uint8 [T, H, W, 3] -> normalised [1, 3, T, R, R]."""
        torch = self.torch
        x = fit_frames(frames_uint8, self.token_grid.input_hw, self.token_grid.fit)
        t = torch.from_numpy(x).to(self.device).permute(3, 0, 1, 2)[None].float() / 255.0
        return ((t - self.mean) / self.std).to(self.dtype)

    def _grid(self, tok, T_tok):
        gh, gw = self.token_grid.grid_hw
        return tok.reshape(T_tok, gh, gw, tok.shape[-1])

    def _n_per(self):
        gh, gw = self.token_grid.grid_hw
        return gh * gw

    # ------------------------------------------------------------------ model passes

    def _encoder(self, x, visible: "torch.Tensor", keep: list[int] | None = None):
        """Encoder on visible tokens only ([N] bool). Returns (normed output, {block: hidden})."""
        m = self.model.encoder
        tok = m.patch_embed(x)
        tok = tok + self.pos_enc[:, : tok.shape[1]].to(x.dtype)       # context-only input: first positions
        h = tok[:, visible]
        hs = {}
        for i, blk in enumerate(m.blocks):
            h = blk(h)
            if keep is not None and i in keep:
                hs[i] = h
        return m.norm(h), hs

    def _decoder(self, enc_out, visible, decode: "torch.Tensor"):
        """Decoder over [visible tokens, mask tokens at ``decode`` positions]; returns (normed hidden, pixels)
        for the decode positions (or for the visible ones if ``decode`` is empty)."""
        m = self.model
        v = m.encoder_to_decoder(enc_out) + self.pos_dec[:, visible].to(enc_out.dtype)
        parts = [v]
        nd = int(decode.sum())
        if nd:
            parts.append(m.mask_token.to(enc_out.dtype) + self.pos_dec[:, decode].to(enc_out.dtype))
        h = self.torch.cat(parts, 1)
        for blk in m.decoder.blocks:
            h = blk(h)
        h = m.decoder.norm(h[:, -nd:] if nd else h)
        return h, m.decoder.head(h)

    def target_pixels(self, x, with_scale: bool = False):
        """Per-patch-normalised pixels, the pre-training target: [N, 2*14*14*3] ordered (t, h, w).
        ``with_scale``: also the per-element variance of the TRUE patch ([N, same]), to turn the
        normalised error back into raw-pixel MSE (0-1 units)."""
        torch = self.torch
        p, tub = self.token_grid.patch, self.token_grid.tubelet
        u = (x.float() * self.std + self.mean)[0]                       # [3, T, H, W] in [0, 1]
        C, T, H, W = u.shape
        u = u.reshape(C, T // tub, tub, H // p, p, W // p, p).permute(1, 3, 5, 2, 4, 6, 0)
        u = u.reshape(-1, tub * p * p, C)                               # [N, (p0 p1 p2), c]
        sd = u.var(1, unbiased=True, keepdim=True).sqrt()
        u = (u - u.mean(1, keepdim=True)) / (sd + 1e-6)
        if with_scale:
            return u.reshape(u.shape[0], -1), ((sd + 1e-6) ** 2).expand_as(u).reshape(u.shape[0], -1)
        return u.reshape(u.shape[0], -1)

    # ------------------------------------------------------------------ spec API

    def encode(self, video, noise_level=None) -> dict[int, Any]:
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        T_tok = x.shape[2] // self.token_grid.tubelet
        allv = torch.ones(T_tok * self._n_per(), dtype=torch.bool, device=x.device)
        with torch.no_grad():
            out, hs = self._encoder(x, allv, keep=self.layers)
            dec, _ = self._decoder(out, allv, torch.zeros_like(allv))
        acts = {l: self._grid(h[0], T_tok) for l, h in hs.items()}
        acts[ENC_NORM] = self._grid(out[0], T_tok)
        acts[FINAL] = self._grid(dec[0], T_tok)
        return acts

    def _future(self, x, context_end_frame):
        torch = self.torch
        tub = self.token_grid.tubelet
        if context_end_frame % tub:
            raise ValueError(f"context_end_frame {context_end_frame} not a multiple of tubelet {tub}")
        n_ctx = (context_end_frame // tub) * self._n_per()
        n_all = (x.shape[2] // tub) * self._n_per()
        vis = torch.zeros(n_all, dtype=torch.bool, device=x.device)
        vis[:n_ctx] = True
        with torch.no_grad():
            out, _ = self._encoder(x[:, :, :context_end_frame], vis[:n_ctx])
            # positions are absolute token indices, so the context encoding equals the full-clip one restricted
            # to context positions only if no future token is visible: true here (context tokens only)
            h, pix = self._decoder(out, vis, ~vis)
        return h[0], pix[0], (n_all - n_ctx) // self._n_per()

    def forecast(self, video, context_end_frame: int) -> dict[int, Any]:
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        h, _, T_post = self._future(x, context_end_frame)
        return {FINAL: self._grid(h, T_post)}

    def pred_real(self, video, context_end_frame: int, n_masks: int = 4, seed: int = 0) -> dict[int, Any]:
        """P-pred-real features (spec §5.2): decoder hidden states (FINAL) for MASKED REAL post-event tokens,
        with every other token of the clip (all pre, the other post tokens) visible to the encoder. Spatial
        positions split into ``n_masks`` seeded groups, group k masked in every post token-frame on pass k.
        Same space and shape [T_post_tok, gh, gw, 512] as forecast()."""
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        per = self._n_per()
        n_ctx = (context_end_frame // self.token_grid.tubelet) * per
        n_all = (x.shape[2] // self.token_grid.tubelet) * per
        out = None
        with torch.no_grad():
            for grp in spatial_mask_groups(per, n_masks, seed):
                tgt = torch.zeros(n_all, dtype=torch.bool, device=x.device)
                for t in range((n_all - n_ctx) // per):
                    tgt[n_ctx + t * per + torch.as_tensor(grp, device=x.device)] = True
                enc, _ = self._encoder(x, ~tgt)
                h, _ = self._decoder(enc, ~tgt, tgt)
                if out is None:
                    out = torch.zeros(n_all - n_ctx, h.shape[-1], dtype=h.dtype, device=h.device)
                out[tgt[n_ctx:]] = h[0]
        return {FINAL: self._grid(out, (n_all - n_ctx) // per)}

    def surprise(self, video, context_end_frame: int, object_masks=None) -> dict[str, float]:
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        tgt, var = self.target_pixels(x, with_scale=True)
        n_ctx = (context_end_frame // self.token_grid.tubelet) * self._n_per()
        _, pix, T_post = self._future(x, context_end_frame)
        d2 = (pix.float() - tgt[n_ctx:]) ** 2
        err, werr = d2.mean(-1), (d2 * var[n_ctx:]).mean(-1)          # normalised / raw-pixel MSE per token
        res = {"fut_mse_all": float(err.mean()), "fut_wmse_all": float(werr.mean())}
        obj = None
        if object_masks is not None:
            obj = torch.as_tensor(np.asarray(object_masks), device=x.device).bool().reshape(-1)
            if obj.any():
                res["fut_mse_obj"] = float(err[obj].mean())
                res["fut_wmse_obj"] = float(werr[obj].mean())
        # pre-training tube masks: same draws for every clip (seeded), scored on masked POST tokens
        n_all = tgt.shape[0]
        per = self._n_per()
        g = torch.Generator(device="cpu").manual_seed(0)
        e_all, e_obj, w_all, w_obj = [], [], [], []
        for _ in range(self.tube_draws):
            keep = torch.randperm(per, generator=g)[: int(round(per * (1 - self.tube_ratio)))]
            vis2d = torch.zeros(per, dtype=torch.bool)
            vis2d[keep] = True
            vis = vis2d.repeat(n_all // per).to(x.device)
            with torch.no_grad():
                out, _ = self._encoder(x, vis)
                _, p2 = self._decoder(out, vis, ~vis)
            d2 = (p2[0].float() - tgt[~vis]) ** 2
            for lst_a, lst_o, e in ((e_all, e_obj, d2.mean(-1)), (w_all, w_obj, (d2 * var[~vis]).mean(-1))):
                full = torch.full((n_all,), float("nan"), device=x.device)
                full[~vis] = e
                post = full[n_ctx:]
                lst_a.append(torch.nanmean(post))
                if obj is not None and obj.any():
                    lst_o.append(torch.nanmean(post[obj]))
        res["tube_mse_all"] = float(torch.stack(e_all).mean())
        res["tube_wmse_all"] = float(torch.stack(w_all).mean())
        if e_obj:
            res["tube_mse_obj"] = float(torch.stack(e_obj).mean())
            res["tube_wmse_obj"] = float(torch.stack(w_obj).mean())
        return res
