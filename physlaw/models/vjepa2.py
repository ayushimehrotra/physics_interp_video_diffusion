"""V-JEPA 2 adapter (spec §4.1, milestone 2): encode / forecast / surprise.

Input. 64 master frames at 24 fps (every 2nd frame of the 48 fps master), each shown
ONCE, placed so the first post-contact frame is input frame 32. With tubelet 2,
token-frames 0-15 are entirely pre-contact and 16-31 entirely post-contact, with no
straddling tubelet. Frames are squashed to 256x256 (no crop) and normalised with the
checkpoint's processor statistics.

forecast / surprise: context-only encoding (differs from the old project).
transformers' ``VJEPA2Model.forward(context_mask=...)`` runs the encoder over the WHOLE
video and only afterwards selects the context tokens. Attention is bidirectional, so those
"context" tokens already contain the post-contact frames, which leaks the future into the
prediction and makes twins' contexts differ. V-JEPA's context encoder sees only the context
tokens. Here the encoder runs on the pre-contact frames alone; its RoPE positions are token
indices, so they are identical to those positions in the full clip. The predictor then
fills in the post-contact positions. The TARGET is the encoder's output on the full clip at
those positions, as in V-JEPA training. Twins share their pre-contact frames byte-for-byte,
so they share the context and the forecast exactly, and differ only in the target.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from physlaw.models.base import ModelAdapter, TokenGridSpec, fit_frames, sample_frames

FINAL = -1   # key for the encoder's final (layer-normed) output: the predictor's target space


class VJEPA2Adapter(ModelAdapter):

    def __init__(self, cfg, model, mean, std, device):
        super().__init__(cfg)
        import torch
        self.torch = torch
        self.model = model
        self.device = device
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        R = int(cfg["input"]["resolution"])
        tg = cfg["token_grid"]
        self.token_grid = TokenGridSpec(input_hw=(R, R), patch=int(tg["patch"]), tubelet=int(tg["tubelet"]),
                                        fit=cfg["input"].get("fit", "squash"))
        self.n_frames = int(cfg["input"]["num_frames"])
        self.fps = int(cfg["input"]["fps"])
        self.dtype = next(model.parameters()).dtype

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "VJEPA2Adapter":
        import torch
        from transformers import AutoVideoProcessor, VJEPA2Model
        dtype = getattr(torch, cfg.get("dtype", "float32"))
        src, kw = (cfg["local_path"], {}) if cfg.get("local_path") else (cfg["hf_id"], {"revision": cfg.get("revision")})
        model = VJEPA2Model.from_pretrained(src, **kw, torch_dtype=dtype,
                                            attn_implementation="sdpa").to(device).eval()
        try:
            proc = AutoVideoProcessor.from_pretrained(src, **kw)
            mean, std = proc.image_mean, proc.image_std
        except Exception:  # noqa: BLE001  (processor config missing: ImageNet stats, which V-JEPA 2 uses)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return cls(cfg, model, mean, std, device)

    # ------------------------------------------------------------------ geometry

    def input_frames(self, labels: dict) -> list[int]:
        pl = labels["physlaw"]
        n_master = labels["scene_constants"]["num_frames"]
        return sample_frames(master_fps=labels["scene_constants"]["frame_rate"], n_master=n_master,
                             fps=self.fps, n_frames=self.n_frames,
                             contact_frame=pl["first_post_contact_frame"], center_on_contact=True)

    def context_frames(self, labels: dict) -> int:
        """Number of leading INPUT frames that are entirely pre-contact, rounded down to a tubelet."""
        fr = self.input_frames(labels)
        k = sum(f < labels["physlaw"]["first_post_contact_frame"] for f in fr)
        return k - k % self.token_grid.tubelet

    def to_tensor(self, frames_uint8: np.ndarray):
        """uint8 [T, H, W, 3] (already the sampled input frames) -> normalised [1, T, 3, R, R]."""
        torch = self.torch
        x = fit_frames(frames_uint8, self.token_grid.input_hw, self.token_grid.fit)
        t = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2)[None].float() / 255.0
        return ((t - self.mean) / self.std).to(self.dtype)

    def _grid(self, h, n_tok_frames: int):
        gh, gw = self.token_grid.grid_hw
        return h.reshape(n_tok_frames, gh, gw, h.shape[-1])

    # ------------------------------------------------------------------ spec interface

    def encode(self, video, noise_level=None) -> dict[int, Any]:
        """Activations [T_tok, gh, gw, D] for every configured layer (block output), plus FINAL."""
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        T_tok = x.shape[1] // self.token_grid.tubelet
        with torch.no_grad():
            out = self.model(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
        hs = out.hidden_states          # (embeddings, block_0, ..., block_{L-1})
        acts = {int(l): self._grid(hs[int(l) + 1][0], T_tok) for l in self.layers}
        acts[FINAL] = self._grid(out.last_hidden_state[0], T_tok)
        return acts

    def forecast(self, video, context_end_frame: int) -> dict[int, Any]:
        """Predicted FINAL-space tokens for token-frames >= context_end_frame / tubelet, from context only."""
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        tub = self.token_grid.tubelet
        if context_end_frame % tub:
            raise ValueError(f"context_end_frame {context_end_frame} not a multiple of tubelet {tub}")
        gh, gw = self.token_grid.grid_hw
        per = gh * gw
        n_ctx, n_all = (context_end_frame // tub) * per, (x.shape[1] // tub) * per
        with torch.no_grad():
            enc = self.model(pixel_values_videos=x[:, :context_end_frame], skip_predictor=True).last_hidden_state
            full = torch.zeros(1, n_all, enc.shape[-1], dtype=enc.dtype, device=enc.device)
            full[:, :n_ctx] = enc            # only these positions are read (context_mask below)
            ctx = [torch.arange(n_ctx, device=enc.device)[None]]
            tgt = [torch.arange(n_ctx, n_all, device=enc.device)[None]]
            pred = self.model.predictor(encoder_hidden_states=full, context_mask=ctx, target_mask=tgt).last_hidden_state
        return {FINAL: self._grid(pred[0], (n_all - n_ctx) // per)}

    def surprise(self, video, context_end_frame: int, object_masks=None) -> dict[str, float]:
        """Predictor error on post-context tokens: L1, MSE, 1-cos; over all tokens and over ``object_masks``
        (bool [T_post_tok, gh, gw], e.g. the union of the balls' ROIs) if given."""
        torch = self.torch
        x = video if torch.is_tensor(video) else self.to_tensor(video)
        pred = self.forecast(x, context_end_frame)[FINAL].float()
        tub = self.token_grid.tubelet
        with torch.no_grad():
            tgt = self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state[0]
        tgt = self._grid(tgt, x.shape[1] // tub)[context_end_frame // tub:].float()
        res = {}
        regions = {"all": torch.ones(pred.shape[:3], dtype=torch.bool, device=pred.device)}
        if object_masks is not None:
            regions["obj"] = torch.as_tensor(np.asarray(object_masks), device=pred.device).bool()
        for tag, m in regions.items():
            if m.sum() == 0:
                continue
            p, t = pred[m], tgt[m]
            res[f"l1_{tag}"] = float((p - t).abs().mean())
            res[f"mse_{tag}"] = float(((p - t) ** 2).mean())
            res[f"cos_{tag}"] = float(1 - torch.nn.functional.cosine_similarity(p, t, dim=-1).mean())
        return res
