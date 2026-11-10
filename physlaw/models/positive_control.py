"""Positive controls (spec §4.4): PC1 model + adapter, the PC training-set loader, and the PC1 pass check.

PC1 is a V-JEPA-style encoder + predictor at ViT-S scale, built from transformers' ``VJEPA2Model`` with a
small config (``arch`` in configs/models/positive_control.yaml). Input format, tubelet (2), patch (16) and
RoPE positions are those of V-JEPA 2 ViT-L, so ``PC1Adapter`` IS ``VJEPA2Adapter`` (encode / forecast /
surprise with context-only encoding) with weights from our checkpoint, and tokens/project.py, pooling and
stages 04-06 run on it unchanged.

Training (scripts/10_train_positive_control.py): masked latent prediction against an EMA target encoder
(L1 loss on the target encoder's final layer-normed tokens, as V-JEPA 2). Each micro-batch draws one mask:
with probability ``p_future`` a FUTURE-FRAME mask (context = token-frames [0, k), target = every later
token, k random), otherwise a V-JEPA spatial tube multi-block mask. The context encoder runs on the
context tokens alone with their true RoPE positions, which for a future mask is exactly what forecast()
does at evaluation (encode the pre-event frames alone, then predict the rest).

PC2 (Wan2.2-TI2V-5B, rank-32 LoRA) has no module of its own: training is in stage 10, evaluation goes
through the Wan2.2 adapter with the config key ``lora: {path, scale}``. Both PCs read the training render
set through :func:`training_clips` / :class:`ClipWindows`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physlaw.models.base import fit_frames, sample_frames
from physlaw.models.vjepa2 import VJEPA2Adapter

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # V-JEPA 2's statistics


# ---------------------------------------------------------------------------------------------- model

def vjepa2_config(cfg):
    """transformers VJEPA2Config for PC1: ``arch`` sizes, V-JEPA 2 ViT-L's input geometry."""
    from transformers import VJEPA2Config
    a, inp, tg = cfg["arch"], cfg["input"], cfg["token_grid"]
    conf = VJEPA2Config(
        hidden_size=a["dim"], num_hidden_layers=a["depth"], num_attention_heads=a["heads"], mlp_ratio=4.0,
        pred_hidden_size=a["pred_dim"], pred_num_hidden_layers=a["pred_depth"],
        pred_num_attention_heads=a["pred_heads"], pred_num_mask_tokens=10, pred_zero_init_mask_tokens=True,
        crop_size=int(inp["resolution"]), image_size=int(inp["resolution"]), frames_per_clip=int(inp["num_frames"]),
        patch_size=int(tg["patch"]), tubelet_size=int(tg["tubelet"]), drop_path_rate=0.0, qkv_bias=True,
        initializer_range=0.02, layer_norm_eps=1e-6)
    conf._attn_implementation = "sdpa"
    return conf


def build_model(cfg, seed: int = 0):
    import torch
    from transformers import VJEPA2Model
    torch.manual_seed(seed)
    return VJEPA2Model(vjepa2_config(cfg))


class PC1Adapter(VJEPA2Adapter):
    """VJEPA2Adapter on the PC1 checkpoint: encoder = the EMA target encoder, predictor = the trained one."""

    @classmethod
    def load(cls, cfg, device: str = "cuda") -> "PC1Adapter":
        import torch
        from transformers import VJEPA2Config, VJEPA2Model
        ck = torch.load(Path(cfg["local_path"]) / "pc1.pt", map_location="cpu", weights_only=True)
        conf = VJEPA2Config.from_dict(json.loads(ck["config"]))
        conf._attn_implementation = "sdpa"
        model = VJEPA2Model(conf)
        model.load_state_dict(ck["eval_model"])
        model = model.to(getattr(torch, cfg.get("dtype", "float32"))).to(device).eval()
        return cls(cfg, model, IMAGENET_MEAN, IMAGENET_STD, device)


def encode_context(encoder, x, idx):
    """Encoder output for the tokens ``idx`` alone (int64 [n]), at their true RoPE positions."""
    h = encoder.embeddings(x)[:, idx]
    pos = idx[None].expand(h.shape[0], -1)
    for layer in encoder.layer:
        h = layer(h, pos)[0]
    return encoder.layernorm(h)


def predict(model, x, ctx, tgt):
    """Predictor output at target tokens ``tgt`` from the online encoder on context tokens ``ctx``."""
    z = encode_context(model.encoder, x, ctx)
    c = model.config
    B, N = x.shape[0], x.shape[1] // c.tubelet_size * (x.shape[-1] // c.patch_size) ** 2
    full = z.new_zeros(B, N, z.shape[-1])
    full[:, ctx] = z
    return model.predictor(encoder_hidden_states=full, context_mask=[ctx[None].expand(B, -1)],
                           target_mask=[tgt[None].expand(B, -1)]).last_hidden_state


def sample_mask(rng, n_tok_frames: int, grid: int, mcfg) -> tuple[np.ndarray, np.ndarray]:
    """(context, target) token indices for one training step (spec §4.4: future-frame masking).

    future (prob ``p_future``): context = token-frames [0, k), k ~ U[future_ctx]; target = the rest.
    tube   (otherwise): V-JEPA multi-block mask, ``n`` rectangles of ``scale`` x grid area (aspect in
           ``aspect``) masked in every token-frame; target = masked tokens, context = the rest."""
    per = grid * grid
    N = n_tok_frames * per
    if rng.random() < float(mcfg["p_future"]):
        lo, hi = mcfg["future_ctx"]
        k = int(rng.integers(lo, min(hi, n_tok_frames - 1) + 1))
        return np.arange(k * per), np.arange(k * per, N)
    n, scale = mcfg["tube_blocks"][int(rng.integers(len(mcfg["tube_blocks"])))]
    m = np.zeros((grid, grid), bool)
    for _ in range(int(n)):
        ar = np.exp(rng.uniform(*np.log(mcfg["aspect"])))
        h = int(np.clip(round(np.sqrt(scale * per * ar)), 1, grid))
        w = int(np.clip(round(np.sqrt(scale * per / ar)), 1, grid))
        y, x = rng.integers(0, grid - h + 1), rng.integers(0, grid - w + 1)
        m[y:y + h, x:x + w] = True
    if m.all():
        m[rng.integers(grid), :] = False
    m = np.broadcast_to(m.reshape(-1), (n_tok_frames, per)).reshape(-1)
    return np.flatnonzero(~m), np.flatnonzero(m)


# ---------------------------------------------------------------------------------------------- data

def event_frame(pl: dict) -> int:
    """First post-event master frame (F1: first post-contact frame; other families: event_frame)."""
    return int(pl["first_post_contact_frame"] if "first_post_contact_frame" in pl else pl["event_frame"])


def is_legal(pl: dict) -> bool:
    """A clip obeys every law: ``physlaw.legal`` if the family writes it, else F1's twin naming (conserve_*)."""
    if "legal" in pl:
        return bool(pl["legal"])
    return pl["twin"].startswith(("conserve", "legal"))


def eval_group_ids(dirs) -> set[str]:
    """Twin group IDs of the evaluation sets (every group directory under each family dir)."""
    from physlaw.config import resolve_path
    out = set()
    for d in dirs:
        d = resolve_path(d)
        if d.is_dir():
            out |= {p.name for p in d.iterdir() if p.is_dir()}
    return out


def training_clips(dcfg, splits) -> list[Path]:
    """Legal clips of the PC training render set (``<root>/<family>/<group>/<twin>/``) in ``splits``.

    Raises if any clip's twin group ID belongs to an evaluation set (``exclude_eval``): the PCs must never
    train on the benchmark's evaluation twins (spec §4.4)."""
    from physlaw.config import resolve_path
    from physlaw.data.io import list_clips
    banned = eval_group_ids(dcfg.get("exclude_eval") or [])
    out, bad = [], []
    for fam in dcfg["families"]:
        for p in list_clips(resolve_path(dcfg["root"]) / fam):
            pl = json.loads((p / "labels.json").read_text())["physlaw"]
            if pl["twin_group_id"] in banned:
                bad.append(f"{fam}/{pl['twin_group_id']}")
            elif is_legal(pl) and pl["split"] in splits:
                out.append(p)
    if bad:
        raise ValueError(f"{len(bad)} training clips have twin group IDs from an evaluation set "
                         f"(e.g. {sorted(set(bad))[:5]}); refusing to train on evaluation twins")
    return out


def read_frames(path: Path, idx: list[int]) -> np.ndarray:
    """uint8 [len(idx), H, W, 3]: only the requested master frames are kept while decoding."""
    import imageio.v3 as iio
    want, last, out = set(idx), max(idx), {}
    for i, f in enumerate(iio.imiter(path / "video.mp4", plugin="FFMPEG")):
        if i in want:
            out[i] = f
        if i >= last:
            break
    return np.stack([out[i] for i in idx])


class ClipWindows:
    """Map-style dataset (a torch DataLoader accepts it) of model-input windows of training clips.

    Item i: (uint8 [n_frames, H, W, 3] at ``hw`` by ``fit``, number of window frames before the event).
    The window is ``n_frames`` at ``fps``, from frame 0 or (``center``) centred on the event shifted by up to
    ±``jitter`` master frames (clamped to the clip); ``flip`` mirrors
    horizontally with p = 0.5 (both leave the physics legal). jitter 0 and no flip: deterministic."""

    def __init__(self, paths, *, fps, n_frames, hw, fit="squash", center=True, jitter=0, flip=False):
        self.paths, self.fps, self.n, self.hw, self.fit = list(paths), int(fps), int(n_frames), tuple(hw), fit
        self.center, self.jitter, self.flip = center, int(jitter), flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        lab = json.loads((p / "labels.json").read_text())
        sc, ev = lab["scene_constants"], event_frame(lab["physlaw"])
        rng = np.random.default_rng() if (self.jitter or self.flip) else None
        shift = int(rng.integers(-self.jitter, self.jitter + 1)) if self.jitter else 0
        fr = sample_frames(master_fps=sc["frame_rate"], n_master=sc["num_frames"], fps=self.fps, n_frames=self.n,
                           contact_frame=ev + shift, center_on_contact=self.center)
        x = fit_frames(read_frames(p, fr), self.hw, self.fit)
        if self.flip and rng.random() < 0.5:
            x = np.ascontiguousarray(x[:, :, ::-1])
        return x, int(sum(f < ev for f in fr))


# ---------------------------------------------------------------------------------------------- pass check

def _find(d, keys: tuple[str, ...], path=()):
    """(path, value) for every number in nested dict ``d`` whose key path ends with ``keys``."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _find(v, keys, path + (k,))
    elif isinstance(d, (int, float)) and path[-len(keys):] == keys:
        yield path, float(d)


def pass_check(results_dir, crit) -> dict:
    """PC1 pass criterion (spec §4.4; values accepted by the user 2026-11-10) on results/<model>/f1/.

    L1: S_family >= ``l1_s_family_min`` for every legal twin in the primary L1 variant ``l1_variant``
        (read from ``variants.<variant>.twins.<twin>.S.family`` or any ``S_family`` key under that variant).
    L2: motion-adjusted win rate >= ``l2_adj_win_min`` for every momentum contrast, under the pre-registered
        covariate set and distance (``motion_adjusted.contrasts.<cov>.<contrast>.<distance>.adj_win_rate``).
        The legal-vs-legal control is reported; if its CI excludes 0.5 the adjustment is insufficient
        (spec §5.3 item 4) and the verdict is INCONCLUSIVE rather than PASS.
    Returns {"verdict": PASS | FAIL | INCONCLUSIVE | MISSING, "l1": ..., "l2": ..., "notes": [...]}."""
    rd = Path(results_dir)
    res = {"l1": {}, "l2": {}, "notes": []}
    ok, missing = True, False
    f1, f2 = rd / "l1.json", rd / "l2.json"
    if f1.exists():
        var = json.loads(f1.read_text()).get("variants", {}).get(crit["l1_variant"], {})
        vals = [(p, v) for p, v in list(_find(var, ("S", "family"))) + list(_find(var, ("S_family",)))
                if any(t in p for t in crit["l1_twins"])]
        for p, v in vals:
            t = next(t for t in crit["l1_twins"] if t in p)
            res["l1"][t] = v
            ok &= bool(np.isfinite(v) and v >= crit["l1_s_family_min"])
        if not vals:
            missing = True
            res["notes"].append(f"no S_family for {crit['l1_twins']} in l1.json variant '{crit['l1_variant']}'")
    else:
        missing = True
        res["notes"].append(f"{f1} missing")
    if f2.exists():
        adj = json.loads(f2.read_text()).get("motion_adjusted", {}).get("contrasts", {}).get(crit["l2_covariates"], {})
        for c in list(crit["l2_contrasts"]) + [crit["l2_control"]]:
            r = adj.get(c, {}).get(crit["l2_distance"])
            if r is None:
                missing = True
                res["notes"].append(f"l2.json: no {crit['l2_covariates']} / {c} / {crit['l2_distance']}")
                continue
            res["l2"][c] = {"adj_win_rate": r["adj_win_rate"], "ci95": r.get("adj_win_rate_ci95")}
            if c != crit["l2_control"]:
                ok &= bool(r["adj_win_rate"] >= crit["l2_adj_win_min"])
            elif r.get("adj_win_rate_ci95") and not (r["adj_win_rate_ci95"][0] <= 0.5 <= r["adj_win_rate_ci95"][1]):
                res["notes"].append("legal-vs-legal control not at chance after adjustment (spec §5.3.4)")
                res["control_off_chance"] = True
    else:
        missing = True
        res["notes"].append(f"{f2} missing")
    res["verdict"] = ("MISSING" if missing else "FAIL" if not ok
                      else "INCONCLUSIVE" if res.get("control_off_chance") else "PASS")
    res["criterion"] = dict(crit)
    return res


def format_pass(res: dict) -> str:
    c = res["criterion"]
    lines = [f"PC1 pass check: {res['verdict']}  (L1 S_family >= {c['l1_s_family_min']} [{c['l1_variant']}], "
             f"L2 adjusted win >= {c['l2_adj_win_min']} [{c['l2_covariates']}, {c['l2_distance']}])"]
    lines += [f"  L1 S_family {t}: {v:+.3f}" for t, v in res["l1"].items()]
    for k, r in res["l2"].items():
        ci = r["ci95"] or [float("nan")] * 2
        lines.append(f"  L2 {k}: {r['adj_win_rate']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    lines += [f"  note: {n}" for n in res["notes"]]
    return "\n".join(lines)
