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


def is_legal(twin: str, summary: dict | None = None) -> bool:
    """A twin obeys every law: the generator's ``legal`` flag if recorded, else F1's naming (conserve_*)."""
    if summary and "legal" in summary:
        return bool(summary["legal"])
    return twin.startswith(("conserve", "legal"))


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
    """Legal, completed clips of the PC training render set (``<root>/<family>/<group>/<twin>/``) in ``splits``,
    read from each group's ``_GROUP_COMPLETE`` record (split, twins, per-twin summaries).

    Raises if any group's twin group ID belongs to an evaluation set (``exclude_eval``): the PCs must never
    train on the benchmark's evaluation twins (spec §4.4)."""
    from physlaw.config import resolve_path
    banned = eval_group_ids(dcfg.get("exclude_eval") or [])
    out, bad = [], []
    for fam in dcfg["families"]:
        for g in sorted((resolve_path(dcfg["root"]) / fam).glob("*/_GROUP_COMPLETE")):
            rec = json.loads(g.read_text())
            gid = rec.get("group", g.parent.name)
            if gid in banned or g.parent.name in banned:
                bad.append(f"{fam}/{gid}")
                continue
            if rec["split"] not in splits:
                continue
            out += [g.parent / t for t in rec["twins"] if is_legal(t, rec.get("summaries", {}).get(t))
                    and (g.parent / t / "_COMPLETE").exists()]
    if bad:
        raise ValueError(f"{len(bad)} training groups have twin group IDs from an evaluation set "
                         f"(e.g. {sorted(bad)[:5]}); refusing to train on evaluation twins")
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

    Item i: (uint8 [n_frames, H, W, 3] at ``hw`` by ``fit``, number of window frames before the event, i).
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
        return x, int(sum(f < ev for f in fr)), i


# ---------------------------------------------------------------------------------------------- pass check

def pass_check(results_dir, crit, untrained_dir=None) -> dict:
    """PC1 pass criterion (spec §4.4, §5.0.5) on results/<model>/ (a family dir results/<model>/f1 is accepted too).

    Momentum (on ``momentum_family``, F1):
      L1: the held-out-selected S_mom (``selected["<law>|<pool>"].momentum.score`` in l1.json; F1's law key is
          "energy", the collision variants carrying the momentum package) >= ``l1_s_mom_min``, for the first pool of
          ``l1_pools`` that L1 reports (the instrument, pred_real or as_is, is the one L1 used: recorded).
      L2: raw win rate >= ``l2_win_min`` for every momentum contrast at the pre-registered distance, AND the model −
          pixel-baseline win rate (``vs_pixel``, paired CI) above 0 for every pixel baseline (the motion control).
          The legal-vs-legal reference contrast is reported only.
    Energy (``energy``): on at least one of ``energy.entries`` ({family: energy_kink entry}, F2 or F7 bounce), the raw
      L2 kink (log surprise, no adjustment) at the pre-registered ``energy.distance`` has K CI > 0 beyond every baseline (l2.json verdict
      "kink beyond baselines"), AND the untrained model's K CI on the same entry and distance contains 0
      (``untrained_dir`` = results/<untrained model>/; missing -> MISSING).
    Returns {"verdict": PASS | FAIL | MISSING, "l1", "l2", "energy", "notes", "criterion"}."""
    rd = Path(results_dir)
    if (rd / "l1.json").exists() or (rd / "l2.json").exists():      # a family dir was given
        rd = rd.parent
    fam = crit.get("momentum_family", "f1")
    res = {"l1": {}, "l2": {}, "energy": {}, "notes": []}
    ok, missing = True, False
    f1, f2 = rd / fam / "l1.json", rd / fam / "l2.json"
    if f1.exists():
        sel = json.loads(f1.read_text()).get("selected", {})
        l1 = json.loads(f1.read_text())
        sel = l1.get("selected", {})
        opt = next((k for p_ in crit["l1_pools"] for k, v in sel.items()
                    if k.split("|")[-1] == p_ and "score" in v.get("momentum", {})), None)
        if opt is None:
            missing = True
            res["notes"].append(f"l1.json has no selected S_mom for any pool of {list(crit['l1_pools'])}")
        else:
            m = sel[opt]["momentum"]
            inst = l1.get("variants", {}).get(m.get("selected"), {}).get("instrument")
            res["l1"] = {"option": opt, "instrument": inst, "S_mom": m["score"],
                         "q95": [m.get("score_q025"), m.get("score_q975")], "n_variants": m.get("n_variants")}
            ok &= bool(np.isfinite(m["score"]) and m["score"] >= crit["l1_s_mom_min"])
    else:
        missing = True
        res["notes"].append(f"{f1} missing")
    if f2.exists():
        con = json.loads(f2.read_text()).get("contrasts", {})
        for c in list(crit["l2_contrasts"]) + [crit["l2_control"]]:
            src = next((k for k in con.get(c, {}) if k != "pixel_baselines"), None)
            r = con[c][src].get(crit["l2_distance"]) if src else None
            if r is None:
                missing = True
                res["notes"].append(f"l2.json: no {c} / {crit['l2_distance']}")
                continue
            gaps = {b: g["model_minus_baseline_ci95"] for b, g in r.get("vs_pixel", {}).items()}
            res["l2"][c] = {"win_rate": r["win_rate"], "ci95": r.get("win_rate_ci95"), "vs_pixel_ci95": gaps}
            if c == crit["l2_control"]:
                continue
            if not gaps:
                missing = True
                res["notes"].append(f"l2.json: {c} has no pixel-baseline gap (run scripts/05_surprise.py)")
            ok &= bool(r["win_rate"] >= crit["l2_win_min"] and all(ci[0] > 0 for ci in gaps.values()))
    else:
        missing = True
        res["notes"].append(f"{f2} missing")
    en = crit.get("energy")
    if en:
        e_ok, e_missing = energy_check(rd, en, untrained_dir, res)
        ok &= e_ok or e_missing
        missing |= e_missing and not e_ok
    res["verdict"] = "MISSING" if missing else "FAIL" if not ok else "PASS"
    res["criterion"] = dict(crit)
    return res


def _kink_rec(path: Path, entry: str, distance: str) -> dict | None:
    if not path.exists():
        return None
    return (json.loads(path.read_text()).get("energy_kink", {}).get(entry, {}).get("distances", {})
            .get(distance))


def energy_check(rd: Path, en, untrained_dir, res: dict) -> tuple[bool, bool]:
    """(passed on some entry, nothing decidable) for the energy criterion; fills res["energy"] and notes."""
    passed, decidable = False, False
    for fam, entry in dict(en["entries"]).items():
        r = _kink_rec(rd / fam / "l2.json", entry, en["distance"])
        if r is None:
            res["notes"].append(f"{fam}/l2.json: no energy_kink {entry} / {en['distance']}")
            continue
        rec = {"K": r["model"]["K"], "K_ci95": r["model"]["K_ci95"], "verdict": r["verdict"]}
        model_ok = r["verdict"] == "kink beyond baselines"
        u = _kink_rec(Path(untrained_dir) / fam / "l2.json", entry, en["distance"]) if untrained_dir else None
        if u is None:
            res["notes"].append(f"{fam}: untrained energy kink missing (pass --untrained results/<untrained model>)")
        else:
            lo, hi = u["model"]["K_ci95"]
            rec.update(untrained_K=u["model"]["K"], untrained_K_ci95=[lo, hi], untrained_contains_0=bool(lo <= 0 <= hi))
            decidable = True
            passed |= model_ok and rec["untrained_contains_0"]
        res["energy"][f"{fam}/{entry}"] = rec
    return passed, not decidable


def format_pass(res: dict) -> str:
    c = res["criterion"]
    lines = [f"PC1 pass check: {res['verdict']}  (L1 S_mom >= {c['l1_s_mom_min']}, "
             f"L2 win >= {c['l2_win_min']} and > every pixel baseline [{c['l2_distance']}]; energy kink)"]
    if res["l1"]:
        lines.append(f"  L1 S_mom ({res['l1']['option']}, {res['l1']['instrument']}, held-out selected): "
                     f"{res['l1']['S_mom']:+.3f}")
    for k, r in res["l2"].items():
        ci = r["ci95"] or [float("nan")] * 2
        gaps = ", ".join(f"{b} [{g[0]:+.2f}, {g[1]:+.2f}]" for b, g in r["vs_pixel_ci95"].items())
        lines.append(f"  L2 {k}: {r['win_rate']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]; model − pixel: {gaps}")
    for k, r in res.get("energy", {}).items():
        u = (f"; untrained K {r['untrained_K']:+.2f} [{r['untrained_K_ci95'][0]:+.2f}, {r['untrained_K_ci95'][1]:+.2f}]"
             if "untrained_K" in r else "; untrained missing")
        lines.append(f"  energy kink {k}: K {r['K']:+.2f} [{r['K_ci95'][0]:+.2f}, {r['K_ci95'][1]:+.2f}] ({r['verdict']}){u}")
    lines += [f"  note: {n}" for n in res["notes"]]
    return "\n".join(lines)
