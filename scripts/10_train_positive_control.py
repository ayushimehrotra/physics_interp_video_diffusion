#!/usr/bin/env python3
"""Stage 10 - train a positive control (spec §4.4) on the PC training render set; the config selects which.

    HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/10_train_positive_control.py \\
        --config configs/models/positive_control.yaml           # PC1: ViT-S V-JEPA-style model
    HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/10_train_positive_control.py \\
        --config configs/models/pc2_wan22_ti2v_5b.yaml          # PC2: Wan2.2-TI2V-5B rank-32 LoRA
    /root/envs/physics_venv/bin/python scripts/10_train_positive_control.py \\
        --config configs/models/positive_control.yaml --check results/pc1-vits/f1   # PC1 pass/fail

Output dir: the config's ``local_path`` (or --out):
  PC1  pc1.pt          eval weights (EMA target encoder + predictor; physlaw.models.positive_control.PC1Adapter)
       train_state.pt  online model, EMA encoder, optimizer, step (resume)
  PC2  lora/           adapter_model.safetensors + adapter_config.json (PEFT format; the Wan adapter merges
                       it via ``lora: {path, scale}``), lora_step<N>/ snapshots every save_every steps
       train_state.pt  LoRA weights, optimizer, step (resume); prompt_embeds.pt (cached text embedding)
  both curves.json (train loss; PC1 also the held-out prediction error), curves.png, timing in curves.json
Resumable: an existing train_state.pt is resumed; --force starts over. --steps overrides train.steps
(e.g. a short test). The data loader refuses clips whose twin group ID is in an evaluation set.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import config_hash, load_config  # noqa: E402
from physlaw.models import positive_control as PC  # noqa: E402


def lr_at(step: int, tr) -> float:
    """Linear warmup, then cosine to final_lr."""
    if step < tr["warmup"]:
        return tr["lr"] * (step + 1) / tr["warmup"]
    q = (step - tr["warmup"]) / max(1, tr["steps"] - tr["warmup"])
    return tr["final_lr"] + 0.5 * (tr["lr"] - tr["final_lr"]) * (1 + math.cos(math.pi * min(q, 1.0)))


def batches(ds, tr, seed):
    """Endless shuffled micro-batches (uint8 [B, T, H, W, 3], frames before the event)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=tr["micro_batch"], shuffle=True, drop_last=True, generator=g,
                                     num_workers=tr["num_workers"], persistent_workers=tr["num_workers"] > 0)
    if len(dl) == 0:
        raise ValueError(f"{len(ds)} training clips < micro_batch {tr['micro_batch']}")
    while True:
        yield from dl


def atomic_save(obj, path: Path):
    import torch
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_curves(out: Path, curves: dict):
    """curves.json + curves.png: training loss | held-out error (small multiples, one y-scale each)."""
    (out / "curves.json").write_text(json.dumps(curves, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, c1, c2 = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
    panels = [("train", "training loss", [("loss", "loss", c1, "-")])]
    if curves.get("holdout"):
        panels.append(("holdout", "held-out prediction error (L1, future tokens)",
                       [("l1_all", "predictor", c1, "-"), ("copy_last", "copy last context frame", c2, "--")]))
    fig, axs = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), facecolor="#fcfcfb", squeeze=False)
    for ax, (key, title, series) in zip(axs[0], panels):
        rows = curves[key]
        ax.set_facecolor("#fcfcfb")
        for k, lab, col, ls in series:
            ax.plot([r["step"] for r in rows], [r[k] for r in rows], color=col, lw=2, ls=ls, label=lab)
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("step", color=ink2)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(colors=ink2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=ink2)
    head = f"{curves['model']} ({curves['config_hash']})"
    if curves.get("code_path_test"):
        head += "  CODE-PATH TEST on stand-in data: not a PC result"
    fig.suptitle(head, color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)


def load_curves(out: Path, cfg, step: int) -> dict:
    base = {"model": cfg.name, "config_hash": config_hash(cfg), "code_path_test": bool(cfg["data"].get("code_path_test")),
            "data_root": str(cfg["data"]["root"]), "families": list(cfg["data"]["families"]), "train": [], "holdout": []}
    f = out / "curves.json"
    if step and f.exists():
        old = json.loads(f.read_text())
        for k in ("train", "holdout"):
            base[k] = [r for r in old.get(k, []) if r["step"] <= step]
        base["timing"] = old.get("timing", {})
    return base


# ============================================================================================== PC1

def train_pc1(cfg, out: Path, a):
    import torch
    tr, dc, inp = cfg["train"], cfg["data"], cfg["input"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R, tub = int(inp["resolution"]), int(cfg["token_grid"]["tubelet"])
    grid, T_tok = R // int(cfg["token_grid"]["patch"]), int(inp["num_frames"]) // tub
    win = dict(fps=inp["fps"], n_frames=inp["num_frames"], hw=(R, R), fit=inp.get("fit", "squash"), center=True)
    train_paths = PC.training_clips(dc, dc["train_splits"])
    hold_paths = PC.training_clips(dc, dc["holdout_splits"])[: dc["n_holdout"]]
    print(f"[10] PC1 {cfg.name}: {len(train_paths)} train clips, {len(hold_paths)} held-out, out={out}", flush=True)
    if dc.get("code_path_test"):
        print("[10] CODE-PATH TEST: stand-in data, not a positive-control result", flush=True)
    train_ds = PC.ClipWindows(train_paths, jitter=dc["jitter"], flip=dc["flip"], **win)
    hold_ds = PC.ClipWindows(hold_paths, **win)

    model = PC.build_model(cfg, cfg["arch"]["init_seed"]).to(dev).train()
    ema = PC.build_model(cfg, cfg["arch"]["init_seed"])
    ema.predictor = model.predictor                     # shared: ema = the eval model (EMA encoder + predictor)
    ema.encoder.load_state_dict(model.encoder.state_dict())
    ema = ema.to(dev)
    ema.encoder.requires_grad_(False).eval()
    if tr["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": tr["weight_decay"]},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                            lr=tr["lr"], betas=tuple(tr["betas"]))
    step, state_f = 0, out / "train_state.pt"
    if state_f.exists() and not a.force:
        st = torch.load(state_f, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        ema.encoder.load_state_dict(st["ema_encoder"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"[10] resumed at step {step}", flush=True)
    curves = load_curves(out, cfg, step)
    n_params = {k: sum(p.numel() for p in getattr(model, k).parameters()) for k in ("encoder", "predictor")}
    curves["n_params"] = n_params
    accum = tr["batch_size"] // tr["micro_batch"]
    rng = np.random.default_rng(tr["seed"] + step)
    it = batches(train_ds, tr, tr["seed"] + step)
    mean = torch.tensor(PC.IMAGENET_MEAN, device=dev).view(1, 1, 3, 1, 1)
    std = torch.tensor(PC.IMAGENET_STD, device=dev).view(1, 1, 3, 1, 1)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    def heldout():
        ad = PC.PC1Adapter(cfg, ema, PC.IMAGENET_MEAN, PC.IMAGENET_STD, dev)
        ad.layers = []                                   # encode(): FINAL only
        model.eval()
        rows = []
        for i in range(len(hold_ds)):
            frames, kin = hold_ds[i]
            k = kin - kin % tub
            x = ad.to_tensor(frames)
            s = ad.surprise(x, k)
            tgt = ad.encode(x)[-1].float()
            s["copy_last"] = float((tgt[k // tub:] - tgt[k // tub - 1]).abs().mean())
            s["target_std"] = float(tgt.reshape(-1, tgt.shape[-1]).std(0).mean())   # collapse watch: -> 0 if collapsed
            rows.append(s)
        model.train()
        return {k: float(np.mean([r[k] for r in rows])) for k in ("l1_all", "mse_all", "cos_all", "copy_last", "target_std")}

    def save():
        atomic_save({"model": model.state_dict(), "ema_encoder": ema.encoder.state_dict(), "opt": opt.state_dict(),
                     "step": step}, state_f)
        atomic_save({"config": model.config.to_json_string(), "eval_model": ema.state_dict(), "step": step,
                     "config_hash": config_hash(cfg)}, out / "pc1.pt")
        write_curves(out, curves)

    steps = a.steps or tr["steps"]
    t0, t_data, n_done = time.time(), 0.0, 0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while step < steps:
        lr = lr_at(step, tr)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for _ in range(accum):
            td = time.time()
            frames, _ = next(it)
            t_data += time.time() - td
            x = ((frames.to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255.0) - mean) / std
            ctx, tgt = (torch.from_numpy(m).to(dev) for m in PC.sample_mask(rng, T_tok, grid, cfg["mask"]))
            with torch.no_grad(), torch.autocast(**amp):
                h = ema.encoder(x).last_hidden_state[:, tgt]
            with torch.autocast(**amp):
                p = PC.predict(model, x, ctx, tgt)
            loss = (p.float() - h.float()).abs().mean() / accum
            loss.backward()
            tot += float(loss.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        m = tr["ema"][0] + (tr["ema"][1] - tr["ema"][0]) * step / tr["steps"]
        with torch.no_grad():
            ep, op = list(ema.encoder.parameters()), list(model.encoder.parameters())
            torch._foreach_mul_(ep, m)
            torch._foreach_add_(ep, op, alpha=1 - m)
        step += 1
        n_done += 1
        if not math.isfinite(tot):
            raise FloatingPointError(f"loss {tot} at step {step}")
        if step % tr["log_every"] == 0 or step == steps:
            el = time.time() - t0
            curves["train"].append({"step": step, "loss": tot, "lr": lr, "ema": m, "grad_norm": gn})
            curves["timing"] = {"s_per_step": el / n_done, "data_wait_s_per_step": t_data / n_done,
                                "batch_size": tr["batch_size"], "device": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
                                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None}
            print(f"[10] step {step}/{steps} loss {tot:.4f} lr {lr:.2e} gn {gn:.2f} "
                  f"{el / n_done:.2f} s/step (data wait {t_data / n_done:.2f})", flush=True)
        if step % tr["eval_every"] == 0 or step == steps:
            h = heldout()
            curves["holdout"].append({"step": step, **h})
            print(f"[10] held-out ({len(hold_ds)} clips): L1 {h['l1_all']:.4f} vs copy-last {h['copy_last']:.4f}, "
                  f"target sd {h['target_std']:.4f}", flush=True)
        if step % tr["save_every"] == 0 or step == steps:
            save()
    print(f"[10] PC1 done at step {step} -> {out}", flush=True)


# ============================================================================================== PC2

def train_pc2(cfg, out: Path, a):
    """RigidBench's recipe (arXiv 2608.15555, Wan2.2-TI2V-5B): rank-32 LoRA (alpha 32) on q/k/v/o and the
    FFN of every block; first-frame pinning (first latent frame clean, timestep 0, excluded from the loss, as
    the TI2V pipeline's expand_timesteps conditioning); flow matching with timesteps uniform over the official
    scheduler's 1,000 shifted training sigmas; fp32 MSE on the velocity (noise - x0)."""
    import torch
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    tr, dc, pt = cfg["train"], cfg["data"], cfg["pc2_train"]
    dev, src, bf16 = "cuda", cfg["local_path"], torch.bfloat16
    H, W = int(pt["height"]), int(pt["width"])
    train_paths = PC.training_clips(dc, dc["train_splits"])
    print(f"[10] PC2 {cfg.name}: {len(train_paths)} train clips, {pt['num_frames']} frames {W}x{H}, out={out}", flush=True)
    if dc.get("code_path_test"):
        print("[10] CODE-PATH TEST: stand-in data, not a positive-control result", flush=True)
    ds = PC.ClipWindows(train_paths, fps=pt["fps"], n_frames=pt["num_frames"], hw=(H, W), center=True,
                        jitter=dc["jitter"], flip=dc["flip"])

    pe_f = out / "prompt_embeds.pt"
    if not pe_f.exists():
        from transformers import AutoTokenizer, UMT5EncoderModel
        te = UMT5EncoderModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=bf16, device_map=dev)
        pipe = WanPipeline(tokenizer=AutoTokenizer.from_pretrained(src, subfolder="tokenizer"), text_encoder=te,
                           vae=None, scheduler=None, transformer=None)
        with torch.no_grad():
            pe, _ = pipe.encode_prompt(cfg["prompt"], do_classifier_free_guidance=False, max_sequence_length=512,
                                       device=dev)          # 512: WanPipeline.__call__ default
        atomic_save({"prompt": cfg["prompt"], "embeds": pe.cpu()}, pe_f)
        del te, pipe
        torch.cuda.empty_cache()
    pe = torch.load(pe_f, weights_only=True)
    assert pe["prompt"] == cfg["prompt"], "prompt changed: delete prompt_embeds.pt"
    pe = pe["embeds"].to(dev, bf16)

    vae = AutoencoderKLWan.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32, device_map=dev).eval()
    vae.requires_grad_(False)
    l_mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    l_istd = 1.0 / torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    sch = UniPCMultistepScheduler.from_pretrained(src, subfolder="scheduler")
    shift, n_train_t = float(sch.config.flow_shift), int(sch.config.num_train_timesteps)
    tf = WanTransformer3DModel.from_pretrained(src, subfolder="transformer", torch_dtype=bf16, device_map=dev)
    tf.requires_grad_(False)
    if tr["grad_checkpointing"]:
        tf.enable_gradient_checkpointing()
    tf.add_adapter(LoraConfig(r=pt["lora_rank"], lora_alpha=pt["lora_alpha"], lora_dropout=0.0,
                              init_lora_weights=True, target_modules=list(pt["lora_targets"])))
    params = [p for p in tf.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()                          # fp32 LoRA weights under bf16 autocast
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"], betas=tuple(tr["betas"]))
    step, state_f = 0, out / "train_state.pt"
    if state_f.exists() and not a.force:
        st = torch.load(state_f, map_location=dev, weights_only=False)
        set_peft_model_state_dict(tf, st["lora"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"[10] resumed at step {step}", flush=True)
    curves = load_curves(out, cfg, step)
    curves["n_trainable"] = sum(p.numel() for p in params)
    accum = tr["batch_size"] // tr["micro_batch"]
    gen = torch.Generator(device=dev).manual_seed(tr["seed"] + step)
    it = batches(ds, tr, tr["seed"] + step)

    def save_lora(d: Path):
        """PEFT format: adapter_model.safetensors + adapter_config.json (r, lora_alpha, targets)."""
        from safetensors.torch import save_file
        d.mkdir(parents=True, exist_ok=True)
        sd = {f"base_model.model.{k}": v.to(bf16).contiguous() for k, v in get_peft_model_state_dict(tf).items()}
        save_file(sd, d / "adapter_model.safetensors.tmp")
        os.replace(d / "adapter_model.safetensors.tmp", d / "adapter_model.safetensors")
        tf.peft_config["default"].save_pretrained(d)

    steps = a.steps or tr["steps"]
    t0, t_data, n_done = time.time(), 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    tf.train()
    while step < steps:
        lr = lr_at(step, tr)
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for _ in range(accum):
            td = time.time()
            frames, _ = next(it)
            t_data += time.time() - td
            v = frames.to(dev).permute(0, 4, 1, 2, 3).float() / 127.5 - 1.0          # [B, 3, T, H, W] in [-1, 1]
            with torch.no_grad():
                x0 = (vae.encode(v).latent_dist.sample(gen) - l_mean) * l_istd
            B, _, F, h, w = x0.shape
            u = (torch.randint(0, n_train_t, (B,), device=dev, generator=gen) + 1).float() / n_train_t
            sig = (shift * u / (1 + (shift - 1) * u)).view(B, 1, 1, 1, 1)           # official flow shift
            eps = torch.randn(x0.shape, device=dev, generator=gen)
            xt = (1 - sig) * x0 + sig * eps
            xt[:, :, 0] = x0[:, :, 0]                                               # first-frame pinning
            p = tf.config.patch_size
            ts = (sig.view(B, 1, 1, 1) * n_train_t).expand(B, F, h // p[1], w // p[2]).clone()
            ts[:, 0] = 0
            with torch.autocast("cuda", dtype=bf16):
                pred = tf(hidden_states=xt.to(bf16), timestep=ts.flatten(1), encoder_hidden_states=pe.expand(B, -1, -1),
                          return_dict=False)[0]
            loss = ((pred.float() - (eps - x0))[:, :, 1:] ** 2).mean() / accum
            loss.backward()
            tot += float(loss.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        n_done += 1
        if not math.isfinite(tot):
            raise FloatingPointError(f"loss {tot} at step {step}")
        if step % tr["log_every"] == 0 or step == steps:
            el = time.time() - t0
            curves["train"].append({"step": step, "loss": tot, "lr": lr, "grad_norm": gn})
            curves["timing"] = {"s_per_step": el / n_done, "data_wait_s_per_step": t_data / n_done,
                                "batch_size": tr["batch_size"], "micro_batch": tr["micro_batch"],
                                "frames": pt["num_frames"], "hw": [H, W], "device": torch.cuda.get_device_name(),
                                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(f"[10] step {step}/{steps} loss {tot:.4f} lr {lr:.2e} gn {gn:.3f} {el / n_done:.1f} s/step "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if step % tr["save_every"] == 0 or step == steps:
            save_lora(out / "lora")
            if step % tr["save_every"] == 0:
                save_lora(out / f"lora_step{step}")
            atomic_save({"lora": get_peft_model_state_dict(tf), "opt": opt.state_dict(), "step": step}, state_f)
            write_curves(out, curves)
    print(f"[10] PC2 done at step {step} -> {out / 'lora'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default: the config's local_path")
    ap.add_argument("--steps", type=int, default=None, help="stop at this step (default train.steps)")
    ap.add_argument("--check", default=None, metavar="RESULTS_DIR",
                    help="PC1: print pass/fail of results/<model>/f1/{l1,l2}.json against pass_criterion; no training")
    ap.add_argument("--force", action="store_true", help="start over even if train_state.pt exists")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.check:
        res = PC.pass_check(a.check, cfg["pass_criterion"])
        print(PC.format_pass(res))
        sys.exit(0 if res["verdict"] == "PASS" else 1)
    out = Path(a.out or cfg["local_path"])
    out.mkdir(parents=True, exist_ok=True)
    which = cfg.get("positive_control")
    print(f"[10] {which} {cfg.name} hash={config_hash(cfg)}", flush=True)
    if which == "pc1":
        train_pc1(cfg, out, a)
    elif which == "pc2":
        train_pc2(cfg, out, a)
    else:
        sys.exit(f"10_train_positive_control.py: config sets positive_control={which!r}; expected pc1 or pc2")


if __name__ == "__main__":
    main()
