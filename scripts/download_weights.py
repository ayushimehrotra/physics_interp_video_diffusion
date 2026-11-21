#!/usr/bin/env python3
"""Download every model's pinned weights into $PHYSLAW_MODELS, exactly where configs/models/*.yaml expect them.

    PHYSLAW_MODELS=/path/to/models /root/envs/physics_venv/bin/python scripts/download_weights.py            # all
    ... scripts/download_weights.py --models physrvg,vjepa2_vitl                                            # subset
    ... scripts/download_weights.py --dry-run                                                                # list only

Each config pins ``hf_id`` + ``revision`` (whole snapshot -> ``local_path``) and, for LoRA / DiT / checkpoint blocks,
one file (-> its ``path``); stage 04 / 07 verify the sha256 pins before loading. Resumable: rerun to finish.
Gated: nvidia/Cosmos-Predict2.5-2B and nvidia/Cosmos-1.0-Guardrail (accept both licences on huggingface.co, then
``hf auth login`` or HF_TOKEN). A failed model is reported and the rest continue. PC1 / PC2 have no hub weights
(trained with scripts/10_train_positive_control.py) and are skipped.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config  # noqa: E402

# Public models the Cosmos guardrail (cosmos-guardrail 0.3.2) loads from the HF cache at run time.
COSMOS_GUARDRAIL = ["nvidia/Cosmos-1.0-Guardrail", "Qwen/Qwen3Guard-Gen-0.6B", "google/siglip-so400m-patch14-384"]


def file_block(hf, b: dict, dry: bool):
    """One file of a repo -> b['path'] (repo file = b['file'], else the longest suffix of path that is in the repo)."""
    dst = Path(b["path"])
    if dst.exists():
        return f"ok {dst}"
    rf = b.get("file")
    if not rf:
        files = hf.list_repo_files(b["hf_id"], revision=b.get("revision"))
        parts = dst.parts
        rf = next((f for i in range(len(parts)) for f in files if f == "/".join(parts[i:])), None)
        if rf is None:
            raise FileNotFoundError(f"no file of {b['hf_id']} matches {dst}")
    if dry:
        return f"get {b['hf_id']}:{rf} -> {dst}"
    from huggingface_hub import hf_hub_download
    src = Path(hf_hub_download(b["hf_id"], rf, revision=b.get("revision")))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    # LoRA: PEFT reads adapter_config.json beside the weights
    side = str(Path(rf).parent / "adapter_config.json").lstrip("./")
    if rf.endswith("adapter_model.safetensors"):
        shutil.copyfile(hf_hub_download(b["hf_id"], side, revision=b.get("revision")), dst.parent / "adapter_config.json")
    return f"got {dst}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list of config stems (default: every configs/models/*.yaml)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("PHYSLAW_MODELS", "/workspace/models")
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    hf = HfApi()
    stems = a.models.split(",") if a.models else sorted(p.stem for p in (REPO / "configs/models").glob("*.yaml"))
    failed = []
    for stem in stems:
        cfg = load_config(REPO / f"configs/models/{stem}.yaml")
        if cfg.get("positive_control"):
            print(f"[skip] {stem}: trained here (stage 10), no hub weights"); continue
        try:
            if cfg.get("hf_id") and cfg.get("local_path"):
                lp = Path(cfg["local_path"])
                if a.dry_run:
                    print(f"[{stem}] snapshot {cfg['hf_id']}@{cfg.get('revision')} -> {lp}")
                else:
                    snapshot_download(cfg["hf_id"], revision=cfg.get("revision"), local_dir=lp)
                    print(f"[{stem}] ok {lp}")
            te = cfg.get("text_encoder") or {}
            if te.get("path") and not Path(te["path"]).exists():
                print(f"[{stem}] note: text encoder from {te['path']} (another model's snapshot; download that model)")
            for blk in ("checkpoint", "dit_weights", "lora"):
                b = cfg.get(blk) or {}
                if b.get("hf_id") and b.get("path"):
                    print(f"[{stem}] {blk}: {file_block(hf, b, a.dry_run)}")
            if stem.startswith("cosmos"):
                for r in COSMOS_GUARDRAIL:
                    if a.dry_run:
                        print(f"[{stem}] guardrail {r} -> HF cache"); continue
                    p = Path(snapshot_download(r))
                    if r.endswith("Cosmos-1.0-Guardrail"):
                        # NLTK refuses data in a group/world-writable directory (e.g. network filesystems):
                        # keep a private copy in ~/nltk_data, which NLTK always searches.
                        dst = Path.home() / "nltk_data"
                        if not (dst / "tokenizers" / "punkt_tab").exists():
                            shutil.copytree(p / "blocklist" / "nltk_data", dst, dirs_exist_ok=True)
                            os.system(f"chmod -R go-w {dst}")
                    print(f"[{stem}] guardrail ok {r}")
        except Exception as e:  # keep going: one gated / missing model must not stop the rest
            failed.append(stem)
            print(f"[FAILED] {stem}: {type(e).__name__}: {str(e)[:300]}")
    print(f"done; failed: {', '.join(failed) or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
