#!/usr/bin/env python3
"""Stage 6 — run one evaluation level and write results/<model>/<family>/<level>.{json,md}.

    /root/envs/physics_venv/bin/python scripts/06_eval.py --model configs/models/vjepa2_vitl.yaml \\
        --level configs/eval/l0_states.yaml --data data/f1
    /root/envs/physics_venv/bin/python scripts/06_eval.py --model configs/models/vjepa2_vitl.yaml \\
        --level configs/eval/l2_surprise.yaml --data data/f1
    ... --level configs/eval/l1_forecast_laws.yaml     (P-pred-real used if stage 4 --pred-real ran)
    ... --level configs/eval/l2b_detection.yaml        (after L2: reads l2.json for the encoded-but-unused index)

    /root/envs/physics_venv/bin/python scripts/06_eval.py --model configs/models/vjepa2_vitl.yaml \\
        --level configs/eval/l3_invariance.yaml --data data/f4
    HF_HUB_OFFLINE=1 /root/envs/physics_venv/bin/python scripts/06_eval.py --model configs/models/vjepa2_vitl.yaml \\
        --level configs/eval/l4_causal.yaml --data data/f3

Inputs: cache/<model>/<family>/ from stage 4 (+ _pixel_baselines.jsonl from stage 5 for L2).
Order per model: L0 (builds _maskfeat_v2.npz, used by L1/L2b/L3/L4), L2, L2b, L1. --out writes elsewhere (test runs).
L3 runs the analyses configured for the family (CPU). L4 loads the model (GPU): it reruns forecast() (and, for
generators, generate()) under activation interventions, cached in --work, then scores them.
L0 also folds in the untrained control's result if results/<untrained>/<family>/l0.json exists
(pass --untrained configs/models/untrained_vitl.yaml to name it).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")          # many single-cell workers beat one fat BLAS

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import config_hash, load_config, resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--untrained", default="configs/models/untrained_vitl.yaml")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None, help="default results/<model>/<family> (use a scratch dir for test runs)")
    ap.add_argument("--cache", default=None, help="stage 4 cache (default cache/<model>/<family>)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--work", default=None, help="L4: intervention cache (default cache/<model>/<family>/_l4)")
    ap.add_argument("--force", action="store_true", help="L4: recompute cached interventions")
    a = ap.parse_args()
    mcfg = load_config(a.model)
    cfg = load_config(a.level, a.set)
    data = resolve_path(a.data)
    cache = Path(a.cache) if a.cache else REPO / "cache" / mcfg.name / data.name
    out_dir = Path(a.out) if a.out else REPO / "results" / mcfg.name / data.name
    out_dir.mkdir(parents=True, exist_ok=True)
    level = cfg["level"]
    t0 = time.time()
    meta = {"model": mcfg.name, "family": data.name, "level": level,
            "model_config_hash": config_hash(mcfg), "eval_config_hash": config_hash(cfg)}

    if level == "l0":
        from physlaw.eval import l0_states
        res = l0_states.run(cache, data, cfg, mcfg, workers=a.workers)
        res.update(meta)
        (out_dir / "l0.json").write_text(json.dumps(res, indent=1))
        unt = None
        ucfg = load_config(a.untrained)
        uf = REPO / "results" / ucfg.name / data.name / "l0.json"
        if uf.exists() and ucfg.name != mcfg.name:
            unt = json.loads(uf.read_text())
        md = l0_states.table(res, unt)
    elif level == "l2":
        from physlaw.eval import l2_surprise
        sources = {mcfg.name: cache / "_index.jsonl"}
        pb = cache / "_pixel_baselines.jsonl"
        if pb.exists():
            sources["pixel_baselines"] = pb
        else:
            print(f"[06] WARNING: no {pb}; run scripts/05_surprise.py first for the copy-last-frame baseline")
        res = l2_surprise.run(sources, data, cfg)
        res.update(meta)
        (out_dir / "l2.json").write_text(json.dumps(res, indent=1))
        md = l2_surprise.table(res)
        if "motion_adjusted" in res:
            md = ("## Motion-adjusted (calibrated on legal-vs-legal pairs)\n\n" + l2_surprise.adjusted_table(res["motion_adjusted"])
                  + "\n\n## Raw contrasts\n\n" + md)
    elif level == "l2b":
        from physlaw.eval import l2b_detection
        l2f = out_dir / "l2.json"
        l2s = json.loads(l2f.read_text()).get("motion_adjusted", {}).get("summary") if l2f.exists() else None
        if l2s is None:
            print(f"[06] WARNING: no motion-adjusted summary in {l2f}; run L2 first for the encoded-but-unused index")
        res = l2b_detection.run(cache, data, cfg, l2s, workers=min(a.workers, 8))
        res.update(meta)
        (out_dir / "l2b.json").write_text(json.dumps(res, indent=1))
        md = l2b_detection.table(res)
    elif level == "l1":
        from physlaw.eval import l1_forecast_laws
        res = l1_forecast_laws.run(cache, data, cfg, mcfg)
        res.update(meta)
        (out_dir / "l1.json").write_text(json.dumps(res, indent=1))
        md = l1_forecast_laws.table(res)
    elif level == "l3":
        from physlaw.eval import l3_invariance
        res = l3_invariance.run(cache, data, cfg, mcfg)
        res.update(meta)
        (out_dir / "l3.json").write_text(json.dumps(res, indent=1, default=float))
        md = l3_invariance.table(res)
    elif level == "l4":
        from physlaw.eval import l4_causal
        res = l4_causal.run(cache, data, cfg, mcfg, work=Path(a.work) if a.work else cache / "_l4", force=a.force)
        res.update(meta)
        (out_dir / "l4.json").write_text(json.dumps(res, indent=1, default=float))
        md = l4_causal.table(res)
    else:
        sys.exit(f"06_eval.py: level {level} not built yet (see docs/RESUME.md)")

    (out_dir / f"{level}.md").write_text(f"# {level.upper()} — {mcfg.name} on {data.name}\n\n{md}\n")
    print(md)
    print(f"[06] {level} done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
