#!/usr/bin/env python3
"""Stage 1 — render a scene family (spec §2, milestone 1 for F1).

Run with the Kubric venv:

    KUBRIC_USE_GPU=1 /root/envs/kubric_venv/bin/python scripts/01_generate_data.py \\
        --config configs/data/f1_collisions.yaml --workers 4

    # pilot: first 8 groups only, into a separate directory
    ... --n-groups 8 --out data/f1_pilot

    # every benchmark family: configs/data/f{1,2,3,4,6,7,8,9,10}_*.yaml (sizes in each config)
    # PC1 training set (legal clips, 256x256, seeds disjoint from the benchmark, asserted):
    ... --config configs/data/pc1_train.yaml [--families f7,f8]

Each twin group runs as its own subprocess (physlaw/data/generate.py), so a
Blender crash costs one group. Resumable and idempotent: groups with
_GROUP_COMPLETE are skipped unless --force; failed groups are retried on the next
run. Every finished group appends one line to <out>/_runlog.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from physlaw.config import config_hash, load_config, resolve_path  # noqa: E402

DATA_CFG = REPO / "configs" / "data"


def run_one(seed: int, a, out: Path) -> dict:
    cmd = [a.python, str(REPO / "physlaw" / "data" / "generate.py"), "--config", a.config,
           "--seed", str(seed), "--out", str(out), *(["--no-render"] if a.no_render else []),
           *(["--force"] if a.force else []), *(["--set", *a.set] if a.set else [])]
    t0 = time.time()
    try:
        env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, cwd=REPO, env=env)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "status": "timeout", "seconds": time.time() - t0}
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("RESULT ")), None)
    rec = json.loads(line[7:]) if line else {"status": "failed", "error": proc.stderr[-1500:]}
    rec.pop("summaries", None)
    rec.update(seed=seed, returncode=proc.returncode, wall_seconds=time.time() - t0)
    return rec


def benchmark_seed_ranges() -> dict:
    """[seed_offset, seed_offset + n_groups) of every benchmark family config."""
    out = {}
    for path in sorted(DATA_CFG.glob("f*.yaml")):
        c = load_config(path)
        out[path.name] = (int(c.seed_offset), int(c.seed_offset) + int(c.n_groups))
    return out


def pc1_jobs(pc, a) -> list:
    """(family config path, overrides, out dir, n_groups) per PC1 family, seed ranges asserted disjoint."""
    bench = benchmark_seed_ranges()
    want = set(a.families.split(",")) if a.families else None
    jobs = []
    for i, (fam, spec) in enumerate(pc.families.items()):
        n = a.n_groups or int(spec.n_groups)
        lo = int(pc.pc1_seed_base) + 100000 * i
        hi = lo + int(spec.n_groups)
        assert int(spec.n_groups) <= 100000, fam
        for name, (b0, b1) in bench.items():
            assert hi <= b0 or lo >= b1, f"PC1 {fam} seeds [{lo}, {hi}) overlap benchmark {name} [{b0}, {b1})"
        if want and fam not in want:
            continue
        sets = [f"{k}={v}" for k, v in pc.overrides.items()] + [f"seed_offset={lo}"] + list(a.set)
        out = (a.out or resolve_path(pc.out_dir)) / fam
        jobs.append((str(DATA_CFG / spec.config), sets, out, n))
    return jobs


def run_family(config: str, sets: list, out: Path, n: int | None, a) -> int:
    cfg = load_config(config, sets)
    out = out if out.is_absolute() else REPO / out
    out.mkdir(parents=True, exist_ok=True)
    n = n or cfg.n_groups
    seeds = [cfg.seed_offset + i for i in range(n)]
    todo = [s for s in seeds if a.force or not (out / f"{cfg.family}_{s:06d}" / "_GROUP_COMPLETE").exists()]
    if a.reverse:
        todo = todo[::-1]
    (out / "_config.json").write_text(json.dumps({"config": cfg, "hash": config_hash(cfg)}, indent=1, default=str))
    print(f"[01] family={cfg.family} groups={n} todo={len(todo)} workers={a.workers} "
          f"render={'no' if a.no_render else 'yes'} out={out} config_hash={config_hash(cfg)} "
          f"KUBRIC_USE_GPU={os.environ.get('KUBRIC_USE_GPU', '')}", flush=True)
    job = argparse.Namespace(**{**vars(a), "config": config, "set": sets})
    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(out / "_runlog.jsonl", "a") as log:
        futs = {ex.submit(run_one, s, job, out): s for s in todo}
        for f in as_completed(futs):
            rec = f.result()
            rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            log.write(json.dumps(rec, default=str) + "\n"); log.flush()
            done += 1
            failed += rec.get("status") not in ("ok", "skipped_exists", "skipped_locked")
            el = time.time() - t0
            print(f"[01] {done}/{len(todo)} {rec.get('group', rec['seed'])} {rec.get('status')} "
                  f"{rec.get('wall_seconds', 0):.0f}s  elapsed {el / 60:.1f} min  "
                  f"eta {el / done * (len(todo) - done) / 60:.1f} min  failed {failed}"
                  + (f"  ERR {rec.get('error', '')[:200]}" if rec.get("status") not in ("ok", "skipped_exists", "skipped_locked") else ""),
                  flush=True)
    print(f"[01] finished {cfg.family}: {done - failed} ok, {failed} failed, {(time.time() - t0) / 60:.1f} min", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <cfg.out_dir>/<family>")
    ap.add_argument("--n-groups", type=int, default=None, help="override cfg.n_groups (per family in pc1 mode)")
    ap.add_argument("--families", default=None, help="pc1 mode: comma-separated subset, e.g. f7,f8")
    ap.add_argument("--workers", type=int, default=6, help="GPU saturates at ~4 (H100); 6 covers CPU-only phases")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--reverse", action="store_true",
                    help="take groups from the end; run a second driver with this on the same --out to add workers")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    if cfg.get("mode") == "pc1_train":
        failed = sum(run_family(c, sets, out, n, a) for c, sets, out, n in pc1_jobs(cfg, a))
    else:
        out = a.out or resolve_path(cfg.out_dir) / cfg.family
        failed = run_family(a.config, list(a.set), out, a.n_groups, a)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
