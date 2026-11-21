#!/usr/bin/env python3
"""Run the PhysLaw benchmark for every model config on every rendered family, spread over all GPUs.

    # what would run (no GPU, no work): one line per task, plus the models that are skipped and why
    /root/envs/physics_venv/bin/python scripts/queues/run_models.py --dry-run
    # everything, every GPU (resumable: rerun the same command; finished tasks are skipped)
    /root/envs/physics_venv/bin/python scripts/queues/run_models.py
    # subsets
    ... --models vjepa2_vitl,physrvg --families f1,f7 --gpus 0,1,2,3
    # smoke: the first N groups per family, into smoke/models/ (never into cache/ or results/)
    ... --smoke 2

Per model (configs/models/<name>.yaml) the task graph follows the adapter's capabilities and the config
(physlaw/models/registry.py, physlaw/models/track_a.py); nothing is model-specific here:

  internals (adapter.internals: V-JEPA 2, VideoMAE-v2, CogVideoX, Wan T2V / TI2V, PhysRVG, VideoREPA, PC1, PC2)
    GPU  04 cache                    every family of the level configs
    GPU  04 --pred-real              continuation models only (``continuation: true``)
    CPU  05 pixel baselines          L2 families
    CPU  06 L0 -> L2, L2b, L3, L1    each level on its config's ``families``; L1 continuation only
    GPU  06 L4                       continuation only, after L0 / L1 / L2 (and the untrained model's L4)
  outputs (adapter.generator and output_track A with a native / variant mode, or output_track B)
    GPU  07 generate -> CPU 08 track -> CPU 09 score       configs/eval/outputs.yaml families
    CPU  08 --real (track B readout validation on the real clips)
  finally CPU 11 aggregate (results/leaderboard.md).

Models whose weights are not on disk are skipped with the reason (Cosmos-Predict2.5 is gated; PC1 / PC2 need stage
10 first). Cosmos runs stage 07 in /root/envs/cosmos_venv (PHYSLAW_COSMOS=1 bash scripts/00_setup_envs.sh).

Scheduling: one GPU task per GPU at a time (a model never shares a GPU), CPU tasks in --cpu-slots parallel slots.
GPU tasks need ``MIN_GPU_GB`` of total GPU memory (peak measured on an A100 80GB, smoke/models/MATRIX.md); a GPU
too small for a model is never given its tasks. Within a model the order is internals before outputs; across
models, the largest first (so the 14B models start while every GPU is free).
State: logs/models/<task>.log per task, logs/models/_done/<task> on success (delete to redo one task, or --redo),
logs/models/_status.tsv (one line per finished task: time, task, exit code, minutes). A failed task skips its
dependants and the run carries on; rerunning retries it.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from physlaw.config import load_config  # noqa: E402

P = os.environ.get("P", "/root/envs/physics_venv/bin/python")
P_COSMOS = os.environ.get("P_COSMOS", "/root/envs/cosmos_venv/bin/python")
LEVELS = {"l0": "l0_states", "l1": "l1_forecast_laws", "l2": "l2_surprise", "l2b": "l2b_detection",
          "l3": "l3_invariance", "l4": "l4_causal"}
UNTRAINED = "untrained_vitl"

# Total GPU memory (GB) a model's GPU tasks need: measured peak (torch max_memory_reserved) over stage 04 / 07 / L4
# in the smoke matrix on an A100 80GB, plus ~15% headroom (smoke/models/MATRIX.md). Same-architecture bigger
# siblings that were not smoke-run are estimated from their weights. Unknown models default to 80.
MIN_GPU_GB = {   # measured peak (smoke, F1) -> value; "est." = not smoke-run here, from a measured sibling / weights
    "vjepa2_vitl": 8, "untrained_vitl": 8, "positive_control": 8, "vjepa2_vitg": 16,          # 3.6 GB; g est.
    "videomae_v2_base": 4, "videomae_v2_giant": 8,                                             # 1.1 / 5.0 GB
    "cogvideox_2b": 24, "cogvideox_5b": 30, "videorepa_5b": 30, "cogvideox15_5b_i2v": 52,     # -, est., 24.8, 43.9
    "wan21_t2v_1_3b": 24, "wan22_ti2v_5b": 42, "physrvg": 34, "wan22_ti2v_5b_physrvg": 34,    # 17.7, 35.5, 27.6, 23.1
    "pc2_wan22_ti2v_5b": 42, "wan21_vace_1_3b": 34,                                            # est., 27.3
    "wan22_i2v_a14b": 76, "wan21_i2v_14b": 52, "ltx_video_13b_distilled": 52,                   # 65.0, 43.4, 42.7
    "wan21_t2v_14b": 80, "wan21_vace_14b": 56, "cosmos_predict25_2b": 48,                       # est. (internals), est., est.
}
# Largest first (GPU tasks are queued in this order); anything else after these.
ORDER = ["wan22_i2v_a14b", "wan21_i2v_14b", "wan21_t2v_14b", "wan21_vace_14b", "ltx_video_13b_distilled",
         "cogvideox15_5b_i2v", "videorepa_5b", "cogvideox_5b", "physrvg", "wan22_ti2v_5b_physrvg", "wan22_ti2v_5b",
         "pc2_wan22_ti2v_5b", "cosmos_predict25_2b", "cogvideox_2b", "wan21_t2v_1_3b", "wan21_vace_1_3b",
         "untrained_vitl", "vjepa2_vitg", "vjepa2_vitl", "positive_control", "videomae_v2_giant",
         "videomae_v2_base"]


@dataclass
class Task:
    name: str
    cmd: list[str]
    gpu: bool
    model: str = ""
    deps: list[str] = field(default_factory=list)
    prio: int = 0


# ----------------------------------------------------------------------------------------------- availability

def weights_missing(stem: str, cfg) -> str | None:
    """Why the model cannot run here (weights / venv not on disk), or None."""
    pc = cfg.get("positive_control")
    if pc == "pc1":
        f = Path(cfg["local_path"]) / "pc1.pt"
        return None if f.exists() else f"{f} missing: train PC1 first (scripts/10_train_positive_control.py)"
    if pc == "pc2":
        f = Path(cfg["lora"]["path"])
        return None if f.exists() else f"{f} missing: train PC2 first (scripts/10_train_positive_control.py)"
    paths = [cfg.get("local_path")]
    for blk in ("text_encoder", "lora", "dit_weights", "checkpoint"):
        b = cfg.get(blk) or {}
        paths.append(b.get("path"))
    for p in filter(None, paths):
        if not Path(p).exists() or (Path(p).is_dir() and not any(Path(p).iterdir())):
            return f"{p} missing" + (" (gated on HF: accept the licence, hf auth login, download)"
                                     if "Cosmos" in str(p) else "")
    if stem.startswith("cosmos") and not Path(P_COSMOS).exists():
        return f"{P_COSMOS} missing (PHYSLAW_COSMOS=1 bash scripts/00_setup_envs.sh)"
    return None


def capabilities(cfg):
    from physlaw.models.registry import adapter_class
    from physlaw.models.track_a import NATIVE_MODES, continues
    cls = adapter_class(cfg)
    track, mode = cfg.get("output_track", "none"), cfg.get("track_a_mode")
    tracks = []
    if cls.generator or cfg.get("track_b_adapter"):
        if track == "A" and mode in NATIVE_MODES:
            tracks.append("a")
        if track == "B":
            tracks.append("b")
    # ``predicts: false`` (encoder-only weights, e.g. VideoMAE-v2 base): no surprise -> no stage 05 / L2
    return dict(internals=bool(cls.internals), continues=continues(cfg), tracks=tracks,
                predicts=cfg.get("predicts", True) is not False)


def data_families(root: Path, only: set[str] | None) -> dict[str, Path]:
    """family -> data dir (the family read from the first labels.json; dirs without a complete group skipped)."""
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        done = sorted(d.glob("*/_GROUP_COMPLETE"))
        if not done:
            continue
        labels = next(done[0].parent.glob("*/labels.json"), None)
        fam = d.name
        if labels is not None:
            fam = (json.loads(labels.read_text()).get("physlaw") or {}).get("family", d.name)
        if (only is None or fam in only or d.name in only) and fam not in out:
            out[fam] = d
    return out


# ----------------------------------------------------------------------------------------------- plan

def plan(models: list[str], fams: dict[str, Path], smoke: int | None, cpu_workers: int):
    level_fams = {lv: set(load_config(REPO / f"configs/eval/{f}.yaml").get("families") or [])
                  for lv, f in LEVELS.items()}
    out_fams = set(load_config(REPO / "configs/eval/outputs.yaml").get("families") or [])
    tasks: dict[str, Task] = {}
    skipped = {}
    sm = REPO / "smoke" / "models"

    def add(t: Task):
        tasks[t.name] = t

    for rank, stem in enumerate(models):
        cfg = load_config(REPO / f"configs/models/{stem}.yaml")
        why = weights_missing(stem, cfg)
        if why:
            skipped[stem] = why
            continue
        cap = capabilities(cfg)
        name = cfg["name"]
        mpath = f"configs/models/{stem}.yaml"
        if not cap["internals"] and not cap["tracks"]:
            skipped[stem] = f"no internals and output_track {cfg.get('output_track')}: nothing to run"
            continue
        prio = len(models) - rank
        for fam, data in fams.items():
            cache = (sm / "cache" / name / fam) if smoke else (REPO / "cache" / name / data.name)
            res = (sm / "results" / name / fam) if smoke else (REPO / "results" / name / data.name)
            io = ["--model", mpath, "--data", str(data)]
            t = f"{stem}.{fam}"
            if cap["internals"] and fam in (level_fams["l0"] | level_fams["l2"]):
                lim = ["--limit", str(smoke)] if smoke else []
                add(Task(f"{t}.04_cache", [P, "scripts/04_cache_activations.py", *io, "--out", str(cache), *lim],
                         True, stem, [], prio))
                if cap["continues"]:
                    add(Task(f"{t}.04_pred_real", [P, "scripts/04_cache_activations.py", *io, "--out", str(cache),
                                                   "--pred-real", *lim], True, stem, [f"{t}.04_cache"], prio))
                ev = [*io, "--cache", str(cache), "--out", str(res), "--workers", str(cpu_workers)]
                if smoke:   # L4 localization reads L2 / the untrained L4 from the smoke results, not results/
                    l4set = ["--set", f"localization.results_root={sm / 'results'}"]
                else:
                    l4set = []

                def level(lv, deps, gpu=False, extra=()):
                    if fam in level_fams[lv]:
                        add(Task(f"{t}.06_{lv}", [P, "scripts/06_eval.py", "--level",
                                                  f"configs/eval/{LEVELS[lv]}.yaml", *ev, *extra],
                                 gpu, stem, deps, prio))
                        return [f"{t}.06_{lv}"]
                    return []
                if fam in level_fams["l2"] and cap["predicts"]:
                    add(Task(f"{t}.05_pixel", [P, "scripts/05_surprise.py", *io, "--cache", str(cache),
                                               "--workers", str(cpu_workers)], False, stem, [f"{t}.04_cache"], prio))
                unt_l0 = [f"{UNTRAINED}.{fam}.06_l0"] if UNTRAINED in models and stem != UNTRAINED else []
                l0 = level("l0", [f"{t}.04_cache"] + unt_l0)
                l2 = level("l2", l0 + [f"{t}.05_pixel"]) if fam in level_fams["l2"] and cap["predicts"] else []
                level("l2b", l0)
                level("l3", l0)
                if cap["continues"]:
                    l1 = level("l1", l0 + [f"{t}.04_pred_real"])
                    unt_l4 = ([f"{UNTRAINED}.{fam}.06_l4"] if UNTRAINED in models and stem != UNTRAINED
                              and fam in level_fams["l4"] else [])
                    level("l4", l0 + l1 + l2 + unt_l4, gpu=True,
                          extra=[*l4set] + (["--work", str(cache / "_l4")] if smoke else []))
            if fam not in out_fams:
                continue
            for tr in cap["tracks"]:
                if tr == "b" and fam == "f4":
                    continue                      # moving background (spec; stage 07 notes)
                gen = cache if tr == "a" else cache.with_name(f"{cache.name}_trackB")
                py = P_COSMOS if stem.startswith("cosmos") else P
                lim = []
                if smoke:
                    lim = (["--n", str(smoke), "--seeds", "0"] if tr == "b"
                           else ["--set", f"track_a.gate.n_clips={smoke}", "track_a.seeds_per_clip=1"])
                deps = [f"{t}.04_cache"] if f"{t}.04_cache" in tasks and tr == "a" else []
                add(Task(f"{t}.07_track{tr}", [py, "scripts/07_generate_outputs.py", *io, "--track", tr,
                                               "--out", str(gen), *lim], True, stem, deps, prio))
                add(Task(f"{t}.08_track{tr}", [P, "scripts/08_track_outputs.py", *io, "--track", tr, "--gen", str(gen)],
                         False, stem, [f"{t}.07_track{tr}"], prio))
                deps9 = [f"{t}.08_track{tr}"]
                if tr == "b":
                    add(Task(f"{t}.08_real", [P, "scripts/08_track_outputs.py", *io, "--track", "b", "--real",
                                              "--out", str(gen)], False, stem, [f"{t}.07_track{tr}"], prio))
                    deps9.append(f"{t}.08_real")
                add(Task(f"{t}.09_track{tr}", [P, "scripts/09_eval_outputs.py", *io, "--track", tr,
                                               "--readouts", str(gen), "--out", str(res)], False, stem, deps9, prio))
    # drop dependencies on tasks that are not planned (e.g. a family outside a level's list)
    for t in tasks.values():
        t.deps = [d for d in t.deps if d in tasks]
    if tasks and not smoke:
        add(Task("all.11_aggregate", [P, "scripts/11_aggregate.py"], False, "", sorted(tasks), -1))
    return tasks, skipped


# ----------------------------------------------------------------------------------------------- run

def gpu_list(arg: str | None) -> list[tuple[str, float]]:
    """[(index, total GB)] of the GPUs to use."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.strip().splitlines()
    gpus = [(i.strip(), float(m) / 1024) for i, m in (ln.split(",") for ln in q)]
    if arg:
        want = arg.split(",")
        gpus = [g for g in gpus if g[0] in want]
    return gpus


class Runner:
    def __init__(self, tasks, gpus, cpu_slots, state: Path, redo: bool, env: dict):
        self.tasks, self.gpus, self.state, self.env = tasks, gpus, state, env
        (state / "_done").mkdir(parents=True, exist_ok=True)
        if redo:
            for t in tasks:
                (state / "_done" / t).unlink(missing_ok=True)
        self.done = {t for t in tasks if (state / "_done" / t).exists()}
        self.failed: set[str] = set()
        self.running: set[str] = set()
        self.lock = threading.Condition()
        self.cpu_sem = threading.Semaphore(cpu_slots)

    def log(self, msg):
        print(f"[run_models] {time.strftime('%F %T')} {msg}", flush=True)

    def blocked(self, t: Task) -> bool:
        return any(d in self.failed or d in self.skipped_dep for d in t.deps)

    def ready(self, t: Task) -> bool:
        return all(d in self.done for d in t.deps)

    def fits(self, t: Task, gb: float) -> bool:
        return gb + 0.5 >= MIN_GPU_GB.get(t.model, 80)

    def pending(self):
        return [t for n, t in self.tasks.items()
                if n not in self.done and n not in self.failed and n not in self.running and n not in self.skipped_dep]

    def execute(self, t: Task, gpu: str | None):
        env = dict(self.env)
        env["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
        logf = self.state / f"{t.name}.log"
        t0 = time.time()
        self.log(f"start {t.name}" + (f" on GPU {gpu}" if gpu is not None else ""))
        with open(logf, "a") as fh:
            fh.write(f"\n### {time.strftime('%F %T')} {shlex.join(t.cmd)}\n")
            fh.flush()
            rc = subprocess.call(t.cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        with self.lock:
            self.running.discard(t.name)
            if rc == 0:
                self.done.add(t.name)
                (self.state / "_done" / t.name).write_text(f"{time.strftime('%F %T')} {mins:.1f} min\n")
            else:
                self.failed.add(t.name)
            with open(self.state / "_status.tsv", "a") as fh:
                fh.write(f"{time.strftime('%F %T')}\t{t.name}\t{rc}\t{mins:.1f}\t{gpu if gpu is not None else 'cpu'}\n")
            self.lock.notify_all()
        self.log(f"{'done' if rc == 0 else 'FAILED'} {t.name} (exit {rc}, {mins:.1f} min)"
                 + ("" if rc == 0 else f" -> {logf}"))

    def _mark_skips(self):
        changed = True
        while changed:
            changed = False
            for n, t in self.tasks.items():
                if n not in self.skipped_dep and n not in self.done and n not in self.running and self.blocked(t):
                    self.skipped_dep.add(n)
                    changed = True

    def _next(self, gpu: str | None, gb: float):
        """The highest-priority ready task for this worker, or None; caller holds the lock."""
        self._mark_skips()
        cands = [t for t in self.pending() if t.gpu == (gpu is not None) and self.ready(t)
                 and (gpu is None or self.fits(t, gb))]
        if not cands:
            return None
        return max(cands, key=lambda t: (t.prio, -list(self.tasks).index(t.name)))

    def worker(self, gpu: str | None, gb: float):
        while True:
            with self.lock:
                while True:
                    t = self._next(gpu, gb)
                    if t is not None:
                        self.running.add(t.name)
                        break
                    mine = [x for x in self.pending() if x.gpu == (gpu is not None)
                            and (gpu is None or self.fits(x, gb))]
                    if not mine:
                        return
                    self.lock.wait(timeout=30)
            if gpu is None:
                with self.cpu_sem:
                    self.execute(t, None)
            else:
                self.execute(t, gpu)

    def run(self, cpu_slots: int):
        self.skipped_dep: set[str] = set()
        threads = [threading.Thread(target=self.worker, args=(g, gb), daemon=True) for g, gb in self.gpus]
        threads += [threading.Thread(target=self.worker, args=(None, 0.0), daemon=True) for _ in range(cpu_slots)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._mark_skips()
        big = max((gb for _, gb in self.gpus), default=0)
        unfit = [n for n, t in self.tasks.items() if t.gpu and not self.fits(t, big) and n not in self.done]
        self.log(f"finished: {len(self.done)} done, {len(self.failed)} failed, {len(self.skipped_dep)} skipped "
                 f"(failed dependency), {len(unfit)} need a bigger GPU")
        for n in sorted(self.failed):
            self.log(f"  FAILED  {n}  -> {self.state / (n + '.log')}")
        for n in sorted(unfit):
            self.log(f"  NO GPU  {n}  (needs {MIN_GPU_GB.get(self.tasks[n].model, 80)} GB)")
        return 1 if self.failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="comma list of config stems (default: every configs/models/*.yaml)")
    ap.add_argument("--families", default=None, help="comma list (default: every data/<dir> with a complete group)")
    ap.add_argument("--data-root", default="data", help="comma list of roots holding <family> dirs")
    ap.add_argument("--gpus", default=None, help="comma list of GPU indices (default: all)")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="treat every GPU as at most this many GB (GPUs shared with other jobs)")
    ap.add_argument("--cpu-slots", type=int, default=None, help="parallel CPU tasks (default: max(2, nCPU // 32))")
    ap.add_argument("--smoke", type=int, default=None, metavar="N", help="first N groups; outputs under smoke/models/")
    ap.add_argument("--state", default=None, help="logs/state dir (default logs/models, smoke: smoke/models/logs)")
    ap.add_argument("--redo", action="store_true", help="forget the _done markers of the planned tasks")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    stems = sorted(p.stem for p in (REPO / "configs/models").glob("*.yaml"))
    if a.models:
        want = a.models.split(",")
        bad = [m for m in want if m not in stems]
        if bad:
            sys.exit(f"unknown model configs: {bad} (configs/models/*.yaml stems)")
        stems = want
    rank = {m: i for i, m in enumerate(ORDER)}
    stems.sort(key=lambda m: rank.get(m, len(ORDER)))
    fams = {}
    for r in a.data_root.split(","):            # earlier roots win for a family found in several
        root = Path(r) if Path(r).is_absolute() else REPO / r
        for f, d in data_families(root, set(a.families.split(",")) if a.families else None).items():
            fams.setdefault(f, d)
    if not fams:
        sys.exit(f"no rendered families under {a.data_root}")
    cpu_slots = a.cpu_slots or max(2, (os.cpu_count() or 8) // 32)
    cpu_workers = max(4, (os.cpu_count() or 8) // cpu_slots)
    tasks, skipped = plan(stems, fams, a.smoke, cpu_workers)
    print(f"families: {', '.join(f'{f}={d}' for f, d in fams.items())}")
    for m, why in skipped.items():
        print(f"SKIP {m}: {why}")
    if a.dry_run:
        for t in tasks.values():
            print(f"{'GPU' if t.gpu else 'CPU'}  {MIN_GPU_GB.get(t.model, 80) if t.gpu else '':>3}  {t.name}"
                  f"   <- {','.join(t.deps) or '-'}\n        {shlex.join(t.cmd)}")
        print(f"{len(tasks)} tasks ({sum(t.gpu for t in tasks.values())} GPU), {len(skipped)} models skipped")
        return
    gpus = gpu_list(a.gpus)
    if a.max_gb:
        gpus = [(g, min(gb, a.max_gb)) for g, gb in gpus]
    if not gpus:
        sys.exit("no GPUs")
    state = Path(a.state) if a.state else (REPO / ("smoke/models/logs" if a.smoke else "logs/models"))
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PHYSLAW_MODELS", "/workspace/models")
    env.setdefault("OMP_NUM_THREADS", "4")
    r = Runner(tasks, gpus, cpu_slots, state, a.redo, env)
    r.log(f"{len(tasks)} tasks ({len(r.done)} already done) on GPUs "
          f"{', '.join(f'{g} ({gb:.0f} GB)' for g, gb in gpus)}, {cpu_slots} CPU slots; state {state}")
    sys.exit(r.run(cpu_slots))


if __name__ == "__main__":
    main()
