#!/usr/bin/env python
"""Run one experiment config. This is the only script an experiment needs.

Three modes, normally run in sequence:

    # 1. invert every simulator scene and store pooled activations
    .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode capture

    # 2. fit linear probes on what was captured
    .venv/bin/python scripts/run_experiment.py configs/momentum_probe.yaml --mode probe

    # generation sweeps (prompts x seeds) are the third mode, and the default
    .venv/bin/python scripts/run_experiment.py configs/red_ball_slope.yaml

All GPUs on the box (the wrapper picks the count):

    ./scripts/run_experiment.sh configs/momentum_probe.yaml --mode capture

``capture`` and ``generate`` shard across ranks and skip finished work, so an
interrupted sweep resumes. ``probe`` is cheap, single-process, and reads only
what capture wrote — so it can be re-run with different targets, or restricted
to a stratum, without touching a GPU.

To check that an inversion actually stayed on the model's trajectory, use
``VideoDiffusionRunner.resample()`` interactively on one scene and compare the
result to the source clip. It is not wired into a sweep flag because it doubles
the cost per scene and is a spot check, not something to run 1000 times.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics_interp import (  # noqa: E402
    VideoDiffusionRunner,
    expand_grid,
    get_dist_info,
    load_config,
    shard,
)
from physics_interp.interp import probe_grid  # noqa: E402
from physics_interp.video import write_json  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a video diffusion experiment grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", type=Path, help="path to an experiment YAML")
    p.add_argument(
        "--mode", choices=("generate", "capture", "probe"), default="generate",
        help="generate videos, capture activations from a dataset, or fit probes",
    )
    p.add_argument(
        "-o", "--output-dir", default=None,
        help="override output_dir from the config",
    )
    p.add_argument(
        "--dataset-dir", default=None,
        help="override dataset_dir from the config (capture mode)",
    )
    p.add_argument(
        "--device", default=None,
        help="override device (default: cuda:LOCAL_RANK)",
    )
    p.add_argument(
        "--model", default=None,
        help="override model.id from the config, e.g. 'vjepa2-vitl'. Encoders "
             "capture in one forward pass instead of an inversion, so the "
             "inversion block of the config is ignored for them. Send each model "
             "to its own --output-dir",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="run at most this many units on this rank (smoke tests)",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="re-run units that are already complete",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and exit without loading the model",
    )
    p.add_argument(
        "--list-sites", action="store_true",
        help="load the model, print every hookable site name, and exit",
    )
    p.add_argument(
        "--keep-going", action="store_true",
        help="continue past a failing unit instead of aborting the rank",
    )
    p.add_argument(
        "--sites", default=None,
        help="capture mode: site pattern to tap, overriding the config (e.g. "
             "'block.*.attn_heads'). Capture cost is set by the diffusion pass, "
             "not the number of taps, so re-capturing at a different site costs "
             "a full run either way — send it to a separate --output-dir.",
    )
    # Probe-mode overrides. Probing is cheap and re-reads captures, so asking a
    # different question should not need a second config file.
    p.add_argument(
        "--patch-sites", default=None,
        help="capture mode, encoders only: also store RAW per-token activations "
             "for these sites, for patch-preserving attentive probes. Expensive "
             "(2.4-12.6 MB per site per scene), so pass a strided subset of "
             "layers, e.g. 'block.0.resid_post,block.4.resid_post'",
    )
    p.add_argument(
        "--targets", default=None,
        help="probe mode: comma-separated label targets, overriding the config",
    )
    p.add_argument(
        "--pools", default=None,
        help="probe mode: comma-separated pooled regions, overriding the config. "
             "A '+'-joined spec concatenates pools along the FEATURE axis, e.g. "
             "'A.pre+A.post+B.pre+B.post' hands the probe all four object-local "
             "velocity pools side by side — a relation between two objects cannot "
             "survive a mean over both of them.",
    )
    p.add_argument(
        "--where", default=None,
        help="probe mode: restrict scenes, as comma-separated key=value pairs "
             "(e.g. 'stratum=matched_momentum'). Values true/false are parsed "
             "as booleans, digits as numbers.",
    )
    p.add_argument(
        "--tag", default=None,
        help="probe mode: suffix for the results file, so several experiments "
             "off one config do not overwrite each other",
    )
    return p.parse_args(argv)


def parse_where(spec: str | None) -> dict:
    """Parse 'a=b,c=true' into {'a': 'b', 'c': True}."""
    out: dict = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        value = value.strip()
        if value.lower() in ("true", "false"):
            parsed: object = value.lower() == "true"
        elif value.lstrip("-").replace(".", "", 1).isdigit():
            parsed = float(value) if "." in value else int(value)
        else:
            parsed = value
        out[key.strip()] = parsed
    return out


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #


def scene_dirs(dataset_dir: Path) -> list[Path]:
    """Complete scenes, in a stable order.

    Discovery is by CONTENT, not by name: any directory holding both `_COMPLETE`
    and `labels.json` is a scene. The older sets all name their scenes
    `scene_00000`, but `conservation_set_v1` names each clip for its cell and twin
    (`elastic_mixed_violating_seed2`) because the name has to identify which of the
    2x2 cells a clip belongs to and which seed it is the twin of. A `scene_*` glob
    silently returned zero scenes for it, which reads as an empty dataset rather
    than as a naming mismatch.
    """
    return sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "_COMPLETE").exists() and (d / "labels.json").exists()
    )


def run_capture(cfg, args, info) -> int:
    dataset_dir = Path(args.dataset_dir or cfg.dataset_dir)
    if not dataset_dir.is_dir():
        info.log(f"dataset_dir {dataset_dir} does not exist")
        return 1

    out_root = Path(cfg.output_dir) / cfg.name / "activations"
    scenes = scene_dirs(dataset_dir)
    mine = shard(scenes, info)
    todo = mine
    if cfg.resume:
        todo = [s for s in mine if not (out_root / f"{s.name}.npz").exists()]
    skipped = len(mine) - len(todo)   # count before --limit, or the two conflate
    if args.limit is not None:
        todo = todo[: args.limit]

    if info.is_main:
        print(
            f"capture '{cfg.name}': {len(scenes)} scene(s) in {dataset_dir} "
            f"across {info.world_size} process(es) -> {out_root}",
            flush=True,
        )
    info.log(f"assigned {len(mine)}, skipping {skipped} done, running {len(todo)}")
    if args.dry_run:
        for s in todo:
            info.log(f"  would invert {s.name}")
        return 0
    if not todo and not args.list_sites:
        info.log("nothing to do")
        return 0

    info.log(f"loading {cfg.model.id} ...")
    runner = VideoDiffusionRunner.from_config(cfg, device=info.device)
    info.log(f"loaded: {runner.describe()}")
    if args.list_sites:
        if info.is_main:
            for name in runner.sites:
                print(name)
        runner.close()
        return 0

    if args.sites:
        cfg.sites = args.sites
    inv = cfg.inversion

    # An encoder has no trajectory to invert: capture is one forward pass, and
    # the control baseline is its own patch embedding rather than a VAE latent.
    from physics_interp.models.encoders import EncoderAdapter  # noqa: E402

    is_encoder = isinstance(runner.adapter, EncoderAdapter)
    if is_encoder:
        sites = cfg.sites
        sites = [sites] if isinstance(sites, str) else list(sites)
        if "embed.out" not in sites:
            sites.append("embed.out")
        patch_sites = ([s.strip() for s in args.patch_sites.split(",")]
                       if args.patch_sites else None)
        info.log(f"encoding in one forward pass, sites={sites!r}"
                 + (f", storing raw patches for {patch_sites}" if patch_sites else ""))
    else:
        patch_sites = None
        sites = cfg.sites
        info.log(
            f"inverting with {inv.num_steps} {inv.solver} steps, capturing at "
            f"{inv.num_captures if inv.capture_steps is None else len(inv.capture_steps)} "
            f"point(s), sites={sites!r}"
        )

    failures: list[tuple[str, str]] = []
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        for i, scene in enumerate(todo, 1):
            info.log(f"[{i}/{len(todo)}] {scene.name}")
            try:
                rec = (runner.encode_scene(scene, sites=sites, patch_sites=patch_sites)
                       if is_encoder
                       else runner.capture_scene(scene, inversion=inv, sites=sites))
                payload = {f"act.{k}": v for k, v in rec["activations"].items()}
                payload.update({f"patch.{k}": v for k, v in rec.get("patches", {}).items()})
                meta = {k: v for k, v in rec.items()
                        if k not in ("activations", "patches")}
                # Labels are large; keep the whole thing so probing can ask a
                # new question later without re-inverting anything.
                payload["meta"] = np.frombuffer(
                    json.dumps(meta, default=str).encode(), dtype=np.uint8
                )
                np.savez_compressed(out_root / f"{scene.name}.npz", **payload)
                info.log(f"    -> {out_root / (scene.name + '.npz')}")
            except Exception:
                failures.append((scene.name, traceback.format_exc()))
                info.log(f"    FAILED {scene.name}:\n{traceback.format_exc()}")
                if not args.keep_going:
                    raise
    finally:
        runner.close()

    return _report_failures(failures, cfg, info, len(todo) - len(failures))


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #


def load_captures(out_root: Path):
    """Read every captured scene into stacked arrays + aligned labels."""
    paths = sorted(out_root.glob("*.npz"))   # capture names each file after its scene dir
    if not paths:
        raise FileNotFoundError(f"no captures in {out_root}; run --mode capture first")

    per_site: dict[str, list[np.ndarray]] = {}
    labels, metas = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            for key in z.files:
                if key.startswith("act."):
                    per_site.setdefault(key[4:], []).append(z[key])
        labels.append(meta["labels"])
        metas.append(meta)

    # A site missing from some scenes would stack into a misaligned array and
    # silently pair scene i's activations with scene j's labels.
    n = len(paths)
    ragged = {s: len(v) for s, v in per_site.items() if len(v) != n}
    if ragged:
        raise ValueError(f"site(s) captured for only some scenes: {ragged} of {n}")

    return {s: np.stack(v) for s, v in per_site.items()}, labels, metas


def run_probe(cfg, args) -> int:
    out_root = Path(cfg.output_dir) / cfg.name / "activations"
    activations, labels, metas = load_captures(out_root)
    pool_names = metas[0]["pools"]

    targets = args.targets.split(",") if args.targets else cfg.probe.targets
    pools = ([p.strip() for p in args.pools.split(",")] if args.pools
             else cfg.probe.pools)
    where = parse_where(args.where) if args.where else (cfg.probe.where or None)

    print(
        f"probe '{cfg.name}': {len(labels)} scene(s), {len(activations)} site(s), "
        f"{len(metas[0]['captured_steps'])} capture step(s), pools={pool_names}"
        + (f", where={where}" if where else ""),
        flush=True,
    )

    rows = probe_grid(
        activations,
        labels,
        targets=[t.strip() for t in targets],
        pools=pools,
        pool_names=pool_names,
        alphas=cfg.probe.alphas,
        train_fraction=cfg.probe.train_fraction,
        split_seed=cfg.probe.split_seed,
        standardize=cfg.probe.standardize,
        where=where,
    )

    suffix = f"_{args.tag}" if args.tag else ""
    dest = Path(cfg.output_dir) / cfg.name / f"probes{suffix}.json"
    write_json(
        {
            "config": {"probe": {**cfg.probe.__dict__, "pools": pools},
                       "inversion": cfg.inversion.__dict__},
            "targets": targets,
            "pools": pools,
            "where": where,
            "num_scenes": len(labels),
            "noise_levels": metas[0]["noise_levels"],
            "results": rows,
        },
        dest,
    )

    _print_summary(rows, metas[0])
    print(f"\nfull grid ({len(rows)} fits) -> {dest}", flush=True)
    return 0


# Sites that are baselines rather than model internals, and so are excluded from
# the argmax and reported separately. `vae.latent` is the diffusion pipeline's raw
# autoencoder latent; `embed.out` is an encoder's patch embedding before any
# block. Both answer "was this already in the pixels", which is the question a
# positive result has to survive.
CONTROL_SITES = ("vae.latent", "embed.out")


def _best_and_null(group, score_key, shuf_key):
    """Winning cell, its selection-matched permutation null, and a p-value.

    `best` is an argmax over the whole (site x step) grid on held-out data, so
    the control beside it has to be the argmax of that *same* grid on permuted
    labels. Reporting the winning cell's own shuffled score instead — one draw
    rather than the max of ~279 — understates the null by the entire winner's
    curse, which is how a chance-level plausibility result printed shuf=0.42
    against acc=0.68 while the grid-max null was really 0.63.

    `p_perm` is the fraction of shuffled cells reaching the winner's score. It
    is a lower bound on the true p-value: the cells are one permutation draw
    each and are heavily correlated across neighbouring blocks and steps, so
    the effective number of independent draws is well under the grid size.
    """
    real = [r for r in group if r["site"] not in CONTROL_SITES] or group
    score = lambda r: np.nan_to_num(r.get(score_key, np.nan), nan=-9e9)
    best = max(real, key=score)
    shuf = np.array([np.nan_to_num(r.get(shuf_key, np.nan), nan=-9e9) for r in real])
    null = float(shuf.max()) if shuf.size else float("nan")
    pval = float((shuf >= score(best)).mean()) if shuf.size else float("nan")
    vae = max((r.get(score_key, float("nan")) for r in group
               if r["site"] in CONTROL_SITES), default=float("nan"))
    return best, null, pval, vae


def _print_summary(rows, meta) -> None:
    """Best fit per (target, pool), with the controls beside it.

    Binary targets are scored and ranked by accuracy rather than R-squared: a
    ridge fit can separate two classes cleanly while having an unimpressive
    R-squared, so ranking those by R-squared would pick the wrong block.
    """
    if not rows:
        print("no probes fit — every pool was empty or too few scenes")
        return

    levels = meta["noise_levels"]
    by_key: dict[tuple, list] = {}
    for r in rows:
        by_key.setdefault((r["target"], r["pool"]), []).append(r)

    binary = {t for (t, _), g in by_key.items() if "accuracy" in g[0]}
    if binary:
        print(f"\n{'target':<24} {'pool':<10} {'best site':<22} {'noise':>6} "
              f"{'acc':>7} {'bal':>7} {'shuf*':>7} {'p_perm':>7} {'VAE':>7}")
        print("-" * 104)
        for (target, pool), group in sorted(by_key.items()):
            if target not in binary:
                continue
            best, null, pval, vae = _best_and_null(
                group, "accuracy", "accuracy_shuffled")
            noise = levels[best["step_index"]] if best["step_index"] < len(levels) else 0.0
            print(f"{target:<24} {pool:<10} {best['site']:<22} {noise:>6.2f} "
                  f"{best['accuracy']:>7.3f} {best['balanced_accuracy']:>7.3f} "
                  f"{null:>7.3f} {pval:>7.3f} {vae:>7.3f}")
        print("\nacc/bal are held-out accuracy and balanced accuracy; chance is 0.50.")

    continuous = [k for k in by_key if k[0] not in binary]
    if continuous:
        print(f"\n{'target':<24} {'pool':<10} {'best site':<22} {'noise':>6} "
              f"{'R2':>7} {'shuf*':>7} {'p_perm':>7} {'VAE':>7}")
        print("-" * 104)
        for (target, pool) in sorted(continuous):
            best, null, pval, vae_r2 = _best_and_null(
                by_key[(target, pool)], "r2", "r2_shuffled")
            noise = levels[best["step_index"]] if best["step_index"] < len(levels) else 0.0
            print(f"{target:<24} {pool:<10} {best['site']:<22} {noise:>6.2f} "
                  f"{best['r2']:>7.3f} {null:>7.3f} {pval:>7.3f} {vae_r2:>7.3f}")

    print(
        "\n'shuf*' is the permutation control selected the SAME way as the winner: the max over the\n"
        "whole (site x step) grid on shuffled labels, not the winning cell's own shuffled score.\n"
        "'best site' is an argmax on held-out data, so it carries a winner's curse that only a\n"
        "grid-max null accounts for — read the margin as best minus shuf*, never best minus 0.5/0.\n"
        "'p_perm' is the share of shuffled cells reaching the winner, and is a LOWER bound: the\n"
        "cells are one draw each and correlated, so effective independent draws << grid size.\n"
        "'VAE' is the raw autoencoder latent, which carries no physical objective. A block only\n"
        "tells you something if it clears shuf* and VAE both."
    )


# --------------------------------------------------------------------------- #
# generate (the original grid)
# --------------------------------------------------------------------------- #


def run_generate(cfg, args, info) -> int:
    specs = expand_grid(cfg)
    mine = shard(specs, info)

    if cfg.resume:
        todo = [s for s in mine if not s.is_done(cfg.output_dir)]
        skipped = len(mine) - len(todo)
    else:
        todo, skipped = mine, 0
    if args.limit is not None:
        todo = todo[: args.limit]

    if info.is_main:
        print(
            f"experiment '{cfg.name}': {len(cfg.prompts)} prompt(s) x "
            f"{len(cfg.seeds)} seed(s) = {len(specs)} run(s) across "
            f"{info.world_size} process(es) -> {cfg.output_dir}",
            flush=True,
        )
    info.log(f"assigned {len(mine)}, skipping {skipped} done, running {len(todo)}")

    if args.dry_run:
        for s in todo:
            info.log(f"  would run {s.name}: {s.prompt.text[:70]!r}")
        return 0
    if not todo and not args.list_sites:
        info.log("nothing to do")
        return 0

    info.log(f"loading {cfg.model.id} ...")
    runner = VideoDiffusionRunner.from_config(cfg, device=info.device)
    info.log(f"loaded: {runner.describe()}")

    if args.list_sites:
        if info.is_main:
            for name in runner.sites:
                print(name)
        runner.close()
        return 0

    failures: list[tuple[str, str]] = []
    try:
        for i, spec in enumerate(todo, 1):
            info.log(f"[{i}/{len(todo)}] {spec.name} :: {spec.prompt.text[:60]!r}")
            try:
                out = runner.run(spec, cfg.output_dir)
                info.log(f"    -> {out}")
            except Exception:
                failures.append((spec.name, traceback.format_exc()))
                info.log(f"    FAILED {spec.name}:\n{traceback.format_exc()}")
                if not args.keep_going:
                    raise
    finally:
        runner.close()

    return _report_failures(failures, cfg, info, len(todo) - len(failures))


def _report_failures(failures, cfg, info, completed: int = 0) -> int:
    if not failures:
        info.log(f"done: {completed} unit(s), no failures")
        return 0
    # A failure report per rank, so a sweep's damage is auditable after the
    # fact rather than lost in interleaved stdout.
    report = Path(cfg.output_dir) / cfg.name / f"failures_rank{info.rank}.json"
    write_json([{"unit": n, "traceback": tb} for n, tb in failures], report)
    info.log(f"{len(failures)} unit(s) failed; report written to {report}")
    return 1


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model:
        cfg.model.id = args.model
        cfg.model.repo = None      # let the adapter pick its own default repo
    if args.no_resume:
        cfg.resume = False

    if args.mode == "probe":
        return run_probe(cfg, args)

    info = get_dist_info(args.device)
    if args.mode == "capture":
        return run_capture(cfg, args, info)
    return run_generate(cfg, args, info)


if __name__ == "__main__":
    raise SystemExit(main())
