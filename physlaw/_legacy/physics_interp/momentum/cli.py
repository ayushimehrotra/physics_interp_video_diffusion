"""Shared command-line plumbing for the protocol scripts under ``scripts/``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .captures import ClipSet, load_clipset

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "docs" / "artifacts"


def base_parser(doc: str, *, pools: bool = True, site: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", required=True, nargs="+",
                   help="capture dir(s), one per model: outputs/<set>/<model>/activations")
    p.add_argument("--names", nargs="+", default=None, help="model names, one per --captures")
    p.add_argument("--step", type=int, default=None, help="step index for captures with a step axis")
    if pools:
        p.add_argument("--pools", nargs="+", default=["A.all+B.all", "all.pre"],
                       help="pools to run; 'concat' = A.all+B.all (object-local)")
    if site:
        p.add_argument("--site", default="ingredients",
                       help="'ingredients' (label-free best), 'last', 'embed', 'all', or a site name")
    p.add_argument("--draws", type=int, default=4000, help="permutation draws for the null")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="load only the first N clips (smoke tests)")
    p.add_argument("--out", type=Path, default=None, help="artifact JSON path")
    return p


def load_sets(args) -> dict[str, ClipSet]:
    names = args.names or [Path(c).parent.name if Path(c).name == "activations" else Path(c).name
                           for c in args.captures]
    if len(names) != len(args.captures):
        sys.exit("--names must match --captures")
    sets = {}
    for name, c in zip(names, args.captures):
        cs = load_clipset(c, step=args.step, limit=args.limit)
        print(f"[{name}] {cs.describe()}", flush=True)
        sets[name] = cs
    return sets


def sites_for(cs: ClipSet, spec: str) -> list[str]:
    if spec == "all":
        return cs.sites
    return [spec] if spec in cs.sites else [spec]   # resolved later by choose_site


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_artifact(payload: dict, path: Path | None, default_name: str) -> Path:
    path = path or (ARTIFACTS / default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=1))
    print(f"\nartifact -> {path}", flush=True)
    return path


def fmt_auc(rec: dict | None) -> str:
    if not rec or not np.isfinite(rec.get("auc", np.nan)):
        return "   —   "
    return f"{rec['auc']:.3f} (p={rec['p']:.4f})"
