"""The capture format and the clip-set index.

One ``.npz`` per clip, written by ``scripts/capture_encoder.py`` (and, with a
leading step axis, by the older CogVideoX inversion capture)::

    act.<site>   float16 [pools, dim]            (or [steps, pools, dim])
    meta         uint8   JSON bytes: {labels, pools, sites, model, frames, ...}

Sites are ``embed.out`` then ``block.<i>.resid_post``; pools are
``{A, B, all} x {pre, post, all}`` as ``"A.post"`` etc. A pool whose region is
empty (object out of frame) is an all-zero row and is reported as missing.

Composite pools are built here, at analysis time, by concatenation:
``"A.all+B.all"`` is the object-local pool the register calls ``concat``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import physics

__all__ = ["ClipSet", "load_clipset", "site_depth", "POOL_ALIASES", "write_capture"]

POOL_ALIASES = {
    "concat": "A.all+B.all",
    "object-local": "A.all+B.all",
    "frame-wide": "all.pre",
}

_SITE_RE = re.compile(r"^block\.(\d+)\.")


def site_order_key(site: str) -> tuple[int, int]:
    if site == "embed.out":
        return (0, -1)
    m = _SITE_RE.match(site)
    return (1, int(m.group(1))) if m else (2, 0)


def site_depth(sites: Sequence[str]) -> dict[str, float]:
    """Depth as a fraction of the network: 0 = patch embedding, 1 = last block."""
    ordered = sorted(sites, key=site_order_key)
    n = max(len(ordered) - 1, 1)
    return {s: i / n for i, s in enumerate(ordered)}


def write_capture(path: Path, activations: dict[str, np.ndarray], meta: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"act.{k}": np.asarray(v, dtype=np.float16) for k, v in activations.items()}
    payload["meta"] = np.frombuffer(json.dumps(meta, default=str).encode(), dtype=np.uint8)
    np.savez_compressed(path, **payload)
    return path


@dataclass
class ClipSet:
    """Every captured clip of one set under one model, aligned by index."""

    root: Path
    clip_ids: list[str]
    labels: list[dict]
    metas: list[dict]
    pools: list[str]
    acts: dict[str, np.ndarray]           # site -> [n, pools, dim] float32
    seed: np.ndarray = field(init=False)
    arm: np.ndarray = field(init=False)
    violates: np.ndarray = field(init=False)
    restitution: np.ndarray = field(init=False)
    stratum: np.ndarray = field(init=False)

    def __post_init__(self):
        self.seed = np.array([physics.clip_seed(l) for l in self.labels])
        self.arm = np.array([physics.clip_arm(l) for l in self.labels])
        self.violates = np.array([bool(l.get("violates_momentum", False)) for l in self.labels])
        self.restitution = np.array([physics.clip_restitution(l) for l in self.labels])
        self.stratum = np.where(self.restitution > 0.5, "elastic", "inelastic")

    # ---- access ----------------------------------------------------------- #

    @property
    def sites(self) -> list[str]:
        return sorted(self.acts, key=site_order_key)

    @property
    def n(self) -> int:
        return len(self.clip_ids)

    def depth(self) -> dict[str, float]:
        return site_depth(self.sites)

    def resolve_pool(self, pool: str) -> list[str]:
        pool = POOL_ALIASES.get(pool, pool)
        parts = pool.split("+")
        missing = [p for p in parts if p not in self.pools]
        if missing:
            raise KeyError(f"pool(s) {missing} not captured; have {self.pools}")
        return parts

    def features(self, site: str, pool: str) -> tuple[np.ndarray, np.ndarray]:
        """``(X [n, dim*k], valid [n])`` for a (composite) pool at a site."""
        parts = self.resolve_pool(pool)
        arr = self.acts[site]
        cols = [arr[:, self.pools.index(p), :] for p in parts]
        X = np.concatenate(cols, axis=1).astype(np.float64)
        valid = np.all([np.abs(c).sum(1) > 0 for c in cols], axis=0)
        return X, valid

    def label_array(self, fn) -> np.ndarray:
        return np.stack([np.atleast_1d(np.asarray(fn(l), dtype=np.float64)) for l in self.labels])

    def subset(self, mask: np.ndarray) -> "ClipSet":
        idx = np.flatnonzero(np.asarray(mask, bool))
        return ClipSet(
            root=self.root,
            clip_ids=[self.clip_ids[i] for i in idx],
            labels=[self.labels[i] for i in idx],
            metas=[self.metas[i] for i in idx],
            pools=self.pools,
            acts={s: a[idx] for s, a in self.acts.items()},
        )

    def by_seed(self) -> dict[int, np.ndarray]:
        return {int(s): np.flatnonzero(self.seed == s) for s in np.unique(self.seed)}

    def twins(self) -> list[tuple[int, int]]:
        """``(honest_index, violating_index)`` for every seed that has both."""
        out = []
        for s, idx in self.by_seed().items():
            h = [i for i in idx if self.arm[i] == "honest"]
            v = [i for i in idx if self.arm[i] == "violating"]
            if len(h) == 1 and len(v) == 1:
                out.append((h[0], v[0]))
        return out

    def describe(self) -> str:
        strata = {s: int((self.stratum == s).sum()) for s in ("elastic", "inelastic")}
        arms = {a: int((self.arm == a).sum()) for a in np.unique(self.arm)}
        return (f"{self.n} clips, {len(np.unique(self.seed))} seeds, {len(self.sites)} sites, "
                f"pools={self.pools}, strata={strata}, arms={arms}, dim={next(iter(self.acts.values())).shape[-1]}")


def load_clipset(root: str | Path, *, step: int | None = None, sites: Iterable[str] | None = None,
                 limit: int | None = None) -> ClipSet:
    """Read every ``*.npz`` under ``root``.

    ``step`` selects along a leading step axis when captures have one (the
    CogVideoX inversion format); ``-1`` is the last captured step. ``sites``
    restricts what is loaded, which matters for the 41-site giant captures.
    """
    root = Path(root)
    paths = sorted(p for p in root.glob("*.npz"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no captures under {root}")
    wanted = set(sites) if sites is not None else None
    per_site: dict[str, list[np.ndarray]] = {}
    labels, metas, ids = [], [], []
    pools = None
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            for key in z.files:
                if not key.startswith("act."):
                    continue
                site = key[4:]
                if wanted is not None and site not in wanted:
                    continue
                arr = z[key]
                if arr.ndim == 3:
                    if step is None:
                        raise ValueError(f"{path.name} has a step axis {arr.shape}; pass step=")
                    arr = arr[step]
                per_site.setdefault(site, []).append(arr.astype(np.float32))
        if pools is None:
            pools = list(meta["pools"])
        elif list(meta["pools"]) != pools:
            raise ValueError(f"{path.name} has pools {meta['pools']} != {pools}")
        labels.append(meta["labels"])
        metas.append({k: v for k, v in meta.items() if k != "labels"})
        ids.append(path.stem)
    n = len(paths)
    ragged = {s: len(v) for s, v in per_site.items() if len(v) != n}
    if ragged:
        raise ValueError(f"site(s) captured for only some clips: {ragged} of {n}")
    return ClipSet(root=root, clip_ids=ids, labels=labels, metas=metas, pools=pools,
                   acts={s: np.stack(v) for s, v in per_site.items()})
