"""YAML configs with attribute access, ``key.path=value`` overrides and a stable hash.

Deliberately dependency-light (PyYAML only) so the same loader runs in the Kubric
venv and the torch venv. Paths in configs are relative to the repo root unless
absolute; ``${env:NAME,default}`` is substituted from the environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO = Path(__file__).resolve().parents[1]
_ENV = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


class Cfg(dict):
    """dict with attribute access; nested dicts become Cfg."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    if isinstance(x, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), x)
    return x


def _set(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        cfg = cfg.setdefault(k, Cfg())
    cfg[keys[-1]] = _wrap(yaml.safe_load(value))


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Cfg:
    path = Path(path)
    if not path.is_absolute():
        path = (REPO / path) if not path.exists() else path
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("defaults", None)
    cfg = load_config(path.parent / base) if base else Cfg()
    _deep_update(cfg, _wrap(raw))
    for ov in overrides:
        k, _, v = ov.partition("=")
        _set(cfg, k, v)
    cfg["_source"] = str(path)
    return cfg


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


LOCATION_KEYS = ("local_path", "path")   # where weights live on disk: not part of a model's identity
PIN_KEYS = ("sha256",                     # top-level per-file weight hashes: they verify the pinned revision, so
                                          # adding them (2026-11-10) leaves existing caches' hashes unchanged
            "output_track", "track_a_mode", "track_a", "track_b_adapter")   # output-track routing (2026-11-11):
                                          # which generation path stage 07 uses, not what stage 4 caches


def _strip_locations(d):
    if isinstance(d, dict):
        return {k: _strip_locations(v) for k, v in d.items() if k not in LOCATION_KEYS}
    return d


def config_hash(cfg: dict) -> str:
    """Hash of the config's content. File locations (``local_path``, ``checkpoint.path``) are excluded:
    the weights are identified by pinned revisions / sha256, so moving them (e.g. to another pod's disk)
    does not change the hash. (Before 2026-11-05 locations were included; videomaev2-giant's caches carry
    the old hash 2c78b257e18c; the same config and weights (sha256 5773c881...) now hash to 4531024ff411)"""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in PIN_KEYS}
    return hashlib.sha256(json.dumps(_strip_locations(clean), sort_keys=True, default=str).encode()).hexdigest()[:12]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p
