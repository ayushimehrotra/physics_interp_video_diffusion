"""ROI and frame-wide mean pooling over token windows (spec §3).

Pools: ``roi_A``, ``roi_B``, ``roi_pair`` (union of the two ROIs) and ``frame``.
Windows: ``pre`` / ``post`` (token-frames entirely before / after contact) and ``all``.
"""
from __future__ import annotations

import numpy as np

POOLS = ("roi_A", "roi_B", "roi_pair", "frame", "contact")
WINDOWS = ("pre", "post", "all")
CONTACT_DILATE = 2        # token rings added around the balls' last pre-contact ROIs


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation of a [gh, gw] mask by a (2r+1)^2 square, no scipy."""
    out = m.copy()
    gh, gw = m.shape
    p = np.pad(m, r)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= p[dy:dy + gh, dx:dx + gw]
    return out


def contact_mask(roi_a: np.ndarray, roi_b: np.ndarray, phases: list[str], dilate: int = CONTACT_DILATE) -> np.ndarray:
    """Leak-free object pool: the union of both balls' ROIs in the LAST pre-contact token-frame, dilated by
    ``dilate`` tokens, held fixed over every token-frame. It is built from pre-event information only, so
    pooling a forecast with it cannot reveal where the balls go (unlike roi_*.post, which sits at the TRUE
    post-event location), while staying local to the collision (unlike ``frame``)."""
    ph = np.asarray(phases)
    pre = np.flatnonzero(ph == "pre")
    t = int(pre[-1]) if len(pre) else 0
    base = _dilate(roi_a[t] | roi_b[t], dilate)
    return np.broadcast_to(base, roi_a.shape).copy()


def pool_weights(objects: dict, phases: list[str]) -> dict[str, np.ndarray]:
    """``{f"{pool}.{window}": weights [T_tok, gh, gw]}``, each summing to 1 (or all-zero if empty)."""
    roi_a, roi_b = objects["A"].roi, objects["B"].roi
    masks = {"roi_A": roi_a, "roi_B": roi_b, "roi_pair": roi_a | roi_b,
             "frame": np.ones_like(roi_a, dtype=bool), "contact": contact_mask(roi_a, roi_b, phases)}
    ph = np.array(phases)
    win = {"pre": ph == "pre", "post": ph == "post", "all": np.ones(len(ph), bool)}
    out = {}
    for p, m in masks.items():
        for w, sel in win.items():
            wt = (m & sel[:, None, None]).astype(np.float32)
            s = wt.sum()
            out[f"{p}.{w}"] = wt / s if s > 0 else wt
    return out


def pool(acts: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """acts [T_tok, gh, gw, D] -> {name: [D]} (NaN vector for an empty pool)."""
    T, gh, gw, D = acts.shape
    flat = acts.reshape(-1, D).astype(np.float32)
    out = {}
    for k, w in weights.items():
        w = w.reshape(-1)
        out[k] = (w @ flat) if w.sum() > 0 else np.full(D, np.nan, np.float32)
    return out
