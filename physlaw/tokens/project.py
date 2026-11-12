"""Object masks -> token-grid indices per model (spec §3).

Input: the master instance mask ``uint8 [T_master, H, W]`` (0 floor, 1 A, 2 B),
the master-frame indices a model is fed, and its :class:`TokenGridSpec`.
Output, per object and per token-frame:

* ``occupancy``  fraction of each token's pixels covered by the object (averaged over
                 the tubelet's frames) — what the old project's pools used;
* ``center``, ``radius`` the projected object centre and equivalent-disk radius,
                 in token units;
* ``roi``        boolean token mask: tokens whose centre lies within
                 ``roi_mult * radius`` of the projected centre (Kneifl et al.: 1.5x).

The ROI is a disk in token space, not the mask itself. Its pooled activations are
therefore much less a readout of the mask's shape than mask-weighted pools, but they
still carry position and size; that is why eval configs keep the mask-only floor
(old project §20).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physlaw.models.base import TokenGridSpec


@dataclass
class ObjectTokens:
    occupancy: np.ndarray     # [T_tok, gh, gw] float32
    center: np.ndarray        # [T_tok, 2] (row, col) in token units; nan if absent
    radius: np.ndarray        # [T_tok] in token units; 0 if absent
    roi: np.ndarray           # [T_tok, gh, gw] bool


def _fit_map(H: int, W: int, spec: TokenGridSpec):
    """Affine map master pixel (y, x) -> input pixel (y', x') for the spec's fit."""
    h, w = spec.input_hw
    if spec.fit == "squash":
        return (h / H, 0.0), (w / W, 0.0)
    s = min(h / H, w / W)
    return (s, (h - H * s) / 2.0), (s, (w - W * s) / 2.0)


def roi_object_ids(labels: dict) -> tuple[tuple[str, int], ...]:
    """``(("A", seg id), ("B", seg id))`` for the pools: the first two NON-aux objects of the labels (F9 rods and the
    F11 prop are never pooled); a family with one scored object gets B = -1 (no pixel matches: empty roi_B)."""
    objs = labels["physlaw"].get("objects") or {"A": {}, "B": {}}      # minimal labels: the two balls, ids 1 and 2
    sids = [int(o.get("segmentation_id", i + 1)) for i, o in enumerate(objs.values()) if not o.get("aux")]
    return (("A", sids[0]), ("B", sids[1] if len(sids) > 1 else -1))


def project_masks(instance: np.ndarray, frames: list[int], spec: TokenGridSpec,
                  object_ids=(("A", 1), ("B", 2)), roi_mult: float = 1.5) -> dict[str, ObjectTokens]:
    T, H, W = instance.shape
    gh, gw = spec.grid_hw
    (sy, oy), (sx, ox) = _fit_map(H, W, spec)
    # token index of every master pixel (pixel centres)
    ty = np.floor(((np.arange(H) + 0.5) * sy + oy) / spec.patch).astype(int)
    tx = np.floor(((np.arange(W) + 0.5) * sx + ox) / spec.patch).astype(int)
    valid = (ty[:, None] >= 0) & (ty[:, None] < gh) & (tx[None, :] >= 0) & (tx[None, :] < gw)
    cell = np.where(valid, ty[:, None] * gw + tx[None, :], -1).ravel()
    pix_per_cell = np.bincount(cell[cell >= 0], minlength=gh * gw).reshape(gh, gw).astype(np.float32)
    tf = spec.token_frames(len(frames))
    cy, cx = np.meshgrid(np.arange(gh) + 0.5, np.arange(gw) + 0.5, indexing="ij")
    out = {}
    for name, sid in object_ids:
        occ = np.zeros((len(tf), gh, gw), np.float32)
        ctr = np.full((len(tf), 2), np.nan)
        rad = np.zeros(len(tf))
        for j, idx in enumerate(tf):
            ys_all, xs_all, n = [], [], 0
            for k in idx:
                hit = (instance[frames[k]] == sid)
                flat = cell[hit.ravel()]
                flat = flat[flat >= 0]
                occ[j] += np.bincount(flat, minlength=gh * gw).reshape(gh, gw)
                ys, xs = np.nonzero(hit)
                ys_all.append(ys); xs_all.append(xs); n += ys.size
            occ[j] /= np.maximum(pix_per_cell, 1) * len(idx)
            if n:
                ys, xs = np.concatenate(ys_all), np.concatenate(xs_all)
                ctr[j] = [((ys.mean() + 0.5) * sy + oy) / spec.patch, ((xs.mean() + 0.5) * sx + ox) / spec.patch]
                area_tok = (n / len(idx)) * (sy * sx) / spec.patch ** 2
                rad[j] = np.sqrt(area_tok / np.pi)
        roi = np.zeros_like(occ, dtype=bool)
        for j in range(len(tf)):
            if rad[j] > 0:
                roi[j] = (cy - ctr[j, 0]) ** 2 + (cx - ctr[j, 1]) ** 2 <= (roi_mult * rad[j]) ** 2
                if not roi[j].any():   # object smaller than a token: keep its centre token
                    roi[j, min(int(ctr[j, 0]), gh - 1), min(int(ctr[j, 1]), gw - 1)] = True
        out[name] = ObjectTokens(occ, ctr, rad, roi)
    return out


def token_frame_phase(frames: list[int], spec: TokenGridSpec, first_post_contact_frame: int) -> list[str]:
    """'pre' if every master frame of the token-frame is before contact, 'post' if every one is
    at/after first_post_contact_frame, else 'straddle' (excluded from pre/post windows)."""
    out = []
    for idx in spec.token_frames(len(frames)):
        fs = [frames[k] for k in idx]
        out.append("pre" if max(fs) < first_post_contact_frame else
                   "post" if min(fs) >= first_post_contact_frame else "straddle")
    return out


def group_object_region(clips, frames: list[int], spec: TokenGridSpec | None, t_end: float,
                        master_fps: float) -> np.ndarray:
    """ONE object region per twin group for the L2 "obj" surprise (``spec``: token grid, [gh, gw]) and the pixel
    baselines (``spec`` None: master pixels, [H, W]): the union over the group's twins of their object masks (the
    two scored, non-aux objects of :func:`roi_object_ids`; token grid: their ROIs) at every post-event frame inside
    the group's clean window (time <= ``t_end``, the min over the twins of clean_until_s), held fixed over all
    post-event frames. Every twin of a group is scored on the same region, so the region cannot carry any one twin's
    outcome. ``clips``: iterable of (instance masks [T_master, H, W], labels), one per twin (loaded lazily);
    ``frames``: the master frames the model sees (the same for every twin of a group). A token-frame (frame) is in
    the window when it is wholly post-event and its last master frame is no later than ``t_end``; if none is, the
    first post-event one."""
    region = None
    for inst, labels in clips:
        fpc = int(labels["physlaw"]["first_post_contact_frame"])
        ids = [sid for _, sid in roi_object_ids(labels)]
        if spec is None:
            idx = [[i] for i in range(len(frames))]
            post = [frames[i] >= fpc for i in range(len(frames))]
            obj = np.isin(inst[frames], ids)
        else:
            idx = spec.token_frames(len(frames))
            post = [p == "post" for p in token_frame_phase(frames, spec, fpc)]
            o = project_masks(inst, frames, spec, roi_object_ids(labels))
            obj = o["A"].roi | o["B"].roi
        t_last = np.array([max(frames[i] for i in ix) / master_fps for ix in idx])
        keep = np.asarray(post) & (t_last <= t_end + 1e-9)
        if not keep.any() and any(post):
            keep[post.index(True)] = True
        u = obj[keep].any(0)
        region = u if region is None else region | u
    return region
