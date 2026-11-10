"""Shared probe library for the replication: ridge and Adam linear probes, nested CV.

Protocol (both probe types, identical):
  * outer folds grouped by `groups` (seed / KE group), 5 by default;
  * inside each outer-train set, ONE inner grouped split (25%) picks the
    hyperparameters (ridge alpha; Adam lr x weight-decay x epoch) per TARGET GROUP;
  * standardisation (X and Y) is fit on the rows being trained; constant features zeroed;
  * cell (site x pool) selection is nested too: each outer fold picks the cell with the
    best inner-validation score, so the "best cell" number carries no winner's curse.

Adam probe: AdamW, zero init, minibatch, MSE on standardised targets, early stopping
picked on the inner split, then refit on the full outer-train rows for the chosen epochs.
All configs and all target columns train in ONE batched tensor: with MSE each output
column's weights are independent (and Adam's state is per element), so this is exactly
equivalent to separate fits.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path("/workspace/meta/physics_interp_video_diffusion")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "common"))
from physics_interp.momentum.captures import load_clipset  # noqa: E402
from physics_interp.momentum.stats import grouped_folds  # noqa: E402
from physics_interp.interp import extract_label  # noqa: E402
from _fast_cv import DualRidge  # noqa: E402

FAST = Path("/root/fast/outputs")
ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)
ADAM_LRS = (1e-3, 3e-3, 1e-2)
ADAM_WDS = (0.0, 1e-3, 1e-2, 1e-1)
EPOCHS = 100
BATCH = 64
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))


def capture_dir(name: str) -> Path:
    """Prefer the local /root copy; fall back to /workspace."""
    for root in (FAST, REPO / "outputs"):
        p = root / name / "momentum_probe" / "activations"
        if p.exists() and any(p.glob("*.npz")):
            return p
    raise FileNotFoundError(name)


def load(name: str, sites=None):
    return load_clipset(capture_dir(name), sites=sites)


def label_matrix(labels: list[dict], target: str) -> np.ndarray:
    rows = [np.atleast_1d(np.asarray(extract_label(l, target), dtype=np.float64)) for l in labels]
    return np.stack(rows)


# --------------------------------------------------------------------------- standardise
def _std_x(X, rows):
    mu = X[rows].mean(0)
    sd = X[rows].std(0)
    live = sd > 1e-8
    return mu, np.where(live, 1.0 / np.where(live, sd, 1.0), 0.0)


def _std_y(Y, rows):
    mu = Y[rows].mean(0)
    sd = Y[rows].std(0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def _col_r2(y, p):
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((y - p) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)


def group_r2(y, p, groups_cols):
    """Pooled R^2 per target group (vector targets pool their columns)."""
    out = []
    for cols in groups_cols:
        yy, pp = y[:, cols], p[:, cols]
        ss = ((yy - yy.mean(0)) ** 2).sum()
        out.append(1.0 - ((yy - pp) ** 2).sum() / ss if ss > 0 else np.nan)
    return np.array(out)


# --------------------------------------------------------------------------- ridge
class RidgeProbe:
    name = "ridge"

    def select_and_fit(self, X, Y, fit_rows, val_rows, groups_cols):
        """Pick alpha per target group on val_rows, refit on fit_rows+val_rows."""
        dr = DualRidge(X, Y, fit_rows)
        preds = dr.predict(X, val_rows, ALPHAS)
        scores = np.stack([group_r2(Y[val_rows], p, groups_cols) for p in preds])  # [A, G]
        best = np.nanargmax(np.nan_to_num(scores, nan=-np.inf), axis=0)
        all_rows = np.concatenate([fit_rows, val_rows])
        full = DualRidge(X, Y, all_rows)
        chosen = [ALPHAS[i] for i in best]
        return _RidgeModel(full, chosen, groups_cols), scores.max(0)


class _RidgeModel:
    def __init__(self, dr, alphas, groups_cols):
        self.dr, self.alphas, self.groups_cols = dr, alphas, groups_cols

    def predict(self, X, rows):
        uniq = sorted(set(self.alphas))
        P = dict(zip(uniq, self.dr.predict(X, rows, uniq)))
        out = np.zeros((len(rows), sum(len(c) for c in self.groups_cols)))
        for a, cols in zip(self.alphas, self.groups_cols):
            out[:, cols] = P[a][:, cols]
        return out


# --------------------------------------------------------------------------- adam
def _adam_run(Xtr, Ytr, Xva, Yva, lrs, wds, epochs, seed, device="cpu"):
    """Train C configs x k columns jointly. Returns W[C,d,k], b[C,1,k], val_mse[E,C,k]."""
    g = torch.Generator().manual_seed(seed)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Ytr = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    n, d = Xtr.shape
    k = Ytr.shape[1]
    C = len(lrs)
    W = torch.zeros(C, d, k, device=device)
    b = torch.zeros(C, 1, k, device=device)
    mW, vW = torch.zeros_like(W), torch.zeros_like(W)
    mb, vb = torch.zeros_like(b), torch.zeros_like(b)
    lr = torch.tensor(lrs, dtype=torch.float32, device=device).view(C, 1, 1)
    wd = torch.tensor(wds, dtype=torch.float32, device=device).view(C, 1, 1)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    curve = None
    if Xva is not None:
        Xva = torch.as_tensor(Xva, dtype=torch.float32, device=device)
        Yva = torch.as_tensor(Yva, dtype=torch.float32, device=device)
        curve = np.zeros((epochs, C, k), dtype=np.float32)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)  # CPU generator, moved
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            err = torch.einsum("nd,cdk->cnk", xb, W) + b - yb  # [C,B,k]
            gW = (2.0 / len(idx)) * torch.einsum("nd,cnk->cdk", xb, err)
            gb = 2.0 * err.mean(1, keepdim=True)
            t += 1
            for p, gp, m, v, decay in ((W, gW, mW, vW, True), (b, gb, mb, vb, False)):
                m.mul_(b1).add_(gp, alpha=1 - b1)
                v.mul_(b2).addcmul_(gp, gp, value=1 - b2)
                step = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
                if decay:
                    p.mul_(1 - lr * wd)
                p.sub_(lr * step)
        if curve is not None:
            pv = torch.einsum("nd,cdk->cnk", Xva, W) + b
            curve[ep] = ((pv - Yva) ** 2).mean(1).cpu().numpy()
    return W.cpu().numpy(), b.cpu().numpy(), curve


class AdamProbe:
    name = "adam"

    def __init__(self, lrs=ADAM_LRS, wds=ADAM_WDS, epochs=EPOCHS, seed=0, device=None):
        self.device = device or os.environ.get("ADAM_DEVICE", "cpu")
        grid = [(lr, wd) for lr in lrs for wd in wds]
        self.lrs = [g[0] for g in grid]
        self.wds = [g[1] for g in grid]
        self.epochs, self.seed = epochs, seed

    def select_and_fit(self, X, Y, fit_rows, val_rows, groups_cols):
        mx, sx = _std_x(X, fit_rows)
        my, sy = _std_y(Y, fit_rows)
        Xf, Xv = (X[fit_rows] - mx) * sx, (X[val_rows] - mx) * sx
        Yf, Yv = (Y[fit_rows] - my) / sy, (Y[val_rows] - my) / sy
        _, _, curve = _adam_run(Xf, Yf, Xv, Yv, self.lrs, self.wds, self.epochs, self.seed, self.device)
        # per target group: (epoch, config) minimising summed val MSE over the group's columns
        choices, scores = [], []
        var_v = Yv.var(0) + 1e-12
        for cols in groups_cols:
            c = curve[:, :, cols].sum(-1)  # [E, C]
            e, ci = np.unravel_index(np.argmin(c), c.shape)
            choices.append((int(e) + 1, int(ci)))
            r2v = 1.0 - curve[e, ci, cols].sum() / var_v[cols].sum()
            scores.append(r2v)
        # refit on fit+val for each distinct (epochs, config); run the longest once per config
        all_rows = np.concatenate([fit_rows, val_rows])
        mx, sx = _std_x(X, all_rows)
        my, sy = _std_y(Y, all_rows)
        Xa, Ya = (X[all_rows] - mx) * sx, (Y[all_rows] - my) / sy
        k = Y.shape[1]
        Wf = np.zeros((X.shape[1], k), dtype=np.float32)
        bf = np.zeros((1, k), dtype=np.float32)
        by_cfg: dict[tuple[int, int], list] = {}
        for (ep, ci), cols in zip(choices, groups_cols):
            by_cfg.setdefault((ep, ci), []).extend(cols)
        for (ep, ci), cols in by_cfg.items():
            W, b, _ = _adam_run(Xa, Ya[:, cols], None, None, [self.lrs[ci]], [self.wds[ci]], ep,
                                self.seed + 1, self.device)
            Wf[:, cols], bf[:, cols] = W[0], b[0]
        return _LinModel(mx, sx, my, sy, Wf, bf), np.array(scores)


@dataclass
class _LinModel:
    mx: np.ndarray
    sx: np.ndarray
    my: np.ndarray
    sy: np.ndarray
    W: np.ndarray
    b: np.ndarray

    def predict(self, X, rows):
        return (((X[rows] - self.mx) * self.sx) @ self.W + self.b) * self.sy + self.my


PROBES = {"ridge": RidgeProbe, "adam": AdamProbe}


# --------------------------------------------------------------------------- CV drivers
def inner_split(groups, train, seed):
    inner = grouped_folds(np.asarray(groups)[train], n_folds=4, seed=seed)
    val = train[inner[0]]
    return np.setdiff1d(train, val), val


def targets_matrix(labels, targets):
    """Stack targets column-wise; return Y and column index lists per target."""
    mats, cols, c = [], [], 0
    for t in targets:
        m = label_matrix(labels, t)
        mats.append(m)
        cols.append(list(range(c, c + m.shape[1])))
        c += m.shape[1]
    return np.hstack(mats), cols


def nested_oof(cells: dict, Y, groups_cols, groups, probe, *, n_folds=5, seed=0, fit_mask=None):
    """For every cell: out-of-fold predictions + per-fold inner-val score.

    Returns {cell: {"pred": [n,k], "val": [folds, G]}}, folds list.
    """
    n = Y.shape[0]
    fit_mask = np.ones(n, bool) if fit_mask is None else fit_mask
    folds = grouped_folds(np.asarray(groups), n_folds, seed)
    out = {}
    for cname, X in cells.items():
        pred = np.full(Y.shape, np.nan)
        vals = []
        for fi, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            train = train[fit_mask[train]]
            fit, val = inner_split(groups, train, seed + 100 + fi)
            model, vscore = probe.select_and_fit(X, Y, fit, val, groups_cols)
            pred[test] = model.predict(X, test)
            vals.append(vscore)
        out[cname] = {"pred": pred, "val": np.array(vals)}
    return out, folds


def nested_best(res: dict, folds, gi: int, cols):
    """Per fold pick the cell with the best inner-val score for target group gi."""
    names = list(res)
    n = next(iter(res.values()))["pred"].shape[0]
    pred = np.full((n, len(cols)), np.nan)
    picks = []
    for fi, test in enumerate(folds):
        best = max(names, key=lambda c: np.nan_to_num(res[c]["val"][fi, gi], nan=-np.inf))
        picks.append(best)
        pred[test] = res[best]["pred"][test][:, cols]
    return pred, picks


def cells_of(cs, pools, sites=None):
    sites = sites or cs.sites
    cells = {}
    for s in sites:
        for p in pools:
            X, ok = cs.features(s, p)
            cells[f"{s}|{p}"] = X.astype(np.float32)
    return cells


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))


# --------------------------------------------------------------------------- RAM cache
SHM = Path("/dev/shm/fast/cache")


class Cached:
    """A capture set loaded from the /dev/shm cache written by cache.py."""

    def __init__(self, name: str):
        meta = json.loads((SHM / f"{name}.meta.json").read_text())
        self.name = name
        self.sites, self.pools = meta["sites"], meta["pools"]
        self.clip_ids, self.labels = meta["clip_ids"], meta["labels"]
        self.A = np.load(SHM / f"{name}.acts.npy", mmap_mode="r")  # [n, sites, pools, dim] f16
        self.n = len(self.labels)
        self.seed = np.array([l.get("seed") for l in self.labels])

    def features(self, site: str, pool: str) -> np.ndarray:
        si = self.sites.index(site)
        parts = [np.asarray(self.A[:, si, self.pools.index(p)], dtype=np.float32)
                 for p in pool.split("+")]
        return np.concatenate(parts, 1)

    def cells(self, pools, sites=None):
        return {f"{s}|{p}": self.features(s, p) for s in (sites or self.sites) for p in pools}

    def subset_rows(self, mask):
        return np.flatnonzero(mask)


def group_ids(cs) -> np.ndarray:
    """CV groups: KE design group for ke_factor_set, else simulation seed (twins share it)."""
    out = []
    for l in cs.labels:
        kd = l.get("ke_design")
        out.append(kd["group"] if kd else f"s{l.get('seed')}")
    return np.array(out)
