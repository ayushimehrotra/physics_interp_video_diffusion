"""Binary energy probes on `conservation_set_v1`: does the encoder know the clip is losing energy?

THE QUESTION. Momentum conservation is the law this set was built for; energy is the
*other* law, and §42c pinned elasticity precisely so that it could be asked. The
elastic honest cell conserves kinetic energy exactly (KE ratio 1.000 on all 75 clips)
and the inelastic honest cell loses it on all 75 -- so "is this video losing energy"
is a clean 75/75 binary on the honest half of the set.

THE GRID. Nine cells, the full crossing the captures already store: region {all, A, B}
x phase {pre, post, all}, every site, both encoders. No cell is selected and hidden --
the whole grid is reported, and the headline is priced against a max-over-grid null.

FOUR TARGETS, because "losing energy" is three different questions on this set:

  energy_loss_honest      honest clips only, 75/75. Elastic (KE ratio == 1.000) vs
                          inelastic (KE ratio 0.098-0.992). THE TARGET AS ASKED.
  energy_loss_all         all 300 clips, KE_post < 0.99*KE_pre. The violating twins'
                          post velocities are overwritten, so their energy moves in
                          both directions and the label stops tracking elasticity.
  energy_loss_violating   violating clips only, 90 lose / 60 do not. Here restitution
                          and energy loss are DECOUPLED -- this is the cell where a
                          probe cannot pass by reading how fast the balls separate.
  energy_created          violating clips, KE ratio > 1.01: 57 clips where energy is
                          created out of nothing. Impossible physics, in the pixels.

THE CEILING THAT MAKES THE HEADLINE READABLE. `e = |v_rel_post . n| / |v_rel_pre . n|`
is MASS-FREE -- any observer who can see the two balls separate can compute it with no
notion of mass, energy or momentum anywhere. On `energy_loss_honest` it separates the
classes at **AUC 1.000**, because there the label IS `e < 1`. So on that target a probe
at 0.95 has demonstrated it can see relative speed, and nothing more; §12's rule
applies in full. The mass-free oracle is printed next to every number for exactly this
reason, and it is why `energy_loss_violating` is the cell worth reading: mass-free
scalars top out at 0.775 there, so a probe that beats that has combined speed with
something else.

THREE CONTROLS, none optional on this set:

1. GROUPED FOLDS, by seed. Each seed contributes an honest and a violating twin that
   share their entire pre-contact segment. Split by clip and a probe memorises one
   twin's deflection and flags the other; §42b measured that trap at 0.787
   discrimination on data with no leak in it. Every fold here is grouped, and the
   permutation null permutes whole seeds.

2. THE MASK-ONLY FLOOR (§20), which matters more here than on any previous target.
   Object-local pooling picks its tokens with the ground-truth segmentation, so the
   *choice of region* is a channel independent of the tokens' contents -- and energy
   loss is a purely post-contact kinematic fact that the mask carries perfectly. Run
   with `--mask-floor` to fit the identical grid on the pooling weights with no
   activations anywhere. The `all.*` rows are the built-in check: those masks are
   all-ones and identical in every scene, so they must read 0.5.

3. A MAX-OVER-GRID PERMUTATION NULL. Reporting 9 pools x 25 sites and quoting the best
   is a selection the null has to price. Labels are permuted at the seed level, the
   whole grid is re-scored inside each draw, and the max over cells forms the null for
   the observed max (Westfall-Young). Per-cell p-values are reported too, uncorrected
   and labelled as such.

ONE THING NO CELL HERE CAN TEST. VideoMAE and V-JEPA2 attend bidirectionally over all
frames, so a `pre` token has already attended to the post-contact ones (§4). A high
`.pre` number is the model reading the aftermath through attention, not predicting it.

    .venv/bin/python scripts/kinetic_energy/probe_energy.py                 # the 9-cell grid
    .venv/bin/python scripts/kinetic_energy/probe_energy.py --mask-floor    # §20's floor, same grid
    .venv/bin/python scripts/kinetic_energy/probe_energy.py --transfer      # honest -> violating
    .venv/bin/python scripts/kinetic_energy/probe_energy.py --vs-kinematics # probe vs a camera
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# numpy only, plus Pillow for --mask-floor. Nothing from `physics_interp` is
# imported: this script reads what capture already wrote, so it needs neither the
# model adapters nor torch.

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
POOLS = ["all.pre", "all.post", "all.all",
         "A.pre", "A.post", "A.all",
         "B.pre", "B.post", "B.all"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _midrank(S: np.ndarray) -> np.ndarray:
    """Column-wise ranks with ties at their midrank, vectorised over columns.

    Ties are not a corner case here. The `all.*` mask-floor rows are all-ones and
    identical in every scene, so the probe's predictions there are exactly
    constant -- and only midranking makes that come out at the 0.500 the control
    is supposed to report. Ordinal ranks would hand it an arbitrary number.
    """
    n, P = S.shape
    order = np.argsort(S, axis=0, kind="mergesort")
    Ss = np.take_along_axis(S, order, axis=0)
    pos = np.arange(n)[:, None]
    is_start = np.empty(S.shape, bool)
    is_start[0] = True
    is_start[1:] = Ss[1:] != Ss[:-1]
    is_end = np.empty(S.shape, bool)
    is_end[-1] = True
    is_end[:-1] = Ss[1:] != Ss[:-1]
    run_start = np.maximum.accumulate(np.where(is_start, pos, 0), axis=0)
    run_end = np.minimum.accumulate(np.where(is_end, pos, n - 1)[::-1], axis=0)[::-1]
    ranks = np.empty(S.shape, float)
    np.put_along_axis(ranks, order, (run_start + run_end) / 2.0 + 1.0, axis=0)
    return ranks


def auc_cols(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Rank AUC per column of `S [n, P]` against the matching column of `Y [n, P]`.

    The permutation null carries every draw as a column, so scoring one column at a
    time in Python was the whole runtime; this is the same statistic, vectorised.
    """
    Y = np.asarray(Y, bool)
    ranks = _midrank(np.asarray(S, float))
    n1 = Y.sum(axis=0).astype(float)
    n0 = Y.shape[0] - n1
    pos_rank_sum = np.where(Y, ranks, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (pos_rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.where((n1 == 0) | (n0 == 0), np.nan, out)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Scalar convenience wrapper."""
    return float(auc_cols(np.asarray(y, bool)[:, None], np.asarray(s, float)[:, None])[0])


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    tpr = pred[y].mean() if y.any() else np.nan
    tnr = (~pred[~y]).mean() if (~y).any() else np.nan
    return float((tpr + tnr) / 2)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# ridge, with the label axis left free
# --------------------------------------------------------------------------- #

class RidgeSplit:
    """One (train, test) split of one cell, eigendecomposed once.

    The permutation null re-scores the entire grid inside every draw, so the fit
    count is (cells x sites x alphas x folds x draws). What makes that tractable is
    that the eigendecomposition of the training Gram matrix does not depend on the
    LABEL: permuting y changes only a matmul. So `Y` here is `[n_train, P]` holding
    every permutation as a column, and one call returns every alpha's predictions
    for every draw at once.
    """

    def __init__(self, Xtr: np.ndarray, Xte: np.ndarray):
        mu = Xtr.mean(0)
        sd = Xtr.std(0)
        # A feature that is CONSTANT across scenes must contribute exactly zero,
        # and `sd + 1e-6` does not achieve that: `Xtr - mu` is float noise around
        # 1e-19 rather than a true zero, and dividing it by 1e-6 lifts it to 1e-13,
        # which then ranks the scenes. The `all.*` mask-floor rows are exactly this
        # case -- identical in every scene, so they MUST read AUC 0.500 -- and they
        # came out at 0.399 before this guard: a number that looks like a result and
        # is entirely the last bits of a rounding error. Dead features are zeroed.
        dead = sd <= 1e-9 * (np.abs(Xtr).mean(0) + 1e-30)
        scale = np.where(dead, 1.0, sd + 1e-6)
        Ztr = (Xtr - mu) / scale
        Zte = (Xte - mu) / scale
        Ztr[:, dead] = 0.0
        Zte[:, dead] = 0.0
        xm = Ztr.mean(0)
        Zc = Ztr - xm
        w, V = np.linalg.eigh(Zc @ Zc.T)
        self.w, self.V = w, V
        self.Ktr = (Zc @ Zc.T) @ V
        self.Kte = ((Zte - xm) @ Zc.T) @ V

    def predict(self, Y: np.ndarray) -> dict[float, np.ndarray]:
        """`Y [n_train, P]` -> {alpha: test predictions `[n_test, P]`}."""
        ym = Y.mean(0)
        A = self.V.T @ (Y - ym)
        return {a: self.Kte @ (A / (self.w + a)[:, None]) + ym for a in ALPHAS}

    def predict_train(self, Y: np.ndarray) -> dict[float, np.ndarray]:
        ym = Y.mean(0)
        A = self.V.T @ (Y - ym)
        return {a: self.Ktr @ (A / (self.w + a)[:, None]) + ym for a in ALPHAS}


def grouped_folds(groups: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """k folds of INDICES, split on unique group ids so no group straddles a fold."""
    uniq = rng.permutation(np.unique(groups))
    return [np.flatnonzero(np.isin(groups, g)) for g in np.array_split(uniq, k)]


# --------------------------------------------------------------------------- #
# the probe: nested grouped CV, every permutation carried along
# --------------------------------------------------------------------------- #

def cell_oof_scores(X: np.ndarray, Ys: np.ndarray, groups: np.ndarray,
                    seed: int, n_out: int = 5, n_in: int = 3) -> np.ndarray:
    """Out-of-fold scores for one cell, for every label column of `Ys [n, P]`.

    Alpha is chosen inside the training half only, per outer fold, per column --
    a permuted label gets the same freedom to pick its ridge penalty as the real
    one, which is what makes the null a null of the whole procedure.
    """
    X = np.asarray(X, np.float64)
    n, P = Ys.shape
    rng = np.random.default_rng(seed)
    outer = grouped_folds(groups, n_out, rng)
    oof = np.zeros((n, P))
    for k in range(n_out):
        te = outer[k]
        tr = np.concatenate([outer[j] for j in range(n_out) if j != k])
        inner_rng = np.random.default_rng(seed + 1000 + k)
        inner = grouped_folds(groups[tr], n_in, inner_rng)
        # inner folds index into `tr`
        val_score = {a: np.zeros(P) for a in ALPHAS}
        for f in range(n_in):
            iva = tr[inner[f]]
            itr = np.concatenate([tr[inner[j]] for j in range(n_in) if j != f])
            preds = RidgeSplit(X[itr], X[iva]).predict(Ys[itr])
            Yva = Ys[iva] > 0.5
            for a, Pv in preds.items():
                val_score[a] += np.nan_to_num(auc_cols(Yva, Pv), nan=0.5)
        best = np.argmax(np.stack([val_score[a] for a in ALPHAS]), axis=0)  # [P]
        preds = RidgeSplit(X[tr], X[te]).predict(Ys[tr])
        stack = np.stack([preds[a] for a in ALPHAS])                        # [A, nte, P]
        # SUBTRACT THE FOLD'S TRAINING-LABEL MEAN before pooling folds together.
        # A ridge prediction is `Kte @ w + mean(y_train)`, and pooling raw scores
        # across folds turns that intercept into a comparison BETWEEN folds: a fold
        # whose TEST half holds more positives has fewer left in its train half, so
        # its whole block of scores shifts DOWN. Pooled AUC then reads systematically
        # below chance for any cell carrying no signal -- the mask floor's `all.*`
        # rows, which are identical in every scene and must read exactly 0.500, came
        # out at 0.399 and 0.364 before this. The intercept IS the training mean
        # (fitted values on centred features average to it exactly), so removing it
        # uses no test information whatever, and a dead cell now scores every clip
        # 0.0 -- all tied, AUC 0.500 by construction rather than by luck.
        oof[te] = stack[best, :, np.arange(P)].T - Ys[tr].mean(0)[None, :]
    return oof


def cell_oof_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        seed: int, n_out: int = 5, n_in: int = 3) -> np.ndarray:
    """The same protocol scored by R2, for the continuous positive controls."""
    X = np.asarray(X, np.float64)
    rng = np.random.default_rng(seed)
    outer = grouped_folds(groups, n_out, rng)
    oof = np.zeros_like(y)
    for k in range(n_out):
        te = outer[k]
        tr = np.concatenate([outer[j] for j in range(n_out) if j != k])
        inner = grouped_folds(groups[tr], n_in, np.random.default_rng(seed + 1000 + k))
        score = {a: 0.0 for a in ALPHAS}
        for f in range(n_in):
            iva = tr[inner[f]]
            itr = np.concatenate([tr[inner[j]] for j in range(n_in) if j != f])
            for a, Pv in RidgeSplit(X[itr], X[iva]).predict(y[itr]).items():
                score[a] += r2(y[iva], Pv)
        best = max(ALPHAS, key=lambda a: score[a])
        oof[te] = RidgeSplit(X[tr], X[te]).predict(y[tr])[best]
    return oof


def oof_threshold_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           seed: int, n_out: int = 5) -> float:
    """Balanced accuracy with the decision threshold taken from TRAINING fits only.

    A ridge on a 0/1 label has no calibrated 0.5; thresholding the out-of-fold
    scores at their own median would use the test labels' base rate. The threshold
    is instead the training prior's quantile of the training fitted values.
    """
    X = np.asarray(X, np.float64)
    rng = np.random.default_rng(seed)
    outer = grouped_folds(groups, n_out, rng)
    pred = np.zeros(len(y), bool)
    for k in range(n_out):
        te = outer[k]
        tr = np.concatenate([outer[j] for j in range(n_out) if j != k])
        split = RidgeSplit(X[tr], X[te])
        Y = y[tr].reshape(-1, 1).astype(float)
        a = 10.0
        thr = np.quantile(split.predict_train(Y)[a].ravel(), 1 - y[tr].mean())
        pred[te] = split.predict(Y)[a].ravel() > thr
    return balanced_accuracy(y, pred)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_captures(outdir: str):
    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no captures in {outdir}")
    store: dict[str, list] = {}
    labels, pools, sites = [], None, None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        m = json.loads(bytes(z["meta"]).decode())
        pools = m["pools"]
        labels.append(m["labels"])
        if sites is None:
            sites = [k[4:] for k in z.files if k.startswith("act.")]
        for s in sites:
            store.setdefault(s, []).append(z["act." + s][0])
    acts = {s: np.stack(v).astype(np.float32) for s, v in store.items()}
    return acts, labels, pools, [Path(p).stem for p in paths]


def scene_facts(L: dict) -> dict:
    c = L["collision"]
    n = np.array(c["contact_normal"][:2], float)
    n = n / max(np.linalg.norm(n), 1e-12)
    vAp = np.array(c["pre"]["velocity_A"][:2], float)
    vBp = np.array(c["pre"]["velocity_B"][:2], float)
    vAq = np.array(c["post"]["velocity_A"][:2], float)
    vBq = np.array(c["post"]["velocity_B"][:2], float)
    up, uq = (vAp - vBp) @ n, (vAq - vBq) @ n
    return dict(
        scene_id=L["scene_id"], seed=L["seed"], e=L["restitution"],
        mixed=L["objects"]["A"]["material"] != L["objects"]["B"]["material"],
        viol=bool(L["violates_momentum"]),
        ke_pre=float(c["pre"]["kinetic_energy"]),
        ke_post=float(c["post"]["kinetic_energy"]),
        ke_ratio=float(c["kinetic_energy_ratio"]),
        # mass-free kinematic scalars: the ceiling a pure observer already has
        e_meas=abs(uq) / max(abs(up), 1e-12),
        rel_speed_post=float(np.linalg.norm(vAq - vBq)),
        rel_speed_pre=float(np.linalg.norm(vAp - vBp)),
        speed_sum_post=float(np.linalg.norm(vAq) + np.linalg.norm(vBq)),
        speed_sum_pre=float(np.linalg.norm(vAp) + np.linalg.norm(vBp)),
        # --- the --fixed-e ladder: visible kinematics, then visible size, then mass
        sA=float(np.linalg.norm(vAp)), sB=float(np.linalg.norm(vBp)),
        sAq=float(np.linalg.norm(vAq)), sBq=float(np.linalg.norm(vBq)),
        rel_pre=float(np.linalg.norm(vAp - vBp)),
        rel_post=float(np.linalg.norm(vAq - vBq)),
        cos_n=float(abs(up) / max(np.linalg.norm(vAp - vBp), 1e-12)),
        un=float(abs(up)),
        imp=float(abs(L["sampled_parameters"]["impact_parameter_frac"])),
        rA=float(L["objects"]["A"]["radius"]), rB=float(L["objects"]["B"]["radius"]),
        mass_A=float(L["objects"]["A"]["mass"]), mass_B=float(L["objects"]["B"]["mass"]),
        log_mass_ratio=float(np.log(L["objects"]["A"]["mass"] / L["objects"]["B"]["mass"])),
        log_total_mass=float(np.log(L["objects"]["A"]["mass"] + L["objects"]["B"]["mass"])),
    )


TARGETS = {
    "energy_loss_honest": dict(
        select=lambda F: np.array([not f["viol"] for f in F]),
        label=lambda F: np.array([f["ke_ratio"] < 1 - 1e-6 for f in F]),
        note="honest clips only; elastic (KE ratio 1.000) vs inelastic. THE TARGET AS ASKED. "
             "Mass-free ceiling is 1.000, so this measures whether relative speed is legible.",
    ),
    "energy_loss_all": dict(
        select=lambda F: np.ones(len(F), bool),
        label=lambda F: np.array([f["ke_ratio"] < 0.99 for f in F]),
        note="all 300 clips; the violating twins decouple the label from elasticity.",
    ),
    "energy_loss_violating": dict(
        select=lambda F: np.array([f["viol"] for f in F]),
        label=lambda F: np.array([f["ke_ratio"] < 0.99 for f in F]),
        note="violating clips only; restitution and energy loss are decoupled here. "
             "THE CELL WORTH READING.",
    ),
    "energy_created": dict(
        select=lambda F: np.array([f["viol"] for f in F]),
        label=lambda F: np.array([f["ke_ratio"] > 1.01 for f in F]),
        note="violating clips only; energy created out of nothing -- impossible physics.",
    ),
}

MASS_FREE = ["e_meas", "rel_speed_post", "rel_speed_pre", "speed_sum_post", "speed_sum_pre"]
NEEDS_MASS = ["ke_pre", "ke_post"]


def oracle_aucs(F: list[dict], sel: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for name in MASS_FREE + NEEDS_MASS:
        v = np.array([f[name] for f in F])[sel]
        a = auc(y, v)
        out[name] = max(a, 1 - a)  # sign is free to an observer who may flip it
    return out


# --------------------------------------------------------------------------- #
# §20's mask-only floor, without the torch dependency
# --------------------------------------------------------------------------- #

def _nearest_resize(seg: np.ndarray, size: int) -> np.ndarray:
    """Replicates `torch.nn.functional.interpolate(..., mode='nearest')` index maths."""
    t, h, w = seg.shape
    iy = np.minimum((np.arange(size) * (h / size)).astype(np.int64), h - 1)
    ix = np.minimum((np.arange(size) * (w / size)).astype(np.int64), w - 1)
    return seg[:, iy][:, :, ix]


def _latent_occupancy(seg: np.ndarray, frames: int, height: int, width: int,
                      tubelet: int) -> np.ndarray:
    """Per-latent-cell occupancy for ids (1, 2); mirrors `masks_to_latent_grid`."""
    t_pix, h_pix, w_pix = seg.shape
    fy, fx = h_pix // height, w_pix // width
    out = np.zeros((2, frames, height, width), np.float32)
    counts = np.zeros(frames, np.int64)
    for t in range(t_pix):
        lat = t // tubelet
        counts[lat] += 1
        for k, sid in enumerate((1, 2)):
            block = (seg[t] == sid).reshape(height, fy, width, fx)
            out[k, lat] += block.mean(axis=(1, 3))
    return out / np.maximum(counts, 1)[None, :, None, None]


def mask_floor_cells(outdir: str, dataset_dir: str, stems: list[str]) -> dict:
    """The pooling weights themselves, as features, with no activations anywhere.

    Returns cells keyed `mask-<tag>/<pool>` for tag in {normalised, occupancy}.
    `normalised` is the row the recorder actually pooled with (sums to 1, so the
    object's AREA is divided out); `occupancy` is the same row before that
    division and is the strictly stronger control, since it is the normalised row
    times a per-scene scalar the probe is then free to use.
    """
    from PIL import Image

    paths = sorted(glob.glob(f"{outdir}/momentum_probe/activations/*.npz"))
    meta = json.loads(bytes(np.load(paths[0], allow_pickle=True)["meta"]).decode())
    lay = meta["metadata"]["layout"]
    frames, height, width = lay["frames"], lay["height"], lay["width"]
    size = meta["metadata"]["model"]["input_size"]
    idx = meta["frame_indices"]
    tubelet = len(idx) // frames
    phases = {k: tuple(v) for k, v in meta["phases"].items()}

    norm_rows, occ_rows = [], []
    for stem in stems:
        files = sorted((Path(dataset_dir) / stem / "segmentation").glob("segmentation_*.png"))
        seg = np.stack([np.asarray(Image.open(files[i])) for i in idx])
        if seg.ndim == 4:
            seg = seg[..., 0]
        occ = _latent_occupancy(_nearest_resize(seg, size), frames, height, width, tubelet)
        regions = {"A": occ[0], "B": occ[1],
                   "all": np.ones((frames, height, width), np.float32)}
        nr, orr = {}, {}
        for rname, o in regions.items():
            for pname, (start, end) in phases.items():
                masked = np.zeros_like(o)
                masked[start:end] = o[start:end]
                flat = masked.reshape(-1)
                total = flat.sum()
                orr[f"{rname}.{pname}"] = flat
                nr[f"{rname}.{pname}"] = flat / total if total > 0 else flat
        norm_rows.append(nr)
        occ_rows.append(orr)

    cells = {}
    for tag, rows in (("normalised", norm_rows), ("occupancy", occ_rows)):
        for p in POOLS:
            cells[f"mask-{tag}/{p}"] = np.stack([r[p] for r in rows]).astype(np.float64)
    return cells




# --------------------------------------------------------------------------- #
# the transfer test: does an "energy" direction survive when elasticity stops
# predicting energy loss?
# --------------------------------------------------------------------------- #
#
# `energy_loss_honest` cannot distinguish an energy detector from a separation-speed
# detector, because on honest clips the label IS `e < 1` and `e` is mass-free and
# visible (oracle AUC 1.000). The violating twins break that identity: their post
# velocities are overwritten, so 90 of 150 lose energy while `e` predicts which ones
# at only 0.636.
#
# So: FIT on honest clips, PREDICT on violating ones, grouped by seed so a scene's
# own twin is never in the training half. Two outcomes, and they are different
# claims about the model:
#
#   ~0.64, i.e. what `e` alone transfers at   the probe learned relative speed.
#   materially above it                       the probe learned something that
#                                             combines speed with size or mass --
#                                             an energy-shaped quantity.
#
# The null is exact and free: the score vector does not depend on the TEST labels,
# so permuting them re-scores the same numbers.


def transfer_scores(X: np.ndarray, y_tr: np.ndarray, tr_idx: np.ndarray,
                    te_idx: np.ndarray, seeds: np.ndarray, seed: int,
                    graded: bool = False, n_out: int = 5, n_in: int = 3) -> np.ndarray:
    """Scores for `te_idx`, from probes fitted only on `tr_idx` of OTHER seeds.

    `graded` trains on the CONTINUOUS honest KE ratio instead of the 0/1 label and
    selects alpha by R2. Without it the obvious objection to a failed transfer is
    that the training label was binary and threw away the size of the energy loss;
    the graded route hands the probe every drop of that and asks again.

    Returns `(centred, raw)`. AUC pools scores across folds and so needs the fold
    intercept gone; R2 is scored against the KE ratio's own scale and so needs it
    kept. Returning one array and using it for both is how a graded R2 of +0.30
    turned into -3.91 -- the predictions were correct and simply no longer had a
    mean.
    """
    X = np.asarray(X, np.float64)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(seeds))
    out = np.zeros(len(te_idx))
    raw = np.zeros(len(te_idx))
    for fold in np.array_split(uniq, n_out):
        te_sel = np.flatnonzero(np.isin(seeds[te_idx], fold))
        tr_sel = np.flatnonzero(~np.isin(seeds[tr_idx], fold))
        if len(te_sel) == 0 or len(tr_sel) == 0:
            continue
        tr = tr_idx[tr_sel]
        te = te_idx[te_sel]
        # alpha on an inner grouped split of the TRAINING (honest) clips only
        inner = grouped_folds(seeds[tr], n_in, np.random.default_rng(seed + 7))
        score = {a: 0.0 for a in ALPHAS}
        for f in range(n_in):
            iva = tr[inner[f]]
            itr = np.concatenate([tr[inner[j]] for j in range(n_in) if j != f])
            Y = y_tr[np.searchsorted(tr_idx, itr)].reshape(-1, 1).astype(float)
            yv = y_tr[np.searchsorted(tr_idx, iva)]
            for a, Pv in RidgeSplit(X[itr], X[iva]).predict(Y).items():
                score[a] += (r2(yv.reshape(-1, 1).astype(float), Pv) if graded
                             else np.nan_to_num(auc(yv > 0.5, Pv.ravel()), nan=0.5))
        best = max(ALPHAS, key=lambda a: score[a])
        Y = y_tr[tr_sel].reshape(-1, 1).astype(float)
        pred = RidgeSplit(X[tr], X[te]).predict(Y)[best].ravel()
        raw[te_sel] = pred
        # same fold-intercept removal as `cell_oof_scores` -- see the note there
        out[te_sel] = pred - Y.mean()
    return out, raw


def run_transfer(outdir: str, base_cells: dict, F: list[dict], args) -> dict:
    seeds = np.array([f["seed"] for f in F])
    honest = np.array([not f["viol"] for f in F])
    viol = ~honest
    tr_idx = np.flatnonzero(honest)
    te_idx = np.flatnonzero(viol)
    kr = np.array([f["ke_ratio"] for f in F])
    y_bin = (kr < 1 - 1e-6)[honest]            # honest: elastic vs inelastic
    y_grd = kr[honest]                          # honest: how much energy survived
    y_te = (kr < 0.99)[viol]                    # violating: did it actually lose energy
    kr_te = kr[viol]

    # what a mass-free scalar transfers at, with its SIGN fixed on the honest half
    orc = {}
    for name in MASS_FREE + NEEDS_MASS:
        v = np.array([f[name] for f in F])
        sgn = 1.0 if auc(y_bin, v[honest]) >= 0.5 else -1.0
        orc[name] = auc(y_te, sgn * v[viol])
    mf = max(orc[k] for k in MASS_FREE)

    print(f"\n  --- TRANSFER: fit on {honest.sum()} honest, predict {viol.sum()} violating")
    print(f"      test label = KE ratio < 0.99  ({int(y_te.sum())} lose / {int((~y_te).sum())} do not)")
    print("      transferred kinematic oracles (sign fixed on the honest half):")
    print("        mass-free: " + "  ".join(f"{k} {orc[k]:.3f}" for k in MASS_FREE))
    print("        needs mass: " + "  ".join(f"{k} {orc[k]:.3f}" for k in NEEDS_MASS))
    print(f"        >>> MASS-FREE TRANSFER CEILING {mf:.3f} <<<")

    rng = np.random.default_rng(args.seed)
    perms = np.stack([rng.permutation(y_te) for _ in range(args.n_perm)], axis=1)
    out = {"oracles": orc, "mass_free_ceiling": mf, "routes": {}}
    names = list(base_cells)

    for route, y_fit, graded in (("direct", y_bin.astype(float), False),
                                 ("graded", y_grd, True)):
        obs, nulls, r2s = {}, {}, {}
        for cname, X in base_cells.items():
            sc, raw = transfer_scores(X, y_fit, tr_idx, te_idx, seeds, args.seed,
                                      graded=graded)
            # a graded probe predicts KE RATIO, so a LOW prediction means energy lost
            sc = -sc if graded else sc
            obs[cname] = auc(y_te, sc)
            nulls[cname] = auc_cols(perms > 0.5, np.repeat(sc[:, None], args.n_perm, axis=1))
            # R2 on the RAW predictions: the KE ratio has a scale, and the centred
            # score does not.
            r2s[cname] = r2(kr_te.reshape(-1, 1), raw.reshape(-1, 1)) if graded else float("nan")
        O = np.array([obs[c] for c in names])
        NL = np.stack([nulls[c] for c in names])
        null_max = NL.max(axis=0)
        node = {"per_pool": {}}
        print(f"      [{route}] per-pool best over sites (transferred AUC):")
        for p in POOLS:
            idxs = [i for i, c in enumerate(names) if c.endswith("/" + p)]
            j = idxs[int(np.argmax(O[idxs]))]
            pv = float((NL[j] >= O[j]).mean())
            flag = "   ** beats mass-free ceiling **" if O[j] > mf else ""
            print(f"        {p:9s} {O[j]:.3f}  p={pv:.3f}  null {NL[j].mean():.3f}   "
                  f"{names[j].split('/')[0]}{flag}")
            node["per_pool"][p] = {"best_cell": names[j], "auc": float(O[j]),
                                   "p_uncorrected": pv, "null_mean": float(NL[j].mean())}
        b = int(np.argmax(O))
        p_max = float((null_max >= O[b]).mean())
        extra = ""
        if graded:
            best_r2 = max(r2s.values())
            extra = f"   best R2 on the violating clips' actual KE ratio {best_r2:+.3f}"
            node["best_r2_ke_ratio_violating"] = float(best_r2)
        print(f"      [{route}] GRID BEST {O[b]:.3f} at {names[b]}   max-over-grid null "
              f"{null_max.mean():.3f} (95th {np.percentile(null_max, 95):.3f})  "
              f"p_corrected = {p_max:.3f}{extra}")
        node["grid_best"] = {"cell": names[b], "auc": float(O[b]), "p_corrected": p_max,
                             "null_max_p95": float(np.percentile(null_max, 95))}
        node["full_grid"] = {c: float(obs[c]) for c in names}
        out["routes"][route] = node
    return out



# --------------------------------------------------------------------------- #
# --vs-kinematics: is the probe reading ENERGY, or how fast the balls separate?
# --------------------------------------------------------------------------- #
#
# Printing the probe's AUC next to the |AUC| of the best single visible scalar is
# suggestive, not a test: different estimators, different feature counts, and the
# margins that decide this are small (0.795 against 0.779 on the violating cell).
#
# So a KINEMATIC OBSERVER is fitted through the identical pipeline -- same grouped
# folds, same ridge, same nested alpha search -- on nothing but mass-free scalars,
# and the two are compared by a PAIRED BOOTSTRAP over seeds. Seeds, not clips: the
# twins share a pre-contact segment, so resampling clips would count two views of
# one scene as two draws and shrink the interval.
#
# BOTH SIDES ARE OPTIMISTIC, deliberately and asymmetrically. The probe's cell is
# the argmax over the whole grid chosen on the very data the comparison is scored
# on -- a winner's curse worth roughly +0.05 here, handed to the probe. The
# observer gets no search at all. An interval that still contains zero is therefore
# a conservative negative: the probe was given every advantage and did not use it.

def kinematic_features(F: list[dict], names: list[str]) -> np.ndarray:
    """The named scalars, standardised, plus every pairwise product.

    The products matter. The probe is a linear map over 768 or 1024 dimensions and
    can express interactions the representation happens to hold; a kinematic
    observer restricted to five raw scalars could lose the comparison on functional
    form alone rather than on information. Squares and cross-terms remove that
    excuse -- the observer now spans the full quadratic in what a camera can see.
    """
    X = np.stack([[f[n] for n in names] for f in F]).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cols = [X]
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            cols.append((X[:, i] * X[:, j])[:, None])
    Q = np.concatenate(cols, axis=1)
    return (Q - Q.mean(0)) / (Q.std(0) + 1e-12)


def bootstrap_delta(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                    groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Paired bootstrap over GROUPS of `AUC(s_a) - AUC(s_b)`."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    slots = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([slots[g] for g in pick])
        if y[idx].all() or not y[idx].any():
            continue
        deltas.append(auc(y[idx], s_a[idx]) - auc(y[idx], s_b[idx]))
    d = np.array(deltas)
    return {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_probe_higher": float((d > 0).mean())}


def run_vs_kinematics(base_cells: dict, F: list[dict], want: list[str],
                      all_seeds: np.ndarray, args) -> dict:
    out = {}
    for tname in want:
        spec = TARGETS[tname]
        sel = spec["select"](F)
        y = spec["label"](F)[sel]
        groups = all_seeds[sel]
        Fs = [f for f, k in zip(F, sel) if k]
        Y = y.astype(float).reshape(-1, 1)

        s_free = cell_oof_scores(kinematic_features(Fs, MASS_FREE), Y, groups, args.seed)[:, 0]
        s_all = cell_oof_scores(kinematic_features(Fs, MASS_FREE + NEEDS_MASS), Y,
                                groups, args.seed)[:, 0]

        best_auc, best_name, best_s = -np.inf, None, None
        for cname, X in base_cells.items():
            sc = cell_oof_scores(X[sel], Y, groups, args.seed)[:, 0]
            a = auc(y, sc)
            if a > best_auc:
                best_auc, best_name, best_s = a, cname, sc

        a_free, a_all = auc(y, s_free), auc(y, s_all)
        bf = bootstrap_delta(y, best_s, s_free, groups, args.n_boot, args.seed)
        ba = bootstrap_delta(y, best_s, s_all, groups, args.n_boot, args.seed)

        print(f"\n  --- {tname}   n={sel.sum()}  pos={int(y.sum())}")
        print(f"      probe, best of {len(base_cells)} cells      {best_auc:.3f}   {best_name}")
        print(f"      kinematic observer, MASS-FREE      {a_free:.3f}   "
              f"(quadratic in {len(MASS_FREE)} visible scalars)")
        print(f"      kinematic observer, + true energy  {a_all:.3f}   "
              f"(adds KE pre/post, which need the masses)")
        print(f"      probe - mass-free   {bf['mean']:+.3f}  "
              f"95% CI [{bf['lo']:+.3f}, {bf['hi']:+.3f}]  "
              f"P(probe higher) = {bf['p_probe_higher']:.3f}"
              f"{'   ** probe wins **' if bf['lo'] > 0 else ''}")
        print(f"      probe - full        {ba['mean']:+.3f}  "
              f"95% CI [{ba['lo']:+.3f}, {ba['hi']:+.3f}]  "
              f"P(probe higher) = {ba['p_probe_higher']:.3f}")

        out[tname] = {"n": int(sel.sum()), "pos": int(y.sum()),
                      "probe": {"auc": float(best_auc), "cell": best_name},
                      "kinematic_mass_free": float(a_free),
                      "kinematic_with_energy": float(a_all),
                      "delta_vs_mass_free": bf, "delta_vs_full": ba}
    return out



# --------------------------------------------------------------------------- #
# --fixed-e: the magnitude question, and why it is mass-free too
# --------------------------------------------------------------------------- #
#
# The obvious objection to the binary result is that masses vary hugely here (ratio
# 0.059-24.2, radius AND density both moving), so surely HOW MUCH energy is lost
# needs them. It does not, and the reason is an identity rather than a measurement.
#
# Momentum conservation gives `m_A dv_A = -m_B dv_B`, hence
#
#     m_A / m_B  =  |dv_B| / |dv_A|
#
# with both deflections plainly visible. Verified on all 150 honest clips at max
# relative error 1.02e-15. Substituting that ratio for the true masses -- the scale
# cancels out of a ratio of energies -- reconstructs the KE ratio from visible
# velocities alone at R2 = 1.000000. So there is no mass-requiring target anywhere on
# the honest half, at any mass range.
#
# This mode measures it anyway, because the argument deserves a number. It pins `e` at
# 0.15 (where the KE ratio still spans 0.098-0.992) and races the probe against a
# three-rung observer ladder:
#
#   K1  velocities and impact geometry only -- no size, no mass
#   K2  K1 + the two radii: everything a camera sees, still no density
#   K3  K2 + log mass ratio, log total mass, true KE_pre -- needs the labels
#
# READ THE LADDER'S ORDERING WITH CARE. K1 > K2 > K3 comes out descending, and that is
# CAPACITY, not information: the quadratic expansion runs 54 / 77 / 119 columns against
# n = 75, so each rung overfits harder than the last. It does not mean mass hurts.

FIXED_E_K1 = ["sA", "sB", "sAq", "sBq", "rel_pre", "rel_post", "cos_n", "un", "imp"]
FIXED_E_K2 = FIXED_E_K1 + ["rA", "rB"]
FIXED_E_K3 = FIXED_E_K2 + ["log_mass_ratio", "log_total_mass", "ke_pre"]


def bootstrap_delta_r2(y, pred_a, pred_b, groups, n_boot, seed):
    """Paired bootstrap over GROUPS of `R2(pred_a) - R2(pred_b)`."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    slots = {g: np.flatnonzero(groups == g) for g in uniq}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        i = np.concatenate([slots[g] for g in pick])
        d.append(r2(y[i], pred_a[i]) - r2(y[i], pred_b[i]))
    d = np.array(d)
    return {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p_probe_higher": float((d > 0).mean())}


def identity_check(labels: list[dict]) -> dict:
    """`m_A/m_B == |dv_B|/|dv_A|`, and the KE ratio rebuilt with no mass anywhere."""
    ratio_err, pred, true = [], [], []
    for L in labels:
        if L["violates_momentum"]:
            continue
        c = L["collision"]
        vAp = np.array(c["pre"]["velocity_A"][:2], float)
        vBp = np.array(c["pre"]["velocity_B"][:2], float)
        vAq = np.array(c["post"]["velocity_A"][:2], float)
        vBq = np.array(c["post"]["velocity_B"][:2], float)
        q = np.linalg.norm(vBq - vBp) / np.linalg.norm(vAq - vAp)   # = m_A/m_B, visible
        t = L["objects"]["A"]["mass"] / L["objects"]["B"]["mass"]
        ratio_err.append(abs(q - t) / t)
        pred.append((q * vAq @ vAq + vBq @ vBq) / (q * vAp @ vAp + vBp @ vBp))
        true.append(c["kinetic_energy_ratio"])
    pred, true = np.array(pred), np.array(true)
    return {"n_honest": len(pred),
            "mass_ratio_max_rel_error": float(np.max(ratio_err)),
            "ke_ratio_reconstruction_r2": r2(true.reshape(-1, 1), pred.reshape(-1, 1)),
            "ke_ratio_max_abs_error": float(np.max(np.abs(pred - true)))}


def run_fixed_e(base_cells: dict, F: list[dict], labels: list[dict],
                all_seeds: np.ndarray, args) -> dict:
    idc = identity_check(labels)
    print(f"\n  --- THE IDENTITY, on {idc['n_honest']} honest clips")
    print(f"      m_A/m_B  vs  |dv_B|/|dv_A|      max relative error "
          f"{idc['mass_ratio_max_rel_error']:.2e}")
    print(f"      KE ratio from VISIBLE VELOCITIES ONLY, no mass anywhere:  "
          f"R2 = {idc['ke_ratio_reconstruction_r2']:.6f}  "
          f"(max abs error {idc['ke_ratio_max_abs_error']:.2e})")

    sel = np.array([(not f["viol"]) and f["e"] == 0.15 for f in F])
    groups = all_seeds[sel]
    Fs = [f for f, k in zip(F, sel) if k]
    kr = np.array([f["ke_ratio"] for f in F])
    y = kr[sel].reshape(-1, 1)
    print(f"\n  --- FIXED e = 0.15, honest only, n={sel.sum()}   "
          f"KE ratio {y.min():.3f}-{y.max():.3f}, sd {y.std():.3f}")
    print("      e is constant, so every bit of this variation is mass ratio x geometry")

    tiers = {"K1 kinematics only": FIXED_E_K1,
             "K2 + radii (all visible)": FIXED_E_K2,
             "K3 + true masses": FIXED_E_K3}
    preds = {t: cell_oof_regression(kinematic_features(Fs, f), y, groups, args.seed)
             for t, f in tiers.items()}

    best = (-np.inf, None, None)
    for cname, X in base_cells.items():
        pr = cell_oof_regression(X[sel], y, groups, args.seed)
        sc = r2(y, pr)
        if sc > best[0]:
            best = (sc, cname, pr)
    print(f"      probe, best of {len(base_cells)} cells        R2 {best[0]:+.3f}   {best[1]}")
    out = {"identity": idc, "n": int(sel.sum()),
           "probe": {"r2": float(best[0]), "cell": best[1]}, "observers": {}, "deltas": {}}
    for t, pr in preds.items():
        out["observers"][t] = r2(y, pr)
        print(f"      observer {t:26s} R2 {r2(y, pr):+.3f}")
    for t, pr in preds.items():
        d = bootstrap_delta_r2(y, best[2], pr, groups, args.n_boot, args.seed)
        flag = ("   ** probe wins **" if d["lo"] > 0
                else "   ** probe loses **" if d["hi"] < 0 else "")
        print(f"      probe - {t:26s} {d['mean']:+.3f}  "
              f"95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
              f"P(probe higher) = {d['p_probe_higher']:.3f}{flag}")
        out["deltas"][t] = d

    # mass is not the bottleneck, and this is the line that rules that out
    print("      is MASS even there?  (grouped 5-fold, best cell, R2)")
    for tag, v in (("log mass_A", np.log([f["mass_A"] for f in F])),
                   ("log mass_B", np.log([f["mass_B"] for f in F]))):
        yv = np.asarray(v).reshape(-1, 1)
        b = max((r2(yv, cell_oof_regression(X, yv, all_seeds, args.seed)), n)
                for n, X in base_cells.items())
        print(f"         {tag:12s} {b[0]:+.3f}   {b[1]}")
        out.setdefault("mass_decodability", {})[tag] = {"r2": float(b[0]), "cell": b[1]}
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def permuted_labels(y: np.ndarray, groups: np.ndarray, n_perm: int,
                    seed: int) -> np.ndarray:
    """`[n, n_perm + 1]`; column 0 is the observed label.

    Whole GROUPS are permuted, never clips. A seed's honest and violating twins
    share their pre-contact segment, so shuffling within a seed would leave the
    permuted label predictable from features the null is meant to strip.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    slots = [np.flatnonzero(groups == g) for g in uniq]
    cols = [y.astype(float)]
    for _ in range(n_perm):
        perm = rng.permutation(len(uniq))
        z = np.zeros(len(y))
        for dst, src in enumerate(perm):
            # groups are equal-sized on every target here (1 or 2 clips); assign
            # the source group's labels into the destination group's slots.
            s, d = slots[src], slots[dst]
            z[d] = y[s][:len(d)] if len(s) >= len(d) else np.resize(y[s], len(d))
        cols.append(z)
    return np.stack(cols, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+",
                    default=["outputs/cons_videomae-base", "outputs/cons_vjepa2-vitl"])
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-dir", default="datasets/conservation_set_v1")
    ap.add_argument("--fixed-e", action="store_true",
                    help="the MAGNITUDE question at e pinned to 0.15, plus the\nidentity that makes it mass-free (see above)")
    ap.add_argument("--vs-kinematics", action="store_true",
                    help="compare the probe against a kinematic observer "
                         "fitted through the identical pipeline (see above)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for --vs-kinematics")
    ap.add_argument("--transfer", action="store_true",
                    help="fit on honest clips, predict violating ones (see above)")
    ap.add_argument("--mask-floor", action="store_true",
                    help="replace activations with the pooling masks (§20 control)")
    ap.add_argument("-o", "--out", default="docs/artifacts/energy_probes.json")
    args = ap.parse_args()

    want = [t.strip() for t in args.targets.split(",") if t.strip()]
    report: dict = {"pools": POOLS, "n_perm": args.n_perm, "targets": {}}

    for outdir in args.outputs:
        acts, labels, pools, stems = load_captures(outdir)
        F = [scene_facts(L) for L in labels]
        sites = list(acts)
        if args.mask_floor:
            base_cells = mask_floor_cells(outdir, args.dataset_dir, stems)
        else:
            base_cells = {f"{s}/{p}": acts[s][:, pools.index(p), :]
                          for s in sites for p in POOLS}
        print(f"\n{'=' * 88}\n{outdir}   {len(F)} clips, {len(sites)} sites, "
              f"{len(base_cells)} cells{'   [MASK-ONLY FLOOR]' if args.mask_floor else ''}")

        all_seeds = np.array([f["seed"] for f in F])

        if args.fixed_e:
            report.setdefault("fixed_e", {})[outdir] = run_fixed_e(
                base_cells, F, labels, all_seeds, args)
            continue

        if args.vs_kinematics:
            report.setdefault("vs_kinematics", {})[outdir] = run_vs_kinematics(
                base_cells, F, want, all_seeds, args)
            continue

        if args.transfer:
            report.setdefault("transfer", {})[outdir] = run_transfer(outdir, base_cells, F, args)
            continue

        for tname in want:
            spec = TARGETS[tname]
            sel = spec["select"](F)
            y_full = spec["label"](F)
            y = y_full[sel]
            groups = all_seeds[sel]
            cells = {n: X[sel] for n, X in base_cells.items()}
            Ys = permuted_labels(y, groups, args.n_perm, args.seed)

            orc = oracle_aucs(F, sel, y)
            mf = max(orc[k] for k in MASS_FREE)
            print(f"\n  --- {tname}   n={sel.sum()}  pos={int(y.sum())}  neg={int((~y).sum())}")
            print(f"      {spec['note']}")
            print("      kinematic oracles (|AUC| from ONE visible scalar):")
            print("        mass-free: " + "  ".join(f"{k} {orc[k]:.3f}" for k in MASS_FREE))
            print("        needs mass: " + "  ".join(f"{k} {orc[k]:.3f}" for k in NEEDS_MASS))
            print(f"        >>> MASS-FREE CEILING {mf:.3f} <<<")

            grid = {}
            for cname, X in cells.items():
                oof = cell_oof_scores(X, Ys, groups, args.seed)
                grid[cname] = auc_cols(np.repeat(y[:, None], Ys.shape[1], axis=1), oof)
            A = np.stack([grid[c] for c in cells])                    # [C, P+1]
            obs, null = A[:, 0], A[:, 1:]
            null_max = null.max(axis=0)
            best_i = int(np.argmax(obs))
            best_name = list(cells)[best_i]
            p_max = float((null_max >= obs[best_i]).mean())

            print(f"      per-pool best over sites (AUC, uncorrected p, null mean):")
            per_pool = {}
            for p in POOLS:
                idxs = [i for i, c in enumerate(cells) if c.endswith("/" + p)]
                j = idxs[int(np.argmax(obs[idxs]))]
                cn = list(cells)[j]
                pv = float((null[j] >= obs[j]).mean())
                per_pool[p] = {"best_cell": cn, "auc": float(obs[j]), "p_uncorrected": pv,
                               "null_mean": float(null[j].mean()),
                               "null_p95": float(np.percentile(null[j], 95))}
                flag = "" if obs[j] <= mf + 1e-9 else "   ** beats mass-free ceiling **"
                print(f"        {p:9s} {obs[j]:.3f}  p={pv:.3f}  null {null[j].mean():.3f} "
                      f"(95th {np.percentile(null[j], 95):.3f})   {cn.split('/')[0]}{flag}")
            bal = oof_threshold_accuracy(cells[best_name], y, groups, args.seed)
            print(f"      GRID BEST {obs[best_i]:.3f} at {best_name}  "
                  f"balanced acc {bal:.3f}  max-over-grid null mean {null_max.mean():.3f} "
                  f"(95th {np.percentile(null_max, 95):.3f})  p_corrected = {p_max:.3f}")

            node = report["targets"].setdefault(tname, {})
            node["note"] = spec["note"]
            node[outdir] = {
                "n": int(sel.sum()), "pos": int(y.sum()),
                "oracles": orc, "mass_free_ceiling": mf,
                "per_pool": per_pool,
                "grid_best": {"cell": best_name, "auc": float(obs[best_i]),
                              "balanced_accuracy": bal,
                              "null_max_mean": float(null_max.mean()),
                              "null_max_p95": float(np.percentile(null_max, 95)),
                              "p_corrected": p_max},
                "full_grid": {c: float(obs[i]) for i, c in enumerate(cells)},
            }

        if not args.mask_floor:
            # positive controls: the captures are known-good if these read high
            print("\n  positive controls (grouped 5-fold R2, best cell):")
            for label, vec in (("kinetic_energy_pre", np.array([f["ke_pre"] for f in F])),
                               ("ke_ratio", np.array([f["ke_ratio"] for f in F]))):
                yv = vec.reshape(-1, 1)
                best = max(
                    (r2(yv, cell_oof_regression(X, yv, all_seeds, args.seed)), n)
                    for n, X in base_cells.items())
                print(f"     {label:20s} {best[0]:+.3f}   {best[1]}")
                report.setdefault("controls", {}).setdefault(outdir, {})[label] = \
                    {"r2": best[0], "cell": best[1]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
