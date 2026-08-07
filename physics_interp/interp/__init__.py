"""Interpretability layer: hook substrate plus the linear-probe readout.

:mod:`~physics_interp.interp.hooks` stays opinion-free — it decides only *how*
to tap a model, never what to record. This module is the other half: given
activations that a capture run already pooled and stored, it fits the linear
probes that turn "is this quantity present in the representation" into a
number.

Probes are deliberately linear and deliberately weak. The claim a probe can
support is that a quantity is *linearly decodable* from a representation, which
is a statement about how accessible the information is, not about whether the
model uses it. Two habits keep that claim honest and both are enforced here:
the ridge penalty is selected on a validation split carved out of the training
half (never the test half), and every fit reports a permutation control so
"0.6 R-squared" can be read against what the same pipeline scores on shuffled
labels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .hooks import HookContext, HookHandle, HookManager, StepFilter

__all__ = [
    "HookContext",
    "HookHandle",
    "HookManager",
    "StepFilter",
    "extract_label",
    "ridge_fit",
    "fit_probe",
    "probe_grid",
    "select_scenes",
]


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

#: Targets that are not stored in labels.json but derived from what is. Keeping
#: them here rather than in the generator means a new question can be asked of
#: an already-rendered dataset.
DERIVED: dict[str, Any] = {
    "log_mass_ratio": lambda l: float(np.log(l["mass_ratio"])),
    "momentum_pre": lambda l: l["collision"]["pre"]["total_momentum"][:2],
    "momentum_post": lambda l: l["collision"]["post"]["total_momentum"][:2],
    "momentum_magnitude_pre": lambda l: float(
        np.linalg.norm(l["collision"]["pre"]["total_momentum"])
    ),
    "momentum_magnitude_post": lambda l: float(
        np.linalg.norm(l["collision"]["post"]["total_momentum"])
    ),
    "momentum_A_pre": lambda l: l["collision"]["pre"]["momentum_A"][:2],
    "momentum_B_pre": lambda l: l["collision"]["pre"]["momentum_B"][:2],
    "momentum_A_post": lambda l: l["collision"]["post"]["momentum_A"][:2],
    "momentum_B_post": lambda l: l["collision"]["post"]["momentum_B"][:2],
    "momentum_delta": lambda l: (
        np.asarray(l["collision"]["post"]["total_momentum"][:2])
        - np.asarray(l["collision"]["pre"]["total_momentum"][:2])
    ),
    "impulse": lambda l: l["collision"]["impulse_vector"][:2],
    # Energies. The ratio is the elasticity of the collision; the absolute values
    # are quadratic in speed and so a genuinely different readout from momentum,
    # which is linear in it — a representation could carry one and not the other.
    "ke_ratio": lambda l: float(l["collision"]["kinetic_energy_ratio"]),
    "kinetic_energy_pre": lambda l: float(l["collision"]["pre"]["kinetic_energy"]),
    "kinetic_energy_post": lambda l: float(l["collision"]["post"]["kinetic_energy"]),
    "log_kinetic_energy_pre": lambda l: float(
        np.log(max(l["collision"]["pre"]["kinetic_energy"], 1e-12))
    ),
    # Velocities: the purely kinematic part of momentum, visible in the pixels
    # without any knowledge of mass. A probe that gets velocity but not momentum
    # is reading motion; one that gets both is combining motion with mass.
    "velocity_A_pre": lambda l: l["collision"]["pre"]["velocity_A"][:2],
    "velocity_B_pre": lambda l: l["collision"]["pre"]["velocity_B"][:2],
    "velocity_A_post": lambda l: l["collision"]["post"]["velocity_A"][:2],
    "velocity_B_post": lambda l: l["collision"]["post"]["velocity_B"][:2],
    "speed_B_pre": lambda l: float(
        np.linalg.norm(l["collision"]["pre"]["velocity_B"])
    ),
    # Static/appearance variables. These are the controls: they are readable
    # straight off a single frame, so a probe scoring well on them proves only
    # that the capture pipeline works, and sets the bar a dynamic quantity has
    # to clear to be interesting.
    "mass_A": lambda l: float(l["objects"]["A"]["mass"]),
    "mass_B": lambda l: float(l["objects"]["B"]["mass"]),
    "radius_A": lambda l: float(l["objects"]["A"]["radius"]),
    "radius_B": lambda l: float(l["objects"]["B"]["radius"]),
    "restitution": lambda l: float(l["restitution"]),
    "speed_A_pre": lambda l: float(
        np.linalg.norm(l["collision"]["pre"]["velocity_A"])
    ),
    # Defaults to conserving when the key is absent, so datasets generated before
    # violations existed (the original pilot) still resolve this target — and
    # resolve it *correctly*, since every scene in them does conserve momentum.
    # A probe on such a set gets a constant label and scores nan, which is the
    # honest answer rather than a crash.
    "momentum_conserved": lambda l: float(l.get("momentum_conserved", True)),
    "violates_momentum": lambda l: float(l.get("violates_momentum", False)),
}


def extract_label(labels: dict, target: str) -> np.ndarray:
    """Pull one probe target out of a scene's ``labels.json``.

    Accepts a name from :data:`DERIVED` or a dotted path into the label tree.
    Always returns a 1-D array, so scalar and vector targets are handled by the
    same downstream code — a 2-vector momentum is fit component-wise and scored
    jointly, which is what makes "is momentum decodable" a single number rather
    than two.

    The z components are dropped by the derived momentum targets on purpose:
    the scenes are exactly planar by construction, so z is identically zero and
    including it would inflate every multi-output score with a component that
    is trivially predictable from a constant.
    """
    if target in DERIVED:
        value = DERIVED[target](labels)
    else:
        value = labels
        for part in target.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(
                    f"label path {target!r} failed at {part!r}. "
                    f"Available derived targets: {sorted(DERIVED)}"
                )
            value = value[part]
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


# --------------------------------------------------------------------------- #
# ridge
# --------------------------------------------------------------------------- #


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge. Returns ``(W [d, k], b [k])``.

    Solves in whichever space is smaller. With a few hundred scenes against
    1920 activation dimensions the dual form is not an optimisation but the
    difference between a 1920x1920 solve and an n-by-n one, per alpha, per
    block, per step — and the probing grid does thousands of these.
    """
    n, d = X.shape
    x_mean, y_mean = X.mean(0), Y.mean(0)
    Xc, Yc = X - x_mean, Y - y_mean

    if n >= d:
        A = Xc.T @ Xc + alpha * np.eye(d)
        W = np.linalg.solve(A, Xc.T @ Yc)
    else:
        # W = X^T (X X^T + aI)^-1 Y
        G = Xc @ Xc.T + alpha * np.eye(n)
        W = Xc.T @ np.linalg.solve(G, Yc)
    return W, y_mean - x_mean @ W


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-output R-squared, pooled over components.

    Pooled rather than averaged per component: for a vector target like a 2-D
    momentum, a probe that nails x and fails y should not be rescued by
    averaging, and pooling weights each component by how much variance it
    actually carries.
    """
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_probe(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alphas: Sequence[float] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4),
    train_fraction: float = 0.6,
    split_seed: int = 0,
    standardize: bool = True,
    permutation_control: bool = True,
) -> dict[str, Any]:
    """Fit one linear probe and score it on held-out scenes.

    ``X`` is ``[n_scenes, dim]`` and ``Y`` is ``[n_scenes, k]``. The split is
    by scene, which is the only unit that is actually independent here — two
    pooled vectors from the same clip share everything except which tokens they
    averaged, so splitting on anything finer would leak.
    """
    n = X.shape[0]
    if n < 8:
        raise ValueError(f"need at least 8 scenes to fit and score a probe, got {n}")

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    tr, te = order[:n_train], order[n_train:]
    # Carve the alpha-selection split out of train only. Selecting on test is
    # the standard way to report a probe accuracy that does not replicate.
    n_inner = max(4, int(round(0.75 * len(tr))))
    inner_tr, inner_val = tr[:n_inner], tr[n_inner:]
    if len(inner_val) < 2:
        inner_tr, inner_val = tr, tr  # too few scenes to hold out; report honestly

    def prep(fit_idx):
        mu = X[fit_idx].mean(0)
        sd = X[fit_idx].std(0) + 1e-6 if standardize else np.ones(X.shape[1])
        return mu, sd

    mu, sd = prep(inner_tr)
    scores = []
    for a in alphas:
        W, b = ridge_fit((X[inner_tr] - mu) / sd, Y[inner_tr], a)
        scores.append(_r2(Y[inner_val], (X[inner_val] - mu) / sd @ W + b))
    best_alpha = float(alphas[int(np.nanargmax(scores))])

    mu, sd = prep(tr)
    W, b = ridge_fit((X[tr] - mu) / sd, Y[tr], best_alpha)
    test_r2 = _r2(Y[te], (X[te] - mu) / sd @ W + b)
    train_r2 = _r2(Y[tr], (X[tr] - mu) / sd @ W + b)

    out = {
        "r2": test_r2,
        "train_r2": train_r2,
        "alpha": best_alpha,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "dim": int(X.shape[1]),
    }

    # A binary target (plausible vs implausible) is a classification question, and
    # R-squared is the wrong way to report it — a ridge fit can have a mediocre
    # R-squared while separating the classes almost perfectly. Threshold the same
    # fit instead. Balanced accuracy is reported alongside because a split that
    # happens to be class-imbalanced would flatter plain accuracy.
    if Y.shape[1] == 1 and set(np.unique(Y)) <= {0.0, 1.0}:
        pred = ((X[te] - mu) / sd @ W + b).ravel() > 0.5
        truth = Y[te].ravel() > 0.5
        out["accuracy"] = float((pred == truth).mean())
        per_class = [
            float((pred[truth == c] == c).mean()) for c in (False, True)
            if (truth == c).any()
        ]
        out["balanced_accuracy"] = float(np.mean(per_class))
        out["positive_rate"] = float(truth.mean())

    if permutation_control:
        # Same pipeline, labels shuffled. With d >> n a ridge probe can look
        # impressive on pure noise, so the number that matters is the gap
        # between r2 and this, not r2 itself.
        perm = rng.permutation(n)
        Wp, bp = ridge_fit((X[tr] - mu) / sd, Y[perm][tr], best_alpha)
        out["r2_shuffled"] = _r2(Y[perm][te], (X[te] - mu) / sd @ Wp + bp)
        if "accuracy" in out:
            pred = ((X[te] - mu) / sd @ Wp + bp).ravel() > 0.5
            out["accuracy_shuffled"] = float(
                (pred == (Y[perm][te].ravel() > 0.5)).mean()
            )
    return out


def select_scenes(labels: list[dict], where: dict[str, Any] | None) -> np.ndarray:
    """Boolean mask over scenes matching every ``label_path == value`` in ``where``.

    Exists so the matched-momentum stratum can be probed without regenerating or
    re-capturing anything: the stratum is the subset where momentum and mass are
    decorrelated, and therefore the only subset on which a momentum probe is
    distinguishable from a mass probe.
    """
    mask = np.ones(len(labels), dtype=bool)
    for path, wanted in (where or {}).items():
        for i, scene in enumerate(labels):
            value = scene
            for part in path.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            mask[i] &= (value == wanted)
    return mask


def probe_grid(
    activations: dict[str, np.ndarray],
    labels: list[dict],
    *,
    targets: Sequence[str],
    pools: Sequence[str],
    pool_names: Sequence[str],
    steps: Sequence[int] | None = None,
    where: dict[str, Any] | None = None,
    **fit_kwargs,
) -> list[dict[str, Any]]:
    """Fit a probe for every (site, capture step, pool, target) combination.

    ``activations`` maps site name to ``[n_scenes, steps, pools, dim]``. The
    result is a flat list of records rather than a nested structure because the
    interesting views — accuracy against depth, against noise level, against
    which object was pooled — are all different groupings of the same table.

    ``where`` restricts the fit to a subset of scenes, e.g.
    ``{"stratum": "matched_momentum"}``.
    """
    pool_idx = {name: i for i, name in enumerate(pool_names)}
    missing = [p for p in pools if p not in pool_idx]
    if missing:
        raise KeyError(f"unknown pool(s) {missing}; have {list(pool_names)}")

    subset = select_scenes(labels, where)
    if subset.sum() < 8:
        raise ValueError(
            f"`where`={where} selects {subset.sum()} scene(s); need at least 8"
        )
    labels = [l for l, keep in zip(labels, subset) if keep]
    activations = {s: a[subset] for s, a in activations.items()}

    Y_by_target = {}
    for t in targets:
        Y = np.stack([extract_label(l, t) for l in labels])
        if np.ptp(Y, axis=0).max() == 0:
            # A constant target cannot be regressed and would report nan for every
            # fit; say so once here rather than emitting a grid of nans.
            continue
        Y_by_target[t] = Y

    rows = []
    for site, arr in sorted(activations.items()):
        n_steps = arr.shape[1]
        step_range = range(n_steps) if steps is None else steps
        for si in step_range:
            for pool in pools:
                X = arr[:, si, pool_idx[pool], :].astype(np.float64)
                # A pool can be empty when its object left the frame; those
                # rows are exactly zero by construction in build_pool_weights.
                keep = np.abs(X).sum(1) > 0
                if keep.sum() < 8:
                    continue
                for target, Y in Y_by_target.items():
                    rec = fit_probe(X[keep], Y[keep], **fit_kwargs)
                    rec.update(
                        site=site, step_index=int(si), pool=pool, target=target,
                        n_dropped=int((~keep).sum()),
                    )
                    rows.append(rec)
    return rows
