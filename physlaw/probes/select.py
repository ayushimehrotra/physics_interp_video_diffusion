"""Held-out selection rule and multiple-comparison correction (spec §5 shared protocol, §7).

Selection rule: whenever a "best" layer, pool, noise level or distance is reported, it is chosen on
one set of held-out groups and scored on a DIFFERENT set. ``heldout_select`` splits the twin groups
into two random halves, picks the variant with the best score on one half, scores that variant on
the other half, and does so both ways for ``n_splits`` random splits. Scores are computed from
quantities that are already out-of-fold for every group (cross-fitted probe predictions, or zero-fit
surprise), so both halves are held out from every fit; the halves only separate selection from
reporting. The reported score is the mean over the 2 x n_splits evaluations; the selected variant is
the most frequent pick, with how often each variant was picked.

Holm: step-down family-wise correction over the contrasts of one family and level.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np


def holm(p: Mapping[Hashable, float]) -> dict:
    """Holm-adjusted p-values (monotone, capped at 1). NaN p-values are passed through."""
    keys = [k for k in p if np.isfinite(p[k])]
    order = sorted(keys, key=lambda k: p[k])
    m, run, out = len(order), 0.0, {k: float("nan") for k in p}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def group_halves(groups: Sequence, n_splits: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    ug = np.unique(np.asarray(groups))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(ug)
        out.append((np.sort(perm[: len(ug) // 2]), np.sort(perm[len(ug) // 2:])))
    return out


def heldout_select(score: Callable[[Hashable, np.ndarray], float], variants: Sequence[Hashable],
                   groups: Sequence, *, n_splits: int = 50, seed: int = 0, higher: bool = True) -> dict:
    """``score(variant, group_ids)`` -> the variant's score on those groups (NaN allowed).
    Returns {selected, selected_counts, score, score_sd, score_q025, score_q975, n_variants,
    n_groups, n_select_groups, n_eval_groups, n_splits}."""
    variants = list(variants)
    sign = 1.0 if higher else -1.0
    picks, evals = [], []
    halves = group_halves(groups, n_splits, seed)
    for a, b in halves:
        for sel, ev in ((a, b), (b, a)):
            if len(variants) == 1:
                best = variants[0]
            else:
                s = sign * np.array([score(v, sel) for v in variants], float)
                best = variants[int(np.argmax(np.where(np.isnan(s), -np.inf, s)))]
            picks.append(best)
            evals.append(score(best, ev))
    ev = np.array(evals, float)
    counts = Counter(picks)
    return {"selected": counts.most_common(1)[0][0], "selected_counts": {str(k): c for k, c in counts.most_common()},
            "score": float(np.nanmean(ev)), "score_sd": float(np.nanstd(ev)),
            "score_q025": float(np.nanquantile(ev, 0.025)), "score_q975": float(np.nanquantile(ev, 0.975)),
            "n_variants": len(variants), "n_groups": int(len(np.unique(np.asarray(groups)))),
            "n_select_groups": int(len(halves[0][0])), "n_eval_groups": int(len(halves[0][1])),
            "n_splits": n_splits}


def protocol_note(n_groups: int, k_folds: int, n_variants: int, fold_seeds: int | None = None) -> str:
    """One line for every reported number (spec §7): groups, test groups per fold, variants searched."""
    s = f"n_groups={n_groups}, test groups/fold~{n_groups / k_folds:.0f} ({k_folds}-fold"
    s += f" x {fold_seeds} seeds)" if fold_seeds else ")"
    return s + f", variants searched={n_variants}"
