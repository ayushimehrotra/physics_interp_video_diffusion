#!/usr/bin/env python
"""Do the models make LAWFUL errors? — the §14 dataset used as an instrument.

The question every probe in this project has failed to answer is whether the
model *applies* conservation or merely decodes the kinematics that a conserved
quantity is computed from. Probing cannot separate those: a readout can always
do the arithmetic itself (§26), and any label a probe could succeed on is by
definition a function of what is visible (§12).

This asks it a different way, and does not need a decodable label at all.

    If the model carries a conserved momentum, then its estimate of p_post is
    not an independent readout — it is tied to its own estimate of p_pre. So
    the two estimates' ERRORS should be shared, and on scenes where the
    simulator violated conservation the model should be wrong IN THE DIRECTION
    OF THE LAW: it should under-register the violation.

    If the model merely decodes velocities at two times, the two errors are
    unrelated and the violation is registered at whatever fidelity the readout
    affords, with no directional bias.

`probe_set_v3_latent` is the right instrument for exactly the reason §14 filed
it as a dead end. Its violations keep `dv_A` and `dv_B` antiparallel and its
masses are invisible, so the violation is undetectable from the pixels by ANY
observer — §14's null is a theorem about the dataset. That is what makes the
test clean: no information about the violation is available in the scene, so a
systematic lawful bias cannot be the model reading the answer off the clip. It
can only be the model's own constraint.

WHAT IS MEASURED
----------------
A single readout `w` is fit to map activations -> total momentum, trained on
CONSERVING scenes only and TIED across time: the same `w` is applied to the
pre-contact pool and the post-contact pool. Tying is the whole point — with two
separately fit readouts there is no shared quantity to talk about.

Three statistics. Read them in this order, and note which are void on v3.

  rho    PARTIAL corr(eps_pre, eps_post) given the true momentum, on held-out
         CONSERVING scenes. The partial is essential: p_pre == p_post exactly on
         a conserving scene, so both errors contain the same -p term, and a weak
         readout whose output is near-constant gives raw rho -> 1 with nothing in
         the model doing any work (measured: 0.998 at `embed.out`, R2 0.003).
         Conditioning removes that term. Null: permute which scene's post-error
         pairs with which scene's pre-error. Grid-max over (site, step, pool).

  beta   slope of dp_hat on true dp over VIOLATING scenes. The number to compare
         it to is `r2_post`, NOT 1.0 -- a readout of fidelity r attenuates any
         signal to roughly r. Law-use predicts the CHANGE is attenuated more than
         the level is, i.e. beta well below r2_post.

  pull   the component of eps_post along the direction that would restore
         conservation, over sum |dp|. Null: random unit directions.

  VOID ON v3: `beta` and `pull` both require the model to be ABLE to register
  the violation, and §14 proves no observer can -- the deflections stay
  antiparallel and the masses are invisible. A readout that fails to register an
  unregisterable violation is not evidence of anything. Those two arms need a set
  where dp is inferable, i.e. the `visible_mass` arm. On v3, read `rho` only.

CONTROLS THAT DECIDE IT
-----------------------
Pre-contact and post-contact pools come from the SAME clip, so both errors
carry a shared scene-level appearance component. That alone produces rho > 0
with no physics involved. Three things separate the two readings, and none of
them is optional:

  * the pre-transformer site (`embed.out` / `vae.latent`). Sharing that is
    already present before a block has run is appearance, not construction.
  * the depth curve. §34's `momentum_delta` rose from ~0 at the embedding to a
    mid-to-late peak; if rho is real it should do the same.
  * --placebo ke. Kinetic energy is the same SHAPE of quantity as momentum —
    bilinear in mass and velocity, read from the same pools — and is NOT
    conserved here (restitution < 1). If rho is as high for KE, what is being
    measured is generic scene-level error sharing and there is no conservation
    result. This is the control the whole script stands on.
  * --min-r2, the readout floor. A cell with no working readout cannot answer
    the question and must be excluded from SELECTION as well as from reporting,
    or the argmax lands on a degenerate cell.

RESULT AS OF 2026-08-22 — NEGATIVE, both encoders. rho is high (0.92-0.96
against per-cell nulls of 0.14-0.19) but the KE placebo matches it (0.89-0.96),
and on VideoMAE the placebo is already at 0.835 in block 0 where momentum is at
0.396. The sharing is generic, between two pools of the same clip. See
docs/RESULTS.md §6.

USAGE
-----
    python scripts/lawful_errors.py outputs_v3_videomae-base/momentum_probe \
        --n-perm 500 --out lawful_errors.json
    python scripts/lawful_errors.py outputs_v3_videomae-base/momentum_probe \
        --placebo ke

No GPU. Reads only what `--mode capture` already wrote.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_captures(act_dir: Path):
    """Stack every captured scene. Mirrors run_experiment.load_captures.

    Returns (activations {site: [n, steps, pools, dim]}, labels, meta0).
    """
    paths = sorted(act_dir.glob("scene_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no captures in {act_dir}; run --mode capture first")

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

    n = len(paths)
    ragged = {s: len(v) for s, v in per_site.items() if len(v) != n}
    if ragged:
        raise ValueError(f"site(s) captured for only some scenes: {ragged} of {n}")
    return {s: np.stack(v) for s, v in per_site.items()}, labels, metas[0]


def targets_from(labels: list[dict], kind: str):
    """(Y_pre, Y_post) for the tied readout, plus a human name.

    momentum — the 2-vector total, z dropped (the scenes are exactly planar).
    ke       — the placebo: same bilinear shape, same pools, NOT conserved.
    """
    if kind == "momentum":
        pre = np.stack([np.asarray(l["collision"]["pre"]["total_momentum"][:2], float)
                        for l in labels])
        post = np.stack([np.asarray(l["collision"]["post"]["total_momentum"][:2], float)
                         for l in labels])
        return pre, post, "total_momentum"
    if kind == "ke":
        pre = np.array([[float(l["collision"]["pre"]["kinetic_energy"])] for l in labels])
        post = np.array([[float(l["collision"]["post"]["kinetic_energy"])] for l in labels])
        return pre, post, "kinetic_energy"
    raise ValueError(f"unknown target kind {kind!r}")


def conserved_mask(labels: list[dict]) -> np.ndarray:
    """True where the scene conserves momentum.

    Defaults to conserving when the key is absent, matching DERIVED's
    `momentum_conserved`, so a pre-violation dataset still resolves.
    """
    return np.array([bool(l.get("momentum_conserved", True)) for l in labels])


def pool_pairs(pool_names: list[str]) -> list[tuple[str, str, int, int]]:
    """(name_pre, name_post, idx_pre, idx_post) for every pool with both phases.

    The tied readout needs the same feature space at two times, which is what a
    `X.pre` / `X.post` pair is: same site, same dim, different token subset.
    """
    idx = {n: i for i, n in enumerate(pool_names)}
    out = []
    for name in pool_names:
        if name.endswith(".pre"):
            post = name[:-4] + ".post"
            if post in idx:
                out.append((name, post, idx[name], idx[post]))
    return out


# --------------------------------------------------------------------------- #
# the tied readout
# --------------------------------------------------------------------------- #

def _ridge(X, Y, alpha):
    """Closed-form ridge in whichever space is smaller. Returns (W, b)."""
    n, d = X.shape
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    if n >= d:
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)
    else:
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(n), Yc)
    return W, ym - xm @ W


def _r2(y, yhat) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_tied(X_pre, X_post, Y_pre, Y_post, train, *, alphas=ALPHAS, seed=0):
    """One readout, fit on the stacked (pre, post) rows of the TRAIN scenes.

    Stacking is what ties the weights: the fit sees both phases as samples of
    the same activation -> momentum map, so a single direction has to serve
    both. Alpha is chosen on an inner carve-out of train, never on test —
    same discipline as `fit_probe`.
    """
    Xt = np.vstack([X_pre[train], X_post[train]])
    Yt = np.vstack([Y_pre[train], Y_post[train]])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    n_inner = max(4, int(round(0.75 * len(train))))
    inner_tr = np.concatenate([order[:n_inner], order[:n_inner] + len(train)])
    inner_va = np.concatenate([order[n_inner:], order[n_inner:] + len(train)])
    if len(inner_va) < 4:
        inner_tr = inner_va = np.arange(len(Xt))

    mu, sd = Xt[inner_tr].mean(0), Xt[inner_tr].std(0) + 1e-6
    scores = []
    for a in alphas:
        W, b = _ridge((Xt[inner_tr] - mu) / sd, Yt[inner_tr], a)
        scores.append(_r2(Yt[inner_va], (Xt[inner_va] - mu) / sd @ W + b))
    alpha = float(alphas[int(np.nanargmax(scores))])

    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    W, b = _ridge((Xt - mu) / sd, Yt, alpha)
    apply = lambda X: (X - mu) / sd @ W + b
    return apply, alpha, _r2(Yt, apply(Xt))


# --------------------------------------------------------------------------- #
# the three statistics
# --------------------------------------------------------------------------- #

def _residualize(E, Z):
    """Remove the part of E that any linear function of Z explains (OLS, intercept)."""
    A = np.hstack([Z, np.ones((len(Z), 1))])
    return E - A @ np.linalg.lstsq(A, E, rcond=None)[0]


def error_sharing(eps_pre, eps_post, p_true=None) -> float:
    """PARTIAL corr(eps_pre, eps_post) controlling for the true target.

    The raw correlation is degenerate here and the first run proved it. On a
    conserving scene ``p_pre == p_post`` exactly, so both errors contain the
    same ``-p`` term; when the readout is weak its output is near-constant and
    the errors ARE the target, giving rho -> 1 with nothing in the model doing
    any work. Measured: rho = 0.998 at `embed.out` on a readout scoring R2
    0.003. That is target correlation, not error sharing.

    Conditioning on the true momentum removes exactly that term and leaves the
    part of each error the target cannot explain -- which is the only part a
    shared internal quantity could be responsible for.

    Each component is centred and scaled on its own before pooling, so a
    component carrying more variance cannot dominate the correlation.
    """
    a, b = eps_pre.copy(), eps_post.copy()
    if p_true is not None:
        a, b = _residualize(a, p_true), _residualize(b, p_true)
    for M in (a, b):
        M -= M.mean(0)
        M /= M.std(0) + 1e-12
    return float((a.ravel() * b.ravel()).mean())


def sharing_null(eps_pre, eps_post, p_true, n_perm, rng) -> np.ndarray:
    """Null for rho: break the scene pairing, keep both marginals intact.

    Shuffling labels would be the wrong null here — the statistic is not about
    a label at all, it is about whether scene i's two errors belong together.
    Residualisation happens before the permutation, so the null sees the same
    conditioned quantities the observed statistic does.
    """
    a, b = _residualize(eps_pre, p_true), _residualize(eps_post, p_true)
    return np.array([
        error_sharing(a, b[rng.permutation(len(b))]) for _ in range(n_perm)
    ])


def conservation_pull(eps_post, dp_true, n_perm, rng):
    """How much of the violation the model failed to register, lawfully.

    `u` is the unit vector along -dp: the direction the model's post estimate
    would have to be dragged for the scene to look conserved. Projecting the
    post-readout error onto it gives the share of the violation that went
    unregistered in the lawful direction. Under independent decoding the error
    has no preferred direction, so the null is the same projection onto random
    unit vectors.

    Aggregated as a RATIO OF SUMS rather than a mean of per-scene ratios: |dp|
    varies over an order of magnitude here and the small-|dp| scenes gave the
    per-scene form a heavy tail (a null p95 of 6.18 on the first run).
    """
    mag = np.linalg.norm(dp_true, axis=1)
    keep = mag > 1e-9
    if keep.sum() < 4 or eps_post.shape[1] < 2:
        return float("nan"), np.array([]), int(keep.sum())

    e, m = eps_post[keep], mag[keep]
    u = -dp_true[keep] / m[:, None]
    obs = float((e * u).sum() / m.sum())

    null = []
    for _ in range(n_perm):
        r = rng.normal(size=u.shape)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        null.append(float((e * r).sum() / m.sum()))
    return obs, np.asarray(null), int(keep.sum())


def delta_slope(dp_hat, dp_true, n_boot=0, rng=None):
    """OLS slope of the decoded change on the true change, pooled, with intercept.

    THE NUMBER TO COMPARE IT TO IS `r2_post`, NOT 1.0. A ridge readout of
    fidelity r attenuates any signal it reads to roughly r, so beta < 1 on its
    own says only that the probe is imperfect. Law-use predicts something
    sharper: the model's post estimate is dragged toward its own pre estimate,
    so the CHANGE is attenuated *more* than the level is — beta well below
    r2_post. Independent decoding predicts beta ~= r2_post.
    """
    x, y = dp_true.ravel(), dp_hat.ravel()
    if x.std() < 1e-12:
        return float("nan"), (float("nan"), float("nan"))
    beta = float(np.polyfit(x, y, 1)[0])
    if not n_boot:
        return beta, (float("nan"), float("nan"))
    n = len(dp_true)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = dp_true[idx].ravel(), dp_hat[idx].ravel()
        if xb.std() > 1e-12:
            draws.append(np.polyfit(xb, yb, 1)[0])
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return beta, (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #

def run_grid(acts, labels, meta, *, kind, train_fraction, split_seed, n_perm):
    Y_pre, Y_post, target_name = targets_from(labels, kind)
    cons = conserved_mask(labels)
    pools = pool_pairs(meta["pools"])
    if not pools:
        raise ValueError(f"no .pre/.post pool pair in {meta['pools']}")

    n = len(labels)
    rng_split = np.random.default_rng(split_seed)
    order = rng_split.permutation(n)
    n_train = int(round(train_fraction * n))
    train_all, test_all = order[:n_train], order[n_train:]

    # Fit on conserving scenes ONLY — the readout has to learn what momentum
    # looks like where the law holds, or the violations teach it the exception.
    train = train_all[cons[train_all]]
    test_cons = test_all[cons[test_all]]
    test_viol = test_all[~cons[test_all]]
    if len(train) < 8 or len(test_cons) < 8:
        raise ValueError(
            f"need >=8 conserving scenes each side; got {len(train)} train / "
            f"{len(test_cons)} test"
        )

    rows = []
    for site, arr in sorted(acts.items()):
        for si in range(arr.shape[1]):
            for name_pre, name_post, i_pre, i_post in pools:
                X_pre = arr[:, si, i_pre, :].astype(np.float64)
                X_post = arr[:, si, i_post, :].astype(np.float64)
                # Empty pools are exactly zero by construction (object left frame).
                live = (np.abs(X_pre).sum(1) > 0) & (np.abs(X_post).sum(1) > 0)
                tr = np.array([i for i in train if live[i]])
                tc = np.array([i for i in test_cons if live[i]])
                tv = np.array([i for i in test_viol if live[i]])
                if len(tr) < 8 or len(tc) < 8:
                    continue

                apply, alpha, r2_tied = fit_tied(
                    X_pre, X_post, Y_pre, Y_post, tr, seed=split_seed
                )

                eps_pre = apply(X_pre[tc]) - Y_pre[tc]
                eps_post = apply(X_post[tc]) - Y_post[tc]
                rng = np.random.default_rng(split_seed + 1)
                # p_pre == p_post on a conserving scene, so either serves as the
                # target to condition on.
                rho = error_sharing(eps_pre, eps_post, Y_pre[tc])
                null = sharing_null(eps_pre, eps_post, Y_pre[tc], n_perm, rng)

                r2_pre = _r2(Y_pre[tc], apply(X_pre[tc]))
                r2_post = _r2(Y_post[tc], apply(X_post[tc]))
                rec = {
                    "site": site, "step_index": int(si),
                    "pool_pre": name_pre, "pool_post": name_post,
                    "alpha": alpha, "n_train": int(len(tr)),
                    "n_test_conserving": int(len(tc)), "n_test_violating": int(len(tv)),
                    "r2_tied_train": r2_tied,
                    "r2_pre": r2_pre, "r2_post": r2_post,
                    # Nothing below is interpretable without a working readout;
                    # this is what `--min-r2` filters on.
                    "readout_quality": float(min(r2_pre, r2_post)),
                    "rho": rho,
                    "rho_null_mean": float(null.mean()),
                    "rho_null_p95": float(np.percentile(null, 95)),
                }

                if len(tv) >= 4:
                    e_post_v = apply(X_post[tv]) - Y_post[tv]
                    dp_true = Y_post[tv] - Y_pre[tv]
                    dp_hat = apply(X_post[tv]) - apply(X_pre[tv])
                    pull, pull_null, n_used = conservation_pull(
                        e_post_v, dp_true, n_perm, rng
                    )
                    beta, (blo, bhi) = delta_slope(dp_hat, dp_true, 400, rng)
                    rec.update(
                        pull=pull,
                        pull_null_p95=(float(np.percentile(pull_null, 95))
                                       if len(pull_null) else float("nan")),
                        n_pull=n_used,
                        beta=beta, beta_ci_lo=blo, beta_ci_hi=bhi,
                        # Law-use predicts the CHANGE is attenuated more than the
                        # level: negative gap. Independent decoding predicts ~0.
                        beta_minus_r2post=beta - r2_post,
                    )
                rows.append(rec)
    return rows, target_name


def grid_max_null(acts, labels, meta, *, kind, train_fraction, split_seed, n_perm,
                  min_r2=0.2):
    """Grid-max null for rho — the §34 protocol, not a per-cell null.

    `best rho` is an argmax over every (site, step, pool) cell, so it carries a
    winner's curse that a per-cell null does not price in. Permute the scene
    pairing, recompute EVERY cell, take the max, repeat. Read the margin as
    best - grid_max_p95, never best - 0.
    """
    Y_pre, Y_post, _ = targets_from(labels, kind)
    cons = conserved_mask(labels)
    pools = pool_pairs(meta["pools"])

    n = len(labels)
    order = np.random.default_rng(split_seed).permutation(n)
    n_train = int(round(train_fraction * n))
    train_all, test_all = order[:n_train], order[n_train:]
    train = train_all[cons[train_all]]
    test_cons = test_all[cons[test_all]]

    # Fit every cell once; only the pairing is permuted, so the readouts and
    # their residuals are reusable and the null costs one pass, not n_perm.
    cells = []
    for site, arr in sorted(acts.items()):
        for si in range(arr.shape[1]):
            for _, _, i_pre, i_post in pools:
                X_pre = arr[:, si, i_pre, :].astype(np.float64)
                X_post = arr[:, si, i_post, :].astype(np.float64)
                live = (np.abs(X_pre).sum(1) > 0) & (np.abs(X_post).sum(1) > 0)
                tr = np.array([i for i in train if live[i]])
                tc = np.array([i for i in test_cons if live[i]])
                if len(tr) < 8 or len(tc) < 8:
                    continue
                apply, _, _ = fit_tied(X_pre, X_post, Y_pre, Y_post, tr, seed=split_seed)
                if min(_r2(Y_pre[tc], apply(X_pre[tc])),
                       _r2(Y_post[tc], apply(X_post[tc]))) < min_r2:
                    continue
                cells.append((_residualize(apply(X_pre[tc]) - Y_pre[tc], Y_pre[tc]),
                              _residualize(apply(X_post[tc]) - Y_post[tc], Y_pre[tc])))
    if not cells:
        return np.array([])

    rng = np.random.default_rng(split_seed + 99)
    maxima = []
    for _ in range(n_perm):
        maxima.append(max(
            error_sharing(e_pre, e_post[rng.permutation(len(e_post))])
            for e_pre, e_post in cells
        ))
    return np.asarray(maxima)


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("capture_dir", type=Path,
                   help="e.g. outputs_v3_videomae-base/momentum_probe")
    p.add_argument("--placebo", choices=("momentum", "ke"), default="momentum",
                   help="'ke' runs the not-conserved control. Read the two together.")
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--min-r2", type=float, default=0.2,
                   help="readout floor. Cells below it cannot answer the question: "
                        "with no working readout the errors ARE the target and "
                        "every statistic degenerates (rho 0.998 at R2 0.003).")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-grid-null", action="store_true",
                   help="per-cell nulls only; faster, and NOT publishable on its own")
    args = p.parse_args(argv)

    act_dir = args.capture_dir / "activations"
    acts, labels, meta = load_captures(act_dir)
    n_cons = int(conserved_mask(labels).sum())
    print(f"{len(labels)} scenes ({n_cons} conserving / {len(labels) - n_cons} "
          f"violating), {len(acts)} site(s), {meta['pools']}", flush=True)

    rows, target_name = run_grid(
        acts, labels, meta, kind=args.placebo,
        train_fraction=args.train_fraction, split_seed=args.split_seed,
        n_perm=args.n_perm,
    )
    if not rows:
        print("no cell had enough live scenes to fit")
        return 1

    grid_null = None
    if not args.skip_grid_null:
        print(f"grid-max null, {args.n_perm} permutations ...", flush=True)
        gm = grid_max_null(
            acts, labels, meta, kind=args.placebo,
            train_fraction=args.train_fraction, split_seed=args.split_seed,
            n_perm=args.n_perm, min_r2=args.min_r2,
        )
        grid_null = None if not len(gm) else {"mean": float(gm.mean()), "p95": float(np.percentile(gm, 95)),
                     "max": float(gm.max())}

    # A cell with no working readout cannot answer the question: its errors ARE
    # the target, and every statistic below degenerates. Select on readout
    # quality, never on the statistic being tested.
    valid = [r for r in rows if r["readout_quality"] >= args.min_r2]
    print(f"\ntarget: {target_name}   (placebo mode: {args.placebo})")
    print(f"{len(valid)}/{len(rows)} cells have a working tied readout "
          f"(min R2 >= {args.min_r2})")
    if not valid:
        print("\nNo cell clears the readout floor -- nothing here is interpretable.")
        return 1

    best = max(valid, key=lambda r: r["readout_quality"])
    print(f"\nbest tied readout: {best['site']} step {best['step_index']} "
          f"{best['pool_pre']}->{best['pool_post']}   alpha {best['alpha']:g}")
    print(f"  R2   pre {best['r2_pre']:+.3f}   post {best['r2_post']:+.3f}")
    print(f"  rho  {best['rho']:+.3f}   (per-cell null mean {best['rho_null_mean']:+.3f}, "
          f"p95 {best['rho_null_p95']:+.3f})")
    if grid_null:
        print(f"       GRID-MAX null mean {grid_null['mean']:+.3f}  "
              f"p95 {grid_null['p95']:+.3f}   <-- read the margin against this")
    if "beta" in best:
        print(f"  beta {best['beta']:+.3f}  95% CI [{best['beta_ci_lo']:+.3f}, "
              f"{best['beta_ci_hi']:+.3f}]   vs r2_post {best['r2_post']:+.3f}")
        print(f"       gap {best['beta_minus_r2post']:+.3f}   "
              f"<-- law-use predicts NEGATIVE, independent decoding ~0")
        print(f"  pull {best['pull']:+.3f}  (null p95 {best['pull_null_p95']:+.3f}, "
              f"n={best['n_pull']})")

    top_rho = max(valid, key=lambda r: r["rho"])
    if top_rho is not best:
        print(f"\nhighest rho among valid cells: {top_rho['rho']:+.3f} at "
              f"{top_rho['site']} {top_rho['pool_pre']}->{top_rho['pool_post']} "
              f"(R2 {top_rho['readout_quality']:+.3f})")

    # The controls, printed every run so they cannot be skipped.
    pre_t = [r for r in valid if r["site"] in ("embed.out", "vae.latent")]
    if pre_t:
        b = max(pre_t, key=lambda r: r["rho"])
        print(f"\nPRE-TRANSFORMER  {b['site']}: rho {b['rho']:+.3f}  "
              f"-- sharing already here is appearance, not construction")
    else:
        print("\nPRE-TRANSFORMER  no cell clears the readout floor (expected: the "
              "embedding should not decode momentum)")

    def _key(s):
        m = re.search(r"block\.(\d+)", s)
        return (1, int(m.group(1))) if m else (0, 0)
    print("\ndepth curve  (cells with a working readout only)")
    print(f"  {'site':<28} {'rho':>7} {'R2':>7} {'beta':>7}")
    for site in sorted({r["site"] for r in valid}, key=_key):
        b = max((r for r in valid if r["site"] == site),
                key=lambda r: r["readout_quality"])
        print(f"  {site:<28} {b['rho']:+7.3f} {b['readout_quality']:+7.3f} "
              f"{b.get('beta', float('nan')):+7.3f}")

    if args.placebo == "momentum":
        print("\nNOT A RESULT until you have run --placebo ke and rho is LOWER there.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "capture_dir": str(args.capture_dir), "target": target_name,
            "placebo_mode": args.placebo, "n_scenes": len(labels),
            "n_conserving": n_cons, "n_perm": args.n_perm,
            "split_seed": args.split_seed, "train_fraction": args.train_fraction,
            "grid_max_null": grid_null, "results": rows,
        }, indent=2))
        print(f"\n{len(rows)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
