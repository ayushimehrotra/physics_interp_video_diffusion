"""Joseph et al.'s possible-vs-impossible probe, replicated exactly, on our
conserving-vs-violating label.

`interpreting_physics.pdf` (Joseph et al., *Interpreting Physics in Video World
Models*, arXiv 2602.07050), §3.2 + §4.1 + Appendix B + Appendix C.11. Their
IntPhys probe is NOT the ridge-on-a-0/1-target that `fit_probe` uses, and the
differences are not cosmetic — see the table in §36 of CLAUDE.md. Reproduced here:

  probe          single linear layer f(h) = W h + b, binary LOGISTIC regression
                 trained with CROSS-ENTROPY loss                     (App C.11)
  pooling        spatiotemporally pooled activations, no masks        (App B, §3.2)
  optimiser      Adam                                                (App B)
  sweep          20 configs: lr {1e-4,3e-4,1e-3,3e-3,5e-3}
                 x weight decay {0.01,0.1,0.4,0.8}, best on validation  (App B)
  evaluation     5-fold GROUPED cross-validation, mean +- sd across folds (App B)
  epochs         50 for the IntPhys probe                            (App C.11)
  metrics        accuracy AND AUC, chance = 50%                      (App C.11)
  layers         every layer of the residual stream                  (§3.2)

ONE THING CANNOT BE REPLICATED ON OUR DATA, and it is structural. Their groups
are *matched possible/impossible video pairs* that "differ only at a single
breakpoint frame" (§4.1); the grouping exists so a pair never straddles the
train/test boundary. Our v1 scenes are independent draws with no matched partner,
so there is no group to hold out and the grouped CV degenerates to stratified CV.
Reported honestly by `--grouping`; building matched pairs needs a re-render.

MASK-POOLED VARIANTS ARE A DELIBERATE DEVIATION FROM THE PAPER. Joseph et al.
never pool with ground-truth segmentation (§33a: no masks, bounding boxes, crops
or per-object coordinates anywhere in the paper); their pooling is either the
frame-wide mean or learned. Object-local pooling is forced by our multi-object
design, and it carries §20's leak: the *choice* of region is an information
channel supplied by ground truth, independent of the tokens' contents. So every
mask-pooled cell here is reported against `--mask-floor`, which runs the IDENTICAL
logistic protocol on the pooling weights alone with no activations anywhere.
An object-local number must be read against that floor, not against 0.5.

Usage:
    .venv/bin/python scripts/joseph_intphys_probe.py \
        --outputs outputs_videomae-base outputs_vjepa2-vitl \
        --pools all.all,A.pre,A.post,B.pre,B.post,A.pre+B.pre \
        --mask-floor -o docs/artifacts/joseph_intphys.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)          # for mask_only_floor, imported lazily

# Appendix B: "a hyperparameter sweep over 20 configurations".
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
WEIGHT_DECAYS = (0.01, 0.1, 0.4, 0.8)
EPOCHS = 50          # App C.11, for the IntPhys probe specifically
N_FOLDS = 5          # App B: "5-fold grouped cross-validation"


def auc(score: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based AUC. Ties get midranks, which `argsort(argsort(.))` does not
    give, so ties are averaged explicitly — a probe that collapses to a constant
    would otherwise score 1.0 or 0.0 rather than 0.5."""
    score, truth = np.asarray(score, float), np.asarray(truth, bool)
    if truth.all() or not truth.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = int(truth.sum())
    n_neg = len(truth) - n_pos
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def standardize(fit: np.ndarray, *apply: np.ndarray):
    """Z-score against `fit`'s moments, with dead columns forced to exactly 0.

    `(x - mu) / (sd + 1e-6)` is the usual guard, but it is wrong for mask inputs:
    a token that is zero in every scene has sd = 0, so the guard divides residual
    float noise by 1e-6 and hands the optimiser values ~1e10. Object masks are
    mostly such tokens — the object occupies a small part of the grid — so without
    this the floor probe trains on garbage and its number means nothing. Columns
    with no variance carry no information by definition; zero them.
    """
    mu, sd = fit.mean(0), fit.std(0)
    live = sd > 1e-8
    out = []
    for X in apply:
        Z = np.zeros_like(X, dtype=np.float64)
        Z[:, live] = (X[:, live] - mu[live]) / sd[live]
        out.append(Z)
    return out if len(out) > 1 else out[0]


def train_logistic(Xtr, ytr, Xte, lr, wd, seed, device):
    """One linear logistic probe, Adam + BCE, exactly EPOCHS passes.

    Full-batch: their n is far larger than ours, and at n=150 minibatching adds a
    stochasticity that the paper does not specify. Returns test logits.
    """
    torch.manual_seed(seed)
    xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    xte = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    lin = torch.nn.Linear(xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(lin(xtr).ravel(), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        return lin(xte).ravel().cpu().numpy()


def train_mse(Xtr, ytr, Xte, lr, wd, seed, device):
    """One linear probe with MSE loss, Adam, exactly EPOCHS passes (App C.11).

    Targets are z-scored on TRAIN before fitting and the prediction is mapped
    back. Without it a fixed learning-rate grid means something different for
    every target — mu has sd 0.07 and theta sd 0.07 rad while `accel_measured`
    has sd 0.04, so an unscaled MSE would let the sweep's lr stand in for the
    target's units. Ridge has no such sensitivity, which is itself one of the
    differences being measured here.
    """
    torch.manual_seed(seed)
    ytr = np.asarray(ytr, dtype=np.float64)
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-12
    xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor((ytr - mu) / sd, dtype=torch.float32, device=device)
    xte = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    lin = torch.nn.Linear(xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(EPOCHS):
        opt.zero_grad()
        ((lin(xtr) - ytr_t) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        return lin(xte).cpu().numpy() * sd + mu


def r2_score(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.ndim == 1:
        y, p = y[:, None], p.reshape(-1, 1)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def kfold(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Plain k-fold for continuous targets. Stratification needs classes."""
    idx = np.random.default_rng(seed).permutation(n)
    return [np.array(sorted(f)) for f in np.array_split(idx, n_folds)]


def probe_layer_regression(X, Y, seed: int, device: str) -> dict:
    """The paper's protocol on a continuous target: MSE linear probe, 20-config
    sweep selected on an inner validation split, 5-fold CV, mean +- sd of R2."""
    folds = kfold(len(Y), N_FOLDS, seed)
    scores, picks = [], []
    for f in range(N_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[k] for k in range(N_FOLDS) if k != f])
        perm = np.random.default_rng(seed + f).permutation(len(tr))
        n_inner = int(round(0.8 * len(tr)))
        inner_tr, inner_va = tr[perm[:n_inner]], tr[perm[n_inner:]]

        Ztr, Zva = standardize(X[inner_tr], X[inner_tr], X[inner_va])
        best = (-np.inf, LEARNING_RATES[0], WEIGHT_DECAYS[0])
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                pred = train_mse(Ztr, Y[inner_tr], Zva, lr, wd, seed, device)
                sc = r2_score(Y[inner_va], pred)
                if np.isfinite(sc) and sc > best[0]:
                    best = (sc, lr, wd)
        _, lr, wd = best

        Ztr, Zte = standardize(X[tr], X[tr], X[te])
        scores.append(r2_score(Y[te], train_mse(Ztr, Y[tr], Zte, lr, wd, seed, device)))
        picks.append({"lr": lr, "weight_decay": wd})
    return {"r2_mean": float(np.mean(scores)), "r2_sd": float(np.std(scores)),
            "r2_folds": scores, "selected": picks}


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Class-balanced folds. Grouped CV needs groups; without matched pairs the
    honest degenerate case is stratification, which at least keeps every fold's
    chance rate at the same 0.5 the paper's is."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % n_folds].append(i)
    return [np.array(sorted(f)) for f in folds]


def probe_layer(X: np.ndarray, y: np.ndarray, seed: int, device: str) -> dict:
    """5-fold CV with the 20-config sweep selected on an inner validation split.

    The sweep is run INSIDE each fold on a split carved out of that fold's train
    half. Selecting it on the fold's test half is the standard way to report a
    probe accuracy that does not replicate, and with 20 configs x 13-25 layers
    the inflation is not small.
    """
    folds = stratified_folds(y, N_FOLDS, seed)
    accs, aucs, picks = [], [], []
    for f in range(N_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[k] for k in range(N_FOLDS) if k != f])
        rng = np.random.default_rng(seed + f)
        perm = rng.permutation(len(tr))
        n_inner = int(round(0.8 * len(tr)))
        inner_tr, inner_va = tr[perm[:n_inner]], tr[perm[n_inner:]]

        Ztr, Zva = standardize(X[inner_tr], X[inner_tr], X[inner_va])
        best = (-np.inf, LEARNING_RATES[0], WEIGHT_DECAYS[0])
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                logit = train_logistic(Ztr, y[inner_tr], Zva, lr, wd, seed, device)
                score = auc(logit, y[inner_va] > 0.5)
                if np.isfinite(score) and score > best[0]:
                    best = (score, lr, wd)
        _, lr, wd = best

        Ztr, Zte = standardize(X[tr], X[tr], X[te])
        logit = train_logistic(Ztr, y[tr], Zte, lr, wd, seed, device)
        truth = y[te] > 0.5
        accs.append(float(((logit > 0.0) == truth).mean()))
        aucs.append(auc(logit, truth))
        picks.append({"lr": lr, "weight_decay": wd})
    return {
        "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
        "auc_mean": float(np.nanmean(aucs)), "auc_sd": float(np.nanstd(aucs)),
        "accuracy_folds": accs, "auc_folds": aucs, "selected": picks,
    }


def load(outdir: str):
    """Returns (acts by site as [scenes, pools, dim], labels, pool_names).

    Loaded once for every pool rather than once per pool: the npz read dominates
    a run over the full pool list, and the arrays are small enough to hold.
    """
    acts, labels, pool_names = {}, [], None
    for f in sorted(glob.glob(f"{outdir}/momentum_probe/activations/scene_*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(bytes(z["meta"]).decode())
        pool_names = meta["pools"]
        labels.append(meta["labels"])
        for k in z.files:
            if k.startswith("act."):
                acts.setdefault(k[4:], []).append(np.asarray(z[k], np.float64))
    return {k: np.stack(v)[:, 0] for k, v in acts.items()}, labels, pool_names


def pool_matrix(arr: np.ndarray, spec: str, pool_names: list[str]) -> np.ndarray:
    """`arr` is [scenes, pools, dim]; '+' concatenates on the FEATURE axis.

    Concatenation rather than a mean over both regions: a relation between two
    objects cannot survive averaging them onto the same coordinates.
    """
    idx = {n: i for i, n in enumerate(pool_names)}
    return np.concatenate([arr[:, idx[p], :] for p in spec.split("+")], axis=-1)


def mask_tensors(outdir: str, dataset_dir: str):
    """§20's control inputs: the pooling weights themselves, no activations.

    Rebuilt from each capture's own metadata by `mask_only_floor.scene_mask_rows`,
    so they are the masks the capture actually pooled with, by construction.

    Built ONCE for every pool rather than once per pool spec. The masks do not
    depend on the spec — slicing does — and the read is ~50 segmentation PNGs per
    scene over a network mount, so rebuilding per spec multiplied the dominant
    cost by the length of the pool list for no new information.

    Returns ``(normalised, occupancy, pool_names)``, each ``[scenes, pools, tokens]``.
    Normalising fixes the row sum at 1 and so discards the object's AREA, which is
    a channel of its own; occupancy keeps it and is the strictly stronger control.
    """
    from mask_only_floor import scene_mask_rows  # noqa: PLC0415

    norm, occ, names = [], [], None
    for f in sorted(glob.glob(f"{outdir}/momentum_probe/activations/scene_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
        stem = os.path.basename(f)[:-4]
        w, raw, nm = scene_mask_rows(Path(dataset_dir) / stem, meta)
        if names is None:
            names = nm
        elif nm != names:
            raise ValueError(f"{stem} pools {nm} != {names}")
        norm.append(w)
        occ.append(raw)
    return np.stack(norm), np.stack(occ), names


def site_order(site: str) -> tuple[int, int]:
    if site.startswith("block."):
        return (1, int(site.split(".")[1]))
    return (0, -1)


def run_target(outdir, target, Y, binary, acts, sites, pool_names, pools, args, report):
    """One target across every pool and layer, under the paper's protocol.

    Dispatches on the target's own type rather than on a flag: their IntPhys
    probe is logistic and their motion-variable probes are MSE, and which one
    applies is a property of the label, not a choice.
    """
    kind = "logistic/CE (App C.11), acc+AUC" if binary else "MSE (App C.11), R2"
    print(f"\n  {'='*90}\n  target={target}   [{kind}]   n={len(Y)}\n  {'='*90}")
    store = report["models"][outdir].setdefault(target, {})

    floors = {}
    if args.mask_floor:
        print(f"  \u00a720 mask-only floor (no activations; identical protocol "
              f"on the pooling weights)")
        Mn_all, Mo_all, mask_names = mask_tensors(outdir, args.dataset_dir)
        for spec in pools:
            fn = _probe(pool_matrix(Mn_all, spec, mask_names), Y, binary, args)
            fo = _probe(pool_matrix(Mo_all, spec, mask_names), Y, binary, args)
            floors[spec] = {"normalised": fn, "occupancy": fo}
            print(f"    {spec:<18}normalised {_headline(fn, binary):>8.3f}"
                  f"    occupancy {_headline(fo, binary):>8.3f}")
        store["_mask_floor"] = floors

    for spec in pools:
        if spec.split("+")[0] not in pool_names:
            print(f"\n  pool={spec}   NOT CAPTURED in {outdir} - skipped")
            continue
        print(f"\n  pool={spec}   dim={pool_matrix(acts[sites[0]], spec, pool_names).shape[1]}")
        print(f"  {'layer':<22}" + (f"{'accuracy':>18}{'AUC':>18}" if binary
                                    else f"{'R2 (mean +- sd over 5 folds)':>30}"))
        rows = {}
        for site in sites:
            X = pool_matrix(acts[site], spec, pool_names)
            keep = np.abs(X).sum(1) > 0        # an empty pool is exactly zero
            r = _probe(X[keep], Y[keep], binary, args)
            r["n_dropped"] = int((~keep).sum())
            rows[site] = r
            if binary:
                print(f"  {site:<22}{r['accuracy_mean']:>11.3f} +-{r['accuracy_sd']:.3f}"
                      f"{r['auc_mean']:>11.3f} +-{r['auc_sd']:.3f}")
            else:
                print(f"  {site:<22}{r['r2_mean']:>23.3f} +-{r['r2_sd']:.3f}")
        best = max(rows, key=lambda s_: _headline(rows[s_], binary))
        line = f"    BEST {_headline(rows[best], binary):.3f} at {best}"
        if spec in floors:
            fl = max(_headline(floors[spec]["normalised"], binary),
                     _headline(floors[spec]["occupancy"], binary))
            line += f"   (mask-only floor {fl:.3f}, margin {_headline(rows[best], binary)-fl:+.3f})"
        print(line)
        store[spec] = rows


def load_patches(outdir: str):
    """Raw per-token activations from a `--patch-sites` capture.

    Returns ``({site: [scenes, tokens, dim]}, labels)``. Kept fp16 on disk and
    cast to float32 here: fp16 sums overflow (§18's silent `inf`), and the
    probe trains in float32 anyway.
    """
    per_site, labels = {}, []
    for f in sorted(glob.glob(f"{outdir}/momentum_probe/activations/scene_*.npz")):
        with np.load(f, allow_pickle=False) as z:
            labels.append(json.loads(bytes(z["meta"]).decode())["labels"])
            for k in z.files:
                if k.startswith("patch."):
                    per_site.setdefault(k[6:], []).append(np.asarray(z[k], np.float32))
    return {k: np.stack(v) for k, v in per_site.items()}, labels


def run_attentive(outdir, targets, args, report):
    """Joseph et al.'s second probe: patch-preserving attentive-MLP (§3.2).

    Reported against TWO controls at the SAME input, because otherwise the
    comparison confounds the probe with the pooling:

      * `mean+ridge`  - the token mean of these very tokens, read by `fit_probe`.
        This is the honest linear baseline: same information, linear readout.
      * `shuffled`    - `fit_attentive_probe`'s own permutation control. A head
        with a learned query and an MLP on 90 training scenes can fit noise.

    A target the linear control reads near ceiling is the instrument check. If
    the attentive probe cannot match it, its number on a HARD target bounds
    nothing (§34g found exactly that on v1: `velocity_A_pre` 0.413 attentive
    against 0.974 linear).
    """
    from physics_interp.interp import extract_label, fit_probe, fit_attentive_probe  # noqa: PLC0415

    patches, labels = load_patches(outdir)
    if not patches:
        raise SystemExit(f"{outdir} holds no patch.* arrays - capture it with --patch-sites")
    sites = sorted(patches, key=site_order)
    n_tok, dim = patches[sites[0]].shape[1:]
    print(f"\n{'#'*96}\n# ATTENTIVE-MLP  {outdir}   n={len(labels)}  "
          f"tokens={n_tok}  dim={dim}\n{'#'*96}")
    store = report.setdefault("attentive", {}).setdefault(outdir, {})

    for target in targets:
        Y = np.stack([np.asarray(extract_label(l, target), float) for l in labels])
        print(f"\n  target={target}")
        print(f"  {'layer':<22}{'attentive':>11}{'shuf':>9}{'mean+ridge':>13}{'gap':>8}")
        for site in sites:
            X = patches[site]
            for ep in args.attentive_epochs:
                a = fit_attentive_probe(X, Y, epochs=ep, device=args.device)
                lin = fit_probe(X.mean(1), Y, permutation_control=False)
                tag = f"{site}" + (f" [{ep}ep]" if len(args.attentive_epochs) > 1 else "")
                print(f"  {tag:<22}{a['r2']:>11.3f}{a['r2_shuffled']:>9.3f}"
                      f"{lin['r2']:>13.3f}{a['r2'] - lin['r2']:>8.3f}")
                store.setdefault(target, {})[f"{site}@{ep}"] = {
                    "attentive_r2": a["r2"], "attentive_shuffled": a["r2_shuffled"],
                    "mean_ridge_r2": lin["r2"], "epochs": ep}


def _probe(X, Y, binary: bool, args):
    if binary:
        return probe_layer(X, Y.ravel(), args.seed, args.device)
    return probe_layer_regression(X, Y, args.seed, args.device)


def _headline(r: dict, binary: bool) -> float:
    return r["auc_mean"] if binary else r["r2_mean"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--pools", default="all.all",
                    help="comma-separated; '+' concatenates on the feature axis. "
                         "'all.all' is the frame-wide space-time mean the paper uses")
    ap.add_argument("--dataset-dir", default="momentum_dataset/probe_set")
    ap.add_argument("--targets", default="momentum_conserved",
                    help="comma-separated. `momentum_conserved` (or any 0/1 "
                         "target) uses their logistic/cross-entropy probe and is "
                         "scored by accuracy+AUC; anything else uses their MSE "
                         "regression probe (App C.11) and is scored by R2")
    ap.add_argument("--mask-floor", action="store_true",
                    help="also run the identical protocol on the pooling masks "
                         "alone (\u00a720). Mandatory reading for any object-local cell")
    ap.add_argument("--attentive", action="store_true",
                    help="run their patch-preserving attentive-MLP probe on "
                         "`patch.*` arrays instead of the pooled linear probe. "
                         "Needs a capture made with --patch-sites")
    ap.add_argument("--attentive-epochs", default="120",
                    help="comma-separated; more than one runs a sweep, which is "
                         "how you tell 'the model lacks it' from 'the probe is "
                         "undertrained' (\u00a734g)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    report = {"protocol": "joseph_et_al_intphys", "pools": pools,
              "targets": targets,
              "grouping": "stratified (no matched pairs exist in this dataset; "
                          "the paper's grouped CV holds out matched "
                          "possible/impossible pairs)",
              "epochs": EPOCHS, "n_folds": N_FOLDS,
              "mask_floor": bool(args.mask_floor),
              "sweep": {"lr": list(LEARNING_RATES), "weight_decay": list(WEIGHT_DECAYS)},
              "models": {}}

    if args.attentive:
        args.attentive_epochs = [int(e) for e in args.attentive_epochs.split(",")]
        report["attentive_epochs"] = args.attentive_epochs
        for outdir in args.outputs:
            run_attentive(outdir, targets, args, report)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(report, fh, indent=1)
            print(f"\nwrote {args.out}")
        return

    from physics_interp.interp import extract_label  # noqa: PLC0415

    for outdir in args.outputs:
        acts, labels, pool_names = load(outdir)
        sites = sorted(acts, key=site_order)
        print(f"\n{'#'*96}\n# {outdir}   n={len(labels)}\n{'#'*96}")
        report["models"][outdir] = {}

        for target in targets:
            Y = np.stack([np.asarray(extract_label(l, target), float) for l in labels])
            binary = Y.shape[1] == 1 and set(np.unique(Y)) <= {0.0, 1.0}
            run_target(outdir, target, Y, binary, acts, sites, pool_names,
                       pools, args, report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
