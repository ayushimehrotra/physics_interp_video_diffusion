"""L0 states (spec §5): can per-object states be decoded from pooled activations?

Targets, per object o in {A, B} and phase in {pre, post}:
  velocity (2-D world xy, one block), speed, direction (cos, sin), log_mass, radius.
Grid per target: layers x {roi_<o>.<phase>, roi_pair.<phase>, frame.<phase>}.
Score: out-of-fold R^2, grouped 5-fold by twin group, 12 fold seeds, nested alpha
(physlaw.probes.grid, identical to ridge.fit_probe).

Baselines, through the same pipeline:
  * observers   obs_vel / obs_speed / obs_quad / obs_normal (true pre-contact kinematics)
  * mask floor  the pooling weights themselves (flattened [T, gh, gw]) as features, for the best
                cell's pool (old project §20: a ROI pool leaks state through where it sits). If
                token features carried position only, every probe on the pooled vector would be
                linear in these weights, so this is the ceiling of a content-free model.
                Frame pools have uniform weights (floor 0).
  * tracker     both balls' token-space centre, radius, presence per token-frame of the window
                (an ideal tracker on the masks; a reference for what is visible, not a floor)
  * untrained   the same grid on the untrained control (separate run; merged in the report)
  * latent      DiTs (spec §5.1): the model's own noised VAE latent z_t, same pooling, per SNR (cache keys
                ``pooled_latent`` [P, S, C'] + ``latent_keys``); skipped if absent
  * null        500 group-block permutations, max over the target's grid -> corrected p.

Selection rule (spec §5): the reported best cell is chosen on half of the groups and scored on the
other half from the cross-fitted predictions (probes.select.heldout_select); the in-sample max over the
grid is kept as ``best_cell`` / ``r2`` (the old protocol) with the permutation p, which corrects for it.
Holm across targets on the corrected p.
One-twin-per-group refit (spec §5.1): with < ``one_twin_refit_below`` groups, post-event targets are
refit at the selected cell on one twin per group (each twin type in turn, averaged): pre targets repeat
across a group's twins, post ones do not, which inflates post R² at small n.
Velocity protocol check (spec §5.1; ``velocity_protocols.rigidbench.enabled``): RigidBench's
unregularised linear probe trained with Adam on the own-ball ROI pools, per layer, selected by the rule.

Note (F1, legacy appearance arm): density is 1.5 or 6.0 and tied to the material, so mass is
visible through size AND material; the tracker (size only) bounds the size part.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from physlaw.probes.grid import grid_max_p, grid_probe, group_block_perms
from physlaw.probes.observer import OBSERVERS, observer_matrix
from physlaw.probes.ridge import adam_linear_probe, r2_score
from physlaw.probes.select import heldout_select, holm

TARGETS = ("velocity", "speed", "direction", "log_mass", "radius")
OBJECTS = ("A", "B")
PHASES = ("pre", "post")
POOL_KINDS = ("roi_own", "roi_pair", "frame")

_G: dict = {}          # fork-shared state for workers


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def load_index(cache_dir: Path) -> list[dict]:
    recs = {}
    for line in open(cache_dir / "_index.jsonl"):
        r = json.loads(line)
        recs[r["clip"]] = r                      # last write wins (re-runs append)
    return [recs[k] for k in sorted(recs)]


def complete_groups(recs: list[dict]) -> list[dict]:
    """Keep only groups that hold the full twin set (needed for group-block permutations)."""
    twins_of: dict[str, set] = {}
    for r in recs:
        twins_of.setdefault(r["group"], set()).add(r["twin"])
    full = max(len(s) for s in twins_of.values())
    return [r for r in recs if len(twins_of[r["group"]]) == full]


def load_pooled(cache_dir: Path, recs: list[dict]):
    """(acts [n, P, L, D], pools, layers, latent {"latent/snr<±d>": [n, P, C']}). DiT caches may carry the
    noised-input-latent baseline as ``pooled_latent`` [P, S, C'] + ``latent_keys`` (stage 4, models agent);
    absent in older caches -> {}."""
    first = np.load(cache_dir / recs[0]["file"])
    pools = [str(p) for p in first["pools"]]
    layers = [int(l) if str(l).lstrip("-").isdigit() else str(l) for l in first["layers"]]
    lkeys = [str(k) for k in first["latent_keys"]] if "pooled_latent" in first.files else []
    A, L = [], []
    for r in recs:
        z = np.load(cache_dir / r["file"])
        A.append(z["pooled"])
        if lkeys:
            L.append(z["pooled_latent"])
    lat = {}
    if lkeys:
        L = np.stack(L).astype(np.float32)                       # [n, P, S, C']
        lat = {k: L[:, :, i] for i, k in enumerate(lkeys)}
    return np.stack(A).astype(np.float32), pools, layers, lat


def targets(labels: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for o in OBJECTS:
        for ph in PHASES:
            v = np.array([l["physlaw"][f"{ph}_velocity"][o] for l in labels], float)[:, :2]
            s = np.linalg.norm(v, axis=1)
            out[f"{o}.{ph}.velocity"] = v
            out[f"{o}.{ph}.speed"] = s
            out[f"{o}.{ph}.direction"] = v / np.maximum(s, 1e-9)[:, None]
            out[f"{o}.{ph}.log_mass"] = np.log([l["physlaw"]["objects"][o]["mass"] for l in labels])
            out[f"{o}.{ph}.radius"] = np.array([l["physlaw"]["objects"][o]["radius"] for l in labels])
    return out


def _mask_feat_one(args):
    clip_dir, frames, spec, fpc = args
    from physlaw.tokens.pool import pool_weights
    from physlaw.tokens.project import project_masks, token_frame_phase
    inst = np.load(Path(clip_dir) / "masks.npz")["instance"]
    obj = project_masks(inst, list(frames), spec)
    phases = token_frame_phase(list(frames), spec, fpc)
    ph = np.array(phases)
    out = {}
    for w in PHASES:     # tracker: both balls' token-space centre, radius, presence per token-frame
        cols = []
        for o in OBJECTS:
            c = obj[o].center[ph == w]
            present = ~np.isnan(c[:, 0])
            cols += [np.nan_to_num(c[:, 0]), np.nan_to_num(c[:, 1]), obj[o].radius[ph == w], present.astype(float)]
        out[f"geom.{w}"] = np.concatenate(cols)
    W = pool_weights(obj, phases)     # mask floor: the pooling weights of each ROI pool, flattened
    for name, wt in W.items():
        if name.startswith("roi_") and not name.endswith(".all"):
            out[f"w.{name}"] = wt.reshape(-1).astype(np.float32)
    return out


def mask_features(cache_dir: Path, data_dir: Path, recs: list[dict], labels: list[dict], spec,
                  workers: int = 32) -> dict[str, np.ndarray]:
    """{'geom.pre'|'geom.post': [n, F], 'w.<pool>': [n, T*gh*gw]}, cached in <cache_dir>/_maskfeat_v2.npz."""
    f = cache_dir / "_maskfeat_v2.npz"
    clips = [r["clip"] for r in recs]
    if f.exists():
        z = np.load(f)
        if list(z["clips"]) == clips:
            return {k: z[k] for k in z.files if k != "clips"}
    jobs = [(str(data_dir / r["clip"]), np.load(cache_dir / r["file"])["frames"], spec,
             l["physlaw"]["first_post_contact_frame"]) for r, l in zip(recs, labels)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_mask_feat_one, jobs, chunksize=4))
    out = {}
    for k in res[0]:   # token-frame counts can differ by one if contact jitters; pad to max
        L = max(len(r[k]) for r in res)
        out[k] = np.stack([np.pad(r[k], (0, L - len(r[k]))) for r in res])
    np.savez(f, clips=np.array(clips), **out)
    return out


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def _lname(l) -> str:
    """Cell label for a layer key: int layers -> "L8"; diffusion keys are already "snr0/L15"."""
    return f"L{l}" if isinstance(l, int) else str(l)


def _pool_name(kind: str, o: str, ph: str) -> str:
    return {"roi_own": f"roi_{o}", "roi_pair": "roi_pair", "frame": "frame"}[kind] + f".{ph}"


def _run_cell(job):
    key, xfun, block_names = job
    X = _G["X"](xfun)
    blocks = {b: _G["Y"][b] for b in block_names}
    return key, grid_probe(X, blocks, _G["groups"], alphas=_G["alphas"], k=_G["k"],
                           fold_seeds=_G["fold_seeds"], perms=_G["perms"] if xfun[0] == "act" else None)


def _features(xfun):
    kind = xfun[0]
    if kind == "act":
        _, pi, li = xfun
        return _G["acts"][:, pi, li]
    if kind == "obs":
        return _G["obs"][xfun[1]]
    if kind == "mask":
        return _G["mask"][xfun[1]]
    if kind == "lat":
        return _G["lat"][xfun[2]][:, xfun[1]]
    raise KeyError(kind)


def _run_refit(job):
    """One-twin-per-group refit of one block on one feature cell: R² per twin type."""
    xfun, b, fold_seeds = job
    X, Y, tw = _G["X"](xfun), _G["Y"][b], _G["twins"]
    out = {}
    for t in sorted(set(tw)):
        rows = np.where(tw == t)[0]
        out[t] = grid_probe(X[rows], {b: Y[rows]}, _G["groups"][rows], alphas=_G["alphas"], k=_G["k"],
                            fold_seeds=fold_seeds)[b]["score_mean"]
    return job, out


def _run_rigidbench(job):
    xfun, b, kw = job
    return job, adam_linear_probe(_G["X"](xfun), _G["Y"][b], _G["groups"], k=_G["k"], **kw)


def _select(cells: dict, Y: np.ndarray, groups: np.ndarray, n_splits: int) -> dict:
    """Held-out selection over cells {name: {"oof": [fold_seeds, n, k]}}: R² on a half of the groups, averaged
    over fold seeds (like the cell's own score; an average of the seeds' predictions would be an ensemble)."""
    rows = {}

    def score(c, gids):
        key = gids.tobytes()
        if key not in rows:
            rows[key] = np.isin(groups, gids)
        r = rows[key]
        return float(np.mean([r2_score(Y[r], o[r]) for o in cells[c]["oof"]]))
    return heldout_select(score, list(cells), groups, n_splits=n_splits)


def run(cache_dir: Path, data_dir: Path, cfg, model_cfg, workers: int = 48, log=print) -> dict:
    """Returns the L0 result dict (also the per-layer curves)."""
    from physlaw.models.base import TokenGridSpec
    probe = cfg["probe"]
    recs = complete_groups(load_index(cache_dir))
    labels = [json.loads((data_dir / r["clip"] / "labels.json").read_text()) for r in recs]
    acts, pools, layers, lat = load_pooled(cache_dir, recs)
    groups = np.array([r["group"] for r in recs])
    twins = np.array([r["twin"] for r in recs])
    n_groups = len(set(groups))
    log(f"[L0] {len(recs)} clips / {n_groups} groups; acts {acts.shape}; layers {layers}; latent {sorted(lat)}")

    from physlaw.models.base import token_spec
    spec = token_spec(model_cfg)
    mask = mask_features(cache_dir, data_dir, recs, labels, spec, workers=min(workers, 32))
    Y = targets(labels)
    want = [t for t in cfg.get("targets", TARGETS)]
    Y = {k: v for k, v in Y.items() if k.split(".")[2] in want}

    _G.update(X=_features, Y=Y, groups=groups, twins=twins, acts=acts, mask=mask, lat=lat,
              obs={o: observer_matrix(labels, o) for o in OBSERVERS},
              alphas=tuple(float(x) for x in probe["alphas"]), k=int(probe["k_folds"]), fold_seeds=int(probe["fold_seeds"]),
              perms=group_block_perms(groups, twins, int(probe["n_permutations"]), seed=0))

    # one job per distinct feature matrix, carrying every target block that searches it
    jobs: dict[tuple, list[str]] = {}
    for b in Y:
        o, ph, _ = b.split(".")
        for kind in POOL_KINDS:
            pi = pools.index(_pool_name(kind, o, ph))
            for li in range(len(layers)):
                jobs.setdefault(("act", pi, li), []).append(b)
            for lk in lat:
                jobs.setdefault(("lat", pi, lk), []).append(b)
        for obs in OBSERVERS:
            jobs.setdefault(("obs", obs), []).append(b)
        jobs.setdefault(("mask", f"geom.{ph}"), []).append(b)
        for kind in ("roi_own", "roi_pair"):
            jobs.setdefault(("mask", f"w.{_pool_name(kind, o, ph)}"), []).append(b)
    log(f"[L0] {len(jobs)} feature cells x their targets; {len(_G['perms'])} permutations; {workers} workers")

    res: dict[tuple, dict] = {}
    with ProcessPoolExecutor(workers) as ex:   # fork: workers inherit _G
        for i, (key, r) in enumerate(ex.map(_run_cell, [(k, k, v) for k, v in jobs.items()], chunksize=1)):
            res[key] = r
            if (i + 1) % 25 == 0:
                log(f"[L0]   {i + 1}/{len(jobs)} cells")

    n_splits = int(cfg.get("selection", {}).get("n_splits", 50))
    out = {"n_clips": len(recs), "n_groups": n_groups, "layers": layers, "targets": {},
           "protocol": {"k_folds": int(probe["k_folds"]), "fold_seeds": int(probe["fold_seeds"]),
                        "test_groups_per_fold": n_groups / int(probe["k_folds"]),
                        "n_permutations": int(probe["n_permutations"]), "selection_splits": n_splits}}
    for b in Y:
        o, ph, _ = b.split(".")
        cells = {}
        for kind in POOL_KINDS:
            pname = _pool_name(kind, o, ph)
            pi = pools.index(pname)
            for li, l in enumerate(layers):
                cells[f"{_lname(l)}/{pname}"] = res[("act", pi, li)][b]
        best_cv = max(cells, key=lambda c: cells[c]["score_mean"])
        best_q, p, null_max = grid_max_p({c: v["quick"] for c, v in cells.items()},
                                         {c: v["null"] for c, v in cells.items()})
        curve = {kind: [cells[f"{_lname(l)}/{_pool_name(kind, o, ph)}"]["score_mean"] for l in layers] for kind in POOL_KINDS}
        obs = {k: res[("obs", k)][b] for k in OBSERVERS}
        best_obs = max(obs, key=lambda k: obs[k]["score_mean"])
        geom = res[("mask", f"geom.{ph}")][b]
        bpool = best_cv.rsplit("/", 1)[1]          # cell = "<layer key>/<pool>"; diffusion keys contain "/"
        floor = res[("mask", f"w.{bpool}")][b] if ("mask", f"w.{bpool}") in res else None
        out["targets"][b] = {
            "best_cell": best_cv, "r2": cells[best_cv]["score_mean"], "r2_sd": cells[best_cv]["score_sd"],
            "p_corrected": p, "null_max_q95": float(np.quantile(null_max, 0.95)),
            "best_cell_quick": best_q, "curve": curve,
            "observer": {"best": best_obs, "r2": obs[best_obs]["score_mean"],
                         "all": {k: v["score_mean"] for k, v in obs.items()}},
            # floor for the best cell's own pool (frame pools have uniform weights: floor 0)
            "mask_floor": {"pool": bpool, "r2": floor["score_mean"] if floor else 0.0,
                           "r2_sd": floor["score_sd"] if floor else 0.0},
            "mask_floor_by_pool": {k[1][2:]: res[k][b]["score_mean"] for k in res
                                   if k[0] == "mask" and k[1].startswith("w.") and b in res[k]},
            "tracker": {"r2": geom["score_mean"], "r2_sd": geom["score_sd"]},
            "heldout": _select(cells, Y[b], groups, n_splits),
        }
        if lat:
            lcells = {f"{lk}/{_pool_name(kind, o, ph)}": res[("lat", pools.index(_pool_name(kind, o, ph)), lk)][b]
                      for kind in POOL_KINDS for lk in lat}
            out["targets"][b]["latent_baseline"] = _select(lcells, Y[b], groups, n_splits)
    hp = holm({b: t["p_corrected"] for b, t in out["targets"].items()})
    for b, t in out["targets"].items():
        t["p_holm"] = hp[b]

    # one-twin-per-group refit of post-event targets at the selected cell (and its pool's mask floor)
    if n_groups < int(cfg.get("one_twin_refit_below", 200)):
        rjobs = []
        for b, t in out["targets"].items():
            if b.split(".")[1] != "post":
                continue
            lk, pname = t["heldout"]["selected"].rsplit("/", 1)
            li = [_lname(l) for l in layers].index(lk)
            rjobs.append((("act", pools.index(pname), li), b, int(probe["fold_seeds"])))
            if ("mask", f"w.{pname}") in res:
                rjobs.append((("mask", f"w.{pname}"), b, int(probe["fold_seeds"])))
        with ProcessPoolExecutor(workers) as ex:
            for (xfun, b, _), r in ex.map(_run_refit, rjobs):
                key = "one_twin_refit" if xfun[0] == "act" else "one_twin_refit_mask_floor"
                out["targets"][b][key] = {"r2": float(np.mean(list(r.values()))), "by_twin": r,
                                          "n_groups": n_groups}
        log(f"[L0] one-twin refit: {len(rjobs)} fits")

    rb = cfg.get("velocity_protocols", {}).get("rigidbench", {})
    if rb.get("enabled"):
        kw = {k: rb[k] for k in ("epochs", "lr", "batch", "fold_seeds") if k in rb}
        rjobs = [(("act", pools.index(_pool_name("roi_own", b.split(".")[0], b.split(".")[1])), li), b, kw)
                 for b in Y if b.endswith(".velocity") for li in range(len(layers))]
        rres: dict = {}
        with ProcessPoolExecutor(workers) as ex:
            for (xfun, b, _), r in ex.map(_run_rigidbench, rjobs):
                rres.setdefault(b, {})[f"{_lname(layers[xfun[2]])}/{pools[xfun[1]]}"] = r
        for b, cells in rres.items():
            o, ph, _ = b.split(".")
            pname = _pool_name("roi_own", o, ph)
            ridge = {f"{_lname(l)}/{pname}": res[("act", pools.index(pname), li)][b] for li, l in enumerate(layers)}
            out["targets"][b]["velocity_protocol"] = {
                "rigidbench": {**_select(cells, Y[b], groups, n_splits), **{k: rb.get(k) for k in kw},
                               "per_cell": {c: v["score_mean"] for c, v in cells.items()}},
                "ridge_same_pool": _select(ridge, Y[b], groups, n_splits)}
        log(f"[L0] RigidBench protocol: {len(rjobs)} fits")
    return out


def table(res: dict, untrained: dict | None = None) -> str:
    pr = res.get("protocol", {})
    n_var = {b: t["heldout"]["n_variants"] for b, t in res["targets"].items() if "heldout" in t}
    rows = [f"n_groups = {res['n_groups']}, {pr.get('k_folds', 5)}-fold x {pr.get('fold_seeds', '?')} seeds "
            f"(~{pr.get('test_groups_per_fold', res['n_groups'] / 5):.0f} test groups per fold), "
            f"{pr.get('n_permutations', '?')} permutations; variants searched per target: "
            f"{sorted(set(n_var.values())) or '?'} (layers x pools). Held-out R²: cell selected on half the groups, "
            f"scored on the other half ({pr.get('selection_splits', '?')} random splits x 2); in-sample max R² is the "
            "old protocol (its permutation p corrects for the search).", "",
            "| target | held-out R² (selected cell, picks) | in-sample best cell: R² (sd) | untrained held-out R² | "
            "one-twin refit R² (mask floor) | mask floor (same pool) | latent baseline | tracker | best observer | "
            "null q95 | p_corr | p_Holm |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, t in res["targets"].items():
        ut = untrained["targets"].get(b, {}) if untrained else {}
        u = ut.get("heldout", {}).get("score", ut.get("r2"))
        h = t.get("heldout")
        hs = (f"{h['score']:.3f} ({h['selected']}, {h['selected_counts'][str(h['selected'])]}/{2 * h['n_splits']})"
              if h else "")
        ot = t.get("one_twin_refit")
        of = t.get("one_twin_refit_mask_floor")
        os_ = f"{ot['r2']:.3f} ({of['r2']:.3f})" if ot and of else (f"{ot['r2']:.3f}" if ot else "")
        lb = t.get("latent_baseline")
        lbs = f"{lb['score']:.3f} ({lb['selected']})" if lb else ""
        rows.append(f"| {b} | {hs} | {t['best_cell']}: {t['r2']:.3f} ({t['r2_sd']:.3f}) | "
                    f"{'' if u is None else f'{u:.3f}'} | {os_} | {t['mask_floor']['r2']:.3f} | "
                    f"{lbs} | {t['tracker']['r2']:.3f} | "
                    f"{t['observer']['r2']:.3f} ({t['observer']['best']}) | {t['null_max_q95']:.3f} | "
                    f"{t['p_corrected']:.3g} | {t.get('p_holm', float('nan')):.3g} |")
    vp = {b: t["velocity_protocol"] for b, t in res["targets"].items() if "velocity_protocol" in t}
    if vp:
        rows += ["", "Velocity protocol check (spec §5.1), own-ball ROI pool, layer selected by the held-out rule:", "",
                 "| target | ridge (this benchmark) held-out R² (layer) | RigidBench (Adam, unregularised) held-out R² (layer) |",
                 "|---|---|---|"]
        for b, v in vp.items():
            r, a = v["ridge_same_pool"], v["rigidbench"]
            rows.append(f"| {b} | {r['score']:.3f} ({r['selected']}) | {a['score']:.3f} ({a['selected']}) |")
    return "\n".join(rows)
