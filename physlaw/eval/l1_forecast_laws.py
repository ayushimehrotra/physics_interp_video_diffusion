"""L1 forecast law consistency (spec §5.0.4): do the model's OWN predictions of the post-event state obey the laws?

forecast() gives the model's prediction of the post-event token-frames from pre-event frames only (cached by stage 4,
one per (group, pre_segment): :func:`forecast_path`). It is decoded into states and scored with the law statistics
of ``physlaw.eval.laws`` (spec §5.0.1) under the probe rules R1–R5 (§5.0.3).

Decoding (R1, one recipe for every law; the shared implementation physlaw.probes.ridge.r1_decode / instrument_folds).
Ridge probes from pooled features to the velocity at the event instant (target per phase), cross-fitted over twin
groups (``k_folds`` x ``fold_seeds_l1``), each with a fold-proper inverse calibration to true units (ŷ ≈ c + d y per
output, fitted on inner out-of-fold predictions inside the training groups, decode (ŷ - c) / d), which restores the
measurement scale (E[decode | y] = y) so ratio statistics are not shrunk toward the mean. Every instrument also reports its selectivity control tasks (Hewitt & Liang:
``probe_r2[object]["selectivity"]`` {scene, object}). The instrument (recorded per variant as
``instrument``; per-object calibrated pre and post R² in ``probe_r2``):
  * "pred_real" (PRIMARY wherever the cache has ``<clip>.pred_real.npz``: encoders with a predictor / decoder, V-JEPA 2,
    VideoMAE, PC1): post rows = the model's own forecast-space outputs for the masked REAL post-event frames, full clip
    as context. One probe per object per SPACE: pre on the real pre pool (encoder space), post on the pred-real pool
    (predictor space), never one linear map across the two. The forecast lives in the predictor space: read as is.
    Diagnostic ``floor_encoder``: the real post frames through an encoder-space post probe;
  * "as_is" (no pred-real, e.g. diffusion: pre and post pools of the same layer): ONE probe per object trained on
    pre and post rows jointly; the forecast (same layer) read as is.
The same map decodes
  * the PRE state (real pre-event pool) and the floor POST state (the post rows, out-of-fold), R2;
  * the forecast (as is: every law, momentum included) and, as a diagnostic only, the forecast moment-matched per
    feature onto the post rows' per-group means (``forecast_mm``; no labels; :func:`r1_decode`). Moment matching
    cannot tell a feature shift from physics: it absorbs any eps / acceleration ratio the model applies to every
    group, mapping it onto the real ladder's mean (tests/test_l1_events.py);
  * the mask-only floor (R3, roi pools): the same pipeline on the pool's placement weights alone, always reported
    beside an roi result.
Controls: the real-frame floor, the mass-blind rule (S_mom's scale), each law's null (ratio laws), the mask-only floor
(roi pools: every roi score must BEAT it, M0), and three negative-control forecasts per law (H4, :data:`CONTROLS`):
(i) ctl_static, the last context state carried over the post window (per-token-frame kinds: the last context
token-frame's features repeated over the forecast's token-frames; pooled collisions: the pre-event pool, which carries
the same state at constant velocity), (ii) ctl_other_forecast, another set's forecast (same twin type), (iii)
ctl_other_real, another set's REAL post-event features (same twin type), each decoded by the identical instrument
(out of fold for the set the features come from) and scored with the same statistic on THIS clip's pre state and
windows. A ridge probe outputs roughly the conditional mean of its target, and its real training rows are mostly
legal, so features that say little about the outcome can still decode to a law-consistent one: the ``verdict`` of
every score allows a law claim only when the forecast beats all three (paired group-bootstrap CI of the difference
above 0; an undefined control, e.g. gated out, counts as beaten) and the mask-only floor where present ("probe
fills in" otherwise). Where the law's statistic does not depend on the scene (gravity g_ratio: every set falls at the
same g), (ii) and (iii) obey the law by construction, so L1 alone cannot support that claim. The untrained model is a
separate run of this level on its cache (same json keys). No permutation nulls here: held-out selection over spaces
and group-bootstrap CIs only. P-forecast (a probe trained on the forecast itself) is never used: such a probe could do
the physics itself.
Forecast-dependent filters (M1): the momentum interaction gate and the energy gates keep a clip only if the forecast
reacts; every package reports ``pass_share`` (the eligible clips the forecast gate keeps; eligible = the floor sees the
event) and a sensitivity score counting each removed clip as a failure (S_mom: S_FLOOR = −1, ``S_sensitivity``;
energy: as creating energy, ``creation_rate_sensitivity`` = share x creation + (1 − share)). The ratio laws have no
forecast-dependent filter (their windows are positional).

Scores (one primary number per law; every one reported as forecast / floor / observers):
  * momentum (F1, F2, F4; not F3, :data:`F3_NOTE`) S_mom: dv = decoded post − DECODED pre velocities (M4: one
    calibrated instrument per object reads both; masses and the gate unit from the labels; ``S_true_pre`` = the old
    true-pre mix, a diagnostic); interaction gate D_fc >= kappa D_floor (D = |q_A| + |q_B|, q_i = m_i dv_i) and
    D_floor >= kappa 2 mu u_n (true pre); on gated momentum-legal clips with mass ratio >= min_mass_ratio,
    eta = |q_A + q_B| / (|q_A| + |q_B|) pooled as a ratio of sums; S_mom = 1 − (eta_fc − eta_floor) / (eta_blind −
    eta_floor), eta_blind = the mass-blind rule on the true outcome; by mass-ratio bin and kappa, group bootstrap CI,
    per-group sums for :func:`momentum_on_common_groups`. Secondary: S_family (distance to the legal collision family,
    d_null = 1). Mass-free antiparallel and normal alignment are SANITY checks only (N1: they equal 1 for the legal
    collision, the mass-blind rule and every momentum-edited twin), never a law score.
  * energy (every exchange event): collisions F1–F4 from the pooled pre / post pools (velocity constant on both sides:
    g = 0, frictionless floor, so the labelled event-instant target is the velocity of every free frame) with the
    S_mom gate (a pass-through forecast, eps = −1, moves no momentum and never passes it, N1); the F7 bounce, F8 pad
    and F9 swing as per-token-frame kinds (H3: no window averages): bounce = velocities from line fits over the free
    token-frames on each side (after the release, ball in frame; post: up to the second floor contact), BOTH
    extrapolated to the contact time, gate |dv_fc| >= kappa |dv_floor|, |dv_floor| >= kappa |dv_true| (dv =
    v_post(te) − v_pre(te), a vector); pad and swing = sqrt(h' / h) from decoded apex heights (the F8 path is
    piecewise: ramp, fillet, flat, pad, ramp; no straight part holds >= 3 token-frames on both sides of the pad, and
    on a frictionless rail sqrt(h' / h) = s' / s exactly), gate on the rise. On gated legal clips (eps_true <= 1):
      tau = the ``tau_quantile`` of |eps_floor − eps_true| on the real legal clips (probe noise);
      creation rate P(eps_hat > 1 + tau) (PRIMARY) and lower-tail rate P(eps_hat < eps_lo,g − tau), eps_lo,g = per set
      the lowest FLOOR-decoded eps of its gated legal twins (:func:`energy_scores`; excess decoding noise inflates
      BOTH tails, a dissipative bias only the lower one);
      median eps_hat (the expected eps: the hidden parameter, a diagnostic).
    Floor check (R2, ``floor_check``): slope and R² of eps_floor on eps_true over every real twin; energy is "unmeasurable" for a
    model/pool whose slope CI contains 0 or R² < ``slope_r2_min``. R5: eps <= 1 in a forecast is necessary, not
    sufficient (any dissipative or static forecast satisfies it); the primary energy evidence is the L2 kink.
  * the ratio laws, one per-token-frame kind each (``RATIO_LAW`` / ``KINDS``): gravity (F7 projectile / drop),
    friction (F6 slide / pair: decel_const, stop_hold, mass_ratio from the pre, moving and stop windows), inertia (F6
    glide: momentum's inertia_speed, inertia_turn), oscillation (F9 pair: period_ratio, mass_period_ratio from
    zero crossings of the decoded offsets), rolling (F10), statics (F11: tip_consistency, alpha_ratio with the clip's
    own g), angular_momentum (F12: L_ratio and momentum's eta). Velocities, angular velocities and positions from
    per-token-frame pools (section "per-token-frame kinematics": one calibrated instrument per object and quantity on
    the real token-frames, least-squares lines over >= 3 token-frames per window); per statistic of law.stats, S =
    laws.ratio_score(T_fc, T_legal, T_null) (1 legal, 0 the law's null_outcome; the linear form 1 − |T − T_legal| /
    |T_null − T_legal| where the null statistic is 0; the unit form 1 − |T − T_legal| where it is undefined: glide
    inertia, eta, statics), per clip clipped to [−1, 1] and −1 where the clip's labelled statistic is defined but the
    decoded one is not (non-finite, or <= 0 in the log form: M2), median over legal clips (tip_consistency: the
    accuracy, the mean, floor 0), for the forecast, the floor, the labelled truth and the H4 controls.
Pools: frame, contact and roi (object patches) are all scored (collisions). roi pools each object's ROI at its TRUE
post location, so its placement leaks where the object went: every roi score (S_mom, creation rate) must beat the
same score of the mask-only floor (identical pipeline on the pooling weights; M0, ``verdict.beats_mask_only``); frame
and contact pools carry no placement and the per-token-frame laws use the frame pool only. Group bootstrap 95% CIs. Spaces (DiTs: one per (SNR, layer)): S_mom is selected
held-out (probes.select.heldout_select); energy and the ratio laws take the space whose FLOOR instrument is best
(forecast-blind, so no selection bias on the forecast score).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physlaw.eval import laws
from physlaw.eval.l0_states import complete_groups, load_index, object_class_ids
from physlaw.probes.ridge import (DEFAULT_ALPHAS, apply_instrument, instrument_folds, instrument_selectivity,
                                  r1_decode, r2_score)
from physlaw.probes.select import heldout_select

LEGAL = ("conserve_e1", "conserve_e015")       # F1's momentum-legal twins (others: labels momentum_conserved)
COLLISION_FAMILIES = ("f1", "f2", "f3", "f4")
ENERGY_EVENTS = ("collision", "bounce", "pad")
MASS_HIDDEN_FAMILIES = ("f3",)   # every mass ratio of a set shares its pre-event frames: no L1 momentum score (H1)
F3_NOTE = ("F3 (hidden mass): the five mass ratios of a set have identical look and radius (masses by density at a "
           "fixed geometric mean: data/twins.f3_twins, data/generate.py) and byte-identical pre-event frames, so ONE "
           "forecast serves all five ratios and cannot depend on the ratio: S_mom would fail by construction and is "
           "not scored. The mass-free checks (antiparallel, normal alignment) cannot see a momentum edit (N1), so F3 "
           "has no L1 momentum law score: it is an L2 / L4 family (the mass enters through the shown outcome or a "
           "steered mass direction). Energy (eps, mass-free) is still scored.")
# H4 negative-control forecasts (every law): decoded through the identical instrument, scored with the same statistic
CONTROLS = ("ctl_static", "ctl_other_forecast", "ctl_other_real")
CONTROL_DOC = {"ctl_static": "the last observed (context) state carried over the post window",
               "ctl_other_forecast": "another set's forecast (same twin type)",
               "ctl_other_real": "another set's REAL post-event features (same twin type)"}
SANITY_NOTE = ("sanity only (cannot detect the drift edit): antiparallel and normal alignment are 1 for the legal "
               "collision, the mass-blind rule and every momentum-edited twin (N1); never a law score")
POOLS = {"frame": lambda o: "frame", "contact": lambda o: "contact", "roi": lambda o: f"roi_{o}"}
MASK_FLOOR_POOLS = ("roi",)        # pools whose placement can carry the answer: mask-only floor reported
WEAK_FLOOR = 0.8          # d_floor above this: S_family uninformative
WEAK_MOM_FLOOR = 0.8      # eta_floor above this x eta_blind: S_mom uninformative (‡)
# min_mass_ratio 1.25: below it the mass-blind rule has eta_blind < 0.11, nearly as good as physics.
MOM_DEFAULTS = {"kappa": 0.25, "gate_ref": "floor", "kappa_sensitivity": (0.1, 0.25, 0.5), "min_mass_ratio": 1.25,
                "mass_ratio_bins": (1.25, 2.0, 4.0)}
ENERGY_DEFAULTS = {"kappa": 0.25, "tau_quantile": 0.95, "slope_r2_min": 0.5}
_MOM_KEYS = ("fc_num", "fc_den", "fl_num", "fl_den", "bl_num", "bl_den")
NAN = float("nan")
R5_NOTE = ("eps <= 1 in a forecast is necessary, not sufficient: any dissipative or static forecast satisfies it, and "
           "a threshold on eps is decodable from any correlate of the dose (spec §5.0.3 R5); the primary energy "
           "evidence is the L2 kink")


def _S(fc, fl, ref):
    """1 − (fc − fl) / (ref − fl), clipped to [−1, 1]; NaN where ref <= fl. Works on arrays."""
    fc, fl, ref = np.broadcast_arrays(*(np.asarray(a, float) for a in (fc, fl, ref)))
    d = ref - fl
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, np.clip(1 - (fc - fl) / d, -1, 1), np.nan)


def _boot_ci(stat, groups, n_boot, rng) -> list:
    """Group-bootstrap 95% CI of ``stat(row_indices)`` (rows resampled as whole groups)."""
    if not n_boot or len(groups) == 0:
        return [NAN, NAN]
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    b = np.array([stat(np.concatenate([by[j] for j in rng.integers(0, len(by), len(by))])) for _ in range(n_boot)])
    return [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))] if np.isfinite(b).any() else [NAN, NAN]


def _sensitivity(score, share, worst: float = laws.S_FLOOR) -> float:
    """M1: a score with every clip the forecast-dependent gate removed counted as a FAILURE: share x score + (1 −
    share) x worst. S scores (higher better): worst = laws.S_FLOOR (−1); violation rates (energy creation): worst =
    1 (a removed clip counts as creating energy). share 0 (every clip removed): worst; otherwise NaN when either
    input is."""
    score, share = float(score), float(share)
    if share == 0:
        return float(worst)
    return float(share * score + (1 - share) * worst) if np.isfinite(score) and np.isfinite(share) else NAN


def paired_ci(stat_a, stat_b, groups, n_boot, rng) -> list:
    """Group-bootstrap 95% CI of stat_a(rows) − stat_b(rows) on the SAME resampled groups (a resample where only b is
    undefined counts as +inf: a shows the event, b does not)."""
    if not n_boot or len(groups) == 0:
        return [NAN, NAN]
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    d = []
    for _ in range(n_boot):
        r = np.concatenate([by[j] for j in rng.integers(0, len(by), len(by))])
        a, b = stat_a(r), stat_b(r)
        d.append(np.inf if np.isfinite(a) and not np.isfinite(b) else a - b)
    d = np.array(d, float)
    return [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))] if np.isfinite(d).any() else [NAN, NAN]


def verdict(fc, controls: dict, mask_only=None, higher_better: bool = True, diff_ci: dict | None = None) -> dict:
    """H4 / M0: a law claim needs the forecast to BEAT every negative-control forecast (:data:`CONTROLS`, decoded by
    the identical instrument and scored with the same statistic) and, for roi pools, the mask-only floor. Beat =
    the group-bootstrap CI of the paired difference (forecast − control for S scores, control − forecast for
    violation rates; ``diff_ci`` {name: [lo, hi]}) lies above 0, or, without a CI, the point estimate is better. A
    control with an undefined score (e.g. gated out: it shows no event) counts as beaten. ``higher_better``: S
    scores; False for violation rates (energy creation). If a control scores as well as the forecast, the probe (or
    the pooling mask) can fill in law-consistent answers without the model: "probe fills in"."""
    fc = float(fc) if fc is not None else NAN
    diff_ci = diff_ci or {}

    def beat(name, c):
        if c is None or not np.isfinite(float(c)):
            return True
        ci = diff_ci.get(name)
        if ci is not None and np.isfinite(ci[0]):
            return bool(ci[0] > 0)
        return fc > float(c) if higher_better else fc < float(c)
    beats = {k_: bool(np.isfinite(fc) and beat(k_, v)) for k_, v in controls.items()}
    out = {"controls": {k_: (float(v) if v is not None else NAN) for k_, v in controls.items()}, "beats": beats,
           "beats_mask_only": None if mask_only is None else bool(np.isfinite(fc) and beat("mask_only", mask_only)),
           "diff_ci95": {k_: [float(x) for x in v] for k_, v in diff_ci.items()}, "higher_better": higher_better}
    lost = [k_ for k_, b in beats.items() if not b]
    if not np.isfinite(fc):
        out["claim_allowed"], out["verdict"] = False, "no forecast score"
    elif lost:
        out["claim_allowed"], out["verdict"] = False, "probe fills in (" + ", ".join(lost) + " scores as well)"
    elif out["beats_mask_only"] is False:
        out["claim_allowed"], out["verdict"] = False, "mask placement explains it (mask-only scores as well)"
    elif len(beats) < len(CONTROLS):
        out["claim_allowed"], out["verdict"] = False, "controls missing"
    else:
        out["claim_allowed"], out["verdict"] = True, "beats every negative control"
    return out


def control_partners(clips) -> np.ndarray:
    """H4 controls (ii) / (iii): per clip, the row of ANOTHER set's clip with the same twin type (the next set in sorted
    order holding that twin, cyclically); -1 where no other set has it."""
    by: dict = {}
    for i, c in enumerate(clips):
        by.setdefault(c.split("/")[1], []).append(i)
    out = np.full(len(clips), -1)
    for rows in by.values():
        rows = sorted(rows, key=lambda i: clips[i])
        if len({clips[i].split("/")[0] for i in rows}) < 2:
            continue
        for k_, i in enumerate(rows):
            j = next(rows[(k_ + m) % len(rows)] for m in range(1, len(rows) + 1)
                     if clips[rows[(k_ + m) % len(rows)]].split("/")[0] != clips[i].split("/")[0])
            out[i] = j
    return out


def _take(A: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """A[rows], NaN where rows == -1."""
    out = np.asarray(A, float)[np.maximum(rows, 0)].copy()
    out[rows < 0] = NAN
    return out


def pooled_controls(pre: dict, dec: dict, clips) -> None:
    """Adds the H4 negative controls to the pooled decodes (in place): post velocities ``dec[ctl]`` and their pre
    decodes ``pre[ctl]`` (the clip's own decoded pre state, as the forecast's). ctl_static: the decoded pre-event state
    carried over (the identical joint map reads the pre-event pool; collisions move at constant velocity, so the last
    context token-frame and the pre-window pool carry the same state); ctl_other_forecast / ctl_other_real: another
    set's decoded forecast / real post-event pool (same twin type; each decoded out of fold by the identical post
    instrument)."""
    part = control_partners(clips)
    dec["ctl_static"] = np.asarray(pre["forecast"], float).copy()
    dec["ctl_other_forecast"] = _take(dec["forecast"], part)
    dec["ctl_other_real"] = _take(dec["floor"], part)
    for c in CONTROLS:
        pre[c] = pre["forecast"]


# --------------------------------------------------------------------------- forecast cache files
#
# Stage 4 caches one encoder forecast per (group, pre_segment): ``<group>/_forecast.npz`` for groups with one segment,
# ``<group>/_forecast_<pre_segment>.npz`` per segment otherwise (F4 boost levels). Diffusion caches pool each twin with
# its own segment's sampled forecast into the twin's fc_pooled.

def clip_segment(labels: dict) -> str:
    """A clip's ``physlaw.pre_segment`` ("0" when unlabelled: one pre-event state per group)."""
    return str(labels["physlaw"].get("pre_segment", "0"))


def forecast_segments(seg: dict) -> dict:
    """{"<group>/<twin>": pre_segment} -> {"<group>/<twin>": the segment key of its forecast file}: the clip's own
    pre_segment where its group (among the clips given) has more than one, else None (``_forecast.npz``)."""
    by: dict = {}
    for c, s in seg.items():
        by.setdefault(c.split("/")[0], set()).add(str(s))
    return {c: (str(s) if len(by[c.split("/")[0]]) > 1 else None) for c, s in seg.items()}


def forecast_path(cache_dir, group: str, segment: str | None = None) -> Path:
    """The encoder forecast file of (group, pre_segment) (segment keys from :func:`forecast_segments`)."""
    return Path(cache_dir) / group / ("_forecast.npz" if segment is None else f"_forecast_{segment}.npz")


def check_shared_forecast(where: str, segments) -> None:
    """Guard: one forecast may be shared only by twins with the same pre-event frames (same pre_segment)."""
    s = sorted({str(x) for x in segments})
    if len(s) > 1:
        raise RuntimeError(f"{where}: one forecast would be shared by twins with different pre-event frames "
                           f"(pre_segment {s}); forecasts must be keyed by (group, pre_segment)")


def _load_spaces(cache_dir, recs, clips, model_cfg, mf, seg=None):
    """{space: {"real": {pool: [n, D]} (pre AND post pools), "fc": {post pool: [n, D]}, "pr": {post pool: [n, D]}
    (encoders with ``.pred_real.npz`` sidecars only)}}. ``seg``: clip ->
    pre_segment (None: one forecast per group); encoders read each clip's own forecast file."""
    first = np.load(cache_dir / recs[0]["file"])
    pools = [str(p) for p in first["pools"]]
    base = lambda p: p.split(".")[0]
    PP = tuple(p for p in pools if p.endswith(".post") and (base(p) in ("frame", "contact")
                                                            or (base(p).startswith("roi_") and base(p) != "roi_pair")))
    PR = tuple(p.replace(".post", ".pre") for p in PP if p.replace(".post", ".pre") in pools)
    zs = [np.load(cache_dir / r["file"]) for r in recs]
    if model_cfg.get("kind") == "diffusion" and "fc_pooled" not in first.files:
        raise RuntimeError(f"{cache_dir}: diffusion cache has no forecast (fc_pooled); L1 needs stage 4 pass 2 "
                           "(scripts/04_cache_activations.py without --no-forecast adds it to the existing cache)")
    if "fc_pooled" in first.files:     # diffusion: one space per (SNR, layer) key, pooled per twin by stage 4
        lkeys = [str(l) for l in first["layers"]]
        fkeys = [str(x) for x in first["fc_keys"]]
        if not set(fkeys) & set(lkeys):
            raise RuntimeError(f"{cache_dir}: no forecast key {fkeys} is among the encode layers, so no real rows share "
                               "its space (R1); forecast_layers in configs/models must be a subset of layers")
        return {q: {"real": {p: np.stack([z["pooled"][pools.index(p), lkeys.index(q)] for z in zs]).astype(np.float64)
                             for p in PP + PR},
                    "fc": {p: np.stack([z["fc_pooled"][pools.index(p), fkeys.index(q)] for z in zs]).astype(np.float64)
                           for p in PP}}
                for q in fkeys if q in lkeys}
    if "pooled_final" in first.files:  # forecast space: VideoMAE's decoder (own width) or layer -1 of `pooled`
        get = lambda z, p: z["pooled_final"][pools.index(p)]
    else:
        if -1 not in [int(l) for l in first["layers"]]:
            raise RuntimeError(f"{cache_dir}: no forecast space (layer -1) in the cache: an encoder-only model "
                               "(adapter predicts=False, e.g. VideoMAE-v2 base) has no forecast; L1 does not apply")
        li = [int(l) for l in first["layers"]].index(-1)
        get = lambda z, p: z["pooled"][pools.index(p), li]
    real = {p: np.stack([get(z, p) for z in zs]).astype(np.float64) for p in PP + PR}
    phases = {c: np.array([str(x) for x in z["phases"]]) for c, z in zip(clips, zs)}
    fc = {p: np.zeros_like(real["frame.post"]) for p in PP}
    row = {c: i for i, c in enumerate(clips)}
    key = forecast_segments(seg) if seg is not None else {c: None for c in clips}
    for g, sg in sorted({(c.split("/")[0], key[c]) for c in clips}, key=lambda x: (x[0], str(x[1]))):
        z = np.load(forecast_path(cache_dir, g, sg))
        F = z["final"].astype(np.float32)                      # [T_post, gh, gw, D]
        t0 = int(z["context_frames"]) // int(model_cfg.token_grid.tubelet)
        for c in [c for c in clips if c.startswith(g + "/") and key[c] == sg]:
            i = row[c]
            post = phases[c][t0:] == "post"
            for p in [q for q in PP if q != "frame.post"]:
                w = mf[f"w.{p}"][i].reshape(len(phases[c]), *F.shape[1:3])[t0:] * post[:, None, None]
                fc[p][i] = (w.reshape(-1) / w.sum()) @ F.reshape(-1, F.shape[-1])
            fc["frame.post"][i] = F[post].reshape(-1, F.shape[-1]).mean(0)
    out = {"real": real, "fc": fc}
    prf = [cache_dir / r["file"].replace(".npz", ".pred_real.npz") for r in recs]
    if all(f.exists() for f in prf):      # stage 4 --pred-real: forecast-space outputs for masked real post frames
        zp = [np.load(f) for f in prf]
        pp = [str(p) for p in zp[0]["pools"]]
        out["pr"] = {p: np.stack([z["pooled"][pp.index(p)] for z in zp]).astype(np.float64) for p in PP if p in pp}
    return {"final": out}


# --------------------------------------------------------------------------- clip states

def clip_states(labels: dict, ratio_dt=None) -> tuple[dict, dict, dict, dict]:
    """(pre, post, ctx, info). Collisions: laws.from_labels (planar); other families: io.law_inputs (event_state, the
    F9 swing's apex states; ``ratio_dt`` s after the event for the ratio laws when given; aux bodies dropped).
    ctx["event"] in ENERGY_EVENTS, "swing" or "none". info: objects, event, eps (true eps of the exchange event),
    energy_legal, momentum_legal, legal, law ("energy" for pooled exchange events, else the per-token-frame kind of
    :data:`KINDS`)."""
    pl = labels["physlaw"]
    fam = pl.get("family", "f1")
    if fam in COLLISION_FAMILIES:
        pre, post, ctx = laws.from_labels(labels)
        event, objects = "collision", ["A", "B"]
    else:
        from physlaw.data.io import law_inputs
        event = str(pl.get("event_kind"))
        pre, post, ctx = law_inputs(labels, None if event in ENERGY_EVENTS else ratio_dt)
        objects = list(pre["objects"])
        if event not in ENERGY_EVENTS + ("swing",):
            event = "none"
    ctx = {**ctx, "event": event}
    info = {"objects": objects, "event": event, "family": fam, "twin": pl["twin"],
            "legal": bool(pl.get("legal", ctx.get("legal", pl["twin"] in LEGAL)))}
    if event in ENERGY_EVENTS + ("swing",):
        eps = pl.get("epsilon")
        info["eps"] = float(eps) if eps is not None else float(laws.epsilon(pre, post, ctx))
        info["energy_legal"] = info["eps"] <= 1 + 1e-9
    if event == "collision":
        mc = pl.get("momentum_conserved", not pl["twin"].startswith("mom_violate"))
        info["momentum_legal"] = bool(mc) and info["energy_legal"]
    # collisions: pooled exchange-event pools; every other family (F7 bounce and F8 pad included, H3: velocities
    # extrapolated to the event / apex heights) a per-token-frame kind
    info["law"] = "energy" if event == "collision" else ratio_kind(fam, str(pl.get("subfamily", "")))
    return pre, post, ctx, info


# --------------------------------------------------------------------------- momentum (S_mom, unchanged)

def _mom_clip(pre, post, ctx, v_fc, v_fl, vp_fc=None, vp_fl=None) -> dict:
    """S_mom ingredients of one clip: momentum created (num) and moved (den) by the decoded forecast, the floor decode
    of the real frames and the mass-blind reference (on the clip's TRUE outcome); mass-free diagnostics. ``vp_fc`` /
    ``vp_fl``: the DECODED pre-event velocities of each source (M4: dv = decoded post − decoded pre, both read by the
    same calibrated instrument; None = the true pre velocities, the diagnostic ``true_pre`` variant). Masses and the
    gate unit d_min = 2 mu u_n stay labelled (scene properties)."""
    p = pre["p"]
    pf = pre if vp_fc is None else {**pre, "v": np.asarray(vp_fc, float)}
    pl = pre if vp_fl is None else {**pre, "v": np.asarray(vp_fl, float)}
    fc = laws.momentum_exchange(pf, {"v": v_fc, "p": p}, ctx)
    fl = laws.momentum_exchange(pl, {"v": v_fl, "p": p}, ctx)
    fc["d_min"] = laws.momentum_exchange(pre, pre, ctx)["d_min"]
    bl = laws.momentum_exchange(pre, laws.mass_blind_outcome(pre, post, ctx), ctx)
    sf = laws.momentum.stats(pf, {"v": v_fc, "p": p}, ctx)
    sl = laws.momentum.stats(pl, {"v": v_fl, "p": p}, ctx)
    m = np.asarray(ctx["m"], float)
    return {"fc_num": fc["num"], "fc_den": fc["den"], "fl_num": fl["num"], "fl_den": fl["den"],
            "bl_num": bl["num"], "bl_den": bl["den"], "d_min": fc["d_min"], "mass_ratio": float(m.max() / m.min()),
            **{f"{s}_{k_}": float(d[k_]) for s, d in (("fc", sf), ("fl", sl))
               for k_ in ("antiparallel", "normal_A", "normal_B")}}


def _gate(m: dict, src: str, kappa: float, gate_ref: str = "floor") -> bool:
    """Interaction gate. Floor decode: D_fl >= kappa * 2 mu u_n. Forecast: D_fc >= kappa * D_fl (gate_ref "floor")
    or D_fc >= kappa * 2 mu u_n (gate_ref "d_min": absolute; probe noise alone can clear it)."""
    if src == "fl":
        return m["fl_den"] >= kappa * m["d_min"]
    return m["fc_den"] >= kappa * (m["fl_den"] if gate_ref == "floor" else m["d_min"])


def _eta_S(t):
    """t [..., 6] pooled sums in _MOM_KEYS order -> (eta [..., 3] for forecast / floor / blind, S_mom [...])."""
    t = np.asarray(t, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(t[..., 1::2] > 0, t[..., 0::2] / t[..., 1::2], np.nan)
    return eta, _S(eta[..., 0], eta[..., 1], eta[..., 2])


def _mom_group_sums(per, gs_rows, kappa, lo, hi=np.inf, gate_ref="floor"):
    """Per group, the _MOM_KEYS sums over its clips that pass the gate (forecast AND floor) with mass ratio in
    [lo, hi). Returns ([n_groups, 6], clips used per group)."""
    S = np.zeros((len(gs_rows), len(_MOM_KEYS)))
    n = np.zeros(len(gs_rows), int)
    for gi, cs in enumerate(gs_rows):
        for c in cs:
            m = per[c]["mom"]
            if lo <= m["mass_ratio"] < hi and _gate(m, "fc", kappa, gate_ref) and _gate(m, "fl", kappa, gate_ref):
                S[gi] += [m[k_] for k_ in _MOM_KEYS]
                n[gi] += 1
    return S, n


def _momentum_scores(per, gs_rows, n_boot, rng, mcfg, score=True) -> dict:
    """The S_mom package over ``gs_rows`` (per group, the clip keys pooled for it)."""
    kappa, rmin = float(mcfg["kappa"]), float(mcfg["min_mass_ratio"])
    groups = [cs[0].split("/")[0] for cs in gs_rows]
    clips = [c for cs in gs_rows for c in cs]
    gref = str(mcfg.get("gate_ref", "floor"))
    gate = lambda c, src, k_: _gate(per[c]["mom"], src, k_, gref)
    gated = [c for c in clips if gate(c, "fc", kappa) and gate(c, "fl", kappa)]
    diag = lambda k_: float(np.nanmean([per[c]["mom"][k_] for c in gated])) if gated else NAN
    G, n = _mom_group_sums(per, gs_rows, kappa, rmin, gate_ref=gref)
    eta, S = _eta_S(G.sum(0))
    # M1: the forecast gate selects on the outcome. Eligible = mass ratio >= min and the floor sees the event (a
    # model-independent property of the clip); pass_share = the eligible clips the forecast gate keeps
    elig = [c for c in clips if per[c]["mom"]["mass_ratio"] >= rmin and gate(c, "fl", kappa)]
    share = float(np.mean([gate(c, "fc", kappa) for c in elig])) if elig else NAN
    res = {"kappa": kappa, "min_mass_ratio": rmin, "gate_ref": gref,
           "interaction_rate": float(np.mean([gate(c, "fc", kappa) for c in clips])),
           "interaction_rate_floor": float(np.mean([gate(c, "fl", kappa) for c in clips])),
           "pass_share": share, "n_eligible": len(elig),
           "mass_free": {"role": SANITY_NOTE,
                         **{k_: {"forecast": diag(f"fc_{k_}"), "floor": diag(f"fl_{k_}")}
                            for k_ in ("antiparallel", "normal_A", "normal_B")}},
           "n_scored_groups": int((n > 0).sum()), "n_scored_clips": int(n.sum()),
           "eta": {"forecast": float(eta[0]), "floor": float(eta[1]), "mass_blind": float(eta[2])},
           "floor_over_blind": float(eta[1] / eta[2]) if eta[2] > 0 else NAN,
           "S_mom_informative": bool(eta[1] < WEAK_MOM_FLOOR * eta[2]),
           "group_sums": {g: [float(x) for x in G[i]] for i, g in enumerate(groups) if n[i] > 0}}
    if not score:
        return res
    res["S"] = float(S)
    res["S_sensitivity"] = _sensitivity(float(S), share)
    res["S_ci95"] = [NAN, NAN]
    if n_boot and (n > 0).any():
        _, Sb = _eta_S(G[rng.integers(0, len(G), (n_boot, len(G)))].sum(1))
        if np.isfinite(Sb).any():
            res["S_ci95"] = [float(np.nanquantile(Sb, 0.025)), float(np.nanquantile(Sb, 0.975))]
    res["S_by_kappa"] = {str(k_): float(_eta_S(_mom_group_sums(per, gs_rows, float(k_), rmin, gate_ref=gref)[0].sum(0))[1])
                         for k_ in mcfg["kappa_sensitivity"]}
    edges = [float(x) for x in mcfg["mass_ratio_bins"]] + [np.inf]
    res["by_mass_ratio"] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        Gb, nb = _mom_group_sums(per, gs_rows, kappa, lo, hi, gate_ref=gref)
        eb, Sb = _eta_S(Gb.sum(0))
        res["by_mass_ratio"][f"[{lo:g}, {hi:g})"] = {"n_clips": int(nb.sum()), "S": float(Sb), "eta_forecast": float(eb[0]),
                                                     "eta_floor": float(eb[1]), "eta_blind": float(eb[2])}
    return res


def momentum_on_common_groups(entries: list[dict]) -> list[dict]:
    """Re-score S_mom for several models on the groups EVERY one of them scored (the gate otherwise lets each model
    choose its scenes). ``entries``: each model's ``variants[...]["momentum"]``, all at the same kappa."""
    common = sorted(set.intersection(*(set(e["group_sums"]) for e in entries))) if entries else []
    out = []
    for e in entries:
        t = np.array([e["group_sums"][g] for g in common]).sum(0) if common else np.zeros(len(_MOM_KEYS))
        eta, S = _eta_S(t)
        out.append({"n_common_groups": len(common), "S": float(S), "eta_forecast": float(eta[0]),
                    "eta_floor": float(eta[1]), "eta_blind": float(eta[2])})
    return out


def momentum_package(states, clips, dec, n_boot, rng, mcfg, pre=None) -> dict:
    """S_mom (+ S_family, mass-free sanity checks, per-twin etas, observers, the H4 negative controls and the verdict)
    of one (space, pool). ``dec``: {source: post velocities [n, 2, 2]} with "floor" and "forecast" (+ observers and
    controls). ``pre``: {source: DECODED pre velocities} (M4: S_mom reads dv with one calibrated instrument; a source
    without its own pre decode uses the forecast's); None = the true pre velocities. ``S_true_pre`` (diagnostic):
    the old mix of true pre and decoded post velocities."""
    row = {c: i for i, c in enumerate(clips)}
    ug = sorted({c.split("/")[0] for c in clips})
    by_g = {g: [c for c in clips if c.startswith(g + "/")] for g in ug}
    leg_rows = [cs for cs in ([c for c in by_g[g] if states[c][3]["momentum_legal"]] for g in ug) if cs]
    vp = lambda src, c: None if pre is None else pre.get(src, pre.get("forecast"))[row[c]]

    def per_of(src, v, decoded_pre=True):
        return {c: {"mom": _mom_clip(states[c][0], states[c][1], states[c][2], v[row[c]], dec["floor"][row[c]],
                                     vp(src, c) if decoded_pre else None, vp("floor", c) if decoded_pre else None)}
                for c in clips}
    per_src = {src: per_of(src, v) for src, v in dec.items() if src != "floor"}
    per = per_src["forecast"]
    res = _momentum_scores(per, leg_rows, n_boot, rng, mcfg)
    res["pre_velocities"] = "decoded (same instrument)" if pre is not None else "true"
    if pre is not None:          # M4 diagnostic: true pre-event velocities with the decoded post ones
        tp = _momentum_scores(per_of("forecast", dec["forecast"], False), leg_rows, 0, rng, mcfg)
        res["S_true_pre"], res["eta_true_pre"] = tp["S"], tp["eta"]
    res["by_twin"] = {t: {k_: v for k_, v in _momentum_scores(per, [[c] for g in ug for c in by_g[g] if c.endswith("/" + t)],
                                                              0, rng, mcfg, score=False).items() if k_ != "group_sums"}
                      for t in sorted({c.split("/")[1] for c in clips})}
    res["observers"] = {src: {k_: p[k_] for k_ in ("S", "eta", "interaction_rate", "n_scored_clips", "pass_share")}
                        for src in per_src if src != "forecast"
                        for p in [_momentum_scores(per_src[src], leg_rows, 0, rng, mcfg)]}
    kap, rmin, gref = float(mcfg["kappa"]), float(mcfg["min_mass_ratio"]), str(mcfg.get("gate_ref", "floor"))
    G = {src: _mom_group_sums(per_src[src], leg_rows, kap, rmin, gate_ref=gref)[0] for src in per_src}
    S_of = lambda src: (lambda r: float(_eta_S(G[src][r].sum(0))[1]))
    gi = np.arange(len(leg_rows))
    vs = [c for c in CONTROLS + ("mask_only",) if c in G]
    res["verdict"] = verdict(res["S"], {c: res["observers"][c]["S"] for c in CONTROLS if c in res["observers"]},
                             res["observers"].get("mask_only", {}).get("S"), higher_better=True,
                             diff_ci={c: paired_ci(S_of("forecast"), S_of(c), gi, n_boot, rng) for c in vs})
    # S_family (secondary): distance to the legal collision family, d_null = 1
    fam = lambda v, c: laws.momentum.family_distance({"v": v[row[c]], "p": states[c][0]["p"]}, states[c][0], states[c][2])
    lc = [c for cs in leg_rows for c in cs]
    gl = np.array([c.split("/")[0] for c in lc])
    d_fc = np.array([fam(dec["forecast"], c)[0] for c in lc])
    d_fl = np.array([fam(dec["floor"], c)[0] for c in lc])
    s_fam = lambda ix: float(_S(np.nanmean(d_fc[ix]), np.nanmean(d_fl[ix]), 1.0))
    res["family"] = {"S": s_fam(np.arange(len(lc))), "S_ci95": _boot_ci(s_fam, gl, n_boot, rng),
                     "d_forecast": float(np.nanmean(d_fc)), "d_floor": float(np.nanmean(d_fl)),
                     "informative": bool(np.nanmean(d_fl) <= WEAK_FLOOR)}
    return res


# --------------------------------------------------------------------------- energy (eps of the exchange event)

def energy_scores(eps_true, eps_fl, eps_fc, gate_fl, gate_fc, groups, observers=None, tau_q=0.95, r2_min=0.5,
                  n_boot=0, rng=None) -> dict:
    """Energy package from per-clip arrays (module docstring). ``eps_*`` [n]; ``gate_fl`` [n] the floor decode sees
    the event; ``gate_fc`` [n] the forecast passes the interaction gate; ``observers``: {name: (eps [n], gate [n])}
    scored exactly like the forecast (the H4 controls among them). Legal = eps_true <= 1.

    Lower tail (spec §5.0.4, P(eps_hat < eps_floor − tau)): eps_floor = the FLOOR decode of the same set, i.e. per
    group the lowest floor-decoded eps over its gated legal twins (the same pre-event frames, read by the same
    instrument). Not the floor of the same twin: a forecast is one per (group, pre_segment), shared by every rung of a
    ladder, so a per-twin reference would penalise the forecast for not knowing the hidden rung; and not the lowest
    labelled rung of the whole family (the old code), which is in true units, not the instrument's, and lies below
    every scene's own legal range. A forecast below the dissipation the real twins of its own scene show (beyond
    probe noise tau) is the dissipative bias the tail measures."""
    rng = rng or np.random.default_rng(0)
    eps_true, eps_fl, eps_fc = (np.asarray(x, float) for x in (eps_true, eps_fl, eps_fc))
    groups = np.asarray(groups)
    gate_fl = np.asarray(gate_fl, bool) & np.isfinite(eps_fl)
    legal = eps_true <= 1 + 1e-9
    lf = legal & gate_fl
    out = {"note": R5_NOTE, "tau_quantile": tau_q, "n_legal": int(legal.sum()), "n_floor_legal": int(lf.sum())}
    if not lf.any():
        return {**out, "status": "no gated floor clips", "tau": NAN}
    tau = float(np.quantile(np.abs(eps_fl[lf] - eps_true[lf]), tau_q))
    lo_g = {g: float(eps_fl[lf & (groups == g)].min()) for g in np.unique(groups[lf])}
    eps_lo = np.array([lo_g.get(g, NAN) for g in groups])
    out.update(tau=tau, eps_lo=float(eps_true[legal].min()), eps_lo_floor_median=float(np.median(list(lo_g.values()))),
               lower_tail_ref="per group: the lowest floor-decoded eps over its gated legal twins, minus tau",
               eps_true_median=float(np.median(eps_true[lf])))

    cre_of = {}

    def package(e, ok, name=None):
        ok = np.asarray(ok, bool) & lf & np.isfinite(e)
        ix = np.where(ok)[0]
        rate = lambda f: (lambda r: float(np.mean(f(e[r], eps_lo[r]))) if len(r) else NAN)
        if name is not None:        # creation rate on any row subset (paired bootstrap against the forecast)
            cre_of[name] = lambda r, e=e, ok=ok: (float(np.mean(e[r][ok[r]] > 1 + tau)) if ok[r].any() else NAN)
        cre, low = rate(lambda x, lo: x > 1 + tau), rate(lambda x, lo: x < lo - tau)
        share = float(ok.sum() / lf.sum())
        return {"creation_rate": cre(ix), "creation_ci95": _boot_ci(cre, groups[ix], n_boot, rng),
                "lower_tail_rate": low(ix), "lower_tail_ci95": _boot_ci(low, groups[ix], n_boot, rng),
                "eps_median": float(np.median(e[ix])) if len(ix) else NAN,
                "interaction_rate": share, "pass_share": share,
                # M1: every legal clip the forecast gate removed counted as creating energy
                "creation_rate_sensitivity": _sensitivity(cre(ix), share, 1.0), "n": int(len(ix))}
    out["floor"] = package(eps_fl, lf)
    out["forecast"] = package(eps_fc, gate_fc, "forecast")
    out["observers"] = {k_: package(e, g, k_) for k_, (e, g) in (observers or {}).items()}
    ob = out["observers"]
    vs = [c for c in CONTROLS + ("mask_only",) if c in ob]
    out["verdict"] = verdict(out["forecast"]["creation_rate"], {c: ob[c]["creation_rate"] for c in CONTROLS if c in ob},
                             ob["mask_only"]["creation_rate"] if "mask_only" in ob else None, higher_better=False,
                             diff_ci={c: paired_ci(lambda r: -cre_of["forecast"](r), lambda r, c=c: -cre_of[c](r),
                                                       groups, n_boot, rng) for c in vs})
    # R2 instrument check: does the floor decode see eps at all (every real twin, gated)?
    a = np.where(gate_fl & np.isfinite(eps_true))[0]
    fit = lambda r: (np.polyfit(eps_true[r], eps_fl[r], 1)[0] if np.ptp(eps_true[r]) > 0 else NAN)
    if len(a) > 2 and np.ptp(eps_true[a]) > 0:
        slope, r2 = float(fit(a)), float(np.corrcoef(eps_true[a], eps_fl[a])[0, 1] ** 2)
    else:
        slope, r2 = NAN, NAN
    ci = _boot_ci(fit, groups[a], n_boot, rng) if len(a) > 2 else [NAN, NAN]
    measurable = bool(np.isfinite(slope) and r2 >= r2_min and (not n_boot or ci[0] > 0))
    out["floor_check"] = {"slope": slope, "slope_ci95": ci, "r2": r2, "r2_min": r2_min, "n": int(len(a))}
    out["status"] = "measured" if measurable else "unmeasurable"
    return out


def _eps(v_pre, v_post, ctx) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(laws.epsilon({"v": v_pre}, {"v": v_post}, ctx))


def energy_package(states, clips, pre, dec, gates, ecfg, n_boot, rng) -> dict:
    """eps per clip for every decoded source: ``pre`` / ``dec`` {source: decoded pre / post velocities
    [n, n_obj, d]}, ``gates`` {source: bool [n]}, each including "floor" and "forecast"."""
    eps = {s: np.array([_eps(pre[s][i], v[i], states[c][2]) for i, c in enumerate(clips)]) for s, v in dec.items()}
    obs = {s: (eps[s], gates[s]) for s in dec if s not in ("floor", "forecast")}
    return energy_scores([states[c][3]["eps"] for c in clips], eps["floor"], eps["forecast"], gates["floor"],
                         gates["forecast"], [c.split("/")[0] for c in clips], obs, float(ecfg["tau_quantile"]),
                         float(ecfg["slope_r2_min"]), n_boot, rng)


def collision_gates(states, clips, dec, mcfg, pre=None) -> dict:
    """The S_mom interaction gate per source (floor: D_fl >= kappa 2 mu u_n; others: D >= kappa D_fl), D from the
    decoded pre and post velocities (``pre`` {source: [n, 2, d]}; None: true pre). A pass-through forecast (post =
    pre: eps = −1, which satisfies eps <= 1) moves no momentum, D = 0, and never passes (N1)."""
    kap, gref = float(mcfg["kappa"]), str(mcfg.get("gate_ref", "floor"))
    vp = (lambda s, i: None) if pre is None else (lambda s, i: pre.get(s, pre["forecast"])[i])
    out = {}
    for s, v in dec.items():
        ms = [_mom_clip(*states[c][:3], v[i], dec["floor"][i], vp(s, i), vp("floor", i)) for i, c in enumerate(clips)]
        out[s] = np.array([_gate(m, "fl" if s == "floor" else "fc", kap, gref) for m in ms])
    return out


def _decode_pool(S_, mf, pp, states, clips, ix, groups, mask_floor, k, n_seeds, alphas, ids=None):
    """R1 decodes of one (space, pool) for every object. ``pp``: {object: (pre pool, post pool)}. Post rows: the
    pred-real outputs where the space has them for every object (then one calibrated probe per space: encoder pre
    rows, predictor post rows), else the real post pool (one joint probe: same layer). Returns (pre {source:
    [n, n_obj, d]}, post {source: [n, n_obj, d]}, R² {object: {pre, post}}, instrument); sources floor, forecast
    (as is), forecast_mm (moment-matched diagnostic), floor_encoder (pred_real only: the real post frames through
    an encoder-space post probe) (+ mask_only: the pool's placement weights, identical pipeline)."""
    objs = list(pp)
    pr = S_.get("pr", {})
    instrument = "pred_real" if all(b in pr for _, b in pp.values()) else "as_is"
    pre, post, r2 = {}, {}, {}
    for oi, o in enumerate(objs):
        Yp = np.array([states[c][0]["v"][oi] for c in clips])
        Yq = np.array([states[c][1]["v"][oi] for c in clips])
        Xa = S_["real"][pp[o][0]][ix]
        Xb = (pr if instrument == "pred_real" else S_["real"])[pp[o][1]][ix]
        Xf = S_["fc"][pp[o][1]][ix]
        d = r1_decode(Xa, Xb, Yp, Yq, groups, {"forecast": (Xf, None), "forecast_mm": (Xf, Xf)}, k, n_seeds, alphas,
                      joint=instrument == "as_is")
        r2[o] = {"pre": d["r2_pre"], "post": d["r2_post"]}
        if ids is not None:          # selectivity control tasks of the instrument (probes.ridge, Hewitt & Liang)
            r2[o]["selectivity"] = (instrument_selectivity({"x": (Xa, Xb)}, (Yp, Yq), groups, ids[o], k, 1, alphas)
                                    if instrument == "as_is" else
                                    instrument_selectivity({"x": Xb}, Yq, groups, ids[o], k, 1, alphas))
        got = {"floor": (d["pre"], d["post"]), **{s: (d["pre"], d[s]) for s in ("forecast", "forecast_mm")}}
        if instrument == "pred_real":     # diagnostic: the real post frames through an encoder-space post probe
            de = r1_decode(Xa, S_["real"][pp[o][1]][ix], Yp, Yq, groups, {}, k, n_seeds, alphas, joint=False)
            got["floor_encoder"] = (d["pre"], de["post"])
            r2[o]["post_encoder"] = de["r2_post"]
        if mask_floor:
            w = lambda p: mf[f"w.{p}"][ix].astype(np.float64)
            dm = r1_decode(w(pp[o][0]), w(pp[o][1]), Yp, Yq, groups, {}, k, n_seeds, alphas)
            got["mask_only"] = (dm["pre"], dm["post"])
        for s, (a, b) in got.items():
            pre.setdefault(s, []).append(a)
            post.setdefault(s, []).append(b)
    stack = lambda d: {s: np.stack(v, 1) for s, v in d.items()}
    return stack(pre), stack(post), r2, instrument


# --------------------------------------------------------------------------- ratio statistics

UNIT_STATS = ("tip_consistency", "alpha_ratio", "inertia_speed", "inertia_turn", "eta")
MEAN_STATS = ("tip_consistency",)         # a 0 / 1 statistic: S is the accuracy (mean), not a median
LEGAL_FROM_TRUE = ("alpha_ratio",)        # windowed estimator: the legal value is the labelled clip's own statistic


def ratio_scores(T: dict, T_legal, T_null, groups, n_boot=0, rng=None, agg="median", defined=None) -> dict:
    """One statistic, per clip S = 1 at the legal value, 0 at the law's null: the log form laws.ratio_score(T, T_legal,
    T_null) where the null is a positive ratio, else (a null of 0: Aristotle's g_ratio and axis_ratio, rolling order / slip) the linear
    form 1 − |T − T_legal| / |T_null − T_legal|; where the null statistic is undefined (NaN: the Aristotelian stop's
    turn angle, a no-exchange eta, statics) the unit form 1 − |T − T_legal|. ``agg`` ("median" or "mean", e.g. a 0 / 1
    statistic's accuracy) over clips with a group-bootstrap CI, for every source in ``T`` ({source: [n]}; forecast,
    floor, observers). T_legal may be a scalar or per clip.

    Floor (audit M2): every per-clip S is clipped below at laws.S_FLOOR (−1; an accuracy at 0), and a clip whose
    statistic is DEFINED (``defined`` [n], default: where T_legal is finite; L1 passes "the labelled clip's own
    statistic is finite") but whose T is not finite, or <= 0 in the log form (no logarithm: e.g. a decoded
    deceleration ratio of the wrong sign), gets that floor instead of being dropped from the median."""
    rng = rng or np.random.default_rng(0)
    T_null = np.asarray(T_null, float)
    n = len(T_null)
    T_legal = np.broadcast_to(np.asarray(T_legal, float), (n,))
    groups = np.asarray(groups)
    fin = np.isfinite(T_null)
    log_form = bool(fin.all() and fin.any() and (T_null > 0).all() and (T_legal > 0).all())
    floor = 0.0 if agg == "mean" else laws.S_FLOOR
    ok = np.isfinite(T_legal) & (True if defined is None else np.asarray(defined, bool))
    reds = {}
    out = {"T_legal": float(np.nanmedian(T_legal)) if np.isfinite(T_legal).any() else NAN,
           "form": "log" if log_form else ("linear" if fin.any() else "unit"),
           "T_null_median": float(np.nanmedian(T_null)) if fin.any() else NAN, "agg": agg, "S_floor": floor}
    for src, t in T.items():
        t = np.asarray(t, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if log_form:
                s = np.asarray(laws.ratio_score(t, T_legal, T_null, floor=floor), float)
            else:
                d = np.where(fin, np.abs(T_null - T_legal), 1.0)
                d = np.where(d > 0, d, np.nan)
                s = 1 - np.abs(t - T_legal) / d
                s = np.where(~np.isfinite(t) & np.isfinite(d), floor, np.maximum(s, floor))
        s = np.where(ok, s, np.nan)
        with np.errstate(invalid="ignore"):
            bad = ok & (~np.isfinite(t) | ((t <= 0) if log_form else False))
        f = np.nanmean if agg == "mean" else np.nanmedian
        red = lambda r, s=s: float(f(s[r])) if np.isfinite(s[r]).any() else NAN
        out[src] = {"S": red(np.arange(n)), "S_ci95": _boot_ci(red, groups, n_boot, rng),
                    "T_median": float(np.nanmedian(t)) if np.isfinite(t).any() else NAN, "n_floored": int(bad.sum())}
        reds[src] = red
    if "forecast" in reds:      # H4: forecast − control on the same resampled groups
        out["paired_ci95"] = {c: paired_ci(reds["forecast"], reds[c], groups, n_boot, rng) for c in CONTROLS if c in reds}
    return out


# --------------------------------------------------------------------------- per-token-frame kinematics
#
# Every law without an exchange-event pool readout is a ratio of quantities measured at two instants or over a window
# (accelerations, periods, apex heights, rotations), which one pooled post-event vector cannot carry. Stage 4 caches
# every clip's frame pool per token-frame (``pooled_frame_t`` [T_tok, L, D] + ``token_t_s`` [T_tok]); the encoder
# forecast keeps every forecast token-frame (``_forecast*.npz`` ``final``), diffusion caches ``fc_pooled_frame_t``
# [T_post, K, D] (keys ``fc_keys``). Recipe (R1, one instrument): per object and quantity (velocity v; angular
# velocity w; position p) ONE calibrated ridge trained on the real pre- AND post-event token-frames (target: the
# labelled quantity averaged over the token-frame's master frames), cross-fitted by twin group; it decodes the real
# token-frames out of fold (floor) and the forecast token-frames (fold that held the clip's group out). Windows use
# only FREE token-frames (:func:`token_frame_targets` "ok", from the labels): every master frame of the token-frame
# at or after the release (``constants.release_time_s`` when it precedes the event: F7 drop balls and the F9 swing
# bob are held at rest before it), none after clean_until_s (e.g. a floor contact) and every scored object in frame
# (F7 projectiles enter from off screen). Pre
# window: the free real pre-event token-frames; post window: the free token-frames in event .. clean_until_s
# (friction: moving / stop windows), the forecast's token-frames for the forecast (the same free mask, by position), the real ones
# for the floor; each needs >= MIN_TOKEN_FRAMES. "true" = the same fits on the labelled targets. Per kind (:data:`KINDS`, :func:`token_states`):
#   gravity, rolling, inertia (F6 glide), angular_momentum (F12): least-squares lines v(t) (w(t)) over >= 3
#       token-frames per window: pre v (w) at the event and a_pre, post v and w at the post window's mean time, a_post
#       (ctx["dt"] = 0, so the law reads the fitted accelerations; F12 positions, masses and inertias: the labelled
#       ones at contact). Rolling and F12 read v and w with ONE instrument per object (targets [v, w] stacked: one
#       cell and penalty, per-output inverse calibration; M4);
#   bounce (F7): lines v(t) over the free pre token-frames (after the release) and the free post ones (up to the
#       second floor contact; the forecast up to the group's earliest one), both extrapolated to the contact:
#       eps = laws.epsilon(bounce), dv = |v_post(te) − v_pre(te)| (the gate);
#   friction (F6 slide / pair): lines v(t) over the real pre-event token-frames (a_pre, v at the branch) and over the
#       post token-frames while the block moves ([event, legal stop - stop_margin_s], earlier where the twin itself
#       stops: a_post); v_stop = the mean decoded velocity over the stop window [legal stop + stop_margin_s,
#       clean_until_s] (NaN with < 3 token-frames) -> decel_const, stop_hold, mass_ratio;
#   oscillation (F9 pair): periods from the decoded horizontal offsets from each pivot and their decoded rates
#       (io.zero_crossings): the last half period wholly before the event (real rows), the first wholly after it;
#   statics (F11): the rotation phi = |integral of w . tip_axis| over EXACTLY the tip window [event, event +
#       laws.incline.tip_window] (2 sqrt(a / g_clip), or until the block leaves the ledge; without g_clip the
#       twin-shared time to the group's clean-window end; trapezoid from the pre line's w at the event through the
#       post token-frames, w interpolated at the window's end; radians, time-scale free) and alpha = the slope of
#       w . tip_axis over the window's token-frames (stretched to the first 3 post token-frames when it holds fewer:
#       a slope needs them); g_clip = the DECODED free fall of the block after it leaves the ledge (line of v over the
#       free token-frames wholly after t_separate_s, >= 3; M4: alpha and g_clip then share any common gain), else the
#       labelled one (``g_clip_source``); the half-length from the labels;
#   swing (F9 swing) and pad (F8; law energy): eps = sqrt((h_post − z_b) / (h_pre − z_b)), h_pre / h_post the
#       highest decoded height in the pre / post window, z_b the lowest decoded height over both, each refined by a
#       parabola through its three token-frames (apex heights only, no g); the forecast's window runs to the group's
#       LATEST clean end (the highest rung's apex).

MIN_TOKEN_FRAMES = 3
KIN_DEFAULTS = {"stop_margin_s": 0.1}
RATIO_LAW = {"f6": "friction", "f6/glide": "inertia", "f7": "gravity", "f7/bounce": "bounce", "f8": "pad",
             "f9/pair": "oscillation", "f9/swing": "swing", "f10": "rolling", "f11": "statics",
             "f12": "angular_momentum"}   # every family but the collisions (F1-F4: pooled exchange-event pools)
KINDS = {   # kind: ({law: statistics scored (None: every one in law.legal)}, decoded quantities)
    "gravity": ({"gravity": None}, ("v",)),
    "rolling": ({"rolling": None}, ("v", "w")),
    "friction": ({"friction": None}, ("v",)),
    "inertia": ({"momentum": ("inertia_speed", "inertia_turn")}, ("v",)),
    "oscillation": ({"oscillation": None}, ("p", "v")),
    "statics": ({"statics": None}, ("w", "v")),
    "angular_momentum": ({"angular_momentum": None, "momentum": ("eta",)}, ("v", "w")),
    "swing": ({"energy": ("epsilon",)}, ("p",)),
    "pad": ({"energy": ("epsilon",)}, ("p",)),
    "bounce": ({"energy": ("epsilon",)}, ("v",)),
}
ENERGY_KINDS = ("swing", "pad", "bounce")      # per-token-frame kinds scored as energy (eps of the exchange event)
APEX_KINDS = ("swing", "pad")                  # eps = sqrt(h' / h) from decoded apex heights
JOINT_VW = ("rolling", "angular_momentum")     # velocity and spin read by ONE instrument per object (M4)


def ratio_kind(family: str, subfamily: str = "") -> str | None:
    """The per-token-frame kind (:data:`KINDS`) of a family / sub-family without a pooled exchange event."""
    return RATIO_LAW.get(f"{family}/{subfamily}", RATIO_LAW.get(family))


def _need_frames(where: str, what: str):
    raise RuntimeError(f"L1/L4 ratio laws need per-token-frame pools ({what}) in {where}; re-run stage 4 "
                       "(scripts/04_cache_activations.py --force on this model and family) to add them")


def token_frame_targets(labels: dict, frames_idx, spec, objects, clean: bool = True) -> tuple[np.ndarray, ...]:
    """(t [T_tok] mean master-frame time of each token-frame, V, W, P [T_tok, n_obj, 3], ok [T_tok]): the labelled
    velocity, angular velocity and position averaged over each token-frame's master frames, and whether the token-frame
    is FREE: all its master frames at or after the release (``constants.release_time_s``, when it precedes the event),
    at or before ``clean_until_s`` (a token-frame belongs to the clean window when its LAST master frame does: no
    floor contact averaged in; ``clean=False``: no clean-window cut) and with every object in frame (per-frame
    ``in_frame``)."""
    pf, fps = labels["per_frame"], float(labels["scene_constants"]["frame_rate"])
    pl = labels.get("physlaw") or {}
    t_rel = (pl.get("constants") or {}).get("release_time_s")
    t_rel = float(t_rel) if t_rel is not None and float(t_rel) < float(pl.get("event_time_s", -np.inf)) else -np.inf
    t_clean = float(pl.get("clean_until_s") or np.inf) if clean else np.inf
    frames_idx = [int(f) for f in frames_idx]
    T, V, W, P, OK = [], [], [], [], []
    z3 = [0.0, 0.0, 0.0]
    for idx in spec.token_frames(len(frames_idx)):
        fs = [min(frames_idx[i], len(pf) - 1) for i in idx]
        T.append(np.mean(fs) / fps)
        V.append(np.mean([[pf[f][o]["velocity"] for o in objects] for f in fs], 0))
        W.append(np.mean([[pf[f][o].get("angular_velocity", z3) for o in objects] for f in fs], 0))
        P.append(np.mean([[pf[f][o].get("position", z3) for o in objects] for f in fs], 0))
        OK.append(all(t_rel - 1e-9 <= f / fps <= t_clean + 1e-9 and all(pf[f][o].get("in_frame", True) for o in objects)
                      for f in fs))
    return np.array(T), np.array(V, float), np.array(W, float), np.array(P, float), np.array(OK, bool)


def load_token_frames(cache_dir, recs, clips, labels, model_cfg, seg, objects) -> tuple[dict, dict]:
    """(spaces {space: {"real": [X_i [T_i, D]], "fc": [F_i [T_fc, D]]}}, frames {"t", "phase", "V", "W", "P", "t_fc"}:
    per clip lists). Encoders: the forecast space (layer -1 of ``pooled_frame_t``; forecast tokens frame-pooled per
    token-frame). Diffusion: one space per fc key. Raises (re-run stage 4) when a cache lacks the per-frame pools."""
    from physlaw.models.base import token_spec
    spec = token_spec(model_cfg)
    zs = [np.load(Path(cache_dir) / r["file"]) for r in recs]
    if not all("pooled_frame_t" in z.files and "token_t_s" in z.files for z in zs):
        _need_frames(str(cache_dir), "pooled_frame_t + token_t_s in <twin>.npz")
    lk = [str(l) for l in zs[0]["layers"]]
    diffusion = "fc_pooled" in zs[0].files
    if diffusion and not all("fc_pooled_frame_t" in z.files for z in zs):
        _need_frames(str(cache_dir), "fc_pooled_frame_t [T_post, K, D] in the diffusion <twin>.npz; a run without "
                                     "--force or --no-forecast adds it")
    if not diffusion and "-1" not in lk:
        _need_frames(str(cache_dir), "the forecast space (layer -1) in pooled_frame_t")
    fr = {"t": [], "phase": [], "V": [], "W": [], "P": [], "ok": [], "ok_open": [], "t_fc": [], "ok_fc": [],
          "ok_open_fc": []}
    for c, z in zip(clips, zs):
        t, V, W, P, ok = token_frame_targets(labels[c], z["frames"], spec, objects)
        if len(t) != len(z["token_t_s"]):
            raise RuntimeError(f"{c}: {len(z['token_t_s'])} cached token-frames, the token grid gives {len(t)}")
        fr["t"].append(t); fr["V"].append(V); fr["W"].append(W); fr["P"].append(P); fr["ok"].append(ok)
        fr["ok_open"].append(token_frame_targets(labels[c], z["frames"], spec, objects, clean=False)[4])
        fr["phase"].append(np.array([str(x) for x in z["phases"]]))
    if diffusion:          # a forecast key is a space only when encode() cached the same (SNR, layer) (R1)
        fk = [str(x) for x in zs[0]["fc_keys"]]
        both = [q for q in fk if q in lk]
        if not both:
            raise RuntimeError(f"{cache_dir}: no forecast key {fk} is among the encode layers, so no real token-frame "
                               "rows share its space (R1); forecast_layers in configs/models must be a subset of layers")
        spaces = {q: {"real": [z["pooled_frame_t"][:, lk.index(q)].astype(np.float64) for z in zs],
                      "fc": [z["fc_pooled_frame_t"][:, fk.index(q)].astype(np.float64) for z in zs]} for q in both}
        for i, z in enumerate(zs):
            kt = int(z["context_tokens"])
            fr["t_fc"].append(fr["t"][i][kt: kt + len(z["fc_pooled_frame_t"])])
            fr["ok_fc"].append(fr["ok"][i][kt: kt + len(z["fc_pooled_frame_t"])])
            fr["ok_open_fc"].append(fr["ok_open"][i][kt: kt + len(z["fc_pooled_frame_t"])])
        return spaces, fr
    real, fc, key, memo = [], [], forecast_segments(seg), {}
    for i, (c, z) in enumerate(zip(clips, zs)):
        real.append(z["pooled_frame_t"][:, lk.index("-1")].astype(np.float64))
        f = forecast_path(cache_dir, c.split("/")[0], key[c])
        if f not in memo:
            ff = np.load(f)
            F = ff["final"].astype(np.float64)
            memo = {f: (F.reshape(len(F), -1, F.shape[-1]).mean(1), int(ff["context_frames"]) // spec.tubelet)}
        Fm, t0 = memo[f]
        fc.append(Fm)
        fr["t_fc"].append(fr["t"][i][t0: t0 + len(Fm)])
        fr["ok_fc"].append(fr["ok"][i][t0: t0 + len(Fm)])
        fr["ok_open_fc"].append(fr["ok_open"][i][t0: t0 + len(Fm)])
    return {"final": {"real": real, "fc": fc}}, fr


def token_instrument(X: list, Y: list, phase: list, groups, k=5, n_seeds=4, alphas=DEFAULT_ALPHAS, ids=None) -> dict:
    """The per-token-frame R1 instrument of one (object, quantity): fitted on the real pre and post token-frames of
    every clip (X [T_i, D], Y [T_i, d], phase [T_i]), cross-fitted by group. Returns {"folds", "var", "mean", "r2",
    "real": [decoded [T_i, d] out of fold]} (+ "selectivity" {scene, object} with ``ids`` [n] object classes,
    partialled within (twin slot in the group, token-frame index): the structure every group shares at a position,
    which the scene control keeps); constant target columns pass through at their mean."""
    groups = np.asarray(groups)
    keep = [np.isin(p, ("pre", "post")) for p in phase]
    Xr = np.concatenate([x[m] for x, m in zip(X, keep)])
    Yr = np.concatenate([y[m] for y, m in zip(Y, keep)])
    gr = np.concatenate([np.full(int(m.sum()), g) for g, m in zip(groups, keep)])
    var, mean = np.ptp(Yr, 0) > 1e-12, Yr.mean(0)
    folds = instrument_folds({"x": Xr}, Yr[:, var], gr, None, k, n_seeds, alphas) if var.any() else []
    inst = {"folds": folds, "var": var, "mean": mean}
    inst["real"] = [token_decode(inst, x, g) for x, g in zip(X, groups)]
    D = np.concatenate([d[m] for d, m in zip(inst["real"], keep)])
    inst["r2"] = r2_score(Yr[:, var], D[:, var]) if var.any() else NAN
    if ids is not None and var.any():      # controls partialled within (twin slot, token-frame index): ridge.partial_strata
        idr = np.concatenate([np.full(int(m.sum()), i, dtype=object) for i, m in zip(ids, keep)])
        slot = [int((groups[:i] == g).sum()) for i, g in enumerate(groups)]
        T = max(len(m) for m in keep)
        st = np.concatenate([s_ * T + np.arange(len(m))[m] for s_, m in zip(slot, keep)])
        inst["selectivity"] = instrument_selectivity({"x": Xr}, Yr[:, var], gr, idr, k, 1, alphas, strata=st)
    return inst


def token_decode(inst: dict, X: np.ndarray, group, transform=None) -> np.ndarray:
    """Decode token-frame rows X [T, D] of one clip of ``group`` (any space the instrument reads)."""
    out = np.tile(inst["mean"], (len(X), 1))
    if inst["var"].any() and len(X):
        out[:, inst["var"]] = apply_instrument(inst["folds"], X, np.full(len(X), group), transform)
    return out


def fit_line(t, V, sel, t_eval) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line V(t) over the rows ``sel`` (>= MIN_TOKEN_FRAMES): (V at t_eval, slope). NaN otherwise."""
    t, V = np.asarray(t, float), np.asarray(V, float).reshape(len(t), -1)
    sel = np.asarray(sel, bool) & np.isfinite(V).all(1)
    if sel.sum() < MIN_TOKEN_FRAMES or np.ptp(t[sel]) <= 0:
        return np.full(V.shape[1], NAN), np.full(V.shape[1], NAN)
    A = np.stack([np.ones(sel.sum()), t[sel] - t_eval], 1)
    c = np.linalg.lstsq(A, V[sel], rcond=None)[0]
    return c[0], c[1]


def _windows(t, phase, t_q, te, t_end, ok=None, phase_q=None, ok_q=None):
    """Boolean rows of the pre window (real token-frames: phase "pre" before te, free: ``ok`` [T], default all) and of
    the post window (t_q in [te, t_end], free: ``ok_q`` [T_q], default all; with ``phase_q`` only its "post" rows)."""
    t, t_q = np.asarray(t, float), np.asarray(t_q, float)
    s_pre = (np.asarray(phase) == "pre") & (t < te) & (True if ok is None else np.asarray(ok, bool))
    s_post = (t_q >= te) & (t_q <= t_end + 1e-9) & (True if ok_q is None else np.asarray(ok_q, bool))
    if phase_q is not None:
        s_post &= np.asarray(phase_q) == "post"
    return s_pre, s_post


def clip_kinematics(t, V, phase, t_q, V_q, W_q, te, t_end, ok=None, phase_q=None, W=None, ok_q=None,
                    post_at_event: bool = False) -> dict | None:
    """Pre / post kinematic states of one clip. Pre: the real token-frames (t [T], V [T, n_obj, d] (+ W angular, the
    pre spins), phase [T]) of the pre window, the line extrapolated to the event te; post: (t_q, V_q, W_q angular or
    None) over the post window, the forecast's token-frames or, for the floor / truth, the real ones (``phase_q``:
    only "post" rows), the line read at the post window's mean time (``post_at_event``: extrapolated back to te,
    the velocity just after the event). None when a window has < MIN_TOKEN_FRAMES. ``ok`` / ``ok_q``: the free
    token-frames of each row set (:func:`_windows`)."""
    s_pre, s_post = _windows(t, phase, t_q, te, t_end, ok, phase_q, ok_q)
    if s_pre.sum() < MIN_TOKEN_FRAMES or s_post.sum() < MIN_TOKEN_FRAMES:
        return None
    tc = float(te) if post_at_event else float(np.asarray(t_q, float)[s_post].mean())
    n = V.shape[1]
    pre = [fit_line(t, V[:, o], s_pre, te) for o in range(n)]
    post = [fit_line(t_q, V_q[:, o], s_post, tc) for o in range(n)]
    out = {"pre": {"v": np.stack([p[0] for p in pre]), "a": np.stack([p[1] for p in pre])},
           "post": {"v": np.stack([p[0] for p in post]), "a": np.stack([p[1] for p in post])}}
    if W is not None:
        out["pre"]["w"] = np.stack([fit_line(t, W[:, o], s_pre, te)[0] for o in range(n)])
    if W_q is not None:
        out["post"]["w"] = np.stack([fit_line(t_q, W_q[:, o], s_post, tc)[0] for o in range(n)])
    return out


def _peak(t, y) -> float:
    """The maximum of samples y(t) (t increasing), refined by the parabola through the highest sample and its two
    neighbours (an apex / the bottom of a swing falls between token-frames)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    i = int(np.argmax(y))
    if 0 < i < len(y) - 1:
        c = np.polyfit(t[i - 1: i + 2] - t[i], y[i - 1: i + 2], 2)
        if c[0] < 0:
            return float(c[2] - c[1] ** 2 / (4 * c[0]))
    return float(y[i])


def _half_periods(t, x, v, te, before: bool) -> float:
    """Twice the last (before) / first (after) interval between successive zero crossings of x(t) (rates v) lying
    wholly before / after te; NaN without one."""
    from physlaw.data.io import zero_crossings
    ok = np.isfinite(x) & np.isfinite(v)
    c = zero_crossings(np.asarray(t, float)[ok], np.asarray(x, float)[ok], np.asarray(v, float)[ok])
    iv = [(a, b) for a, b in zip(c[:-1], c[1:]) if (b <= te + 1e-9 if before else a >= te - 1e-9)]
    return 2.0 * float(iv[-1][1] - iv[-1][0] if before else iv[0][1] - iv[0][0]) if iv else NAN


def token_states(kind, ctx, t, Q, phase, t_q, Qq, te, t_end, ok=None, phase_q=None, ok_q=None) -> dict | None:
    """Law inputs of one clip of ``kind`` (:data:`KINDS`) from decoded per-token-frame quantities (section comment).
    Q: {quantity: [T, n_obj, 3]} on the real rows (t, phase); Qq: the post source's rows (t_q; ``phase_q`` given:
    only its "post" rows). Returns {"pre", "post"} law states (friction also "dt" = the post window's mean time after
    the event, post["v_stop"]; swing
    {"eps", "rise"}), or None when a window is too short."""
    if kind in ("gravity", "rolling", "inertia", "angular_momentum"):
        return clip_kinematics(t, Q["v"], phase, t_q, Qq["v"], Qq.get("w"), te, t_end, ok, phase_q, Q.get("w"), ok_q)
    if kind == "bounce":                    # H3: velocities extrapolated to the contact from each side (ballistic)
        out = clip_kinematics(t, Q["v"], phase, t_q, Qq["v"], None, te, t_end, ok, phase_q, None, ok_q,
                              post_at_event=True)
        if out is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            out["eps"] = float(laws.epsilon(out["pre"], out["post"], {**ctx, "event": "bounce"}))
        out["dv"] = float(np.linalg.norm(out["post"]["v"] - out["pre"]["v"]))
        return out
    t, t_q = np.asarray(t, float), np.asarray(t_q, float)
    s_pre, s_post = _windows(t, phase, t_q, te, t_end, ok, phase_q, ok_q)
    if kind == "friction":                  # pre: before the branch; post: moving; v_stop: over the stop window
        m_ = float(ctx.get("stop_margin_s", KIN_DEFAULTS["stop_margin_s"]))
        t_stop = float(ctx.get("t_stop_s", np.inf))
        move_end = min(t_end, min(t_stop, float(ctx.get("t_stop_twin_s", np.inf))) - m_)
        out = clip_kinematics(t, Q["v"], phase, t_q, Qq["v"], None, te, move_end, ok, phase_q, None, ok_q)
        if out is None:
            return None
        _, s_mv = _windows(t, phase, t_q, te, move_end, ok, phase_q, ok_q)
        _, s_st = _windows(t, phase, t_q, t_stop + m_, t_end, ok, phase_q, ok_q)
        V = np.asarray(Qq["v"], float)
        out["post"]["v_stop"] = (V[s_st].mean(0) if s_st.sum() >= MIN_TOKEN_FRAMES
                                 else np.full(V.shape[1:], NAN))
        out["dt"] = float(t_q[s_mv].mean() - te)
        return out
    if kind == "oscillation":               # periods from the horizontal offsets from each pivot
        piv = ctx["pivots"]
        objs = list(ctx.get("objects") or sorted(piv))
        Tb, Ta = [], []
        for o, name in enumerate(objs):
            x0 = float(piv[name][0])
            Tb.append(_half_periods(t[s_pre], Q["p"][s_pre, o, 0] - x0, Q["v"][s_pre, o, 0], te, True))
            Ta.append(_half_periods(t_q[s_post], Qq["p"][s_post, o, 0] - x0, Qq["v"][s_post, o, 0], te, False))
        if not (np.isfinite(Tb).any() and np.isfinite(Ta).any()):
            return None
        z = np.zeros((len(objs), 3))
        return {"pre": {"T": np.array(Tb), "v": z}, "post": {"T": np.array(Ta), "v": z.copy()}}
    if kind == "statics":                   # rotation over the tip window and the angular acceleration
        from physlaw.eval.laws.incline import tip_window
        ax = np.asarray(ctx.get("tip_axis", [0.0, 1.0, 0.0]), float)
        ax = ax / np.linalg.norm(ax)
        t_sep = ctx.get("t_separate_s")
        w_tip = tip_window(ctx.get("g_clip"), ctx.get("half_length", NAN), None if t_sep is None else t_sep - te,
                           float(ctx.get("group_clean_end_s", t_end)) - te)
        r = np.where(s_post)[0]
        r = r[np.argsort(t_q[r])]
        if not np.isfinite(w_tip) or len(r) < MIN_TOKEN_FRAMES or s_pre.sum() < MIN_TOKEN_FRAMES:
            return None
        w0 = float(fit_line(t, Q["w"][:, 0] @ ax, s_pre, te)[0][0])
        wq = Qq["w"][:, 0] @ ax
        # phi over EXACTLY [te, te + w_tip] (time-scale invariant): trapezoid of w(t) from the pre line's value at te
        # through the post token-frames, w linearly interpolated at te + w_tip; a window ending after the last free
        # post token-frame is cut there (phi then a lower bound)
        tt, ww = np.r_[te, t_q[r]], np.r_[w0, wq[r]]
        t1 = min(te + w_tip, float(tt[-1]))
        keep = tt < t1
        tt, ww = np.r_[tt[keep], t1], np.r_[ww[keep], np.interp(t1, tt, ww)]
        phi = abs(0.5 * float(((ww[1:] + ww[:-1]) * np.diff(tt)).sum()))
        # alpha (the INITIAL angular acceleration): slope of w over the tip window's token-frames, stretched to the
        # first MIN_TOKEN_FRAMES post token-frames when the window holds fewer (a slope needs them; alpha grows as
        # the block turns, so a stretched window reads it slightly high)
        ra = r[: max(int((t_q[r] <= te + w_tip + 1e-9).sum()), MIN_TOKEN_FRAMES)]
        sel = np.zeros(len(t_q), bool)
        sel[ra] = True
        alpha = float(fit_line(t_q, wq, sel, te)[1][0])
        z = np.zeros((Qq["w"].shape[1], 3))
        out = {"pre": {"v": z}, "post": {"v": z.copy(), "phi": np.array([phi]), "alpha": alpha * ax[None]},
               "n_tip_frames": int((t_q[r] <= te + w_tip + 1e-9).sum())}
        if "v" in Q and t_sep is not None:   # M4: g_clip from the DECODED free fall after the block leaves the ledge
            ts = float(t_sep) + 0.5 * float(np.median(np.diff(t))) if len(t) > 1 else float(t_sep)
            sf = np.asarray(ok if ok is not None else np.ones(len(t), bool), bool) & (t >= ts) & (t <= t_end + 1e-9)
            a_f = fit_line(t, Q["v"][:, 0], sf, ts)[1]
            if np.isfinite(a_f).all():
                out["g_clip"] = float(np.linalg.norm(a_f))
        return out
    if kind in ("swing", "pad"):            # apex heights above the lowest decoded point
        g = np.asarray(ctx.get("g", [0.0, 0.0, -1.0]), float)
        up = -g / np.linalg.norm(g)
        h, hq = Q["p"][:, 0] @ up, Qq["p"][:, 0] @ up
        if s_pre.sum() < MIN_TOKEN_FRAMES or s_post.sum() < MIN_TOKEN_FRAMES:
            return None
        hb = np.r_[h[s_pre], hq[s_post]]
        tb = np.r_[t[s_pre], t_q[s_post]]
        o = np.argsort(tb)
        zb = -_peak(tb[o], -hb[o])
        a, b = _peak(t[s_pre], h[s_pre]) - zb, _peak(t_q[s_post], hq[s_post]) - zb
        return {"eps": float(np.sqrt(b / a)) if a > 0 and b >= 0 else NAN, "rise": b}
    raise ValueError(f"token_states: kind {kind!r} not in {sorted(KINDS)}")


def null_stats(law, pre: dict, ctx: dict, dt: float | None = None) -> dict:
    """The law's null statistics, the null outcome given as kinematics (a = its velocity change / dt; dt the
    kinematics' own gap (friction) else ctx["dt"], 1 where the clip has none). gravity mass_ratio has no kinematic null
    (0 / 0): the Aristotelian a ~ m (a_heavy / a_light = m_heavy / m_light) stands in."""
    dt = float(dt or ctx.get("dt") or 0.0) or 1.0
    post = law.null_outcome(pre, {**ctx, "dt": dt})
    nv = np.asarray(post["v"], float)
    kin = {**post, "v": nv, "a": np.asarray(post.get("a", (nv - np.asarray(pre["v"], float)) / dt), float),
           "w": np.asarray(post.get("w", np.zeros_like(nv)), float)}
    st = law.stats(pre, kin, {**ctx, "dt": 0.0})
    if law.NAME == "gravity" and "m" in ctx:
        m = np.asarray(ctx["m"], float).reshape(-1)
        st["mass_ratio"] = float(m.max() / m.min()) if len(m) > 1 and m.max() > m.min() else NAN
    return st


def ratio_package(kind, states, clips, kin: dict, n_boot, rng) -> dict:
    """Every statistic of the kind's laws (:data:`KINDS`) on the legal clips with kinematics, per source (``kin``:
    {source: [per clip {"pre", "post"} states or None]}; forecast, floor, true). Positions the decoded states lack
    (F12's centres at contact) are the clip's labelled ones. Statistics in UNIT_STATS take the unit form 1 − |T −
    T_legal| (their null is undefined or 0 / 1), tip_consistency its accuracy, alpha_ratio the labelled clip's own
    windowed value as the legal one."""
    main = [s for s in kin if s not in CONTROLS]
    leg = [i for i, c in enumerate(clips) if states[c][3]["legal"] and all(kin[s][i] is not None for s in main)]
    g = [clips[i].split("/")[0] for i in leg]
    ctx = lambda i: {**states[clips[i]][2], "dt": 0.0, "objects": states[clips[i]][3]["objects"]}
    with_p = lambda i, st: ({"p": states[clips[i]][0]["p"], **st} if "p" in states[clips[i]][0] else st)
    # statics: g_clip decoded from the clip's own free fall where token_states measured one (M4), else labelled
    ctx_of = lambda i, k_: {**ctx(i), **({"g_clip": k_["g_clip"]} if "g_clip" in k_ else {})}
    out = {}
    for law_name, names in KINDS[kind][0].items():
        law = laws.LAWS[law_name]
        T = {s: [law.stats(with_p(i, kin[s][i]["pre"]), with_p(i, kin[s][i]["post"]), ctx_of(i, kin[s][i]))
                 if kin[s][i] is not None else None for i in leg] for s in kin}   # no window: every statistic undefined
        null = [null_stats(law, {**states[clips[i]][0], **kin["true"][i]["pre"]}, ctx(i), kin["true"][i].get("dt"))
                for i in leg]
        for stat in names or law.legal:
            legal = ([t[stat] for t in T["true"]] if stat in LEGAL_FROM_TRUE else law.legal[stat][1])
            tn = [NAN] * len(leg) if stat in UNIT_STATS else [t[stat] for t in null]
            r = ratio_scores({s: [NAN if t is None else t[stat] for t in T[s]] for s in kin}, legal, tn, g, n_boot,
                             rng, "mean" if stat in MEAN_STATS else "median",
                             defined=np.isfinite([t[stat] for t in T["true"]]))
            if "forecast" in r:
                r["verdict"] = verdict(r["forecast"]["S"], {c: r[c]["S"] for c in CONTROLS if c in r},
                                       diff_ci=r.get("paired_ci95"))
            out[stat] = r
    return out


def _last_context_rows(X: np.ndarray, t, t_fc) -> np.ndarray:
    """H4 control (i): the last context token-frame's features (the last real row before the first forecast
    token-frame) repeated over the forecast's token-frames."""
    t, t_fc = np.asarray(t, float), np.asarray(t_fc, float)
    if not len(t_fc):
        return np.zeros((0, X.shape[1]))
    i = int(np.searchsorted(t, t_fc[0] - 1e-9)) - 1
    return np.repeat(X[max(i, 0)][None], len(t_fc), 0)


def kinematics_variants(kind, states, clips, spaces, fr, groups, kcfg, k, n_seeds, alphas, n_boot, rng,
                        ids=None, ecfg=None) -> dict:
    """{space: L1 result} of one per-token-frame kind (see the section comment), with the H4 negative controls
    (:data:`CONTROLS`): (i) ctl_static = the last context token-frame's features repeated over the forecast's
    token-frames, (ii) ctl_other_forecast = another set's forecast token-frames (same twin type), (iii) ctl_other_real
    = another set's REAL token-frames from its own forecast start on; all decoded by the identical instrument (the
    fold that held the features' own set out) and read through this clip's windows."""
    objs = states[clips[0]][3]["objects"]
    qs = KINDS[kind][1]
    n = len(clips)
    ctxs = [{**states[c][2], "objects": objs} for c in clips]
    te = [float(c_.get("event_time_s", 0.0)) for c_ in ctxs]
    tend = [float(c_.get("clean_until_s") or np.inf) for c_ in ctxs]   # friction: its windows (token_states)
    for c_ in ctxs:
        c_["stop_margin_s"] = float(kcfg["stop_margin_s"])
    for c_, g in zip(ctxs, groups):          # the group's clean-window end (min over its twins; statics fallback)
        c_["group_clean_end_s"] = min(float(x.get("clean_until_s") or np.inf) for x, h in zip(ctxs, groups) if h == g)
    true = {q: fr[q.upper()] for q in qs}
    ok = fr.get("ok") or [None] * n
    ok_fc = fr.get("ok_fc") or [None] * n
    tend_fc = tend
    # one forecast per group, shared by every rung of a ladder: apex kinds read it up to the group's LATEST clean end
    # (the highest rung's apex: a forecast's apex after its twin's own is still seen; past its apex it only falls
    # back); the bounce up to the group's EARLIEST clean end (the lowest rung's second contact: a forecast's own
    # second contact must not enter the line fit)
    if kind in APEX_KINDS + ("bounce",):
        pick = max if kind in APEX_KINDS else min
        gx = {g: pick(tend[i] for i in range(n) if groups[i] == g) for g in set(groups)}
        tend_fc = [gx[g] for g in groups]
        ok_fc = fr.get("ok_open_fc") or ok_fc            # free token-frames without the twin's own clean cut
    kinem = lambda i, Q, tq, D, ph, okq, t_end=None: token_states(
        kind, ctxs[i], fr["t"][i], {q: Q[q][i] for q in qs}, fr["phase"][i], tq, {q: D[q][i] for q in qs}, te[i],
        tend[i] if t_end is None else t_end, ok[i], ph, okq)
    kin_true = [kinem(i, true, fr["t"][i], true, fr["phase"][i], ok[i]) for i in range(n)]
    part = control_partners(clips)
    start = [int(np.searchsorted(np.asarray(fr["t"][j], float), float(fr["t_fc"][j][0]) - 1e-9))
             if len(fr["t_fc"][j]) else 0 for j in range(n)]
    blocks = [("vw", ("v", "w"))] if kind in JOINT_VW else [(q, (q,)) for q in qs]
    out = {}
    for space, S_ in spaces.items():
        dec = {s_: {} for s_ in ("real", "fc", "ctl_static")}
        r2, sel = {}, {}
        for name, qq in blocks:
            per = {s_: [] for s_ in dec}
            for oi, o in enumerate(objs):
                Y = [np.concatenate([fr[q.upper()][i][:, oi] for q in qq], -1) for i in range(n)]
                inst = token_instrument(S_["real"], Y, fr["phase"], groups, k, n_seeds, alphas,
                                        None if ids is None else ids[o])
                r2.setdefault(o, {})[name] = inst["r2"]
                if "selectivity" in inst:
                    sel.setdefault(o, {})[name] = inst["selectivity"]
                per["real"].append(inst["real"])
                per["fc"].append([token_decode(inst, S_["fc"][i], groups[i]) for i in range(n)])
                per["ctl_static"].append([token_decode(inst, _last_context_rows(S_["real"][i], fr["t"][i],
                                                                                fr["t_fc"][i]), groups[i])
                                          for i in range(n)])
            for qi, q in enumerate(qq):       # split a joint (v, w) instrument's columns
                for s_ in dec:
                    dec[s_][q] = [np.stack([d[i][:, 3 * qi: 3 * qi + 3] for d in per[s_]], 1) for i in range(n)]
        real, fc = dec["real"], dec["fc"]
        kin = {"true": kin_true,
               "floor": [kinem(i, real, fr["t"][i], real, fr["phase"][i], ok[i]) for i in range(n)],
               "forecast": [kinem(i, real, fr["t_fc"][i], fc, None, ok_fc[i], tend_fc[i]) for i in range(n)],
               "ctl_static": [kinem(i, real, fr["t_fc"][i], dec["ctl_static"], None, ok_fc[i], tend_fc[i])
                              for i in range(n)]}
        for name, src in (("ctl_other_forecast", fc), ("ctl_other_real", real)):
            ks = []
            for i in range(n):
                j = part[i]
                if j < 0:
                    ks.append(None)
                    continue
                s0 = 0 if src is fc else start[j]
                L = min(len(fr["t_fc"][i]), len(src[qs[0]][j]) - s0)
                D = {q: {i: src[q][j][s0: s0 + L]} for q in qs}
                okq = None if ok_fc[i] is None else ok_fc[i][:L]
                ks.append(kinem(i, real, fr["t_fc"][i][:L], D, None, okq, tend_fc[i]) if L > 0 else None)
            kin[name] = ks
        res = {"law": next(iter(KINDS[kind][0])), "instrument": "token_frames", "probe_r2": r2, "selectivity": sel,
               "probe_r2_mean": float(np.nanmean([x for d_ in r2.values() for x in d_.values()])),
               "controls": CONTROL_DOC,
               "n_clips": {"legal": int(sum(states[c][3]["legal"] for c in clips)),
                           **{f"windows_{s}": int(sum(x is not None for x in v)) for s, v in kin.items()}}}
        if kind == "statics":
            res["g_clip_source"] = {s: int(sum(x is not None and "g_clip" in x for x in v)) for s, v in kin.items()}
        if kind in ENERGY_KINDS:
            res["energy"] = _event_energy(kind, states, clips, kin, ecfg or ENERGY_DEFAULTS, n_boot, rng)
        else:
            res["ratio"] = ratio_package(kind, states, clips, kin, n_boot, rng)
        out[space] = res
    return out


def _event_energy(kind, states, clips, kin, ecfg, n_boot, rng) -> dict:
    """Energy package of a per-token-frame exchange event: the F9 swing and the F8 pad from decoded apex heights
    (gate quantity: the rise after the event), the F7 bounce from velocities extrapolated to the contact (gate
    quantity: |dv| = |v_post(te) − v_pre(te)|, a vector, so a bounce at eps = 1 still moves). Gates: the floor's
    quantity >= kappa x the labelled one (the floor sees the event), every other source's >= kappa x the floor's (a
    static forecast neither rises nor changes velocity). The H4 controls are scored as observers."""
    get = lambda s, key: np.array([kin[s][i][key] if kin[s][i] is not None else NAN for i in range(len(clips))])
    kap = float(ecfg["kappa"])
    key = "dv" if kind == "bounce" else "rise"
    m = {s: get(s, key) for s in kin}
    with np.errstate(invalid="ignore"):
        gate_fl = m["floor"] >= kap * m["true"]
        gates = {s: m[s] >= kap * m["floor"] for s in kin if s not in ("true", "floor")}
    obs = {c: (get(c, "eps"), gates[c]) for c in CONTROLS if c in kin}
    r = energy_scores([states[c][3]["eps"] for c in clips], get("floor", "eps"), get("forecast", "eps"), gate_fl,
                      gates["forecast"], [c.split("/")[0] for c in clips], obs, float(ecfg["tau_quantile"]),
                      float(ecfg["slope_r2_min"]), n_boot, rng)
    r["estimator"] = ("velocities extrapolated to the contact by line fits over the free token-frames on each side"
                      if kind == "bounce" else "apex heights above the lowest decoded point, sqrt(h' / h)")
    return r


# --------------------------------------------------------------------------- run

def run(cache_dir: Path, data_dir: Path, cfg, model_cfg, log=print) -> dict:
    probe = cfg["probe"]
    alphas = tuple(float(a) for a in probe.get("alphas", DEFAULT_ALPHAS))
    k, n_seeds = int(probe["k_folds"]), int(cfg.get("fold_seeds_l1", 4))
    n_boot = int(cfg["bootstrap"]["n"])
    n_splits = int(cfg.get("selection", {}).get("n_splits", 50))
    mcfg = {**MOM_DEFAULTS, **dict(cfg.get("momentum", None) or {})}
    ecfg = {**ENERGY_DEFAULTS, **dict(cfg.get("energy", None) or {})}
    kcfg = {**KIN_DEFAULTS, **dict(cfg.get("kinematics", None) or {})}
    recs = complete_groups(load_index(cache_dir), data_dir)
    all_clips = [r["clip"] for r in recs]
    labels = {c: json.loads((Path(data_dir) / c / "labels.json").read_text()) for c in all_clips}
    states = {c: clip_states(labels[c]) for c in all_clips}
    mf = np.load(cache_dir / "_maskfeat_v2.npz")
    if list(mf["clips"]) != all_clips:
        raise RuntimeError("_maskfeat_v2.npz is for another clip list: run L0 first (it builds the weights)")
    kinds = sorted({states[c][3]["law"] for c in all_clips if states[c][3]["law"]})
    spaces = (_load_spaces(cache_dir, recs, all_clips, model_cfg, mf, {c: clip_segment(labels[c]) for c in all_clips})
              if set(kinds) - set(KINDS) else {})
    out = {"n_groups": len({r["group"] for r in recs}), "n_clips": len(all_clips), "laws": kinds,
           "momentum_cfg": mcfg, "energy_cfg": ecfg, "energy_note": R5_NOTE,
           "protocol": {"k_folds": k, "fold_seeds": n_seeds, "n_spaces": len(spaces), "selection_splits": n_splits,
                        "bootstrap": n_boot, "probe": "R1: one calibrated ridge per (space, pool, object) on real pre+post "
                                                      "rows (ratio laws and the F9 swing: per (object, quantity) on real "
                                                      "per-token-frame rows); the forecast read as is"},
           "variants": {}, "selected": {}}
    rng = np.random.default_rng(0)
    for kind in kinds:                    # e.g. F7: bounce groups (energy) and projectile / drop groups (gravity)
        clips = [c for c in all_clips if states[c][3]["law"] == kind]
        ix = np.array([all_clips.index(c) for c in clips])
        objs = states[clips[0]][3]["objects"]
        groups = np.array([c.split("/")[0] for c in clips])
        collision = states[clips[0]][3]["event"] == "collision"
        ids = {o: object_class_ids([labels[c] for c in clips], o) for o in objs}
        if kind in KINDS:          # ratio laws (and the F9 swing): kinematics from per-token-frame pools
            sp, fr = load_token_frames(cache_dir, [recs[i] for i in ix], clips, labels, model_cfg,
                                       {c: clip_segment(labels[c]) for c in clips}, objs)
            for space, res in kinematics_variants(kind, states, clips, sp, fr, groups, kcfg, k, n_seeds, alphas,
                                                  n_boot, rng, ids, ecfg).items():
                out["variants"][f"{kind}|{space}|frame"] = res
                log(f"[L1] {kind}|{space}|frame: " + _summary(res))
            continue
        for space, S_ in spaces.items():
            for pname, pool_of in POOLS.items():
                pp = {o: (f"{pool_of(o)}.pre", f"{pool_of(o)}.post") for o in objs}
                if not all(a in S_["real"] and b in S_["real"] and b in S_["fc"] for a, b in pp.values()):
                    continue
                pre, dec, r2, instrument = _decode_pool(S_, mf, pp, states, clips, ix, groups,
                                                        pname in MASK_FLOOR_POOLS, k, n_seeds, alphas, ids)
                pooled_controls(pre, dec, clips)
                res = {"law": kind, "instrument": instrument, "probe_r2": r2,
                       "probe_r2_mean": float(np.mean([d_[ph] for d_ in r2.values() for ph in ("pre", "post")])),
                       "controls": CONTROL_DOC}
                if kind == "energy":     # the pooled route holds the collisions (F1-F4) only
                    if not collision:
                        raise RuntimeError(f"L1: pooled energy route got event {states[clips[0]][3]['event']}")
                    if states[clips[0]][3]["family"] in MASS_HIDDEN_FAMILIES:
                        res["momentum"] = {"status": "not scored", "note": F3_NOTE}
                    else:
                        res["momentum"] = momentum_package(states, clips, dec, n_boot, rng, mcfg, pre)
                    gates = collision_gates(states, clips, dec, mcfg, pre)
                    res["energy"] = energy_package(states, clips, pre, dec, gates, ecfg, n_boot, rng)
                out["variants"][f"{kind}|{space}|{pname}"] = res
                log(f"[L1] {kind}|{space}|{pname}: " + _summary(res))
    _select(out, states, mcfg, n_splits)
    return out


def _summary(res) -> str:
    s = [f"probe R² {res['probe_r2_mean']:.3f}"]
    if "momentum" in res and "S" in res["momentum"]:
        m = res["momentum"]
        s.append(f"S_mom {_fmt_s(m['S'])} (interaction {m['interaction_rate']:.2f}; {m['verdict']['verdict']})")
    if "energy" in res and "forecast" in res["energy"]:
        e = res["energy"]
        s.append(f"energy {e['status']}: creation fc {e['forecast']['creation_rate']:.2f} / floor "
                 f"{e['floor']['creation_rate']:.2f}")
    for st, r in res.get("ratio", {}).items():
        s.append(f"S_{st} fc {_fmt_s(r['forecast']['S'])} / floor {_fmt_s(r['floor']['S'])}")
    return ", ".join(s)


def _select(out, states, mcfg, n_splits) -> None:
    """Per (law, pool): S_mom held-out over spaces; energy / ratio laws: the space with the best FLOOR instrument."""
    keys = {}
    for v in out["variants"]:
        kind, space, pname = v.split("|")
        keys.setdefault((kind, pname), []).append(v)
    for (kind, pname), vs in keys.items():
        sel = {}
        if "S" in out["variants"][vs[0]].get("momentum", {}):
            def f_mom(v, gids):
                G = np.array([out["variants"][v]["momentum"]["group_sums"].get(g, [0.0] * len(_MOM_KEYS)) for g in gids])
                return float(_eta_S(G.sum(0))[1])
            ug = sorted({c.split("/")[0] for c in states})
            sel["momentum"] = heldout_select(f_mom, vs, ug, n_splits=n_splits)
        inst = lambda v: (out["variants"][v]["energy"].get("floor_check", {}).get("r2", NAN)
                          if "energy" in out["variants"][v] else out["variants"][v]["probe_r2_mean"])
        sc = np.array([inst(v) for v in vs], float)
        sel["floor_best"] = vs[int(np.nanargmax(sc))] if np.isfinite(sc).any() else vs[0]
        out["selected"][f"{kind}|{pname}"] = sel


# --------------------------------------------------------------------------- markdown

def _fmt_s(x: float) -> str:
    return f"{x:+.2f}" if np.isfinite(x) else "undefined"


def _ci(c) -> str:
    return f"[{c[0]:.2f}, {c[1]:.2f}]" if np.all(np.isfinite(c)) else "[n/a]"


def _ctl(d: dict, key: str, fmt=_fmt_s) -> str:
    """'static / other fc / other real' of the H4 controls' ``key`` in d ({source: {key: x}})."""
    return " / ".join(fmt(d[c][key]) if c in d and np.isfinite(d[c].get(key, NAN)) else "—" for c in CONTROLS)


def _vd(v: dict | None) -> str:
    return "" if not v else ("✓ " if v["claim_allowed"] else "✗ ") + v["verdict"]


def table(res: dict) -> str:
    pr = res["protocol"]
    rows = [f"n_groups = {res['n_groups']}, {pr['k_folds']}-fold x {pr['fold_seeds']} seeds cross-fitting; spaces "
            f"searched = {pr['n_spaces']}; bootstrap {pr['bootstrap']} over groups. Probe: {pr['probe']}. "
            "Primary numbers: momentum S_mom = 1 − (η_fc − η_floor) / (η_blind − η_floor) (1 = as conserved as the probe "
            "can tell, 0 = the mass-blind rule; pre AND post velocities decoded by the same instrument); energy creation "
            "rate P(ε̂ > 1 + τ) on gated legal clips (F7 bounce: velocities extrapolated to the contact; F8 pad, F9 "
            "swing: ε̂ from decoded apex heights); ratio laws S = 1 − |log T_fc − log T_legal| / |log T_null − log "
            "T_legal| (module docstring for the linear / unit forms; S clipped to [−1, 1], an undefined forecast "
            "statistic scores −1). Every number beside its floor (the same probe on the real clips). Instrument per "
            "variant: pred_real (the model's forecast-space outputs for masked real post frames; primary where cached) "
            "or as_is (the real post pool of the same layer); the forecast is read as is, fc-mm = moment-matched "
            "(diagnostic only). roi pools sit at the TRUE post location, so an roi score must beat the mask-only floor "
            "(same probe on the pooling mask). H4 negative controls (static / other fc / other real): "
            + "; ".join(f"{c}: {d}" for c, d in CONTROL_DOC.items()) + ". A law claim needs the forecast to beat all "
            "three (paired group-bootstrap CI of the difference above 0) and, for roi, the mask-only floor (verdict "
            "column). Pass share = the eligible clips the forecast-dependent gate keeps; sensitivity = the score with "
            "every removed clip counted as a failure (S −1, creation 1).",
            "", f"Energy (R5): {R5_NOTE}.", ""]
    shown = set()
    for key, sel in res["selected"].items():
        shown.add(sel["momentum"]["selected"] if "momentum" in sel else sel["floor_best"])
        shown.add(sel["floor_best"])
    vis = lambda v: pr["n_spaces"] == 1 or v in shown
    f3 = [v for v, d in res["variants"].items() if d.get("momentum", {}).get("status") == "not scored" and vis(v)]
    if f3:
        rows += [f"Momentum not scored ({', '.join(f3)}): {F3_NOTE}", ""]
    mom = [(v, d) for v, d in res["variants"].items() if "S" in d.get("momentum", {}) and vis(v)]
    if mom:
        rows += ["## Momentum (S_mom; S_family secondary)", "",
                 "| law / space / pool | instrument, probe R² | interaction fc / floor (pass share) | η fc / floor / blind | "
                 "S_mom [95% CI] (clips) | S_mom sensitivity | S_mom true pre (diag.) | S_mom fc-mm | S_mom mask-only | "
                 "S_mom controls static / other fc / other real | verdict | S_family [CI] | "
                 "normal alignment A, B fc / floor (sanity: cannot detect the drift edit) |",
                 "|" + "---|" * 13]
        for v, d in mom:
            m, o, f = d["momentum"], d["momentum"]["observers"], d["momentum"]["family"]
            nm = m["mass_free"]
            rows.append(f"| {v} | {d['instrument']}, {d['probe_r2_mean']:.2f} | {m['interaction_rate']:.2f} / "
                        f"{m['interaction_rate_floor']:.2f} ({m['pass_share']:.2f}) | "
                        f"{m['eta']['forecast']:.3f} / {m['eta']['floor']:.3f} / {m['eta']['mass_blind']:.3f} | "
                        f"{_fmt_s(m['S'])} {_ci(m['S_ci95'])} ({m['n_scored_clips']}){'' if m['S_mom_informative'] else ' ‡'} | "
                        f"{_fmt_s(m['S_sensitivity'])} | {_fmt_s(m.get('S_true_pre', NAN))} | "
                        f"{_fmt_s(o['forecast_mm']['S']) if 'forecast_mm' in o else '—'} | "
                        f"{_fmt_s(o['mask_only']['S']) if 'mask_only' in o else '—'} | {_ctl(o, 'S')} | "
                        f"{_vd(m.get('verdict'))} | {_fmt_s(f['S'])} {_ci(f['S_ci95'])}{'' if f['informative'] else ' †'} | "
                        f"{nm['normal_A']['forecast']:.2f}, {nm['normal_B']['forecast']:.2f} / "
                        f"{nm['normal_A']['floor']:.2f}, {nm['normal_B']['floor']:.2f} |")
        rows += ["", f"‡ η_floor > {WEAK_MOM_FLOOR} η_blind: S_mom has almost no range. † d_floor > {WEAK_FLOOR}: S_family "
                 f"uninformative. Antiparallel / normal alignment: {SANITY_NOTE}. Compare models on common gated groups: "
                 "momentum_on_common_groups.", ""]
    en = [(v, d) for v, d in res["variants"].items() if "energy" in d and vis(v)]
    if en:
        rows += ["## Energy (ε of the exchange event; necessary-not-sufficient)", "",
                 "| law / space / pool | status (floor slope [CI], R²) | τ | creation rate fc [CI] / floor"
                 " / fc-mm / mask-only | creation controls static / other fc / other real | verdict | "
                 "lower tail fc [CI] / floor / mask-only | median ε̂ fc / floor / true | pass share fc "
                 "(creation sensitivity) |", "|---|---|---|---|---|---|---|---|---|"]
        for v, d in en:
            e = d["energy"]
            if "forecast" not in e:
                rows.append(f"| {v} | {e['status']} | | | | | | | |")
                continue
            fc, fl, ob, ins = e["forecast"], e["floor"], e["observers"], e["floor_check"]
            g = lambda k_, key: f" / {ob[k_][key]:.2f}" if k_ in ob else " / —"
            rows.append(f"| {v} | {e['status']} ({ins['slope']:.2f} {_ci(ins['slope_ci95'])}, {ins['r2']:.2f}) | "
                        f"{e['tau']:.2f} | {fc['creation_rate']:.2f} {_ci(fc['creation_ci95'])} / {fl['creation_rate']:.2f}"
                        f"{g('forecast_mm', 'creation_rate')}{g('mask_only', 'creation_rate')} | "
                        f"{_ctl(ob, 'creation_rate', lambda x: f'{x:.2f}')} | {_vd(e.get('verdict'))} | "
                        f"{fc['lower_tail_rate']:.2f} {_ci(fc['lower_tail_ci95'])} / {fl['lower_tail_rate']:.2f}"
                        f"{g('mask_only', 'lower_tail_rate')} | {fc['eps_median']:.2f} / {fl['eps_median']:.2f} / "
                        f"{e['eps_true_median']:.2f} | {fc['pass_share']:.2f} ({fc['creation_rate_sensitivity']:.2f}) |")
        rows += ["", "Lower tail: P(ε̂ < ε_lo − τ), ε_lo = per set the lowest floor-decoded ε of its gated legal twins.", ""]
    ra = [(v, d) for v, d in res["variants"].items() if "ratio" in d and vis(v)]
    if ra:
        rows += ["## Ratio laws (gravity, friction, inertia, oscillation, rolling, statics, angular momentum; "
                 "per-token-frame kinematics, frame pool (no roi: no mask leak); 1 = legal, 0 = the law's null; unit "
                 "form 1 − |T − T_legal| where the null is undefined; tip_consistency: accuracy)",
                 "", "| law / space / pool | probe R² | clips (windows fc) | statistic | S fc [CI] | S floor | S true | "
                 "S controls static / other fc / other real | verdict | T median fc / floor / true / legal / null |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for v, d in ra:
            n = d.get("n_clips", {})
            for st, r in d["ratio"].items():
                rows.append(f"| {v} | {d['probe_r2_mean']:.2f} | {n.get('legal', '')} ({n.get('windows_forecast', '')}) | "
                            f"{st} | {_fmt_s(r['forecast']['S'])} {_ci(r['forecast']['S_ci95'])} | "
                            f"{_fmt_s(r['floor']['S'])} | {_fmt_s(r['true']['S'])} | {_ctl(r, 'S')} | "
                            f"{_vd(r.get('verdict'))} | {r['forecast']['T_median']:.2f} / "
                            f"{r['floor']['T_median']:.2f} / {r['true']['T_median']:.2f} / {r['T_legal']:.2f} / "
                            f"{r['T_null_median']:.2f} |")
    return "\n".join(rows)
