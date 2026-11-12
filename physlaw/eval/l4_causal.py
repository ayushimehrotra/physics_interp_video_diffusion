"""L4 causal propagation (spec §5.0.4, §5.6): ONE manifold-steering routine, run per law (``laws.<law>`` blocks in
configs/eval/l4_causal.yaml; the evaluated family picks the law). Recipe from the predecessor's
elasticity_manifold.py (old docs §56i):

1. Manifold (:func:`fit_manifold`), per site = (layer, pool) for the law's parameter theta. A site is fitted in the
   space and on the token set it is injected into (:func:`site_positions`: roi_<obj>.pre -> that object's context ROI
   tokens, frame.pre -> every context token, frame.post -> every post-event token). Encoders: the model's own pass
   the hook acts on (:func:`site_activations`): pre-event sites from forecast()'s context-only encoder pass, captured
   at the site's tokens (the stage-4 cache encodes the whole clip, whose pre tokens see the future); post-event sites
   (energy: eps is invisible before the event) from the pred-real activations of the real clips' masked post tokens
   at intermediate PREDICTOR blocks (V-JEPA 2 / PC1; VideoMAE: pixel-decoder blocks), the token type forecast()
   fills. DiTs: stage 4's encode() at the site pool and (SNR, layer). ``mode: ladder`` (theta varies inside a scene:
   twin ladders) -- centre each scene's activations on its own mean over its rungs, average across scenes per rung ->
   centroid curve, rungs matched by rank; precondition (:func:`population_curve`): retention = |centroid| / rms
   per-scene residual (~1/sqrt(S) if the scenes' offsets point in unrelated directions) and leave-one-scene-out R² of
   the curve (raw / line / cubic), null = theta shuffled within each scene. ``mode: across`` (theta varies only across
   scenes) -- residualise activations and theta on the group-level ``nuisance`` (OLS), slope of the activations on
   theta; precondition = leave-one-scene-out R² of that line, null = theta shuffled across scenes. A site passes when
   every p <= ``max_p``; steering runs only at the passing layer with the highest retention (across: LOSO R²). The
   curve is a line, cubic only if its LOSO R² beats the line's. DiTs: every (SNR, layer) space the cache provides is
   tested; the injection BLOCK is the one whose best passing SNR space ranks highest, and a curve (with its controls,
   swap span and off-distribution metric) is fitted in EACH SNR space of that block that passes (``injection`` in the
   json: block, cached / fitted SNRs).
2. Steer (:func:`run_interventions`): the hook (:func:`steer`) adds shift(dtheta) = c(theta_ref + dtheta) -
   c(theta_ref) (c the curve, theta_ref the median rung) to every token of the site in the injection block, dtheta
   in the law's ``deltas``; the pooled site vector moves by exactly the shift. Encoders: forecast() rerun (encoder
   block on the context tokens, or predictor / decoder block on the target tokens), no generation; the steered site
   vector is captured (:func:`capture`). DiTs: during sampling, per step (:func:`steer_steps`): the step's timestep
   -> its SNR by the model's schedule (:func:`step_snr_db`) -> the cached SNR nearest in log-SNR; the shift is that
   SNR's curve, and a step whose nearest cached SNR failed the precondition gets none (``injection.steps`` and
   work/<scene>_steps.json record per step the curve used or "skipped"). The swap arm's span is per SNR the same way.
   The patch arm (no fitted curve) uses no cached vector: at every step a paired forward pass of the donor (same
   timestep, same noise draw, sequential) gives its live ROI activations, swapped into the base (:func:`patch_live`).
   The run returns the GENERATED video.
3. Read (:data:`READOUTS`): encoders (and DiT activations) -- the steered forecast decoded with the shared R1
   instrument (physlaw.probes.ridge, one per object, cross-fitted by group; ``_mm`` = forecast moment-matched to the
   real post rows): velocities from the frame.post pool (epsilon, deflection ratio; encoders' post rows are the
   pred-real FINAL outputs when available, the forecast's own space, as in L1), accelerations and spin from the
   per-token-frame pools (gravity, friction, oscillation, rolling, statics, angular momentum: l1_forecast_laws'
   token-frame instrument and line fits, plus the post window's sinusoid period and rotation angle,
   :func:`series_extras`; stage 4 ``pooled_frame_t``); DiTs -- the generated video read with the output track-B
   readout (eval/outputs.READOUT_SELF). Objects flagged ``aux`` (rods, props) are never read.
4. Score vs the LAW's prediction (:func:`score_response`):
     energy    theta = eps (F2 / F7 bounce / F8 / F9 swing ladders) -> eps_hat; law: rises, then SATURATES AT 1
               (:func:`saturation_scores`)
     momentum  theta = log(m_A / m_B) (F1, across, nuisance log(r_A / r_B)) -> log(dv_B.n / -dv_A.n); law slope 1
     gravity   theta = log m_A (F7 drops) -> log a_heavy / a_light; law slope 0 (a predicted NULL)
     friction  theta = log m_A (F6 pair, across) -> log deceleration of A along its motion while it moves (the
               post window ends at io.friction_windows' move_end); law slope 0 (mu g)
     oscillation  theta = log m_A (F9 pair, across) -> log period of A after the branch; law slope 0
     rolling   theta = kappa_A (F10, varies per group) -> log a_A along the incline; law slope -1 / (1 + kappa_ref),
               kappa_ref ``auto`` = the median kappa_A of the fitted scenes
     statics   theta = overhang x (F11 grid, across) -> tipped (rotation over laws.incline.tip_window > TIP_ANGLE); law:
               a STEP at x = 1 (:func:`step_scores`: the steered indicator must switch where theta + dtheta crosses 1)
     angular_momentum  theta = |impact parameter| / bar half-length (F12, across) -> L_ratio; law slope 0 (stays 1)
     galilean  theta = boost speed (F4 ball-boost ladder; camera-boost twins excluded, :func:`law_rows`) -> eps_hat
               (and theta_cm); law slope 0
   Per scene the readout's slope on dtheta (saturation: its plateau), median over scenes with a scene-bootstrap CI,
   against the law's value and against the controls. Steering reliability: the per-scene distribution and the share
   of scenes moving the WRONG way (``per_scene``, ``wrong_way``); the off-distribution check (``off_distribution``:
   Mahalanobis distance of the steered pooled vector to the injection layer's training rows, per dtheta, against the
   held-out real rows' distances); the input-level reference (condition ``input``: the real clip at theta + dtheta,
   ladder: the scene's twin, across: the nearest other scene, read with the same instrument -- does steering beat
   showing it; ``verdict.steering_over_input``).
5. Controls (user-approved, only these): ``passive`` propagation (post-event sites only: the unsteered forecast's
   site vector + the same shift, read in the site's own space by an instrument on the real clips' site activations;
   it isolates what the downstream predictor / decoder blocks add; a pre-event site holds no post-event state to
   read, so it has none), ``random_dir`` (one random direction, shift norm matched per dtheta), energy ``speed`` (the
   aftermath-speed direction at fixed eps: manifold of the post-event speed centred per rung), gravity ``size`` (log
   radius, same mode).
6. Pair arms (``variants[...]["pairs"]``; one routine, shared code :func:`swap_delta`):
     swap   (every law; interchange intervention, Geiger et al. / MIB) the base clip's component in the span of the
            fitted theta-curve replaced by a donor's (theta'); IIA = share of pairs whose readout lands closer to the
            law's outcome at theta' than at theta (statics: the law's tip indicator at theta'), fraction of the
            required change achieved, error ratio with a paired sign-flip p; selectivity = target change / change of
            the law's NON-target quantity (:data:`NONTARGET`), both in floor-sd units. Laws predicting NO change
            (slope 0: gravity, friction, oscillation, angular momentum, Galilean) score INVARIANCE instead
            (:func:`invariance_scores`): the model's own readout of theta (:func:`theta_probe` on the swapped forecast;
            DiTs: on the activations at the swapped block / SNR, ``site_theta``) must move by >= ``min_moved`` of the
            donor difference (as a fraction of its real-video change) while the outcome stays put, invariance ratio =
            median |outcome change| / the change under a legal twin pair the law says DOES move it
            (``invariance_ref``: Galilean e) or the outcome's sd over real clips. Controls
            ``swap_random`` (norm-matched random direction), ``swap_passive``.
     patch  (momentum, ``patch: true``; the predecessor's causal collision experiment, docs §41) A's whole pre-contact
            ROI vector replaced by a matched-geometry donor's whose contact falls in the base's contact token-frame
            (:func:`patch_donors`: complete, time-aligned ROI; ``patch_donors`` in the json logs the excluded
            candidates per scene) (DiTs: the donor's live ROI from a paired pass at each sampling step, same
            timestep and noise draw); B's decoded post velocity against the required
            dv_B' = k ((v_A,donor - v_A) . n) n: error ratio, sign-flip p, and the high-k / low-k slope-ratio
            discriminator (physics vs a constant-gain leak); controls ``patch_random``, ``patch_passive``; pass rule
            :func:`patch_verdict` (``pairs.patch.verdict`` / ``reasons``).
   Slope-0 laws (steering and swap): an unchanged outcome passes only with positive evidence that the push worked
   (:func:`null_verdict`: the model's theta readout moved by >= ``min_effect`` of its real-video change, or a control
   direction moved the outcome); else "inert push (uninformative)". ``readout_validity`` (slope laws): the instrument's
   readout of the REAL clips regressed on the true theta across scenes (must recover the law's slope).
Interventions are cached per scene in ``work/<law>`` (resumable; a scene is rerun when a planned run is missing).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from physlaw.eval.l0_states import event_inputs
from physlaw.eval.l3_invariance import forecast_spaces, get, load_family
from physlaw.models.base import PRED_LAYER
from physlaw.probes.ridge import apply_instrument, grouped_kfold, instrument_folds

LAW_OF_FAMILY = {"f1": "momentum", "f2": "energy", "f3": "momentum", "f4": "galilean", "f6": "friction",
                 "f7": "energy", "f8": "energy", "f9": "oscillation", "f10": "rolling", "f11": "statics",
                 "f12": "angular_momentum"}


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #


def blocks(m, layer="0"):
    """The block list holding layer key ``layer``: DiT transformer blocks; encoder blocks (key "<i>"); predictor
    (V-JEPA 2 / PC1) or pixel-decoder (VideoMAE) blocks (key PRED_LAYER "pred/L<i>")."""
    if hasattr(m, "tr"):                                             # diffusers DiTs
        return getattr(m.tr, "transformer_blocks", None) or m.tr.blocks
    mm = getattr(m, "model", None)
    if str(layer).startswith("pred/"):
        if hasattr(mm, "predictor"):
            return mm.predictor.layer
        if hasattr(mm, "decoder"):
            return mm.decoder.blocks
    elif hasattr(mm, "encoder"):                                     # V-JEPA 2 encoder.layer; VideoMAE encoder.blocks
        return mm.encoder.layer if hasattr(mm.encoder, "layer") else mm.encoder.blocks
    raise TypeError(f"L4: no block list for layer {layer!r} of {type(m).__name__}")


def block_index(layer: str) -> int:
    """Layer key -> block index: "12" -> 12, "snr+8/L15" -> 15, "pred/L3" -> 3."""
    return int(str(layer).rsplit("/L", 1)[-1])


def _shifted(out, idx, delta):
    """A block output (tensor or tuple whose first entry is the hidden states) with ``delta`` added at tokens idx."""
    import torch
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone()
    h[:, idx.to(h.device)] += torch.as_tensor(delta, device=h.device, dtype=h.dtype)
    return (h, *out[1:]) if isinstance(out, tuple) else h


@contextmanager
def steer(m, layer: str, idx, delta):
    """While active, add ``delta`` [D] to token positions ``idx`` of the output of block ``layer``."""
    import torch
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    handle = blocks(m, layer)[block_index(layer)].register_forward_hook(lambda mod, inp, out: _shifted(out, idx, delta))
    try:
        yield
    finally:
        handle.remove()


# DiTs: one curve per cached SNR space, chosen per sampling step


def snr_of(key) -> float | None:
    """The SNR (dB) of a DiT space key "snr+8/L15" (None for any other key: encoder layers, "final", latents)."""
    import re
    mt = re.fullmatch(r"snr([+-]?\d+(?:\.\d+)?)/L\d+", str(key))
    return float(mt.group(1)) if mt else None


def snr_name(db: float) -> str:
    """dB -> the cache's SNR tag ("snr+8", "snr-20")."""
    return f"snr{float(db):+g}"


def _t_value(t) -> float:
    """A sampler timestep (int, float or tensor; per-token timesteps: the noisiest token's) as one float."""
    if hasattr(t, "detach"):
        t = t.detach().float().cpu().numpy()
    return float(np.max(np.asarray(t, float)))


def step_snr_db(m, t) -> float:
    """SNR (dB, 10 log10 SNR) of a sampling step at training timestep t by the model's own schedule: DDPM
    (``m.alphas_cumprod``) 10 log10(a_t / (1 - a_t)); rectified flow (s = t / ``m.n_train``, default 1000)
    20 log10((1 - s) / s), the inverse of dit.flow_t_for_db. Pure noise -> -inf, clean -> +inf."""
    t = _t_value(t)
    ac = getattr(m, "alphas_cumprod", None)
    with np.errstate(divide="ignore", invalid="ignore"):
        if ac is not None:
            a = np.float64(float(ac[int(round(t))]))
            return float(10 * np.log10(a / (1 - a))) if a < 1 else float("inf")
        s = np.float64(t / float(getattr(m, "n_train", 1000)))
        return float(20 * np.log10((1 - s) / s)) if 0 < s < 1 else (float("inf") if s <= 0 else float("-inf"))


def nearest_snr(db: float, cached) -> float:
    """The cached SNR (dB) nearest a step's SNR in log-SNR (dB = 10 log10 SNR, a fixed multiple of log SNR); a pure-
    noise (-inf) / clean (+inf) step takes the lowest / highest."""
    c = np.sort(np.asarray(list(cached), float))
    if np.isinf(db):
        return float(c[0] if db < 0 else c[-1])
    return float(c[int(np.argmin(np.abs(c - db)))])


def step_record(i: int, t: float, db: float, cached, fitted) -> dict:
    """One sampling step's injection: the nearest cached SNR and the curve used there ("snr<dB>"), or "skipped" when
    that SNR has no fitted curve (it failed the precondition)."""
    near = nearest_snr(db, cached)
    return {"step": int(i), "t": float(t), "snr_db": float(db) if np.isfinite(db) else ("-inf" if db < 0 else "+inf"),
            "nearest_snr_db": near, "curve": snr_name(near) if near in {float(x) for x in fitted} else "skipped"}


def snr_schedule(step_dbs, cached, fitted, ts=None) -> list[dict]:
    """:func:`step_record` for every step of a sampler whose steps have SNRs ``step_dbs`` (timesteps ``ts``)."""
    ts = list(range(len(step_dbs))) if ts is None else ts
    return [step_record(i, t, db, cached, fitted) for i, (t, db) in enumerate(zip(ts, step_dbs))]


@contextmanager
def steer_steps(m, layer: str, idx, deltas: dict, cached, log: list):
    """DiTs: :func:`steer` with a shift that depends on the sampling step. ``deltas`` {snr_db: [D]}: one shift per
    cached SNR space whose curve was fitted (it passed the precondition); ``cached``: every SNR the cache provides at
    the block. Each transformer pass ``m._pass(z_t, t, ...)`` maps its timestep to an SNR (:func:`step_snr_db`); the
    block output then gets the shift of the nearest cached SNR, or none when that SNR has no curve. ``log`` gets one
    :func:`step_record` per sampling step (consecutive passes at one timestep, e.g. CFG's unconditional pass, are one
    step)."""
    import torch
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    deltas = {float(k): v for k, v in deltas.items()}
    cur = {"v": None}
    own, orig = "_pass" in vars(m), m._pass

    def _pass(z_t, t, *a, **kw):
        tt = _t_value(t)
        if not log or log[-1]["t"] != tt:
            log.append(step_record(len(log), tt, step_snr_db(m, t), cached, deltas))
        r = log[-1]
        cur["v"] = None if r["curve"] == "skipped" else deltas[r["nearest_snr_db"]]
        try:
            return orig(z_t, t, *a, **kw)
        finally:
            cur["v"] = None

    def hook(mod, inp, out):
        return None if cur["v"] is None else _shifted(out, idx, cur["v"])
    m._pass = _pass
    handle = blocks(m, layer)[block_index(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        if own:
            m._pass = orig
        else:
            del m._pass


class LivePatch:
    """A DiT collision-patch run (plan entry): A's ROI swapped in from ``donor`` (clip dir) by a paired forward pass
    at every sampling step (:func:`patch_live`); ``random_seed``: the norm-matched random-direction control."""

    def __init__(self, donor, random_seed: int | None = None):
        self.donor, self.random_seed = donor, random_seed


def context_gain(m, t, clean: bool = False) -> float:
    """Coefficient of the clean context latents in the sampler's context input at timestep t (the noise term is a
    fixed draw shared by every clip): clean context (timestep-0 context, ``clean`` > 0) 1; DDPM sqrt(abar_t); rectified
    flow 1 - t / n_train."""
    if clean:
        return 1.0
    t = _t_value(t)
    ac = getattr(m, "alphas_cumprod", None)
    if ac is not None:
        return float(np.sqrt(float(ac[int(round(t))])))
    return 1.0 - t / float(getattr(m, "n_train", 1000))


@contextmanager
def paired_pass(m, x, x_donor, k: int, st: dict, log: list, gate=None):
    """The paired donor pass of the DiT patches (:func:`patch_live`, :func:`patch_live_component`): every transformer
    pass ``m._pass(z_t, t, ...)`` of the base's sampler is preceded by the same pass (same timestep, same batch / CFG
    layout and embeddings, same noise draw) on the DONOR's input: z_t with the context latents replaced by the donor's,
    noised to this step with the base's context noise (z_t[:kl] + context_gain(t) (z0_donor - z0_base)[:kl], both from
    ``m._mixed`` with the base's k). ``st["mode"]`` is "donor" during the donor pass and "base" during the base pass
    (hooks read / write ``st["donor"]``, reset after each pair). Sequential: no memory beyond one pass. ``log``: per
    step {step, t, snr_db, donor_t, curve: "patch"}. ``gate(t)`` False: that step runs the base pass alone."""
    import torch
    kl = m.n_context_tokens(k) * int(getattr(m, "lpt", 1))
    with torch.no_grad():
        zd = m._mixed(x_donor, k)[:, :kl].clone()
        dz = (zd - m._mixed(x, k)[:, :kl]).double()             # base last: the sampler's latent memo hits
    own, orig = "_pass" in vars(m), m._pass

    def _pass(z_t, t, *a, **kw):
        if gate is not None and not gate(t):
            return orig(z_t, t, *a, **kw)
        tt = _t_value(t)
        z_d = z_t.clone()
        z_d[:, :kl] = (z_t[:, :kl].double() + context_gain(m, t, bool(kw.get("clean"))) * dz.to(z_t.device)).to(z_t.dtype)
        try:
            st["mode"] = "donor"
            orig(z_d, t, *a, **({**kw, "keep": ()} if "keep" in kw else kw))
            if not log or log[-1]["t"] != tt:
                db = step_snr_db(m, t)
                log.append({"step": len(log), "t": tt, "donor_t": tt, "curve": "patch",
                            "snr_db": float(db) if np.isfinite(db) else ("-inf" if db < 0 else "+inf")})
            st["mode"] = "base"
            return orig(z_t, t, *a, **kw)
        finally:
            st["mode"], st["donor"] = None, None
    m._pass = _pass
    try:
        yield
    finally:
        if own:
            m._pass = orig
        else:
            del m._pass


def _check_donor_context(m, k, idx_donor):
    kt = m.n_context_tokens(k)
    if (np.asarray(idx_donor) >= kt * int(np.prod(m.token_grid.grid_hw))).any():   # never truncate a donor ROI
        raise ValueError(f"L4 patch: donor ROI tokens beyond the base's {kt} context token-frames; donors must "
                         f"have their contact in the base's token-frame")


@contextmanager
def patch_live(m, layer: str, idx, idx_donor, x, x_donor, k: int, log: list, random_seed: int | None = None,
               gate=None):
    """DiT collision patch by a paired forward pass (:func:`paired_pass`: before every base transformer pass, the same
    pass at the same timestep, with the same noise draw and CFG layout, on the donor's context). The donor pass's live
    block output gives A's pooled ROI vector (``idx_donor``: the donor's A ROI tokens, all inside the base's context
    since donors share its contact token-frame, :func:`patch_donors`; else ValueError); in the base pass every base ROI
    token (``idx``) moves by donor - base pooled ROI, so the base's pooled ROI vector becomes the donor's at that
    timestep. Nothing is cached. ``random_seed``: the swap's norm replaced along one fixed random direction (control).
    ``log``: per step {step, t, snr_db, donor_t, curve: "patch"}. ``gate(t)`` (optional; localization's per-SNR sweep):
    steps where it is False run the base pass alone, unpatched."""
    import torch
    ib = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    _check_donor_context(m, k, idx_donor)
    idd = torch.as_tensor(np.asarray(idx_donor), dtype=torch.long)
    st = {"mode": None, "donor": None, "dir": None}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if st["mode"] == "donor":
            st["donor"] = h[:, idd.to(h.device)].float().mean(1)                       # [B, D]: live, this step
            return None
        if st["mode"] != "base" or st["donor"] is None or not len(ib) or not len(idd):
            return None
        d = st["donor"] - h[:, ib.to(h.device)].float().mean(1)
        if random_seed is not None:
            if st["dir"] is None:
                r = np.random.default_rng(random_seed).standard_normal(d.shape[-1])
                st["dir"] = torch.as_tensor(r / np.linalg.norm(r), dtype=torch.float32)
            d = d.norm(dim=-1, keepdim=True) * st["dir"].to(d.device)
        h = h.clone()
        h[:, ib.to(h.device)] += d[:, None].to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
    handle = blocks(m, layer)[block_index(layer)].register_forward_hook(hook)
    try:
        with paired_pass(m, x, x_donor, k, st, log, gate):
            yield
    finally:
        handle.remove()


@contextmanager
def patch_live_component(m, layer: str, comp: str, idx, idx_donor, x, x_donor, k: int, log: list, gate=None,
                         ctx_only: bool = True):
    """DiT component patch by the same paired pass (:func:`paired_pass`): at every (gated) step the donor pass's live
    output of component ``comp`` (attention head / MLP, :func:`comp_edit`) pooled per token-frame over the donor's site
    tokens (``idx_donor``) is captured, and in the base pass that follows (same timestep, same noise draw and CFG
    layout) the base's per-token-frame pooled output over its site tokens (``idx``) is set to it (frames the donor
    lacks unchanged). ``ctx_only``: the donor's site must lie inside the base's context (pre-event sites), else
    ValueError (outside it the paired pass holds the base's noised latents)."""
    import torch
    if ctx_only:
        _check_donor_context(m, k, idx_donor)
    per = int(np.prod(m.token_grid.grid_hw))
    ib, idd = np.asarray(idx), np.asarray(idx_donor)
    fb, fd = ib // per, idd // per
    st = {"mode": None, "donor": None}

    def fn(v):                                       # v [B, N, d]: every video token of the component's output
        if st["mode"] == "donor":
            st["donor"] = {int(f): v[:, torch.as_tensor(idd[fd == f], device=v.device)].mean(1).detach()
                           for f in np.unique(fd)}
            return None
        if st["mode"] != "base" or not st["donor"]:
            return None
        v = v.clone()
        for f in np.unique(fb):
            if int(f) in st["donor"]:
                sel = torch.as_tensor(ib[fb == f], device=v.device)
                v[:, sel] = v[:, sel] + (st["donor"][int(f)] - v[:, sel].mean(1))[:, None].to(v.dtype)
        return v
    with comp_edit(m, layer, comp, None, fn), paired_pass(m, x, x_donor, k, st, log, gate):
        yield


@contextmanager
def capture(m, layer: str, idx, out: list):
    """While active, append to ``out`` the mean over token positions ``idx`` of block ``layer``'s output at every
    call (after a steering hook registered before it: the steered site vector)."""
    import torch
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)

    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        out.append(h[0, idx.to(h.device)].float().mean(0).cpu().numpy())
    handle = blocks(m, layer)[block_index(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# steering sites: the curve is fitted in the space and on the token set it is injected into
# --------------------------------------------------------------------------- #


def site_positions(pool: str) -> str:
    """The injection token set of a site pool = the tokens the pool averages: "<roi>.pre" -> that object's ROI tokens
    in the context ("roi_A"); frame.pre -> every context token ("context"); frame.post -> every post-event token
    ("post": encoders' predictor / decoder targets, DiTs' post latents)."""
    if pool == "frame.post":
        return "post"
    if pool == "frame.pre":
        return "context"
    if pool.startswith("roi_") and pool.endswith(".pre") and pool != "roi_pair.pre":
        return pool[: -len(".pre")]
    raise ValueError(f"L4: pool {pool!r} is not a steering site (roi_<obj>.pre, frame.pre or frame.post)")


def site_index(m, c0, positions: str):
    """(flat token indices of the site in the clip's token grid, input frames, context frames)."""
    from physlaw.tokens.project import project_masks, roi_object_ids
    fr, k = m.input_frames(c0.labels), m.context_frames(c0.labels)
    kt = m.n_context_tokens(k) if hasattr(m, "n_context_tokens") else k // m.token_grid.tubelet
    sel = np.zeros((len(m.token_grid.token_frames(len(fr))), *m.token_grid.grid_hw), bool)
    if positions == "post":
        sel[kt:] = True
    elif positions == "context":
        sel[:kt] = True
    elif positions.startswith("roi_"):
        ob = project_masks(c0.masks, fr, m.token_grid, roi_object_ids(c0.labels))
        sel[:kt] = np.asarray(ob[positions[4:]].roi, bool)[:kt]
    else:
        raise ValueError(f"positions {positions!r}: roi_<obj>, context or post")
    return np.flatnonzero(sel.reshape(-1)), fr, k


def site_layers(m, positions: str) -> list[str]:
    """Encoders' candidate injection layers. Pre-event sites: the configured encoder layers (forecast()'s context-only
    encoder pass). Post-event sites: the predictor (V-JEPA 2 / PC1) or pixel-decoder (VideoMAE) blocks but the last,
    i.e. intermediate layers, so that downstream blocks remain (what passive propagation isolates)."""
    if positions == "post":
        n = len(blocks(m, PRED_LAYER.format(0)))
        return [PRED_LAYER.format(i) for i in range(max(n - 1, 1))]
    n = len(blocks(m, "0"))
    return [str(l) for l in m.layers if 0 <= int(l) < n]


def site_activations(m, data_dir: Path, recs, labels, positions: str, n_masks: int = 4, log=print) -> dict:
    """Encoders: every real clip's activations AT THE STEERING SITE, from the pass the hook acts on, pooled over the
    site's tokens: {"layers": keys, key: [n, D]}. Pre-event sites: forecast()'s context-only encoder pass (the
    stage-4 cache encodes the whole clip, whose pre tokens see the future: another space), captured with the hook at
    the site's tokens (twins sharing a pre_segment share it). Post-event sites: the model's own pred-real activations
    of the clip's real post-event tokens at each predictor / decoder block (masked real targets, the token type
    forecast() fills), plus "final" (the pred-real FINAL output: the readout instrument's post rows)."""
    from contextlib import ExitStack

    import torch
    from physlaw.data.io import load_clip
    from physlaw.eval.l1_forecast_laws import clip_segment
    keys = site_layers(m, positions)
    out, memo = {q: [] for q in keys + (["final"] if positions == "post" else [])}, {}
    for i, (r, l) in enumerate(zip(recs, labels)):
        key = (r["group"], str(clip_segment(l)))
        if positions == "post" or key not in memo:
            c0 = load_clip(data_dir / r["clip"])
            idx, fr, k = site_index(m, c0, positions)
            x = m.to_tensor(c0.frames[fr])
            if positions == "post":
                acts = m.pred_real(x, k, n_masks=n_masks, layers=[block_index(q) for q in keys])
                got = {q: acts[q] for q in keys} | {"final": acts[-1]}
                got = {q: a.reshape(-1, a.shape[-1]).float().mean(0).cpu().numpy() for q, a in got.items()}
            else:
                cap = {q: [] for q in keys}
                with ExitStack() as st, torch.no_grad():
                    for q in keys:
                        st.enter_context(capture(m, q, idx, cap[q]))
                    m.forecast(x, k)
                got = {q: np.mean(v, 0) for q, v in cap.items()}
            memo = {key: got}
        for q, v in memo[key].items():
            out[q].append(v)
        if (i + 1) % 100 == 0:
            log(f"[L4] site activations {i + 1}/{len(recs)}")
    return {"layers": keys, **{q: np.stack(v).astype(np.float64) for q, v in out.items()}}


# --------------------------------------------------------------------------- #
# manifolds and the precondition
# --------------------------------------------------------------------------- #


def _population_stats(R: np.ndarray, e: np.ndarray) -> tuple[float, dict]:
    """R [S, E, d] scene-centred activations on a shared rung grid e [E]: (retention, LOSO R² raw / line / cubic)."""
    S = len(R)
    cen = R.mean(0)
    retention = float(np.mean(np.linalg.norm(cen, axis=1) / np.sqrt((R ** 2).sum(2).mean(0)).clip(1e-300)))
    tot = float((R ** 2).sum())
    res = {"raw": 0.0, "line": 0.0, "cubic": 0.0}
    for s in range(S):
        c = (cen * S - R[s]) / (S - 1)
        res["raw"] += float(((R[s] - c) ** 2).sum())
        for name, deg in (("line", 1), ("cubic", min(3, len(e) - 1))):
            res[name] += float(((R[s] - np.vander(e, deg + 1) @ np.polyfit(e, c, deg)) ** 2).sum())
    return retention, {k: 1 - v / tot for k, v in res.items()}


def _ladder(X, theta, scene):
    """(R [S, E, d] scene-centred, rung grid [E]) over the scenes with the modal number of rungs (rank-matched)."""
    by = {g: np.where(scene == g)[0] for g in sorted(set(scene))}
    n = [len(v) for v in by.values()]
    k = max(set(n), key=n.count)
    R, E = [], []
    for rows in by.values():
        if len(rows) != k:
            continue
        rows = rows[np.argsort(theta[rows])]
        Xs = X[rows].astype(np.float64)
        R.append(Xs - Xs.mean(0))
        E.append(theta[rows])
    return np.stack(R), np.median(np.stack(E), 0)


def population_curve(X: np.ndarray, theta: np.ndarray, scene: np.ndarray, draws: int = 200, seed: int = 0) -> dict | None:
    """Ladder-mode precondition (module docstring, step 1); None with < 3 rungs or < 3 scenes."""
    R, e = _ladder(X, theta, scene)
    if len(e) < 3 or len(R) < 3:
        return None
    ret, r2 = _population_stats(R, e)
    rng = np.random.default_rng(seed)
    null = [_population_stats(np.stack([r[rng.permutation(len(e))] for r in R]), e) for _ in range(draws)]
    p = lambda obs, xs: float((1 + np.sum(np.array(xs) >= obs)) / (1 + len(xs)))
    return {"mode": "ladder", "n_scenes": len(R), "theta_grid": e.tolist(), "retention": ret,
            "retention_null_median": float(np.median([a for a, _ in null])), "retention_iid_reference": 1 / np.sqrt(len(R)),
            "p_retention": p(ret, [a for a, _ in null]), "loso_r2": r2,
            "loso_r2_line_null_median": float(np.median([b["line"] for _, b in null])),
            "p_loso_line": p(r2["line"], [b["line"] for _, b in null]), "p_loso_cubic": p(r2["cubic"], [b["cubic"] for _, b in null])}


def _resid(Y, N):
    """Y minus its OLS fit on [1, N] (N None: minus the mean)."""
    Y = np.asarray(Y, np.float64)
    D = np.ones((len(Y), 1)) if N is None else np.c_[np.ones(len(Y)), np.asarray(N, np.float64).reshape(len(Y), -1)]
    return Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]


def _across_loso(Xr, t, scene) -> float:
    sse = 0.0
    for g in sorted(set(scene)):                     # fixed order: the float sum must not follow the string hash seed
        tr, te = scene != g, scene == g
        b = Xr[tr].T @ t[tr] / max(t[tr] @ t[tr], 1e-300)
        sse += float(((Xr[te] - np.outer(t[te], b)) ** 2).sum())
    return 1 - sse / float((Xr ** 2).sum())


def across_curve(X, theta, scene, nuisance=None, draws: int = 200, seed: int = 0) -> dict:
    """Across-mode precondition: activations and theta residualised on [1, nuisance]; LOSO R² of the line
    X ~ theta vs theta shuffled across scenes (scene-level permutation)."""
    Xr, t = _resid(X, nuisance), _resid(theta, nuisance)
    r2 = _across_loso(Xr, t, scene)
    rng = np.random.default_rng(seed)
    gs = np.array(sorted(set(scene)))
    null = []
    for _ in range(draws):
        perm = dict(zip(gs, rng.permutation(gs)))
        tp = np.array([t[scene == perm[s]].mean() for s in scene]) if len(gs) < len(t) else t[rng.permutation(len(t))]
        null.append(_across_loso(Xr, tp, scene))
    return {"mode": "across", "n_scenes": int(len(gs)), "loso_r2": {"line": r2}, "retention": None,
            "loso_r2_line_null_median": float(np.median(null)),
            "p_loso_line": float((1 + np.sum(np.array(null) >= r2)) / (1 + draws))}


def fit_manifold(X, theta, scene, mode: str = "ladder", nuisance=None) -> dict:
    """The steering curve: {"coef": polynomial coefficients [deg + 1, d] of activation on theta, "theta_ref",
    "deg"}. Ladder: scene-centred centroid curve, cubic if its LOSO R² beats the line's; across: the residualised
    slope (a line)."""
    if mode == "ladder":
        R, e = _ladder(X, theta, scene)
        cen = R.mean(0)
        deg = 1
        if len(e) >= 5 and len(R) >= 3:
            _, r2 = _population_stats(R, e)
            deg = 3 if r2["cubic"] > r2["line"] else 1
        return {"coef": np.polyfit(e, cen, deg), "theta_ref": float(np.median(e)), "deg": deg}
    Xr, t = _resid(X, nuisance), _resid(theta, nuisance)
    b = Xr.T @ t / max(t @ t, 1e-300)
    return {"coef": np.stack([b, np.zeros_like(b)]), "theta_ref": 0.0, "deg": 1}


def shift(man: dict, d: float) -> np.ndarray:
    """Activation displacement for a step d along the curve from theta_ref."""
    c, t0 = man["coef"], man["theta_ref"]
    return np.polyval(c, t0 + d) - np.polyval(c, t0)


def plan_shifts(man: dict, deltas, controls: dict, seed: int = 0) -> dict:
    """{condition: {dtheta: shift vector}}: the law's curve, and norm-matched controls (``random_dir``: one random
    direction; any other entry of ``controls``: a fitted manifold, e.g. ``speed`` / ``size``)."""
    out = {"law": {float(d): shift(man, d) for d in deltas}}
    D = out["law"][float(deltas[0])].shape[-1]
    unit = lambda v: v / max(np.linalg.norm(v), 1e-300)
    sgn = lambda d: 1.0 if d >= 0 else -1.0
    for name, c in controls.items():
        if name == "random_dir":
            r = unit(np.random.default_rng(seed).standard_normal(D))
            out[name] = {d: sgn(d) * np.linalg.norm(v) * r for d, v in out["law"].items()}
        else:
            out[name] = {d: sgn(d) * np.linalg.norm(v) * unit(shift(c, sgn(d))) for d, v in out["law"].items()}
    return out


# --------------------------------------------------------------------------- #
# scoring against the law
# --------------------------------------------------------------------------- #


def _quantiles(v) -> dict:
    """Per-scene distribution (steering reliability): 5 / 25 / 50 / 75 / 95 % quantiles of the finite values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"q{q}": (float(np.quantile(v, q / 100)) if len(v) else None) for q in (5, 25, 50, 75, 95)}


def _nan_guard(fn, a, axis=None):
    """``fn`` (np.nanmean / np.nanmedian) without the empty-slice RuntimeWarning: an all-NaN (or empty) slice gives NaN,
    exactly what fn returns there, and every other slice fn's own value."""
    a = np.asarray(a, float)
    empty = np.isnan(a).all(axis=axis)
    if axis is None:
        return float("nan") if empty else float(fn(a))
    return np.where(empty, np.nan, fn(np.where(np.expand_dims(empty, axis), 0.0, a), axis=axis))


def saturation_scores(curves: dict, alphas, groups, n_top: int = 2, sat_tol: float = 0.1, control: str = "speed",
                      passive: str = "passive", n_boot: int = 1000, seed: int = 0, law: str = "law") -> dict:
    """curves {condition: [G, n_steps] readout per scene and step}. Per condition: value at step 0, saturation =
    median over scenes of the plateau (mean over the n_top largest steps) with a scene-bootstrap CI, rise =
    saturation - value at 0, plateau slope (last step), the per-scene plateau quantiles and the share of scenes moving
    AWAY from 1 (``wrong_way``; steering reliability). The ``law`` condition is compared with every other one
    (paired plateau difference, CI). ``law_like``: saturates within sat_tol of 1, differs from the control's
    plateau, and rises more than passive propagation."""
    alphas = np.asarray(alphas, float)
    order = np.argsort(alphas)
    alphas = alphas[order]
    top = np.arange(len(alphas))[-n_top:]
    z = int(np.argmin(np.abs(alphas)))
    G = len(groups)
    B = np.random.default_rng(seed).integers(0, G, (n_boot, G)) if n_boot and G > 1 else None

    med = lambda v, axis=None: _nan_guard(np.nanmedian, v, axis)     # all-NaN slices -> NaN, without a warning

    def ci(v):
        if B is None:
            return None
        x = med(v[B], 1)
        return [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))] if np.isfinite(x).any() else None
    out, plateau = {}, {}
    for c, Y in curves.items():
        Y = np.asarray(Y, float)[:, order]
        p = _nan_guard(np.nanmean, Y[:, top], 1)
        plateau[c] = p
        sat = med(p)
        rise = p - Y[:, z]
        wrong = np.sign(rise) == -np.sign(1.0 - Y[:, z])           # moves away from 1 (the law's plateau)
        ok = np.isfinite(rise)
        out[c] = {"value_at_0": med(Y[:, z]), "saturation": sat, "saturation_ci95": ci(p),
                  "saturation_minus_1": sat - 1.0, "rise": med(rise),
                  "per_scene": _quantiles(p), "wrong_way": float(wrong[ok].mean()) if ok.any() else float("nan"),
                  "plateau_slope": med((Y[:, -1] - Y[:, -2]) / (alphas[-1] - alphas[-2])) if len(alphas) > 1 else float("nan"),
                  "curve_median": med(Y, 0).tolist()}
    if law in plateau:
        for c in plateau:
            if c != law:
                d = plateau[law] - plateau[c]
                out[law][f"minus_{c}"] = {"median": med(d), "ci95": ci(d)}
        e = out[law]
        cd = e.get(f"minus_{control}", {}).get("ci95")
        out["verdict"] = {"saturates_at_1": abs(e["saturation_minus_1"]) <= sat_tol,
                          "differs_from_control": bool(cd and (cd[0] > 0 or cd[1] < 0)),
                          "exceeds_passive": bool(passive in out and abs(e["rise"]) > abs(out[passive]["rise"])),
                          "sat_tol": sat_tol, "control": control}
        out["verdict"]["law_like"] = all(out["verdict"][k] for k in ("saturates_at_1", "differs_from_control", "exceeds_passive"))
    out["steps"] = alphas.tolist()
    out["n_scenes"] = G
    return out


def slope_scores(curves: dict, deltas, groups, law_value: float, tol: float = 0.25, n_boot: int = 1000, seed: int = 0,
                 law: str = "law") -> dict:
    """Per condition: per-scene OLS slope of the readout on dtheta (finite points, >= 2), median with a scene-
    bootstrap CI, the per-scene quantiles and the share of scenes whose slope has the sign opposite to the law's
    (``wrong_way``; None for a predicted null). Verdict for the ``law`` condition: its CI contains the law's value or lies within ``tol`` of it;
    for a non-zero law value it must also differ from every control (paired CI excludes 0). A predicted null
    (law value 0) is only informative next to a control that DOES move the readout (reported)."""
    x = np.asarray(deltas, float)
    G = len(groups)
    B = np.random.default_rng(seed).integers(0, G, (n_boot, G)) if n_boot and G > 1 else None

    def per_scene(Y):
        out = np.full(len(Y), np.nan)
        for i, y in enumerate(np.asarray(Y, float)):
            ok = np.isfinite(y)
            if ok.sum() >= 2 and np.ptp(x[ok]) > 0:
                out[i] = np.polyfit(x[ok], y[ok], 1)[0]
        return out

    def ci(v):
        if B is None or not np.isfinite(v).any():
            return None
        m = np.nanmedian(v[B], 1)
        return [float(np.nanquantile(m, 0.025)), float(np.nanquantile(m, 0.975))]
    sl = {c: per_scene(Y) for c, Y in curves.items()}
    wrong = lambda s: (float(np.mean(np.sign(s[np.isfinite(s)]) == -np.sign(law_value)))
                       if law_value != 0 and np.isfinite(s).any() else None)
    out = {c: {"slope": float(np.nanmedian(s)), "slope_ci95": ci(s), "n_scenes": int(np.isfinite(s).sum()),
               "per_scene": _quantiles(s), "wrong_way": wrong(s)} for c, s in sl.items()}
    out["law_value"] = float(law_value)
    if law in sl:
        for c in sl:
            if c != law:
                out[law][f"minus_{c}"] = {"median": float(np.nanmedian(sl[law] - sl[c])), "ci95": ci(sl[law] - sl[c])}
        L = out[law]
        ok = (L["slope_ci95"] is not None and L["slope_ci95"][0] <= law_value <= L["slope_ci95"][1]) or \
            abs(L["slope"] - law_value) <= tol
        diff = [L[k]["ci95"] for k in L if k.startswith("minus_") and k != "minus_input"]   # input: a reference
        differs = all(d and (d[0] > 0 or d[1] < 0) for d in diff) if diff else False
        out["verdict"] = {"matches_law": bool(ok), "differs_from_controls": bool(differs), "tol": tol,
                          "law_like": bool(ok and (differs or law_value == 0)),
                          "null_prediction": law_value == 0}
        if "input" in out and np.isfinite(out["input"]["slope"]) and out["input"]["slope"] != 0:
            out["verdict"]["steering_over_input"] = L["slope"] / out["input"]["slope"]     # does steering beat showing it
    return out


def step_scores(curves: dict, deltas, groups, theta0, boundary: float = 1.0, control: str = "random_dir",
                passive: str = "passive", n_boot: int = 1000, seed: int = 0, law: str = "law") -> dict:
    """A law that switches at a boundary (statics: the block tips iff x > 1). curves {condition: [G, n_steps] readout
    indicator (1 = tipped, NaN unmeasured)}, theta0 [G] each scene's own theta: the law's indicator at step d is
    [theta0 + d > boundary]. Per condition: accuracy = share of (scene, step) cells agreeing with the law; step =
    P(indicator | law says 1) - P(indicator | law says 0), pooled over scenes (law 1, an unresponsive readout 0);
    ``switch`` = share of scenes whose steps cross the boundary in which the indicator changes the same way; per-scene
    accuracy quantiles and ``wrong_way`` (share of crossing scenes whose indicator changes the opposite way); scene-
    bootstrap CIs. The law condition's step is compared with every other one (paired CI). ``law_like``: the step's CI
    excludes 0, it exceeds the control's (paired CI > 0) and passive propagation's step."""
    x = np.asarray(deltas, float)
    G = len(groups)
    tgt = (np.asarray(theta0, float)[:, None] + x[None]) > boundary
    B = np.random.default_rng(seed).integers(0, G, (n_boot, G)) if n_boot and G > 1 else None

    def step(Y, rows):
        Yr, Tr = Y[rows], tgt[rows]
        ok = np.isfinite(Yr)
        a, b = Yr[ok & Tr], Yr[ok & ~Tr]
        return float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")

    def ci(f):
        if B is None:
            return None
        v = np.array([f(r) for r in B])
        return [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] if np.isfinite(v).any() else None
    out, Ys = {}, {}
    for c, Y in curves.items():
        Y = np.asarray(Y, float)
        Ys[c] = Y
        ok = np.isfinite(Y)
        acc = np.array([np.mean((Y[i, ok[i]] > 0.5) == tgt[i, ok[i]]) if ok[i].any() else np.nan for i in range(G)])
        cross = [i for i in range(G) if tgt[i, ok[i]].any() and (~tgt[i, ok[i]]).any()]
        chg = [np.nanmean(Y[i, ok[i] & tgt[i]]) - np.nanmean(Y[i, ok[i] & ~tgt[i]]) for i in cross]
        allr = np.arange(G)
        out[c] = {"accuracy": float(np.nanmean(acc)) if np.isfinite(acc).any() else float("nan"),
                  "accuracy_ci95": ci(lambda r: np.nanmean(acc[r]) if np.isfinite(acc[r]).any() else np.nan),
                  "step": step(Y, allr), "step_ci95": ci(lambda r: step(Y, r)), "n_scenes": int(np.isfinite(acc).sum()),
                  "n_crossing": len(cross), "switch": float(np.mean(np.array(chg) > 0.5)) if chg else float("nan"),
                  "per_scene": _quantiles(acc), "wrong_way": float(np.mean(np.array(chg) < 0)) if chg else float("nan"),
                  "curve_median": np.nanmedian(Y, 0).tolist()}
    out["boundary"] = float(boundary)
    if law in Ys:
        for c in Ys:
            if c != law:
                out[law][f"minus_{c}"] = {"median": step(Ys[law], np.arange(G)) - step(Ys[c], np.arange(G)),
                                          "ci95": ci(lambda r: step(Ys[law], r) - step(Ys[c], r))}
        L = out[law]
        sc, cd = L["step_ci95"], L.get(f"minus_{control}", {}).get("ci95")
        out["verdict"] = {"steps_at_boundary": bool(sc and sc[0] > 0) if B is not None else bool(L["step"] > 0),
                          "differs_from_control": bool(cd and cd[0] > 0),
                          "exceeds_passive": bool(passive not in out or not np.isfinite(out[passive]["step"])
                                                  or L["step"] > out[passive]["step"]),
                          "control": control}
        out["verdict"]["law_like"] = all(out["verdict"][k] for k in ("steps_at_boundary", "differs_from_control",
                                                                     "exceeds_passive"))
    out["steps"] = x.tolist()
    return out


def resolve_law_cfg(lc: dict, theta) -> dict:
    """``kappa_ref: auto`` (rolling: kappa varies per group) -> the median theta of the fitted scenes."""
    return {**lc, "kappa_ref": float(np.median(theta))} if lc.get("kappa_ref") == "auto" else lc


def law_value(lc: dict) -> float:
    """The law's predicted slope (rolling: d log a / d kappa = -1 / (1 + kappa_ref))."""
    if lc["predict"] == "rolling":
        return -1.0 / (1.0 + float(lc["kappa_ref"]))
    return float(lc["predict"])


def score_response(curves: dict, deltas, groups, lc: dict, n_boot: int = 1000, theta0=None) -> dict:
    """The law's scoring: ``predict`` saturation (energy), step (statics; ``theta0`` = each scene's own theta), else
    a slope (a number, or ``rolling``)."""
    if lc["predict"] == "saturation":
        return saturation_scores(curves, deltas, groups, int(lc.get("n_top", 2)), float(lc.get("sat_tol", 0.1)),
                                 control=lc.get("control", "speed"), n_boot=n_boot)
    if lc["predict"] == "step":
        return step_scores(curves, deltas, groups, theta0, float(lc.get("boundary", 1.0)),
                           control=lc.get("control", "random_dir"), n_boot=n_boot)
    return slope_scores(curves, deltas, groups, law_value(lc), float(lc.get("tol", 0.25)), n_boot=n_boot)


def ols_slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(x[ok]) > 0 else float("nan")


def push_effect(theta_curves: dict, deltas, groups, real_slope: float, n_boot: int = 1000) -> dict:
    """Did the push move what it pushed? ``theta_curves`` {condition: [G, n_steps]}: the model's own theta readout
    (:func:`theta_probe`) of the steered representation per scene and dtheta. Per condition the readout's slope on
    dtheta (per scene, median, scene-bootstrap CI) and ``fraction_of_real`` = slope / ``real_slope`` (the same
    readout's slope on the true theta over real clips: the change real videos produce)."""
    sc = slope_scores(theta_curves, deltas, groups, 1.0, n_boot=n_boot, law="-")
    rs = float(real_slope) if np.isfinite(real_slope) and real_slope > 0 else float("nan")
    out = {c: {"slope": v["slope"], "slope_ci95": v["slope_ci95"], "fraction_of_real": v["slope"] / rs}
           for c, v in sc.items() if isinstance(v, dict) and "slope" in v}
    return {"real_slope": rs, "by_condition": out}


def null_verdict(res: dict, effect: dict | None, min_effect: float = 0.25, controls=("size", "speed")) -> dict:
    """Steering verdict of a law predicting NO change (slope 0), where an unchanged readout alone cannot pass (an inert
    push changes nothing either). ``res`` = :func:`slope_scores` output (its ``verdict``), ``effect`` =
    :func:`push_effect` (None: no theta readout). "fail" if the outcome moved (the law slope's CI excludes 0 beyond
    ``tol``: not ``matches_law``); else "pass" only with POSITIVE evidence that the push was effective: (a) the same
    push moves the model's theta readout (a readout the law says SHOULD move) by >= ``min_effect`` of its real-video
    change, or (b) a control direction (``controls`` present in ``res``: size, speed) moves the outcome (slope CI
    excludes 0); otherwise "inert push (uninformative)". Sets verdict ``result``, ``effective_push`` and ``law_like``
    (= "pass")."""
    v = res.get("verdict")
    if v is None:
        return res
    law_eff = ((effect or {}).get("by_condition") or {}).get("law") or {}
    fr = law_eff.get("fraction_of_real", float("nan"))
    a = bool(np.isfinite(fr) and fr >= min_effect)
    cm = {c: bool(res[c]["slope_ci95"] and (res[c]["slope_ci95"][0] > 0 or res[c]["slope_ci95"][1] < 0))
          for c in controls if isinstance(res.get(c), dict) and "slope" in res[c]}
    b = any(cm.values())
    v["effective_push"] = {"theta_readout_fraction_of_real": fr, "theta_readout_moved": a, "min_effect": min_effect,
                           "control_moves_outcome": cm, "effective": a or b}
    v["result"] = "fail" if not v["matches_law"] else "pass" if a or b else "inert push (uninformative)"
    v["law_like"] = v["result"] == "pass"
    if effect is not None:
        res["theta_readout"] = effect
    return res


# --------------------------------------------------------------------------- #
# theta, readouts
# --------------------------------------------------------------------------- #


def _eps(labels) -> float:
    from physlaw.eval import laws
    ev = event_inputs(labels)
    return float(laws.epsilon({"v": ev[0]}, {"v": ev[1]}, ev[2])) if ev else float("nan")


def label_value(labels, name: str) -> float:
    """theta / nuisance / control quantities from labels."""
    pl = labels["physlaw"]
    o = pl["objects"]
    if name == "epsilon":
        return _eps(labels)
    if name == "log_mass_ratio":
        return float(np.log(o["A"]["mass"] / o["B"]["mass"]))
    if name == "log_radius_ratio":
        return float(np.log(o["A"]["radius"] / o["B"]["radius"]))
    if name.startswith(("log_mass_", "log_radius_")):          # log_mass_A, log_radius_B, ...
        q, ob = name[4:].rsplit("_", 1)
        return float(np.log(o[ob][q]))
    if name == "impact_parameter_rel":                          # F12: |b| in bar half-lengths
        c = pl["constants"]
        return float(abs(c["impact_parameter"]) / c["bar_half_length"])
    if name == "post_speed":
        ev = event_inputs(labels)
        return float(np.linalg.norm(ev[1], axis=1).sum())
    if name == "kappa_A":
        return float(pl["constants"]["inertia_factors"]["A"])
    v = get(labels, name)
    if v is None:
        raise KeyError(f"L4: no value {name!r} in labels")
    return float(np.linalg.norm(v)) if isinstance(v, list) else float(v)


def log_deflection_ratio(vp, vq, ctx) -> float:
    """log(dv_B . n / -dv_A . n): momentum gives log(m_A / m_B) whatever e; NaN without a real exchange."""
    n = np.asarray(ctx["n"], float)[: vp.shape[-1]]
    a, b = -float((vq[0] - vp[0]) @ n), float((vq[1] - vp[1]) @ n)
    return float(np.log(b / a)) if a > 0 and b > 0 else float("nan")


# ---- momentum: the FITTED deflection ratio (no per-clip sign requirement, no clip discarded)
# Per clip the momentum readout is a point (a, b) = (-dv_A . n, dv_B . n); momentum says b = (m_A / m_B) a for every
# e. Both coordinates are decoded with noise of comparable size, so the ratio is fitted as an errors-in-variables
# (Deming) line through the origin, not as a per-clip log(b / a) (undefined, and the clip discarded, whenever a
# decoded a or b <= 0) nor as an OLS b ~ a (attenuated towards 0 by the noise in a). Deming residual of point i under
# ratio r: (b_i - r a_i) / sqrt(s_b^2 + r^2 s_a^2), s_a^2 / s_b^2 the instrument's decoding-error variances of a and
# b on real clips (lambda = s_b^2 / s_a^2; orthogonal TLS when equal); r = exp(c_unit + beta x) > 0, so the fitted
# log ratio needs no sign condition. Effects (steering, swap, input reference, real-video validity) are changes of
# the fitted log ratio along x, with group-bootstrap CIs.

RATIO_BOUND = 6.0          # |log r| and |beta| bounds of the fit (a ratio beyond e^6 is not a collision readout)


def deflection_ab(vp, vq, ctx) -> tuple[float, float]:
    """(a, b) = (-dv_A . n, dv_B . n) of one clip's (pre, post) velocities: the momentum readout's point."""
    vp, vq = np.asarray(vp, float), np.asarray(vq, float)
    n = np.asarray(ctx["n"], float)[: vp.shape[-1]]
    return -float((vq[0] - vp[0]) @ n), float((vq[1] - vp[1]) @ n)


def fit_log_ratio(a, b, x=None, unit=None, noise=(1.0, 1.0)) -> dict:
    """Deming fit of b = r a, log r = c_unit + beta x (``x`` None: no beta, one ratio per unit; ``unit`` None: one
    intercept). ``noise`` (s_a^2, s_b^2). Non-finite points are dropped (decoding failures only: no sign rule).
    Returns {"beta", "c" {unit: c}, "n_points", "units" [unique units]}."""
    from scipy.optimize import least_squares
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    x = np.zeros(len(a)) if x is None else np.asarray(x, float).ravel()
    has_beta = bool(np.ptp(x[np.isfinite(x)]) > 0) if np.isfinite(x).any() else False
    u = np.zeros(len(a), int) if unit is None else np.asarray(unit).ravel()
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(x)
    a, b, x, u = a[ok], b[ok], x[ok], u[ok]
    units, ui = np.unique(u, return_inverse=True)
    out = {"beta": float("nan"), "c": {}, "n_points": int(ok.sum()), "units": units.tolist()}
    if not len(a) or not np.any(np.abs(a) + np.abs(b) > 0):
        return out
    sa2, sb2 = (max(float(s), 1e-12) for s in noise)
    U = len(units)

    def res(p):
        lr = p[:U][ui] + (p[U] * x if has_beta else 0.0)
        r = np.exp(lr)
        return (b - r * a) / np.sqrt(sb2 + r * r * sa2)
    # start: each unit's TLS-free moment ratio sum|b| / sum|a| (positive), beta 0
    c0 = np.array([np.log(max(np.abs(b[ui == k]).sum(), 1e-9) / max(np.abs(a[ui == k]).sum(), 1e-9)) for k in range(U)])
    p0 = np.r_[np.clip(c0, -RATIO_BOUND + 1e-3, RATIO_BOUND - 1e-3), 0.0] if has_beta else \
        np.clip(c0, -RATIO_BOUND + 1e-3, RATIO_BOUND - 1e-3)
    lb, ub = np.full(len(p0), -RATIO_BOUND), np.full(len(p0), RATIO_BOUND)

    def jac(p):                                        # analytic: d res / d log r, sparse in the unit intercepts
        lr = p[:U][ui] + (p[U] * x if has_beta else 0.0)
        r = np.exp(lr)
        q = sb2 + r * r * sa2
        dr = (-a * np.sqrt(q) - (b - r * a) * r * sa2 / np.sqrt(q)) / q * r      # d res / d lr
        from scipy.sparse import csr_matrix
        rows = np.arange(len(a))
        if has_beta:
            return csr_matrix((np.r_[dr, dr * x], (np.r_[rows, rows], np.r_[ui, np.full(len(a), U)])),
                              shape=(len(a), U + 1))
        return csr_matrix((dr, (rows, ui)), shape=(len(a), U))
    sol = least_squares(res, p0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", tr_solver="lsmr")
    out["c"] = {(k.item() if hasattr(k, "item") else k): float(v) for k, v in zip(units, sol.x[:U])}
    if has_beta:
        out["beta"] = float(sol.x[U])
    return out


def deming_dist2(a, b, r, noise=(1.0, 1.0)) -> np.ndarray:
    """Squared Deming distance of points (a, b) to the line b = r a (in decoding-noise units)."""
    sa2, sb2 = (max(float(s), 1e-12) for s in noise)
    r = np.asarray(r, float)
    return (np.asarray(b, float) - r * np.asarray(a, float)) ** 2 / (sb2 + r * r * sa2)


def ratio_noise(dec_ab, true_ab) -> tuple[float, float]:
    """(s_a^2, s_b^2): the instrument's decoding-error variances of a and b on real clips (decoded vs label truth)."""
    d, t = np.asarray(dec_ab, float).reshape(-1, 2), np.asarray(true_ab, float).reshape(-1, 2)
    e = d - t
    ok = np.isfinite(e).all(1)
    if ok.sum() < 3:
        return 1.0, 1.0
    v = np.var(e[ok], 0)
    return float(max(v[0], 1e-12)), float(max(v[1], 1e-12))


def _scene_draws(G: int, n_boot: int, seed: int = 0):
    return np.random.default_rng(seed).integers(0, G, (n_boot, G)) if n_boot and G > 1 else None


def _ratio_beta(P: np.ndarray, x, scenes, noise) -> float:
    """Joint fit over the scenes ``scenes`` (indices into P [G, n_steps, 2], repeats = distinct units) of
    log r = c_scene + beta x."""
    pts = P[scenes]                                     # [g, n, 2]
    unit = np.repeat(np.arange(len(scenes)), pts.shape[1])
    xx = np.tile(np.asarray(x, float), len(scenes))
    return fit_log_ratio(pts[..., 0], pts[..., 1], xx, unit, noise)["beta"]


def ratio_slope_scores(curves: dict, deltas, groups, law_value: float, tol: float = 0.25, n_boot: int = 1000,
                       noise=(1.0, 1.0), seed: int = 0, law: str = "law") -> dict:
    """:func:`slope_scores` for the fitted-ratio readout. ``curves`` {condition: [G, n_steps, 2] points (a, b) per
    scene and dtheta}. Per condition: ``slope`` = beta of the joint Deming fit log r = c_scene + beta dtheta over every
    finite point (no clip discarded), scene-bootstrap CI (``slope_ci95``), per-scene betas (each scene's own fit) for
    the reliability quantiles / ``wrong_way``, ``n_points``. Paired law - condition differences on the same scene
    draws; the verdict as slope_scores'."""
    x = np.asarray(deltas, float)
    G = len(groups)
    B = _scene_draws(G, n_boot, seed)
    P = {c: np.asarray(Y, float).reshape(G, len(x), 2) for c, Y in curves.items()}
    full = {c: _ratio_beta(p, x, np.arange(G), noise) for c, p in P.items()}
    boot = {c: (np.array([_ratio_beta(p, x, r, noise) for r in B]) if B is not None else None) for c, p in P.items()}
    ci = lambda v: ([float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
                    if v is not None and np.isfinite(v).any() else None)
    out = {}
    for c, p in P.items():
        per = np.array([_ratio_beta(p, x, [g], noise) for g in range(G)])
        fin = np.isfinite(per)
        out[c] = {"slope": full[c], "slope_ci95": ci(boot[c]), "n_scenes": int(np.isfinite(p).all(-1).any(-1).sum()),
                  "n_points": int(np.isfinite(p).all(-1).sum()), "per_scene": _quantiles(per),
                  "wrong_way": (float(np.mean(np.sign(per[fin]) == -np.sign(law_value))) if law_value != 0 and fin.any()
                                else None),
                  "method": "Deming fit log(b / a) = c_scene + beta dtheta"}
    out["law_value"] = float(law_value)
    if law in P:
        for c in P:
            if c != law:
                d = None if B is None else boot[law] - boot[c]
                out[law][f"minus_{c}"] = {"median": full[law] - full[c], "ci95": ci(d)}
        L = out[law]
        ok = (L["slope_ci95"] is not None and L["slope_ci95"][0] <= law_value <= L["slope_ci95"][1]) or \
            abs(L["slope"] - law_value) <= tol
        diff = [L[k]["ci95"] for k in L if isinstance(L[k], dict) and k.startswith("minus_") and k != "minus_input"]
        differs = all(d and (d[0] > 0 or d[1] < 0) for d in diff) if diff else False
        out["verdict"] = {"matches_law": bool(ok), "differs_from_controls": bool(differs), "tol": tol,
                          "law_like": bool(ok and (differs or law_value == 0)), "null_prediction": law_value == 0}
        if "input" in out and np.isfinite(out["input"]["slope"]) and out["input"]["slope"] != 0:
            out["verdict"]["steering_over_input"] = L["slope"] / out["input"]["slope"]
    out["noise"] = {"s_a2": float(noise[0]), "s_b2": float(noise[1])}
    return out


def ratio_validity(ab, theta, law_slope: float, n_boot: int = 1000, noise=(1.0, 1.0)) -> dict:
    """Real-video validity of the fitted-ratio readout: log r = c + beta theta over the real clips (one per scene), one
    Deming fit, bootstrap CI over clips; passes when the CI covers the law's slope AND excludes 0."""
    P, th = np.asarray(ab, float).reshape(-1, 2), np.asarray(theta, float)
    fin = np.isfinite(P).all(1) & np.isfinite(th)
    f = lambda ix: fit_log_ratio(P[ix, 0], P[ix, 1], th[ix], None, noise)["beta"]
    rows = np.flatnonzero(fin)
    ci = _boot_ci(lambda ix: f(rows[ix]), np.arange(len(rows)), n_boot) if len(rows) >= 3 else None
    s = f(rows) if len(rows) >= 3 else float("nan")
    return {"slope": s, "ci95": ci, "law_slope": float(law_slope), "n": int(len(rows)), "n_clips": int(len(P)),
            "covers_law": bool(ci and ci[0] <= law_slope <= ci[1]), "excludes_0": bool(ci and (ci[0] > 0 or ci[1] < 0)),
            "valid": bool(ci and ci[0] <= law_slope <= ci[1] and (ci[0] > 0 or ci[1] < 0)),
            "method": "Deming fit log(b / a) = c + beta theta over the real clips", "noise": list(map(float, noise))}


def ratio_interchange_scores(base: dict, pts: list, req, groups, noise=(1.0, 1.0), n_boot: int = 1000) -> dict:
    """Swap test for the fitted-ratio readout. ``base`` {scene: (a, b)} the unswapped forecast's point, ``pts`` the
    swapped points (a, b) per pair, ``req`` the law's change of log r per pair, ``groups`` each pair's scene. Joint
    Deming fit log r = c_scene + kappa x (x = 0 for the base point, req for the swapped ones): ``fraction_achieved`` =
    kappa (scene-bootstrap CI). Per pair, with the scene's fitted base ratio r0 = e^{c_scene}: IIA = share of pairs
    whose swapped point lies closer (Deming distance) to the law's line r0 e^{req} than to the unchanged line r0
    (group-bootstrap CI), ``error_ratio`` = median distance to the law's line / median distance to the unchanged
    line, the paired sign-flip p of their difference. Sign-free: no point is discarded."""
    g = np.asarray(groups)
    P, rq = np.asarray(pts, float).reshape(-1, 2), np.asarray(req, float)
    sc = sorted(base)
    out = {"n_pairs": int((np.isfinite(P).all(1) & np.isfinite(rq)).sum())}
    if not out["n_pairs"]:
        return out

    def fit(scs):
        a, b, x, u = [], [], [], []
        for k, s in enumerate(scs):
            a.append(base[s][0]); b.append(base[s][1]); x.append(0.0); u.append(k)
            for p in np.flatnonzero(g == s):
                a.append(P[p, 0]); b.append(P[p, 1]); x.append(rq[p]); u.append(k)
        return fit_log_ratio(a, b, x, u, noise)
    f = fit(sc)
    c = {s: f["c"].get(k, np.nan) for k, s in enumerate(sc)}
    r0 = np.exp(np.array([c.get(s, np.nan) for s in g]))
    d_req, d_0 = np.sqrt(deming_dist2(P[:, 0], P[:, 1], r0 * np.exp(rq), noise)), np.sqrt(deming_dist2(P[:, 0], P[:, 1], r0, noise))
    ok = np.isfinite(d_req) & np.isfinite(d_0) & (np.abs(rq) > 1e-12)
    hit = np.where(ok, (d_req < d_0).astype(float), np.nan)
    m = lambda ix: _nan_guard(np.nanmean, hit[ix])
    B = _scene_draws(len(sc), n_boot)
    kb = np.array([fit([sc[i] for i in r])["beta"] for r in B]) if B is not None else None
    out.update(iia=m(np.arange(len(hit))), iia_ci95=_boot_ci(m, g, n_boot), fraction_achieved=f["beta"],
               fraction_achieved_ci95=([float(np.nanquantile(kb, 0.025)), float(np.nanquantile(kb, 0.975))]
                                       if kb is not None and np.isfinite(kb).any() else None),
               error_ratio=float(np.median(d_req[ok]) / np.median(d_0[ok])) if ok.any() else float("nan"),
               p_sign_flip=sign_flip_p(d_req[ok] - d_0[ok]), beats_do_nothing=float(np.mean(d_req[ok] < d_0[ok])),
               abs_change=float(np.nanmedian(d_0[ok])) if ok.any() else float("nan"),
               method="Deming fit log(b / a) = c_scene + kappa required; IIA by Deming distance")
    return out


def _r_eps(vp, vq, ctx):
    from physlaw.eval import laws
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(laws.epsilon({"v": vp}, {"v": vq}, ctx))


def _log(x) -> float:
    return float(np.log(x)) if np.isfinite(x) and x > 0 else float("nan")


def _along(post, o: int) -> float:
    """Rate of change of speed of object o (a . v_hat) from a kinematic state."""
    a, v = np.asarray(post["a"], float)[o], np.asarray(post["v"], float)[o]
    s = np.linalg.norm(v)
    return float(a @ v / s) if s > 0 else float(np.linalg.norm(a))


def _mass_ratio(pre, post, ctx) -> float:
    from physlaw.eval import laws
    return laws.gravity.stats(pre, post, {**ctx, "dt": 0.0})["mass_ratio"]


def _perp(v, ctx, o: int = 0) -> float:
    """|component of object o's post velocity perpendicular to gravity| (horizontal speed)."""
    g = np.asarray(ctx["g"], float)[: len(v[o])]
    g = g / np.linalg.norm(g)
    x = np.asarray(v[o], float)
    return float(np.linalg.norm(x - (x @ g) * g))


def _tangential_B(pre, post, ctx) -> float:
    n = np.asarray(ctx["n"], float)[:2]
    return float(np.asarray(post["v"], float)[1][:2] @ np.array([-n[1], n[0]]))


def sine_period(t, V) -> float:
    """Period of the sinusoid c0 + c1 sin(w t) + c2 cos(w t) best fitting (least squares, grid over w) the component
    of V [T, d] with the largest variance (a pendulum bob's horizontal velocity; its vertical one oscillates at 2 w).
    The grid spans periods from 4 x the window (at least a quarter period must be seen) down to the Nyquist limit.
    NaN with < 4 finite rows."""
    t = np.asarray(t, float)
    if len(t) < 4:
        return float("nan")
    V = np.asarray(V, float).reshape(len(t), -1)
    ok = np.isfinite(V).all(1)
    t, V = t[ok], V[ok]
    if len(t) < 4 or np.ptp(t) <= 0:
        return float("nan")
    y = V[:, int(np.argmax(V.var(0)))]
    span, dt = float(np.ptp(t)), float(np.median(np.diff(np.sort(t))))
    best = (np.inf, float("nan"))
    for w in np.geomspace(np.pi / (2 * span), np.pi / max(dt, 1e-12), 600):
        A = np.stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)], 1)
        r = float(((A @ np.linalg.lstsq(A, y, rcond=None)[0] - y) ** 2).sum())
        if r < best[0]:
            best = (r, w)
    return 2 * np.pi / best[1]


def rotation_angle(t, W) -> float:
    """Rotation angle (rad) over the rows: trapezoid integral of |w| (W [T, 3]) over t; 0 with < 2 rows."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0.0
    s = np.linalg.norm(np.asarray(W, float).reshape(len(t), -1), axis=1)
    return float(0.5 * ((s[1:] + s[:-1]) * np.diff(t)).sum())


def series_extras(t, Q: dict, sel, te: float, w_tip: float = float("nan")) -> dict:
    """Per object, from decoded per-token-frame series over the rows ``sel``: "T" the sinusoid period of the velocity
    (:func:`sine_period`), "phi" the rotation angle from te to te + ``w_tip`` (:func:`rotation_angle`; needs Q["w"];
    ``w_tip`` = laws.incline.tip_window, the tip window in the clip's own time unit; NaN when undefined)."""
    t, sel = np.asarray(t, float), np.asarray(sel, bool)
    n = Q["v"].shape[1]
    out = {"T": np.array([sine_period(t[sel], Q["v"][sel, o]) for o in range(n)])}
    if "w" in Q:
        w = sel & (t >= te) & (t <= te + w_tip + 1e-9)
        out["phi"] = np.array([rotation_angle(t[w], Q["w"][w, o]) if np.isfinite(w_tip) else float("nan")
                               for o in range(n)])
    return out


def clip_tip_window(ctx: dict) -> float:
    """laws.incline.tip_window of one clip from its ctx (g_clip and the half-length measured in the clip, the time from
    the release to leaving the ledge, else the twin-shared time to the group's clean-window end); NaN off F11."""
    from physlaw.eval.laws.incline import tip_window
    if "half_length" not in ctx:
        return float("nan")
    te, ts = float(ctx.get("event_time_s", 0.0)), ctx.get("t_separate_s")
    end = ctx.get("group_clean_end_s", ctx.get("clean_until_s"))
    return tip_window(ctx.get("g_clip"), ctx["half_length"], None if ts is None else float(ts) - te,
                      None if end is None or not np.isfinite(float(end)) else float(end) - te)


def _tipped(q) -> float:
    from physlaw.eval.laws.incline import TIP_ANGLE
    phi = np.asarray(q.get("phi", [np.nan]), float)[0]
    return float(phi > TIP_ANGLE) if np.isfinite(phi) else float("nan")


def _L_ratio(p, q, c) -> float:
    """L_ratio of the decoded states: velocities (and the bar's spin) decoded, positions at the contact instant from
    the labels (ctx["p_event"]); nothing spins before the contact."""
    from physlaw.eval import laws
    P = c.get("p_event")
    if P is None or "w" not in q:
        return float("nan")
    v0 = np.asarray(p["v"], float)
    return laws.angular_momentum.stats({"p": P, "v": v0, "w": np.zeros_like(v0)}, {"p": P, "v": q["v"], "w": q["w"]},
                                       c)["L_ratio"]


def _momentum_norm(q, c) -> float:
    """|sum m v'| (2-D): the system's linear momentum, which no change of the impact parameter may move."""
    m = np.asarray(c["m"], float).reshape(-1)
    return float(np.linalg.norm((m[:, None] * np.asarray(q["v"], float)[:, :2]).sum(0)))


def _row_first(r, key) -> float:
    x = r.get(key)
    return float(np.asarray(x, float).reshape(-1)[0]) if x is not None else float("nan")


def _cm_speed(pre, post, ctx) -> float:
    if ctx.get("event") != "collision":
        return _perp(post["v"], ctx) if ctx.get("event") == "bounce" else float("nan")
    m = np.asarray(ctx["m"], float)
    return float(np.linalg.norm((m[:, None] * np.asarray(post["v"], float)).sum(0) / m.sum()))


# readout name -> (from decoded states (pre, post, ctx), from a track-B row, the per-token-frame quantities it needs:
# () = the frame.post velocity pool)
READOUTS = {
    "epsilon": (lambda p, q, c: _r_eps(p["v"], q["v"], c), lambda r: r.get("epsilon", float("nan")), ()),
    "log_deflection_ratio": (lambda p, q, c: log_deflection_ratio(p["v"], q["v"], c), lambda r: log_deflection_ratio(
        np.asarray(r["velocity_pre"]), np.asarray(r["velocity_post"]), {"n": r["contact_normal"]})
        if "contact_normal" in r else float("nan"), ()),
    "log_mass_ratio_accel": (lambda p, q, c: _log(_mass_ratio(p, q, c)),
                             lambda r: _log(r.get("mass_ratio", float("nan"))), ("v",)),
    "log_accel": (lambda p, q, c: _log(_along(q, 0)), lambda r: _log(_row_first(r, "accel")), ("v", "w")),
    "log_decel_A": (lambda p, q, c: _log(-_along(q, 0)), lambda r: _log(-_row_first(r, "accel")), ("v",)),
    "log_period_A": (lambda p, q, c: _log(np.asarray(q.get("T", [np.nan]), float)[0]),
                     lambda r: _log(_row_first(r, "period_post")), ("v",)),
    "tipped": (lambda p, q, c: _tipped(q), lambda r: float(r["tipped"]) if "tipped" in r else float("nan"), ("v", "w")),
    "L_ratio": (_L_ratio, lambda r: r.get("L_ratio", float("nan")), ("v", "w")),
}
# non-target quantity per law (selectivity of the swap / patch arms): what the law says the intervention must NOT move
NONTARGET = {"momentum": _tangential_B,                          # impulse along n: B's tangential velocity is kept
             "energy": _cm_speed,                                 # eps leaves the CoM velocity (bounce: v_horizontal)
             "gravity": lambda p, q, c: _perp(q["v"], c),          # axis independence: horizontal velocity
             "friction": lambda p, q, c: _log(-_along(q, 1)),      # A's mass leaves B's deceleration
             "oscillation": lambda p, q, c: _log(np.asarray(q.get("T", [np.nan, np.nan]), float)[-1]),  # B's period
             "angular_momentum": lambda p, q, c: _momentum_norm(q, c),   # the impact parameter leaves sum m v
             "rolling": lambda p, q, c: _log(_along(q, 1))}        # A's kappa leaves B's acceleration


def _states_of(labels: list[dict]) -> list[tuple]:
    """(pre, post, ctx) per clip for the readouts: the event's planar velocities (event_inputs) where the clip has an
    exchange event; otherwise io.law_inputs' 3-D states (aux bodies dropped) and ctx (masses, g, k, r, event time,
    clean window; ``p_event`` = the positions at the event, which angular momentum takes from the labels)."""
    from physlaw.eval.l1_forecast_laws import clip_states
    out = []
    for l in labels:
        ev = event_inputs(l)
        if ev is not None:
            out.append(({"v": ev[0]}, {"v": ev[1]}, ev[2]))
        else:
            pre, post, ctx, _ = clip_states(l)
            out.append((pre, post, {**ctx, "p_event": pre.get("p")}))
    return out


# --------------------------------------------------------------------------- #
# decoders: the R1 instrument on steered forecasts (encoders, DiT activations)
# --------------------------------------------------------------------------- #


def _dec(folds, Z, g, t=None) -> np.ndarray:
    """Rows Z ([D] or [m, D]) of one clip of group g through cross-fitted folds."""
    Z = np.atleast_2d(np.asarray(Z, float))
    return apply_instrument(folds, Z, np.full(len(Z), g), t)


def theta_probe(Xq, theta, grp, k: int, alphas):
    """The model's own readout of theta: a probe (shared R1 machinery, cross-fitted by group) from real post-event
    features Xq [n, D] to theta [n] (the swap test's "did the swap move the swapped variable" check). Returns
    (i, z [D]) -> theta_hat of clip i's features z (rows are averaged first: per-token-frame features)."""
    folds = instrument_folds({"x": np.asarray(Xq, float)}, np.asarray(theta, float)[:, None], grp, None, k, 1, alphas)
    return lambda i, z, t=None: float(_dec(folds, np.atleast_2d(np.asarray(z, float)).mean(0), grp[i], t)[0, 0])


def pooled_reader(real: dict, fc: dict, Vp, Vq, grp, k: int, alphas, mm: bool = False, post_rows=None, site=None,
                  theta=None) -> dict:
    """Velocity readouts from the frame.post pool, one instrument per object, cross-fitted by group (a clip is read
    by a fold that never saw its group). Post rows: ``post_rows`` (the model's pred-real FINAL outputs for the real
    post frames: the forecast's own space; then separate pre / post probes, as L1's pred_real instrument) or the real
    frame.post pool (R1 joint with frame.pre). ``mm``: the forecast moment-matched onto the real post rows first
    (diagnostic). ``site`` [n, D]: the real clips' activations at a post-event steering site -> "site_post" (passive
    propagation: the shifted unsteered site read in its own space); ``theta`` [n] -> "theta" (theta_probe).
    {"pre": i -> {"v"}, "post": (i, z [D]) -> {"v"}, "floor": i -> {"v"}, ...}."""
    from physlaw.probes.ridge import moment_match
    Xa = real["frame.pre"]
    Xq = real["frame.post"] if post_rows is None else np.asarray(post_rows, float)
    n_obj = Vp.shape[1]
    fit = lambda X, Y: instrument_folds({"x": X}, Y, grp, None, k, 1, alphas)
    if post_rows is None:
        pre_f = post_f = [fit((Xa, Xq), (Vp[:, o], Vq[:, o])) for o in range(n_obj)]
    else:
        pre_f, post_f = [fit(Xa, Vp[:, o]) for o in range(n_obj)], [fit(Xq, Vq[:, o]) for o in range(n_obj)]
    tf = moment_match(fc["frame.post"], Xq, grp) if mm else None
    dec = lambda F, Z, i, t=None: np.stack([_dec(f, Z, grp[i], t)[0] for f in F])
    out = {"pre": lambda i: {"v": dec(pre_f, Xa[i], i)}, "post": lambda i, z: {"v": dec(post_f, z, i, tf)},
           "floor": lambda i: {"v": dec(post_f, Xq[i], i)}, "field": ""}
    if site is not None:
        site_f = [fit(np.asarray(site, float), Vq[:, o]) for o in range(n_obj)]
        out["site_post"] = lambda i, z: {"v": dec(site_f, z, i)}
    if theta is not None:
        th = theta_probe(Xq, theta, grp, k, alphas)
        out["theta"] = lambda i, z: th(i, z, tf)
        out["theta_floor"] = lambda i: th(i, Xq[i])            # the real clip's own features
    return out


def frames_reader(space: dict, fr: dict, grp, states, readout: str, k: int, alphas, theta=None) -> dict:
    """Kinematic readouts from per-token-frame pools: L1's per-token-frame instrument per object and quantity
    (READOUTS[readout][2]), least-squares lines over the free token-frames of the windows
    (l1_forecast_laws.clip_kinematics), plus each window's sinusoid period "T" and rotation angle "phi"
    (:func:`series_extras`); ``theta`` [n] -> "theta" (theta_probe on each clip's mean real post token-frame).
    {"pre": i -> state, "post": (i, z [T_fc, D]) -> state, "floor": i -> state}."""
    from physlaw.eval.l1_forecast_laws import clip_kinematics, token_decode, token_instrument
    n_obj = fr["V"][0].shape[1]
    qs = READOUTS[readout][2]
    inst = {q: [token_instrument(space["real"], [y[:, o] for y in fr[q.upper()]], fr["phase"], grp, k, 1, alphas)
                for o in range(n_obj)] for q in qs}
    real = {q: [np.stack([inst[q][o]["real"][i] for o in range(n_obj)], 1) for i in range(len(grp))] for q in qs}
    te = [float(s[2].get("event_time_s", 0.0)) for s in states]
    tend = [min(float(s[2].get("clean_until_s") or np.inf), float(s[2].get("move_end", np.inf)))  # F6: moving only
            for s in states]
    wt = [clip_tip_window(s[2]) for s in states]
    ok = fr.get("ok") or [None] * len(grp)
    ok_fc = fr.get("ok_fc") or [None] * len(grp)
    free = lambda m, n: np.ones(n, bool) if m is None else np.asarray(m, bool)[:n]

    def kin(i, tq, Q, ph=None, okq=None):
        r = clip_kinematics(fr["t"][i], real["v"][i], fr["phase"][i], tq, Q["v"], Q.get("w"), te[i], tend[i], ok[i], ph,
                            None, okq)
        r = r or {"pre": {"v": np.full((n_obj, 3), np.nan), "a": np.full((n_obj, 3), np.nan)},
                  "post": {"v": np.full((n_obj, 3), np.nan), "a": np.full((n_obj, 3), np.nan)}}
        tq, t = np.asarray(tq, float), np.asarray(fr["t"][i], float)
        s_post = (tq >= te[i]) & (tq <= tend[i] + 1e-9) & free(okq, len(tq)) & (True if ph is None else np.asarray(ph) == "post")
        s_pre = (np.asarray(fr["phase"][i]) == "pre") & (t < te[i]) & free(ok[i], len(t))
        r["post"].update(series_extras(tq, Q, s_post, te[i], wt[i]))
        r["pre"].update(series_extras(t, {q: real[q][i] for q in qs}, s_pre, te[i], wt[i]))
        return r

    def post(i, z):
        z = np.asarray(z, float).reshape(-1, np.asarray(z).shape[-1])
        Q = {q: np.stack([token_decode(inst[q][o], z, grp[i]) for o in range(n_obj)], 1) for q in qs}
        return kin(i, fr["t_fc"][i][: len(z)], Q, None, None if ok_fc[i] is None else ok_fc[i][: len(z)])["post"]
    realq = lambda i: kin(i, fr["t"][i], {q: real[q][i] for q in qs}, fr["phase"][i], ok[i])
    out = {"pre": lambda i: realq(i)["pre"], "post": post, "floor": lambda i: realq(i)["post"], "field": "|t"}
    if theta is not None:
        Xq = np.stack([x[np.asarray(ph) == "post"].mean(0) for x, ph in zip(space["real"], fr["phase"])])
        out["theta"] = theta_probe(Xq, theta, grp, k, alphas)
        out["theta_floor"] = lambda i: out["theta"](i, Xq[i])
    return out


def decoders(cache_dir, data_dir, recs, labels, mcfg, states, grp, readout: str, k: int, alphas, theta=None,
             site=None, site_final=None) -> dict:
    """{variant: reader} for the decoded arms: velocity readouts -> "frame" and "frame_mm" per forecast space
    (frame.pre / frame.post pools; encoders' post rows = the pred-real FINAL outputs, ``site_final`` from the L4 site
    pass or stage 4's .pred_real.npz sidecars, when present); kinematic readouts -> "frame" per space from
    per-token-frame pools (stage 4 pooled_frame_t; re-run stage 4 when absent). ``site``: the real clips' activations
    at a post-event steering site (passive propagation's reader); ``theta``: the swap test's theta readout."""
    out = {}
    if READOUTS[readout][2]:
        from physlaw.eval.l1_forecast_laws import clip_segment, load_token_frames
        clips = [r["clip"] for r in recs]
        objs = list(states[0][0].get("objects") or sorted(n for n, o in labels[0]["physlaw"]["objects"].items()
                                                         if not o.get("aux")))
        sp, fr = load_token_frames(cache_dir, recs, clips, dict(zip(clips, labels)), mcfg,
                                   {c: clip_segment(l) for c, l in zip(clips, labels)}, objs)
        for name, S_ in sp.items():
            out[name] = frames_reader(S_, fr, grp, states, readout, k, alphas, theta)
        return out
    Vp = np.stack([s[0]["v"] for s in states])
    Vq = np.stack([s[1]["v"] for s in states])
    if site_final is None:
        prf = [Path(cache_dir) / r["file"].replace(".npz", ".pred_real.npz") for r in recs]
        if prf and all(f.exists() for f in prf):
            zp = [np.load(f) for f in prf]
            pp = [str(x) for x in zp[0]["pools"]]
            site_final = np.stack([z["pooled"][pp.index("frame.post")] for z in zp]) if "frame.post" in pp else None
    for name, (real, fc) in forecast_spaces(cache_dir, data_dir, recs, mcfg, list(POOLS)).items():
        pr = site_final if name == "final" else None
        for mm in (False, True):
            out[name + ("_mm" if mm else "")] = {**pooled_reader(real, fc, Vp, Vq, grp, k, alphas, mm, pr, site, theta),
                                                 "space": name}
    return out


# --------------------------------------------------------------------------- #
# pairs: the swap test (subspace interchange) and the causal collision patch (full ROI)
# --------------------------------------------------------------------------- #


def curve_basis(man: dict) -> np.ndarray:
    """Orthonormal basis [D, r] of the steering curve's span (the non-constant polynomial coefficient rows)."""
    C = np.asarray(man["coef"], float)[:-1]
    C = C[np.linalg.norm(C, axis=1) > 1e-12]
    if not len(C):
        return np.zeros((np.asarray(man["coef"]).shape[-1], 0))
    U, s, _ = np.linalg.svd(C.T, full_matrices=False)
    return U[:, s > 1e-9 * s.max()]


def swap_delta(x_base: np.ndarray, x_donor: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """Interchange intervention: the base's component in span(Q) replaced by the donor's (Q None: the whole vector,
    i.e. the full-ROI patch)."""
    d = np.asarray(x_donor, float) - np.asarray(x_base, float)
    return d if Q is None else Q @ (Q.T @ d)


def pick_donors(i, rows, theta, scene, n: int, min_dtheta: float, rng, mode: str, ok=None) -> list[int]:
    """Donor clips for base clip i: ladder -> other rungs of its own scene; across -> other scenes' rows; both with
    |theta_d - theta_i| >= min_dtheta and ``ok(j)`` (e.g. matched contact geometry)."""
    cand = [j for j in rows if j != i and abs(theta[j] - theta[i]) >= min_dtheta and (ok is None or ok(j))
            and ((scene[j] == scene[i]) if mode == "ladder" else (scene[j] != scene[i]))]
    return [int(j) for j in rng.choice(cand, min(n, len(cand)), replace=False)] if cand else []


def contact_token_frame(cache_dir, rec) -> int:
    """Index of the clip's contact token-frame (the first token-frame not entirely pre-contact; stage 4's ``phases``
    in the clip's cache file, in the evaluated model's token grid). A cache without it is an error (the patch arm
    cannot align donors): re-run stage 4."""
    f = Path(cache_dir) / str(rec.get("file", ""))
    z = np.load(f) if rec.get("file") and f.is_file() else None
    if z is None or "phases" not in z.files:
        raise ValueError(f"L4 patch: {f} has no token-frame `phases` (needed to match donors' contact token-frame); "
                         f"re-run stage 4 (scripts/04_cache_activations.py) for this model and family")
    ph = [str(p) for p in z["phases"]]
    return next((i for i, p in enumerate(ph) if p != "pre"), len(ph))


def patch_donors(i, cands, theta, scene, n: int, rng, states, cos_min: float, ctf: dict) -> tuple[list[int], int]:
    """Collision-patch donors of base clip i: other scenes' clips with a matched contact normal (n_i . n_j >=
    ``cos_min``) whose contact falls in the SAME token-frame as the base's (``ctf`` {row: contact token-frame}), so the
    swapped ROI is complete (the same context token-frames) and time-aligned. Returns (donors, number of otherwise
    eligible candidates excluded by the token-frame rule)."""
    n_i = np.asarray(states[i][2]["n"], float)[:2]
    geom = lambda j: float(np.asarray(states[j][2]["n"], float)[:2] @ n_i) >= cos_min
    same = lambda j: ctf[j] == ctf[i]
    eligible = [j for j in cands if j != i and scene[j] != scene[i] and geom(j)]
    return pick_donors(i, cands, theta, scene, n, 0.0, rng, "across", lambda j: geom(j) and same(j)), \
        sum(1 for j in eligible if not same(j))


def sign_flip_p(d, n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip test of mean(d) = 0."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    s = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(d)))
    return float((1 + np.sum(np.abs((s * d).mean(1)) >= abs(d.mean()))) / (n + 1))


def interchange_scores(obs, req, nt=None, sd_t: float = 1.0, sd_nt: float = 1.0) -> dict:
    """Swap-test scores from per-pair readout changes ``obs`` = r(swapped) - r(base) and the law's required change
    ``req`` = pred(theta') - pred(theta): IIA = share of pairs closer to the law's outcome at theta' than at theta
    (|obs - req| < |obs|), fraction of the required change achieved (median obs / req), error ratio median|req - obs|
    / median|req| with the paired sign-flip p of |req - obs| - |req| (the do-nothing baseline); selectivity: median
    |target change| / sd over median |non-target change| / sd (``nt``: per-pair non-target changes)."""
    obs, req = np.asarray(obs, float), np.asarray(req, float)
    ok = np.isfinite(obs) & np.isfinite(req) & (np.abs(req) > 1e-12)
    out = {"n_pairs": int(ok.sum())}
    if ok.any():
        em, en = np.abs(req[ok] - obs[ok]), np.abs(req[ok])
        out.update(iia=float(np.mean(em < np.abs(obs[ok]))), fraction_achieved=float(np.median(obs[ok] / req[ok])),
                   error_ratio=float(np.median(em) / np.median(en)), p_sign_flip=sign_flip_p(em - en),
                   beats_do_nothing=float(np.mean(em < en)))
    out["abs_change"] = float(np.nanmedian(np.abs(obs))) if np.isfinite(obs).any() else float("nan")
    if nt is not None:
        nt = np.asarray(nt, float)
        t_, n_ = np.nanmedian(np.abs(obs)) / sd_t, np.nanmedian(np.abs(nt)) / sd_nt if np.isfinite(nt).any() else np.nan
        out["selectivity"] = {"target_change_sd": float(t_), "nontarget_change_sd": float(n_),
                              "ratio": float(t_ / n_) if np.isfinite(n_) and n_ > 0 else float("nan")}
    return out


def invariance_scores(obs, moved, dtheta, ref: float, ref_kind: str, min_moved: float = 0.25,
                      max_ratio: float = 0.25, real_slope: float = 1.0, control_moved: bool = False) -> dict:
    """Swap test for a law that predicts NO change of the readout (slope 0: gravity, friction, oscillation, angular
    momentum, Galilean), where the required change is 0 and IIA / fraction are undefined. Per pair ``obs`` = the
    outcome's change r(swapped) - r(base), ``moved`` = the change of the model's own readout of theta (theta_probe;
    None: not readable, e.g. generated videos) and ``dtheta`` = theta_donor - theta_base. Scores: the invariance ratio
    median |obs| / ``ref`` (``ref``: median |change of the outcome| under a change the law says DOES move it, or the
    sd of the outcome over real clips; ``ref_kind`` says which), and the readout-moved check median(moved / dtheta) >=
    ``min_moved`` (the swap did change what the model represents of theta: without it invariance is vacuous), as a
    fraction of the real-video change (``real_slope``: the theta readout's slope on the true theta over real clips).
    ``result`` (the slope-0 rule, as :func:`null_verdict`): "fail" if the outcome moved (ratio > ``max_ratio``), else
    "pass" if the swap was effective (the theta readout moved, or ``control_moved``: a control direction at the same
    site moves the outcome), else "inert swap (uninformative)"; ``law_like`` = "pass"."""
    obs = np.asarray(obs, float)
    ok = np.isfinite(obs)
    out = {"n_pairs": int(ok.sum()), "reference": float(ref), "reference_kind": ref_kind}
    if not ok.any():
        return out
    out["abs_change"] = float(np.median(np.abs(obs[ok])))
    out["invariance_ratio"] = out["abs_change"] / ref if np.isfinite(ref) and ref > 0 else float("nan")
    fr = np.full(len(obs), np.nan)
    if moved is not None:
        mv, dt = np.asarray(moved, float), np.asarray(dtheta, float)
        good = np.isfinite(mv) & (np.abs(dt) > 1e-12)
        fr[good] = mv[good] / dt[good]
    rs = float(real_slope) if np.isfinite(real_slope) and real_slope > 0 else float("nan")
    out["readout_moved_fraction"] = float(np.nanmedian(fr)) / rs if np.isfinite(fr).any() else float("nan")
    out["readout_moved"] = bool(out["readout_moved_fraction"] >= min_moved) if np.isfinite(fr).any() else None
    out["invariant"] = bool(np.isfinite(out["invariance_ratio"]) and out["invariance_ratio"] <= max_ratio)
    out["control_moved"] = bool(control_moved)
    out["result"] = ("uninformative (no invariance reference)" if not np.isfinite(out["invariance_ratio"]) else
                     "fail" if not out["invariant"] else
                     "pass" if out["readout_moved"] or control_moved else "inert swap (uninformative)")
    out["law_like"] = out["result"] == "pass"
    out.update(min_moved=min_moved, max_ratio=max_ratio, theta_readout_real_slope=rs)
    return out


def patch_along_n(dvB_obs, vA_base, vA_donor, ctxs, k_gain, k_donor=None, vB_base=None):
    """Per collision-patch pair, along the base's contact normal n: (observed change of B's post velocity obs . n, the
    momentum-required change, du . n = (v_A,donor - v_A) . n). Required: B's post normal velocity is
    v_B . n + k (v_A - v_B) . n, k = (1 + e) m_A / (m_A + m_B); the full-ROI patch brings the donor's A (its velocity
    AND its mass), so req = k_donor (v_A,donor - v_B) . n - k (v_A - v_B) . n (``k_donor``: the gain with the donor's
    m_A against the base's B and e; ``vB_base``: B's pre velocity); without them k (du . n) (A's mass unchanged)."""
    on, rq, du = [], [], []
    kd = k_gain if k_donor is None else k_donor
    vb = [np.zeros(2)] * len(k_gain) if vB_base is None else vB_base
    for o, a, b, c, kk, k2, w in zip(dvB_obs, vA_base, vA_donor, ctxs, k_gain, kd, vb):
        n = np.asarray(c["n"], float)[:2]
        a, b, w = (np.asarray(x, float)[:2] @ n for x in (a, b, w))
        du.append(float(b - a))
        rq.append(float(k2) * (b - w) - float(kk) * (a - w))
        on.append(float(np.asarray(o, float)[:2] @ n))
    return np.array(on), np.array(rq), np.array(du)


def patch_closer(dvB_obs, vA_base, vA_donor, ctxs, k_gain, k_donor=None, vB_base=None) -> np.ndarray:
    """Per pair d = |obs - 0| - |obs - req| along n: > 0 when B's observed change is closer to the momentum-required
    change than to no change (the collision patch's criterion (i))."""
    on, rq, _ = patch_along_n(dvB_obs, vA_base, vA_donor, ctxs, k_gain, k_donor, vB_base)
    return np.abs(on) - np.abs(on - rq)


def _slope_ratio(du, on, hi, ix) -> float:
    """OLS slope of obs . n on du . n in the high-k pairs over that in the low-k pairs, over the pair indices ``ix``
    (repeats kept: a bootstrap draw)."""
    def sl(m):
        j = ix[m[ix]]
        return float(np.polyfit(du[j], on[j], 1)[0]) if len(j) >= 3 and np.ptp(du[j]) > 0 else float("nan")
    lo, h = sl(~hi), sl(hi)
    return h / lo if np.isfinite(lo) and lo != 0 else float("nan")


def patch_scores(dvB_obs, vA_base, vA_donor, ctxs, k_gain, nt=None, sd_t=1.0, sd_nt=1.0, groups=None,
                 n_boot: int = 1000, k_donor=None, vB_base=None) -> dict:
    """Causal collision patch (momentum, the predecessor's §41 recipe): A's pre-contact ROI replaced by a donor's;
    physics requires dv_B' = k ((v_A,donor - v_A) . n) n, k = (1 + e) mu / m_B (contact geometry held fixed: the local
    linearisation; with ``k_donor`` / ``vB_base`` the donor's A mass too, :func:`patch_along_n`). Per pair the observed change of B's decoded post velocity ``dvB_obs`` [P, 2]. Scores: error ratio
    median |req - obs| / median |req| (1 = do nothing), paired sign-flip p of |req - obs| - |req|, IIA (closer to the
    required outcome than to the unpatched one), and the high-k / low-k slope-ratio discriminator: OLS slope of obs . n
    on du_n in the pairs above / below the median k; physics predicts the same ratio computed on the required changes
    (mean k_high / mean k_low when A's mass is unchanged), a constant-gain leak of A's velocity into B's readout 1. ``closer_n`` (the verdict's criterion (i), :func:`patch_verdict`): mean over pairs
    of d = |obs . n| - |obs . n - req . n| with a group-bootstrap CI (``groups``: each pair's base scene) and the paired
    sign-flip p; the slope ratio gets the same bootstrap CI (high / low split fixed at the full sample's median k)."""
    on, rq, du = patch_along_n(dvB_obs, vA_base, vA_donor, ctxs, k_gain, k_donor, vB_base)
    req = np.array([r * np.asarray(c["n"], float)[:2] for r, c in zip(rq, ctxs)]).reshape(-1, 2)
    obs, k_gain = np.asarray(dvB_obs, float).reshape(-1, 2)[:, :2], np.asarray(k_gain, float)
    em, en = np.linalg.norm(req - obs, axis=1), np.linalg.norm(req, axis=1)
    ok = np.isfinite(em) & (en > 1e-12)
    out = {"n_pairs": int(ok.sum())}
    if not ok.any():
        return out
    g = np.asarray(groups if groups is not None else np.arange(len(on)))
    d = np.abs(on) - np.abs(on - rq)
    rows = np.flatnonzero(ok & np.isfinite(d))
    keep = lambda ix: np.asarray(ix)[np.isin(ix, rows)]            # a bootstrap draw's scorable pairs (repeats kept)
    mean_d = lambda ix: _nan_guard(np.nanmean, d[keep(ix)])
    out.update(error_ratio=float(np.median(em[ok]) / np.median(en[ok])), p_sign_flip=sign_flip_p(em[ok] - en[ok]),
               iia=float(np.mean(em[ok] < np.linalg.norm(obs[ok], axis=1))), beats_do_nothing=float(np.mean(em[ok] < en[ok])),
               transfer_along_n=float(np.median(on[ok] / rq[ok])) if (np.abs(rq[ok]) > 0).all() else float("nan"),
               closer_n={"mean": mean_d(rows), "ci95": _boot_ci(mean_d, g, n_boot), "p_sign_flip": sign_flip_p(d[rows]),
                         "what": "|obs.n - 0| - |obs.n - req.n| (> 0: closer to the momentum-required change)"})
    hi = k_gain >= np.median(k_gain[ok])
    slope = lambda m: (float(np.polyfit(du[m], on[m], 1)[0]) if m.sum() >= 3 and np.ptp(du[m]) > 0 else float("nan"))
    lo_s, hi_s = slope(ok & ~hi), slope(ok & hi)
    out["slope_ratio"] = {"slope_low_k": lo_s, "slope_high_k": hi_s, "observed": hi_s / lo_s if lo_s else float("nan"),
                          "observed_ci95": _boot_ci(lambda ix: _slope_ratio(du, on, hi, keep(ix)), g, n_boot),
                          "physics": _slope_ratio(du, rq, hi, rows),
                          "leak": 1.0}
    if nt is not None:
        nt = np.asarray(nt, float)
        t_, n_ = np.nanmedian(np.abs(on)) / sd_t, np.nanmedian(np.abs(nt)) / sd_nt
        out["selectivity"] = {"target_change_sd": float(t_), "nontarget_change_sd": float(n_),
                              "ratio": float(t_ / n_) if n_ > 0 else float("nan")}
    return out


def patch_verdict(arms: dict, d: dict, groups, n_boot: int = 1000, max_p: float = 0.05) -> dict:
    """Pass rule of the collision patch. ``arms`` {condition: patch_scores}, ``d`` {condition: per-pair d of
    :func:`patch_closer`, the pairs in the same order in every condition}, ``groups`` the pairs' base scenes. Pass
    only if (i) B's observed change along n is significantly closer to the momentum-required change than to no change
    (group-bootstrap CI of mean d > 0 and the paired sign-flip p < ``max_p``), (ii) the high- / low-impulse slope
    ratio's CI excludes 1 (a constant leak) and covers the physics ratio, and (iii) on (i) it beats every control
    present (``patch_random``, ``patch_passive``: paired CI of d_patch - d_control > 0; at least one must be present).
    ``verdict`` "pass" / "fail" (``reasons``: each failed criterion) / "uninformative" (no scorable pair)."""
    a = arms.get("patch") or {}
    if not a.get("n_pairs"):
        return {"verdict": "uninformative", "reasons": ["no scorable patch pair"]}
    cl, sr = a.get("closer_n") or {}, a.get("slope_ratio") or {}
    ci, sci = cl.get("ci95"), sr.get("observed_ci95")
    crit = {"closer_to_required": bool(ci and ci[0] > 0 and cl.get("p_sign_flip", 1.0) < max_p),
            "slope_ratio_excludes_leak": bool(sci and not sci[0] <= 1.0 <= sci[1]),
            "slope_ratio_matches_physics": bool(sci and np.isfinite(sr.get("physics", np.nan))
                                                and sci[0] <= sr["physics"] <= sci[1])}
    dp, g = np.asarray(d["patch"], float), np.asarray(groups)
    beats = {}
    for c in ("patch_random", "patch_passive"):
        if c not in d:
            beats[c] = "not run" + (" (pre-event site: no passive arm)" if c == "patch_passive" else "")
            continue
        diff = dp - np.asarray(d[c], float)
        m = lambda ix: _nan_guard(np.nanmean, diff[ix])
        dci = _boot_ci(m, g, n_boot)
        beats[c] = {"mean": m(np.arange(len(diff))), "ci95": dci, "beats": bool(dci and dci[0] > 0)}
    run = [v for v in beats.values() if isinstance(v, dict)]
    crit["beats_controls"] = bool(run and all(v["beats"] for v in run))
    reasons = [k for k, v in crit.items() if not v]
    return {"verdict": "fail" if reasons else "pass", "criteria": crit, "controls": beats, "reasons": reasons,
            "max_p": max_p}


def collision_gain(labels, donor=None) -> float:
    """k = (1 + e) mu / m_B of a collision clip (e: the twin's restitution); ``donor`` labels: with the donor's m_A
    (the full-ROI patch swaps A in)."""
    pl = labels["physlaw"]
    mA, mB = (float(pl["objects"][o]["mass"]) for o in "AB")
    if donor is not None:
        mA = float(donor["physlaw"]["objects"]["A"]["mass"])
    e = pl.get("epsilon", pl.get("e_target", (pl.get("constants") or {}).get("e", 1.0)))
    return (1.0 + float(e)) * (mA * mB / (mA + mB)) / mB


# --------------------------------------------------------------------------- #
# off-distribution check (steering reliability)
# --------------------------------------------------------------------------- #


def mahalanobis(X: np.ndarray, shrink: float = 0.1):
    """d(Z) [m]: Mahalanobis distance to the rows of X under a shrunk covariance (1 - s) S + s tr(S) / D I (invertible
    for D > n), via the SVD of the centred rows."""
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = sv ** 2 / max(len(X) - 1, 1)
    lam = shrink * ev.sum() / X.shape[1] + 1e-300
    w = 1.0 / ((1 - shrink) * ev + lam)

    def d(Z):
        Z = np.atleast_2d(np.asarray(Z, np.float64)) - mu
        P = Z @ Vt.T
        return np.sqrt((P ** 2 * w).sum(1) + ((Z ** 2).sum(1) - (P ** 2).sum(1)).clip(0) / lam)
    return d


def off_distribution(dist: dict, real: np.ndarray) -> dict:
    """``dist`` {condition: {key: [distances of steered pooled vectors]}}, ``real`` held-out real rows' distances:
    per key the median distance and the share beyond the real 95th percentile."""
    real = np.asarray(real, float)
    q95 = float(np.quantile(real, 0.95)) if len(real) else float("nan")
    return {"real_median": float(np.median(real)) if len(real) else float("nan"), "real_q95": q95,
            "by_condition": {c: {f"{k_:g}": {"median": float(np.median(v)), "frac_above_real_q95": float(np.mean(np.array(v) > q95))}
                                 for k_, v in sorted(byk.items())} for c, byk in dist.items()}}


# --------------------------------------------------------------------------- #
# intervention runs (GPU)
# --------------------------------------------------------------------------- #


POOLS = ("frame.pre", "frame.post")


def _pool_forecast(acts: dict) -> dict:
    """forecast tokens {key: [T_post, gh, gw, D]} -> {space: frame.post mean [D], "<space>|t": frame mean per
    forecast token-frame [T_post, D] (kinematic readouts)}."""
    out = {}
    for key, F in acts.items():
        F = F.float().cpu().numpy() if hasattr(F, "cpu") else np.asarray(F, np.float32)
        sp = "final" if str(key) == "-1" else str(key)
        out[sp] = F.reshape(-1, F.shape[-1]).mean(0)
        out[f"{sp}|t"] = F.reshape(len(F), -1, F.shape[-1]).mean(1)
    return out


def run_interventions(m, mcfg, data_dir: Path, scenes: dict, plans: dict, layer: str, work: Path,
                      positions: str = "roi_A", force: bool = False, real_clips: dict | None = None, log=print,
                      snrs=None):
    """For each scene {name: clip dir}: forecast() unsteered and under every {condition: {key: shift}} of plans[scene]
    (key: dtheta for the steering curves, the pair number for swap / patch pairs), the shift added at the site's
    tokens (:func:`site_index`) of block ``layer``. Pooled forecasts -> work/<scene>.npz (keys "<cond>|<key>|<space>"
    and "...|<space>|t"; encoders also "<cond>|<key>|site": the site vector as steered, captured after the shift);
    DiTs: every generated video read with the track B readout -> work/<scene>_readout.json, plus the REAL videos of
    ``real_clips[scene]`` ({name: clip dir}) under "real|<name>" (the input-level and invariance references). A DiT
    shift is {snr_db: [D]} (one per fitted SNR space), injected per sampling step at the nearest of ``snrs`` (every
    SNR the cache provides at the block; :func:`steer_steps`); a DiT patch entry is a :class:`LivePatch` (the donor's
    live ROI from a paired pass per step, :func:`patch_live`); the per-step record of every run (which SNR curve,
    "skipped", or "patch") -> work/<scene>_steps.json {"<cond>|<key>": [step records]}."""
    from contextlib import ExitStack

    import torch
    from physlaw.data.io import load_clip
    from physlaw.eval import outputs as O
    work.mkdir(parents=True, exist_ok=True)
    diffusion = mcfg.get("kind") == "diffusion"
    for si, (sc, cdir) in enumerate(scenes.items()):
        f = work / f"{sc}.npz"
        if f.exists() and not force:
            continue
        c0 = load_clip(cdir)
        idx, fr, k = site_index(m, c0, positions)
        x = m.to_tensor(c0.frames[fr])
        rec, rd, steps = {}, {}, {}
        runs = [("base", 0.0, None)] + [(cond, d, v) for cond, dd in plans[sc].items() for d, v in dd.items() if d != 0]
        for cond, d, v in runs:
            site, slog = [], []
            with ExitStack() as st, torch.no_grad():
                if isinstance(v, LivePatch):           # DiT patch: a paired donor pass at every step
                    cd = load_clip(v.donor)
                    idx_d, fr_d, _ = site_index(m, cd, positions)
                    st.enter_context(patch_live(m, layer, idx, idx_d, x, m.to_tensor(cd.frames[fr_d]), k, slog,
                                                v.random_seed))
                elif isinstance(v, dict):
                    st.enter_context(steer_steps(m, layer, idx, v, snrs if snrs is not None else list(v), slog))
                elif v is not None:
                    st.enter_context(steer(m, layer, idx, v))
                if not diffusion:
                    st.enter_context(capture(m, layer, idx, site))
                acts, vid = m.forecast(x, k, seed=0, return_video=True) if diffusion else (m.forecast(x, k), None)
            for sp, z in _pool_forecast(acts).items():
                key = sp.split("|")
                rec["|".join([cond, f"{d:g}", *key])] = z.astype(np.float32)
            if site:
                rec[f"{cond}|{d:g}|site"] = np.mean(site, 0).astype(np.float32)
            if slog:
                steps[f"{cond}|{d:g}"] = slog
            if vid is not None:
                rd[f"{cond}|{d:g}"] = O._row(O.READOUT_SELF[c0.pl["family"]](vid, c0, float(m.fps), gate="loose"))
        if diffusion:
            for name, cd in (real_clips or {}).get(sc, {}).items():
                cr = load_clip(cd)
                rd[f"real|{name}"] = O._row(O.READOUT_SELF[cr.pl["family"]](O.real_video(cr, float(m.fps)), cr,
                                                                           float(m.fps), gate="loose"))
        np.savez(f, **rec)
        if rd:
            (work / f"{sc}_readout.json").write_text(json.dumps(rd, indent=1, default=float))
        if diffusion:
            (work / f"{sc}_steps.json").write_text(json.dumps(steps, indent=1))
        log(f"[L4] {sc} ({si + 1}/{len(scenes)}): {len(runs)} runs")


# --------------------------------------------------------------------------- #
# the routine
# --------------------------------------------------------------------------- #


PAIR_DEFAULTS = {"swap": True, "n_donors": 4, "min_dtheta_sd": 0.25, "input_tol_sd": 0.15, "patch_cos_min": 0.94,
                 "min_moved": 0.25, "max_invariance": 0.25}


def _scene_key(labels, rec, lc) -> str:
    return "/".join(str(get(labels, k, rec["group"])) for k in lc.get("scene_keys", [])) or rec["group"]


def _controls(lc, X, labels, scene, rows) -> dict:
    ctl = {}
    for name in lc.get("controls", []):
        if name in ("passive",):
            continue
        if name == "random_dir":
            ctl[name] = None
            continue
        cc = lc["control_theta"][name]          # {theta: label name, mode, centre_by: a label value held fixed}
        th = np.array([label_value(labels[i], cc["theta"]) for i in rows])
        N = None
        if cc.get("centre_by"):                  # e.g. the aftermath speed at fixed eps: dummies of the rung
            by = np.array([round(label_value(labels[i], cc["centre_by"]), 6) for i in rows])
            N = (by[:, None] == np.unique(by)[None, 1:]).astype(float)
        ctl[name] = fit_manifold(X[rows], th, scene[rows], cc.get("mode", "ladder"), N)
    return ctl


def required_change(lc: dict, th_base: float, th_donor: float) -> float:
    """The law's change of the readout when theta goes from th_base to th_donor (saturation: eps capped at 1; step:
    the change of the law's indicator [theta > boundary])."""
    if lc["predict"] == "saturation":
        return min(th_donor, 1.0) - min(th_base, 1.0)
    if lc["predict"] == "step":
        b = float(lc.get("boundary", 1.0))
        return float(th_donor > b) - float(th_base > b)
    return law_value(lc) * (th_donor - th_base)


def pair_required(lc: dict, theta, theta_c, i: int, j: int) -> float:
    """The law's readout change for a swap / patch pair (base i, donor j). Slope laws: on theta_c, the theta the curve
    encodes (beyond the nuisance): an interchange of the curve's component swaps theta_c and keeps the base's nuisance
    (e.g. its size), so the implied theta moves by theta_c[j] - theta_c[i], not by the raw difference; saturation / step
    laws: on the raw theta (their boundary is absolute)."""
    if lc["predict"] in ("saturation", "step"):
        return required_change(lc, float(theta[i]), float(theta[j]))
    return law_value(lc) * float(theta_c[j] - theta_c[i])


def _random_like(v, rng):
    r = rng.standard_normal(np.shape(v))
    return r / max(np.linalg.norm(r), 1e-300) * np.linalg.norm(v)


def null_prediction(lc: dict) -> bool:
    """The law predicts NO change of the readout (slope 0): the swap test scores invariance (:func:`invariance_scores`)."""
    return lc["predict"] not in ("saturation", "step") and law_value(lc) == 0


def _ref_pairs(lc: dict, recs, clip_of: dict, scenes) -> dict:
    """``invariance_ref`` [twin a, twin b]: two LEGAL twins of a group that differ in a quantity the law says DOES
    change the readout (Galilean: e, friction: mu). {scene: (row a, row b)} in the group of the scene's clip."""
    ab = lc.get("invariance_ref")
    if not ab:
        return {}
    at = {(r["group"], r["twin"]): j for j, r in enumerate(recs)}
    out = {}
    for s in scenes:
        g = recs[clip_of[s]]["group"]
        if (g, ab[0]) in at and (g, ab[1]) in at:
            out[s] = (at[g, ab[0]], at[g, ab[1]])
    return out


def invariance_reference(refp: dict, value, floor_values, lc: dict) -> tuple[float, str]:
    """(scale, kind) of the invariance ratio: the median |readout(a) - readout(b)| over the groups' reference twin
    pairs (``refp``; the same instrument on the real clips), else the sd of the readout over the real clips."""
    d = np.array([abs(value(a) - value(b)) for a, b in sorted(set(refp.values()))], float)
    if np.isfinite(d).any():
        return float(np.nanmedian(d)), "{} vs {}".format(*lc["invariance_ref"])
    return float(np.nanstd(np.asarray(floor_values, float))), "sd_real"


def encoder_sites(model, data_dir: Path, recs, labels, positions: str, wdir: Path, mcfg, force: bool, log) -> dict:
    """:func:`site_activations`, cached in work/<law>/_site_<positions>.npz (recomputed when the clip list differs)."""
    f = wdir / f"_site_{positions}.npz"
    clips = [r["clip"] for r in recs]
    if f.exists() and not force:
        z = np.load(f)
        if [str(c) for c in z["clips"]] == clips:
            keys = [str(q) for q in z["keys"]]
            return {"layers": [str(q) for q in z["layers"]], **{q: z[f"x{i}"] for i, q in enumerate(keys)}}
    S_ = site_activations(model(), data_dir, recs, labels, positions, int((mcfg.get("pred_real") or {}).get("n_masks", 4)),
                          log)
    keys = [q for q in S_ if q != "layers"]
    wdir.mkdir(parents=True, exist_ok=True)
    np.savez(f, clips=np.array(clips), layers=np.array(S_["layers"]), keys=np.array(keys),
             **{f"x{i}": S_[q] for i, q in enumerate(keys)})
    return S_


def law_rows(labels: list[dict], lc: dict) -> list[int]:
    """Clips of the law's ladder: the configured ``subfamily`` and, for Galilean, ``boost_kind`` (F4: "ball" -- the
    ball-boost twins, u0 included; camera-boost twins are L3's camera_vs_ball comparison, not a theta rung; labels
    written before camera boosts existed carry no boost_kind and are all ball boosts)."""
    return [i for i, l in enumerate(labels)
            if (not lc.get("subfamily") or get(l, "physlaw.subfamily") == lc["subfamily"])
            and (not lc.get("boost_kind") or get(l, "physlaw.boost_kind", "ball") == lc["boost_kind"])]


def run(cache_dir: Path, data_dir: Path, cfg, mcfg, work: Path, force: bool = False, log=print) -> dict:
    """Manifold steering for the law of the evaluated family (``law_of_family``, else :data:`LAW_OF_FAMILY`), with the
    swap-test and (momentum) collision-patch arms, steering reliability and the input-level reference."""
    fam = data_dir.name.split("_")[0]
    law = (cfg.get("law_of_family") or {}).get(fam) or LAW_OF_FAMILY[fam]
    lc = cfg["laws"][law]
    pc = {**PAIR_DEFAULTS, **dict(cfg.get("pairs") or {}), **dict(lc.get("pairs") or {})}
    recs, labels, A, pools, layers = load_family(cache_dir, data_dir)
    keep = law_rows(labels, lc)
    recs, labels, A = [recs[i] for i in keep], [labels[i] for i in keep], A[keep]
    scene = np.array([_scene_key(l, r, lc) for l, r in zip(labels, recs)])
    grp = np.array([r["group"] for r in recs])
    theta = np.array([label_value(l, lc["theta"]) for l in labels])
    fit_rows = np.where(theta <= float(lc["fit_max"]) + 1e-9)[0] if lc.get("fit_max") is not None else np.arange(len(theta))
    lc = resolve_law_cfg(lc, theta[fit_rows])
    nuis = (np.array([[label_value(l, n) for n in lc["nuisance"]] for l in labels]) if lc.get("nuisance") else None)
    theta_c = theta if nuis is None else _resid(theta, nuis)          # the theta the curve encodes (beyond nuisance)
    draws, max_p = int(cfg.get("null_draws", 200)), float(cfg.get("max_p", 0.05))
    diffusion = mcfg.get("kind") == "diffusion"
    positions = site_positions(lc["pool"])
    wdir = work / law
    held = {}

    def model():
        if "m" not in held:
            from physlaw.models.registry import load_adapter
            held["m"] = load_adapter(mcfg, device=cfg.get("device", "cuda"), need="internals")
        return held["m"]

    # 0. site activations: the curve is fitted in the space and on the tokens the shift is injected into
    site_final = None
    if diffusion:        # stage 4's encode() at the site pool, per (SNR, layer): the tokens the sampler's hook steers
        sites = {l: A[:, pools.index(lc["pool"]), layers.index(l)].astype(np.float64) for l in layers if l != "-1"}
    else:                # encoders: the model's own pass the hook acts on (module docstring, step 1)
        S_ = encoder_sites(model, data_dir, recs, labels, positions, wdir, mcfg, force, log)
        sites, site_final = {l: S_[l] for l in S_["layers"]}, S_.get("final")

    # 1. precondition per layer
    pre = {}
    for l, Xl in sites.items():
        pre[l] = (population_curve(Xl[fit_rows], theta[fit_rows], scene[fit_rows], draws) if lc["mode"] == "ladder" else
                  across_curve(Xl[fit_rows], theta[fit_rows], scene[fit_rows], None if nuis is None else nuis[fit_rows],
                               draws))
    passing = [l for l, p in pre.items() if p and all(v <= max_p for k, v in p.items() if k.startswith("p_") and k != "p_loso_cubic")]
    out = {"law": law, "family": fam, "mode": lc["mode"], "theta": lc["theta"], "pool": lc["pool"], "precondition": pre,
           "layers_passing": passing, "deltas": [float(d) for d in lc["deltas"]], "predict": lc["predict"], "pairs_cfg": pc,
           "positions": positions, **({"kappa_ref": lc["kappa_ref"]} if "kappa_ref" in lc else {})}
    if cfg.get("stand_in"):
        out["stand_in"] = cfg["stand_in"]
    rank = lambda l: (-np.inf if not pre.get(l) else pre[l]["retention"] if pre[l].get("retention") is not None
                      else pre[l]["loso_r2"]["line"])
    if diffusion:        # one injection block; a curve per cached SNR space there that passes (module docstring, step 2)
        by_block = {}
        for l in sites:
            if snr_of(l) is not None:
                by_block.setdefault(block_index(l), {})[snr_of(l)] = l
        if cfg.get("layer", "auto") != "auto":
            blk = block_index(str(cfg["layer"]))
        else:
            best = {b: max(rank(l) for l in ks.values() if l in passing) for b, ks in by_block.items()
                    if any(l in passing for l in ks.values())}
            blk = max(best, key=best.get) if best else None
        ks = by_block.get(blk, {})
        cached = sorted(ks)
        fitted = [db for db in cached if ks[db] in passing or cfg.get("steer_without_precondition")]
        layer = max((ks[db] for db in fitted), key=rank, default=None)    # the block's best space (hook site, report)
    else:
        layer = str(cfg["layer"]) if cfg.get("layer", "auto") != "auto" else (max(passing, key=rank) if passing else None)
    if layer is None or (layer not in passing and not cfg.get("steer_without_precondition")):
        out["skipped"] = f"no layer passes the precondition (p <= {max_p})"
        log(f"[L4 {law}] {out['skipped']}")
        return out
    out["layer"] = layer
    if diffusion:        # Xs {snr_db: activations of that SNR space}; a plan entry is then {snr_db: shift}
        Xs = {db: sites[ks[db]] for db in fitted}
        out["injection"] = {"block": blk, "cached_snr_db": cached, "fitted_snr_db": fitted,
                            "spaces": {snr_name(db): ks[db] for db in cached},
                            "rule": "per sampling step: the curve of the cached SNR nearest the step's SNR (log-SNR "
                                    "distance); none where that SNR failed the precondition"}
        pk = lambda per: dict(per)
    else:                # encoders: one space; a plan entry is the shift itself
        Xs = {None: sites[layer]}
        pk = lambda per: per[None]

    # 2. per fold (scenes cross-fitted): manifold + control shifts, swap / patch pairs
    scenes = sorted(set(scene))[: cfg.get("max_groups") or None]
    folds = grouped_kfold(np.array(scenes), min(int(cfg["probe"]["k_folds"]), len(scenes)), 0)
    steer_twin = lc.get("steer_twin")
    clip_of = {}
    for s in scenes:
        rows = [i for i in np.where(scene == s)[0] if not steer_twin or recs[i]["twin"] == steer_twin] or list(np.where(scene == s)[0])
        clip_of[s] = int(rows[0])
    states = _states_of(labels)
    for g in set(grp):                               # the twin-shared clean-window end (statics' fallback tip window)
        rows = np.where(grp == g)[0]
        end = min(float(states[i][2].get("clean_until_s") or np.inf) for i in rows)
        for i in rows:
            states[i][2]["group_clean_end_s"] = end
    rng = np.random.default_rng(0)
    min_dt = float(pc["min_dtheta_sd"]) * float(np.std(theta[fit_rows]))
    # the patch site = A's pre-contact ROI site (encoders: the cached site vectors; DiTs: live, :class:`LivePatch`)
    do_patch = bool(lc.get("patch")) and lc["pool"] == "roi_A.pre"
    Xp = dict(Xs) if do_patch and not diffusion else None
    ctf = {clip_of[s]: contact_token_frame(cache_dir, recs[clip_of[s]]) for s in scenes} if do_patch else {}
    pd_rec = {"rule": "donor contact in the base's contact token-frame", "excluded_other_token_frame": {},
              "donors": {}}
    plans, pairs, mans, md_of, real_d = {}, {}, [], {}, []
    for fi, te in enumerate(folds):
        test = {scenes[i] for i in te}
        tgrp = set(grp[np.isin(scene, list(test))])                     # whole twin groups stay out of training
        tr = fit_rows[~np.isin(grp[fit_rows], list(tgrp))]
        per, Q, md, mrec = {}, {}, {}, {}
        for db, Xd in Xs.items():     # one curve (and controls, swap span, off-distribution metric) per space
            man = fit_manifold(Xd[tr], theta[tr], scene[tr], lc["mode"], None if nuis is None else nuis[tr])
            ctl = _controls(lc, Xd, labels, scene, tr)
            per[db] = plan_shifts(man, lc["deltas"], {k: v for k, v in ctl.items() if v is not None} |
                                  ({"random_dir": None} if "random_dir" in ctl else {}), seed=fi)
            mrec[db] = {"deg": man["deg"], "theta_ref": man["theta_ref"],
                        "shift_norm_per_unit": float(np.linalg.norm(shift(man, 1.0)))}
            Q[db], md[db] = curve_basis(man), mahalanobis(Xd[tr])
            real_d += md[db](Xd[np.isin(scene, list(test))]).tolist()
        p0 = next(iter(per.values()))
        plan = {c: {d: pk({db: per[db][c][d] for db in Xs}) for d in dd} for c, dd in p0.items()}
        mans.append(pk(mrec) if not diffusion else {"by_snr": {snr_name(db): r for db, r in mrec.items()}})
        for s in sorted(test):
            i = clip_of[s]
            md_of[s] = md
            ps = {c: dict(v) for c, v in plan.items()}
            P = []
            if pc["swap"]:            # interchange intervention along the fitted curve's span (per space)
                pool = fit_rows if lc["mode"] == "ladder" else [clip_of[t] for t in scenes]
                for p, j in enumerate(pick_donors(i, pool, theta, scene, int(pc["n_donors"]), min_dt, rng, lc["mode"]), 1):
                    dl = {db: swap_delta(Xd[i], Xd[j], Q[db]) for db, Xd in Xs.items()}
                    ps.setdefault("swap", {})[float(p)] = pk(dl)
                    ps.setdefault("swap_random", {})[float(p)] = pk({db: _random_like(v, rng) for db, v in dl.items()})
                    P.append(("swap", p, j))
            if do_patch:              # causal collision patch: A's whole pre-contact ROI from a matched donor
                dons, n_ex = patch_donors(i, [clip_of[t] for t in scenes], theta, scene, int(pc["n_donors"]), rng,
                                          states, float(pc["patch_cos_min"]), ctf)
                pd_rec["excluded_other_token_frame"][s] = n_ex
                pd_rec["donors"][s] = [recs[j]["clip"] for j in dons]
                for p, j in enumerate(dons, 1):
                    if diffusion:     # the donor's live ROI at each sampling step (paired pass), not a cached vector
                        ps.setdefault("patch", {})[float(p)] = LivePatch(data_dir / recs[j]["clip"])
                        ps.setdefault("patch_random", {})[float(p)] = LivePatch(data_dir / recs[j]["clip"],
                                                                                int(rng.integers(2 ** 31)))
                    else:
                        dl = swap_delta(Xp[None][i], Xp[None][j])
                        ps.setdefault("patch", {})[float(p)] = dl
                        ps.setdefault("patch_random", {})[float(p)] = _random_like(dl, rng)
                    P.append(("patch", p, j))
            plans[s], pairs[s] = ps, P
    refp = _ref_pairs(lc, recs, clip_of, scenes) if null_prediction(lc) else {}
    out.update(manifolds=mans, n_scenes=len(scenes),
               n_pairs={c: sum(1 for P in pairs.values() for x in P if x[0] == c) for c in ("swap", "patch")})
    if do_patch:
        out["patch_donors"] = pd_rec
        log(f"[L4 {law}] patch donors: {sum(pd_rec['excluded_other_token_frame'].values())} candidates excluded "
            f"(contact in another token-frame) over {len(scenes)} scenes; per scene "
            f"{pd_rec['excluded_other_token_frame']}")
    safe = lambda s: s.replace("/", "__")

    def complete(s):
        f = wdir / f"{safe(s)}.npz"
        if not f.exists() or (diffusion and not (wdir / f"{safe(s)}_steps.json").exists()):   # per-SNR runs record steps
            return False
        have = {"|".join(k.split("|")[:2]) for k in np.load(f).files}
        return all(f"{c}|{d:g}" in have for c, dd in plans[s].items() for d in dd if d != 0)
    missing = {safe(s): data_dir / recs[clip_of[s]]["clip"] for s in scenes if force or not complete(s)}
    if missing:
        wdir.mkdir(parents=True, exist_ok=True)
        real_clips = {safe(s): {recs[j]["clip"].replace("/", "__"): data_dir / recs[j]["clip"]
                                for j in [*(np.where(scene == s)[0] if lc["mode"] == "ladder" else [clip_of[s]]),
                                          *refp.get(s, ())]}
                      for s in scenes}
        run_interventions(model(), mcfg, data_dir, missing, {safe(s): plans[s] for s in scenes if safe(s) in missing},
                          layer, wdir, positions, True, real_clips, log, **({"snrs": cached} if diffusion else {}))
    held.clear()
    Z = {s: dict(np.load(wdir / f"{safe(s)}.npz")) for s in scenes}
    if diffusion:        # the per-step injection record (identical schedule in every scene): first run per condition
        fs = wdir / f"{safe(scenes[0])}_steps.json"
        stp = {}
        for k_, v_ in (json.loads(fs.read_text()) if fs.exists() else {}).items():
            stp.setdefault(k_.split("|")[0], v_)
        out["injection"]["steps"] = stp

    # off-distribution: the site vector as steered (encoders: captured by the hook; DiTs: the real site + the shift,
    # per fitted SNR space, each against its own space's training rows)
    dist = {}
    for s in scenes:
        i = clip_of[s]
        for c, dd in plans[s].items():
            if not c.startswith("patch"):
                for d, v in dd.items():
                    z = Z[s].get(f"{c}|{d:g}|site")
                    vs = v if isinstance(v, dict) else {None: v}
                    ds = [md_of[s][None](z)[0]] if z is not None else [md_of[s][db](Xs[db][i] + w)[0] for db, w in vs.items()]
                    dist.setdefault(c, {}).setdefault(d, []).extend(float(x) for x in ds)
    out["off_distribution"] = off_distribution(dist, np.array(real_d))

    # 3. read + 4. score
    deltas = sorted({0.0, *(float(d) for d in lc["deltas"])})
    steer_conds = [c for c in plans[scenes[0]] if not c.startswith(("swap", "patch"))]
    n_boot = int(cfg.get("bootstrap", {}).get("n", 1000))
    site_theta = None
    if diffusion and null_prediction(lc):   # DiTs: the model's own theta at the steered / swapped block and SNR
        db0, kf, al = snr_of(layer), int(cfg["probe"]["k_folds"]), tuple(float(a) for a in cfg["probe"]["alphas"])
        X0 = Xs[db0]
        thp = theta_probe(X0, theta_c, grp, kf, al)
        site_theta = lambda i, v: thp(i, X0[i] + (0.0 if v is None or v.get(db0) is None else np.asarray(v[db0], float)))
        out["site_theta"] = {"space": layer, "what": "R1 theta probe on the real clips' activations at the injection "
                             "block / SNR, applied to real site vector + the run's shift"}
    ctx = dict(law=law, lc=lc, scenes=scenes, clip_of=clip_of, plans=plans, pairs=pairs, theta=theta, theta_c=theta_c,
               scene=scene, states=states, labels=labels, deltas=deltas, conds=steer_conds, n_boot=n_boot, pc=pc,
               fit_rows=fit_rows, recs=recs, refp=refp, positions=positions, layer=layer, site_theta=site_theta,
               min_effect=float(cfg.get("min_effect", 0.25)))
    out["variants"] = {}
    rd_files = {s: wdir / f"{safe(s)}_readout.json" for s in scenes}
    if all(f.exists() for f in rd_files.values()):                      # DiTs: the generated videos
        out["variants"]["generated"] = _score_generated({s: json.loads(f.read_text()) for s, f in rd_files.items()}, **ctx)
    if all(any(k_.startswith("base|0|") and not k_.endswith("|site") for k_ in Z[s]) for s in scenes):   # R1 decodes
        readers = decoders(cache_dir, data_dir, recs, labels, mcfg, states, grp, lc["readout"],
                           int(cfg["probe"]["k_folds"]), tuple(float(a) for a in cfg["probe"]["alphas"]), theta_c,
                           Xs[None] if positions == "post" and not diffusion else None, site_final)
        single = len({r.get("space", n) for n, r in readers.items()}) == 1
        for name, rd in readers.items():
            sp = rd.get("space", name)
            tag = ("frame_t" if rd["field"] else "frame") + ("_mm" if name.endswith("_mm") else "")
            out["variants"][tag if single else f"{sp}|{tag}"] = _score_decoded(rd, sp, Z, **ctx)
    log(f"[L4 {law}] " + "; ".join(f"{v}: {d.get('verdict')}" for v, d in out["variants"].items()))
    lcfg = loc_config(cfg, diffusion)
    if lcfg.get("enabled", True):          # WHERE the law is used (gated: only for a model that passes)
        rd_ = locals().get("readers")
        out["localization"] = run_localization(
            out, lcfg=lcfg, cfg=cfg, mcfg=mcfg, data_dir=data_dir, law=law, lc=lc, pc=pc, recs=recs, labels=labels,
            A=A, pools=pools, layers=layers, sites=sites, theta=theta, theta_c=theta_c, scene=scene, grp=grp,
            fit_rows=fit_rows, clip_of=clip_of, nuis=nuis, states=states, positions=positions, diffusion=diffusion,
            work=wdir / "localize", readers=rd_, cached=cached if diffusion else [], model=model, force=force,
            patch_donors=out.get("patch_donors"), log=log)
    return out


REPO = Path(__file__).resolve().parents[2]


def make_runner(model, mcfg, inputs, readout, positions, work, diffusion, cached_snrs, cached_pool, n_masks, force,
                lcfg, log) -> "ModelRunner":
    """The localization runner on the loaded adapter (tests replace this with a stand-in)."""
    return ModelRunner(model(), inputs, readout, positions, work, diffusion, cached_snrs, cached_pool, n_masks, force,
                       int(lcfg["attribution"]["max_fd_dims"]), float(lcfg["attribution"]["h"]), log)


def run_localization(res: dict, *, lcfg, cfg, mcfg, data_dir, law, lc, pc, recs, labels, A, pools, layers, sites, theta,
                     theta_c, scene, grp, fit_rows, clip_of, nuis, states, positions, diffusion, work, readers, cached,
                     model, force, patch_donors, log) -> dict:
    """``localization`` of l4.json: the gate (:func:`localize_gate`; results_root/<model>/<family>/l2.json for the L2
    kink), then :func:`localize` with the run's sites, readout (encoders: the first decoded variant's R1 instrument,
    space "final" preferred; DiTs: the generated video's track-B readout), the control quantity (``control_quantity``
    per law, e.g. size), the passive control (encoders' post-event sites), the untrained model's step-1 curve
    (results_root/<untrained_model>/<family>/l4.json, when present) and, for DiT momentum, the live patch per space."""
    fam = Path(data_dir).name
    name = getattr(mcfg, "name", None) or (mcfg.get("name") if hasattr(mcfg, "get") else None) or "model"
    root = Path(lcfg["results_root"])
    root = root if root.is_absolute() else REPO / root
    null = null_prediction(lc)
    gate = localize_gate(res, law, null, list(lcfg["gate"]), root / name / fam / "l2.json")
    out = {"gate": gate}
    if not gate["gated_in"]:
        out["skipped"] = gate["reason"]
        log(f"[L4 {law}] localization skipped: {gate['reason']}")
        return out
    if "layer" not in res:
        out["skipped"] = "no steering layer (precondition failed everywhere): no sites to localize"
        return out
    # readout
    dec_fn = READOUTS[lc["readout"]][0]
    readout, rd = None, None
    if diffusion and "generated" in res.get("variants", {}):
        from physlaw.data.io import load_clip
        from physlaw.eval import outputs as O
        row_fn = READOUTS[lc["readout"]][1]

        def video(i, vid):
            c0 = load_clip(Path(data_dir) / recs[i]["clip"])
            row = O._row(O.READOUT_SELF[c0.pl["family"]](vid, c0, float(model().fps), gate="loose"))
            pv = row.get("velocity_post")
            return {"r": float(row_fn(row)) if row.get("measurable") else float("nan"), "theta": float("nan"),
                    "post_v": np.asarray(pv, float) if pv is not None else np.full((2, 2), np.nan)}
        readout = LocReadout(video=video)
    elif readers:
        names = [n for n in readers if not n.endswith("_mm")]
        pick = next((n for n in names if readers[n].get("space", n) == "final"), names[0] if names else None)
        if pick is not None:
            rd = readers[pick]

            def value(i, z, rd=rd):
                post = rd["post"](i, z)
                return {"r": float(dec_fn(rd["pre"](i), post, states[i][2])),
                        "theta": float(rd["theta"](i, z)) if "theta" in rd else float("nan"),
                        "post_v": np.asarray(post.get("v"), float) if post.get("v") is not None else None}
            readout = LocReadout(space=rd.get("space", pick) + rd["field"], value=value)
            out["readout_variant"] = pick
    if readout is None:
        out["skipped"] = "no readout for localization runs (no decoded or generated variant)"
        return out
    # control quantity (not the law's parameter)
    cq = lcfg["control_quantity"].get(law, lcfg["control_quantity"].get("default"))
    try:
        size = np.array([label_value(l_, cq) for l_ in labels], float) if cq else None
        size = size if size is not None and np.isfinite(size).all() and np.ptp(size) > 0 else None
    except (KeyError, TypeError, ZeroDivisionError):
        size = None
    out["control_quantity"] = cq if size is not None else f"{cq}: not available (missing or constant)"
    # passive propagation (encoders' post-event sites, velocity readouts): the swapped site read in its own space
    k_f, alphas = int(cfg["probe"]["k_folds"]), tuple(float(a) for a in cfg["probe"]["alphas"])
    pmemo = {}

    def passive(l):
        if diffusion or positions != "post" or READOUTS[lc["readout"]][2] or rd is None:
            return None
        if l not in pmemo:
            Vq = np.stack([s_[1]["v"] for s_ in states])
            Xl = np.asarray(sites[l], float)
            F = [instrument_folds({"x": Xl}, Vq[:, o], grp, None, k_f, 1, alphas) for o in range(Vq.shape[1])]
            pmemo[l] = lambda i, z: float(dec_fn(rd["pre"](i), {"v": np.stack([_dec(f, z, grp[i])[0] for f in F])},
                                                 states[i][2]))
        return pmemo[l]
    # the untrained model's step-1 curve
    untrained = None
    un = lcfg.get("untrained_model")
    uf = root / str(un) / fam / "l4.json"
    if un and un != name and uf.is_file():
        bl = ((json.loads(uf.read_text()).get("localization") or {}).get("layers") or {}).get("by_layer") or {}
        untrained = {l: v["law"] for l, v in bl.items() if isinstance(v, dict) and "law" in v}
        out["untrained"] = str(uf)
    # DiT momentum: live paired-pass patch per (block, SNR) with the run's matched donors
    patch = None
    if diffusion and patch_donors and positions == "roi_A":
        row_of = {r["clip"]: t for t, r in enumerate(recs)}
        don = {clip_of[s_]: [row_of[c] for c in cs if c in row_of] for s_, cs in patch_donors.get("donors", {}).items()
               if s_ in clip_of}

        def pscore(prs, obs):
            obs = np.asarray(obs, float)
            vA = [states[i][0]["v"][0] for i, _ in prs]
            vD = [states[j][0]["v"][0] for _, j in prs]
            cx = [states[i][2] for i, _ in prs]
            kg = [collision_gain(labels[i]) for i, _ in prs]
            kd = [collision_gain(labels[i], labels[j]) for i, j in prs]
            vB = [states[i][0]["v"][1] for i, _ in prs]
            sc = patch_scores(obs, vA, vD, cx, kg, k_donor=kd, vB_base=vB)
            rq = patch_along_n(obs, vA, vD, cx, kg, kd, vB)[1]
            req = np.array([r * np.asarray(c["n"], float)[:2] for r, c in zip(rq, cx)])
            em = np.linalg.norm(req - obs, axis=1)
            hit = np.where(np.isfinite(em), (em < np.linalg.norm(obs, axis=1)).astype(float), np.nan)
            m_ = lambda ix: _nan_guard(np.nanmean, hit[ix])
            return {**sc, "score": m_(np.arange(len(hit))),
                    "score_ci95": _boot_ci(m_, grp[[i for i, _ in prs]], int(lcfg["bootstrap"]))}
        patch = {"donors": don, "score": pscore}
    # invariance reference (zero-change laws): the main swap test's
    ref = float("nan")
    for v in res.get("variants", {}).values():
        r_ = ((v.get("pairs") or {}).get("swap") or {}).get("reference")
        if r_ is not None and np.isfinite(r_):
            ref = float(r_)
            break
    cached_pool = None
    if diffusion:
        cached_pool = lambda l, p: (A[:, pools.index(p), layers.index(l)].astype(np.float64)
                                    if p in pools and l in layers else None)
    windows = ("pre", "post") if diffusion or positions == "post" else ("pre",)

    def inputs(i):
        from physlaw.data.io import load_clip
        m = model()
        c0 = load_clip(Path(data_dir) / recs[i]["clip"])
        idx, fr, k = site_index(m, c0, positions)
        return {"x": m.to_tensor(c0.frames[fr]), "k": k, "T": len(m.token_grid.token_frames(len(fr))),
                "groups": {"site": idx, **token_groups(m, c0, fr, k, windows, list(lcfg["token_groups"]))}}
    runner = make_runner(model, mcfg, inputs, readout, positions, Path(work), diffusion, cached,
                         cached_pool, int((mcfg.get("pred_real") or {}).get("n_masks", 4)) if hasattr(mcfg, "get") else 4,
                         force, lcfg, log)
    min_dt = float(pc["min_dtheta_sd"]) * float(np.std(theta[fit_rows]))
    out.update(localize(runner, lc=lc, lcfg=lcfg, theta=theta, theta_c=theta_c, scene=scene, grp=grp,
                        fit_rows=fit_rows, clip_of=clip_of, sites=sites, nuis=nuis, size=size, passive=passive,
                        untrained=untrained, patch=patch, ref=ref, max_inv=float(pc["max_invariance"]),
                        min_dt=min_dt, k_folds=int(cfg["probe"]["k_folds"]), diffusion=diffusion, log=log))
    v = out.get("verdict") or {}
    log(f"[L4 {law}] localization: peak {(out.get('layers') or {}).get('peak')}, flagged "
        f"{(out.get('layers') or {}).get('flagged')}, localized layers {v.get('localized_layers')}, components "
        f"{v.get('localized_components')}")
    return out


def _nearest(i, d, theta, scene, cands, tol) -> int | None:
    """The real clip at theta_i + d (ladder: same scene; across: any other scene's clip), nearest within ``tol``."""
    best = min(cands, key=lambda j: abs(theta[j] - theta[i] - d), default=None)
    return best if best is not None and abs(theta[best] - theta[i] - d) <= tol else None


def _site_shift(v, sp):
    """A plan entry's shift in decoded space ``sp``: encoders' vector itself; DiTs' {snr_db: shift} entry of sp's SNR
    (None: no curve was fitted at that SNR)."""
    return v.get(snr_of(sp)) if isinstance(v, dict) else v


def _input_cands(i, s, scene, clip_of, lc):
    return list(np.where(scene == s)[0]) if lc["mode"] == "ladder" else [clip_of[t] for t in clip_of]


def _score_decoded(rd, sp, Z, *, law, lc, scenes, clip_of, plans, pairs, theta, theta_c, scene, states, labels, deltas,
                   conds, n_boot, pc, fit_rows, recs, refp, positions, layer, site_theta=None, min_effect=0.25) -> dict:
    """The steering curves (+ passive, input-level reference) and the pair arms of one decoder. Passive propagation
    (post-event sites only, where the readout reads the steered tokens' successors): the unsteered forecast's site
    vector plus the same shift, read in the site's own space (encoders: ``site_post``, an instrument on the real
    clips' site activations; DiTs: the decoded space when it is the steered layer); pre-event sites have no passive
    arm (their space holds no post-event state to read). ``readout_validity``: the instrument on REAL clips (one per
    scene, the steered twin), its readout regressed on the true theta across scenes (slope laws: the law's slope is
    what a valid readout recovers). Slope-0 laws: :func:`null_verdict` with the model's theta readout of the steered
    forecasts (:func:`push_effect`)."""
    dec_fn = READOUTS[lc["readout"]][0]
    f = rd["field"]
    key = lambda c, d: f"{'base' if d == 0 else c}|{d:g}|{sp}{f}"
    val = lambda i, post: dec_fn(rd["pre"](i), post, states[i][2])
    curves = {c: [[val(clip_of[s], rd["post"](clip_of[s], Z[s][key(c, d)])) for d in deltas] for s in scenes]
              for c in conds}
    pas = None
    same_site = sp == layer or (snr_of(sp) is not None and snr_of(layer) is not None
                                and block_index(sp) == block_index(layer))       # DiTs: any SNR space of the block
    if "passive" in lc.get("controls", []) and positions == "post":
        if "site_post" in rd and all("base|0|site" in Z[s] for s in scenes):
            pas = lambda s, v: rd["site_post"](clip_of[s], Z[s]["base|0|site"].astype(np.float64) + v)
        elif same_site and all(_site_shift(v, sp) is not None for v in plans[scenes[0]]["law"].values()):
            pas = lambda s, v: rd["post"](clip_of[s], Z[s][key("base", 0)].astype(np.float64) + _site_shift(v, sp))
    if pas is not None:
        curves["passive"] = [[val(clip_of[s], pas(s, 0.0 if d == 0 else plans[s]["law"][d])) for d in deltas]
                             for s in scenes]
    tol = float(pc["input_tol_sd"]) * float(np.std(theta[fit_rows]))
    floor = {}

    def fl(j):
        if j not in floor:
            floor[j] = dec_fn(rd["pre"](j), rd["floor"](j), states[j][2])
        return floor[j]

    def inp(s, d):                      # input-level reference: the same dtheta shown through the input
        j = _nearest(clip_of[s], d, theta, scene, _input_cands(clip_of[s], s, scene, clip_of, lc), tol)
        return float("nan") if j is None else fl(j)
    curves["input"] = [[inp(s, d) for d in deltas] for s in scenes]
    res = score_response(curves, deltas, scenes, lc, n_boot, [theta[clip_of[s]] for s in scenes])
    floor_ix = sorted({clip_of[s] for s in scenes})
    if lc["predict"] not in ("saturation", "step"):
        res["readout_validity"] = readout_validity([fl(j) for j in floor_ix], theta[floor_ix], law_value(lc), n_boot)
    ref = invariance_reference(refp, fl, [fl(j) for j in floor_ix], lc) if null_prediction(lc) else None
    th_slope = float("nan")
    if null_prediction(lc) and "theta" in rd:      # did the push move the model's own theta readout?
        th_slope = ols_slope(theta_c, [rd["theta_floor"](j) for j in range(len(theta_c))])
        thc = {c: [[rd["theta"](clip_of[s], Z[s][key(c, d)]) for d in deltas] for s in scenes] for c in conds}
        null_verdict(res, push_effect(thc, deltas, scenes, th_slope, n_boot), min_effect)
    elif null_prediction(lc):
        null_verdict(res, None, min_effect)

    def get(s, c, p):
        """(decoded post state, forecast features) of scene s under pair condition c (p None: unsteered)."""
        if p is None:
            z = Z[s][key("base", 0)]
            return rd["post"](clip_of[s], z), z
        if c.endswith("_passive"):
            return (None if pas is None else pas(s, plans[s][c[: -len("_passive")]][float(p)])), None
        z = Z[s][f"{c}|{p:g}|{sp}{f}"]
        return rd["post"](clip_of[s], z), z
    res["pairs"] = _pair_arms(get, rd, law=law, lc=lc, clip_of=clip_of, pairs=pairs, theta=theta, theta_c=theta_c,
                              states=states, labels=labels, val=val, floor_ix=floor_ix, ref=ref, pc=pc, n_boot=n_boot,
                              th_slope=th_slope, control_moved=_control_moved(res))
    return res


def _control_moved(res: dict) -> bool:
    """Slope-0 laws: the steering verdict's check (b), a control direction moves the outcome (:func:`null_verdict`)."""
    return any(((res.get("verdict") or {}).get("effective_push") or {}).get("control_moves_outcome", {}).values())


def readout_validity(values, theta, law_slope: float, n_boot: int = 1000) -> dict:
    """The readout on REAL clips regressed on the true theta across scenes (one clip per scene): OLS slope with a
    bootstrap CI over clips; ``covers_law``: the CI covers the law's slope (the instrument can see the law's effect)."""
    y, x = np.asarray(values, float), np.asarray(theta, float)
    ok = np.isfinite(y) & np.isfinite(x)
    x, y = x[ok], y[ok]
    out = {"law_slope": float(law_slope), "n": int(ok.sum()), "n_clips": int(len(ok))}
    if len(x) < 3 or np.ptp(x) <= 0:
        return {**out, "slope": float("nan"), "ci95": None, "covers_law": False}
    sl = lambda ix: ols_slope(x[ix], y[ix])
    ci = _boot_ci(sl, np.arange(len(x)), n_boot)
    return {**out, "slope": sl(np.arange(len(x))), "ci95": ci, "intercept": float(np.polyfit(x, y, 1)[1]),
            "r": float(np.corrcoef(x, y)[0, 1]), "covers_law": bool(ci and ci[0] <= law_slope <= ci[1])}


def _pair_arms(get, rd, *, law, lc, clip_of, pairs, theta, theta_c, states, labels, val, floor_ix, ref, pc,
               n_boot: int = 1000, th_slope: float = 1.0, control_moved: bool = False) -> dict:
    """Swap test (subspace interchange; scored against the law's required readout change, or, for a law predicting
    no change, as invariance with the theta-readout check) and the collision patch (full-ROI swap; scored on B's
    decoded post velocity; its pass rule :func:`patch_verdict` -> ``pairs.patch.verdict``), each with its random-
    direction and passive controls."""
    nt_fn = NONTARGET.get(law)
    fl = {j: (rd["pre"](j), rd["floor"](j)) for j in floor_ix}
    sd = lambda g: float(np.nanstd([g(j, *fl[j]) for j in floor_ix])) or 1.0
    sd_t = sd(lambda j, p, q: val(j, q))
    sd_nt = sd(lambda j, p, q: nt_fn(p, q, states[j][2])) if nt_fn else 1.0
    th_fn = rd.get("theta")
    out, dpatch = {}, {}
    for arm in ("swap", "patch"):
        rows = [(s, p, j) for s, P in pairs.items() for c, p, j in P if c == arm]
        if not rows:
            continue
        for var in (arm, f"{arm}_random", f"{arm}_passive"):
            obs, req, nt, mv, dth, dvB, vA, vD, cx, kg, kd, vB = [], [], [], [], [], [], [], [], [], [], [], []
            for s, p, j in rows:
                i = clip_of[s]
                pre, (base, zb) = rd["pre"](i), get(s, "base", None)
                post, z = get(s, var, p)
                if post is None:
                    break
                if arm == "swap":
                    obs.append(val(i, post) - val(i, base))
                    req.append(pair_required(lc, theta, theta_c, i, j))
                    dth.append(theta_c[j] - theta_c[i])
                    mv.append(th_fn(i, z) - th_fn(i, zb) if th_fn is not None and z is not None else float("nan"))
                else:
                    dvB.append(np.asarray(post["v"], float)[1] - np.asarray(base["v"], float)[1])
                    vA.append(states[i][0]["v"][0]); vD.append(states[j][0]["v"][0]); cx.append(states[i][2])
                    kg.append(collision_gain(labels[i])); kd.append(collision_gain(labels[i], labels[j]))
                    vB.append(states[i][0]["v"][1])
                if nt_fn:
                    nt.append(nt_fn(pre, post, states[i][2]) - nt_fn(pre, base, states[i][2]))
            else:
                if arm == "swap" and ref is not None:
                    out[var] = invariance_scores(obs, mv, dth, *ref, float(pc["min_moved"]), float(pc["max_invariance"]),
                                                 th_slope, control_moved and var == "swap")
                elif arm == "swap":
                    out[var] = interchange_scores(obs, req, nt if nt_fn else None, sd_t, sd_nt)
                else:
                    out[var] = patch_scores(dvB, vA, vD, cx, kg, nt if nt_fn else None,
                                            float(np.nanstd([np.asarray(fl[j][1]["v"], float)[1][:2] @ np.asarray(states[j][2]["n"], float)[:2]
                                                             for j in floor_ix])) or 1.0, sd_nt, [s for s, _, _ in rows], n_boot,
                                            kd, vB)
                    dpatch[var] = patch_closer(dvB, vA, vD, cx, kg, kd, vB)
    if dpatch:
        out["patch"].update(patch_verdict(out, dpatch, [s for s, P in pairs.items() for c, _, _ in P if c == "patch"],
                                          n_boot))
    return out


def _score_generated(R, *, law, lc, scenes, clip_of, plans, pairs, theta, theta_c, scene, states, labels, deltas, conds,
                     n_boot, pc, fit_rows, recs, refp, positions, layer, site_theta=None, min_effect=0.25) -> dict:
    """DiTs: the steered GENERATED videos' track-B rows (READOUTS row functions); the input-level reference reads the
    real twins' videos with the same readout ("real|<clip>" rows); so does the invariance reference. The model's own
    theta (slope-0 laws: the swap's readout-moved check and :func:`null_verdict`'s push check) is read from the DiT's
    activations at the swapped / steered block and SNR (``site_theta``: (row, plan entry or None) -> theta_hat of the
    real site vector + that entry's shift, the calibrated R1 theta probe), while the outcome is read from the video."""
    row_fn = READOUTS[lc["readout"]][1]
    v = lambda r: row_fn(r) if r.get("measurable") else float("nan")
    curves = {c: [[v(R[s].get(f"{'base' if d == 0 else c}|{d:g}", {})) for d in deltas] for s in scenes] for c in conds}
    tol = float(pc["input_tol_sd"]) * float(np.std(theta[fit_rows]))
    real = lambda s, j: v(R[s].get(f"real|{recs[j]['clip'].replace('/', '__')}", {}))

    def inp(s, d):
        j = _nearest(clip_of[s], d, theta, scene, _input_cands(clip_of[s], s, scene, clip_of, lc), tol)
        return float("nan") if j is None else real(scene[j], j)
    curves["input"] = [[inp(s, d) for d in deltas] for s in scenes]
    res = score_response(curves, deltas, scenes, lc, n_boot, [theta[clip_of[s]] for s in scenes])
    if lc["predict"] not in ("saturation", "step"):
        fx = sorted({clip_of[s] for s in scenes})
        res["readout_validity"] = readout_validity([real(scene[j], j) for j in fx], theta[fx], law_value(lc), n_boot)
    th_slope = float("nan")
    if null_prediction(lc) and site_theta is not None:
        th_slope = ols_slope(theta_c, [site_theta(j, None) for j in range(len(theta_c))])
        thc = {c: [[site_theta(clip_of[s], None if d == 0 else plans[s][c][d]) for d in deltas] for s in scenes]
               for c in conds}
        null_verdict(res, push_effect(thc, deltas, scenes, th_slope, n_boot), min_effect)
    elif null_prediction(lc):
        null_verdict(res, None, min_effect)
    res["n_measurable"] = int(sum(r.get("measurable", False) for x in R.values() for r in x.values()))
    sw = [(s, p, j) for s, P in pairs.items() for c, p, j in P if c == "swap"]
    if sw:
        req = [pair_required(lc, theta, theta_c, clip_of[s], j) for s, p, j in sw]
        ch = lambda c: [v(R[s].get(f"{c}|{p:g}", {})) - v(R[s].get("base|0", {})) for s, p, _ in sw]
        if null_prediction(lc):          # invariance; theta read from the activations at the swapped block / SNR
            d = [abs(real(s, a) - real(s, b)) for s, (a, b) in refp.items()]
            ref = ((float(np.nanmedian(d)), "{} vs {}".format(*lc["invariance_ref"])) if np.isfinite(d).any() else
                   (float(np.nanstd([real(s, clip_of[s]) for s in scenes])), "sd_real"))
            dth = [theta_c[j] - theta_c[clip_of[s]] for s, _, j in sw]
            mv = lambda c: (None if site_theta is None else
                            [site_theta(clip_of[s], plans[s][c][float(p)]) - site_theta(clip_of[s], None) for s, p, _ in sw])
            res["pairs"] = {c: invariance_scores(ch(c), mv(c), dth, *ref, float(pc["min_moved"]), float(pc["max_invariance"]),
                                                 th_slope if site_theta is not None else 1.0,
                                                 _control_moved(res) and c == "swap")
                            for c in ("swap", "swap_random")}
        else:
            res["pairs"] = {c: interchange_scores(ch(c), req) for c in ("swap", "swap_random")}
    pa = [(s, p, j) for s, P in pairs.items() for c, p, j in P if c == "patch"]
    vB = lambda r: (np.asarray(r["velocity_post"], float)[1][:2] if r.get("measurable") and "velocity_post" in r
                    else np.full(2, np.nan))
    if pa:                               # collision patch on the generated videos: B's tracked post velocity
        dp = {}
        for c in ("patch", "patch_random"):
            args = ([vB(R[s].get(f"{c}|{p:g}", {})) - vB(R[s].get("base|0", {})) for s, p, _ in pa],
                    [states[clip_of[s]][0]["v"][0] for s, _, _ in pa], [states[j][0]["v"][0] for _, _, j in pa],
                    [states[clip_of[s]][2] for s, _, _ in pa], [collision_gain(labels[clip_of[s]]) for s, _, _ in pa])
            don = dict(k_donor=[collision_gain(labels[clip_of[s]], labels[j]) for s, _, j in pa],
                       vB_base=[states[clip_of[s]][0]["v"][1] for s, _, _ in pa])
            res.setdefault("pairs", {})[c] = patch_scores(*args, groups=[s for s, _, _ in pa], n_boot=n_boot, **don)
            dp[c] = patch_closer(*args, **don)
        res["pairs"]["patch"].update(patch_verdict(res["pairs"], dp, [s for s, _, _ in pa], n_boot))
    return res


# --------------------------------------------------------------------------- #
# localization: WHERE the law is used (layers / SNRs, tokens, components, a learned subspace)
# --------------------------------------------------------------------------- #
# One routine (:func:`localize`), driven by the ``localize:`` config block, output key ``localization``. It runs only for
# a model that passes (:func:`localize_gate`: the law's steering or swap test, or the L2 / L2-rep kink verdict "kink
# beyond baselines"); otherwise the json records why it was skipped. Interventions go through a runner
# (:class:`ModelRunner` on a loaded adapter; tests: tiny stand-ins) with one call per (key, clip, edits):
#   ("shift", layer, group, vec)   add vec to every token of a token group at the block output (pooled moves by vec)
#   ("set", layer, comp, target)   set a component's per-token-frame pooled output over the site tokens to target [T, d]
#                                  (attention head h: its slice of the output projection's input; mlp: the MLP output)
#   ("live", layer, donor, seed)   DiTs: the collision patch's paired donor pass (:func:`patch_live`) at that block
#   ("pset", layer, comp, donor)   DiTs: component patch from the donor's LIVE value in a paired pass (same timestep,
#                                  noise draw and CFG layout as the base; :func:`patch_live_component`). Mean-ablation
#                                  targets (knock-out) are the other clips' component outputs averaged over that SNR's
#                                  sampling steps of their own runs (a mean over clips, not a per-pair swap).
# DiT layer keys "snr<dB>/L<b>" gate every edit to the sampling steps whose nearest cached SNR is that SNR.

LOC_CONTROLS = ("random", "shuffled", "size", "passive", "untrained")   # step-1 controls (those present)
LOC_DEFAULTS = {
    "enabled": True, "gate": ["steering", "swap", "l2_kink"], "results_root": "results",
    "untrained_model": "untrained-vitl",
    "control_quantity": {"default": "log_radius_A", "momentum": "log_radius_ratio", "angular_momentum": "log_radius_B"},
    "max_groups": 12, "n_donors": 2, "max_layers": 12, "max_snrs": 4, "top_layers": 2, "run_unflagged": False,
    "token_groups": ["roi_A", "roi_B", "contact", "background", "pre", "post"],
    "component_layers": 1, "max_heads": 16, "top_k": 4, "n_control_components": 2, "max_ko_clips": 48,
    "creation_tau": 0.1, "bootstrap": 500,
    "attribution": {"max_fd_dims": 4096, "h": 1.0e-3},
    "das": {"objective": "auto", "ks": [1, 4, 16], "folds": 2, "steps": 300, "lr": 0.05, "batch": 8,
            "max_train_bases": 24, "max_eval_pairs": 48, "hidden": 32, "pcs": 32, "decoder_steps": 3000,
            "decoder_lr": 0.01, "decoder_wd": 0.01, "max_pairs": 400, "max_model_pairs": 16, "tol": 0.05},
    "dit": {},
}
_LOC_SUB = ("attribution", "das", "control_quantity")


def loc_config(cfg, diffusion: bool = False) -> dict:
    """The ``localize`` block over :data:`LOC_DEFAULTS`; DiTs: its ``dit`` overrides (tighter caps) on top."""
    user = dict(cfg.get("localize") or {})
    out = {**LOC_DEFAULTS, **user}
    for k_ in _LOC_SUB:
        out[k_] = {**LOC_DEFAULTS[k_], **dict(user.get(k_) or {})}
    if diffusion:
        for k_, v in dict(out.get("dit") or {}).items():
            out[k_] = {**out[k_], **dict(v)} if k_ in _LOC_SUB else v
    return out


# ---------------------------------------------------------------- gate


def _swap_passes(pairs, null: bool) -> bool:
    """The variant's swap test passes: zero-change laws its invariance verdict; otherwise IIA above the random
    direction's, beating doing nothing (sign-flip p <= 0.05) with an error ratio < 1."""
    sw, rd = (pairs or {}).get("swap") or {}, (pairs or {}).get("swap_random") or {}
    if null:
        return bool(sw.get("law_like"))
    iia, ri = float(sw.get("iia", np.nan)), float(rd.get("iia", np.nan))
    return bool(np.isfinite(iia) and iia > (ri if np.isfinite(ri) else 0.0) and sw.get("p_sign_flip", 1.0) <= 0.05
                and sw.get("error_ratio", np.inf) < 1.0)


KINK_PASS = ("kink beyond baselines", "representational kink beyond baselines")


def l2_kink_verdicts(path, law: str) -> list[str]:
    """The L2 / L2-rep kink verdicts in an l2.json that speak for ``law``: energy -> energy_kink (every distance) and
    energy_kink_rep; statics -> group_kink (the F11 across-group kink); other laws none."""
    p = Path(path)
    if not p.is_file():
        return []
    r = json.loads(p.read_text())
    v = []
    if law == "energy":
        v += [d.get("verdict") for e in (r.get("energy_kink") or {}).values() for d in (e.get("distances") or {}).values()]
        v += [e.get("verdict") for e in (r.get("energy_kink_rep") or {}).values()]
    elif law == "statics":
        v += [e.get("verdict") for e in (r.get("group_kink") or {}).values()]
    return [str(x) for x in v if x]


def localize_gate(res: dict, law: str, null: bool, use, l2_path) -> dict:
    """Localization runs only for a model that passes: the law's manifold steering (any decoded / generated variant
    law-like), its swap test (:func:`_swap_passes`), or an L2 / L2-rep kink "beyond baselines" (read from the model's
    l2.json when present). ``use``: which of steering / swap / l2_kink count."""
    variants = res.get("variants") or {}
    st = {n: bool((v.get("verdict") or {}).get("law_like")) for n, v in variants.items()}
    sw = {n: _swap_passes(v.get("pairs"), null) for n, v in variants.items()}
    kv = l2_kink_verdicts(l2_path, law)
    tests = {"steering": any(st.values()), "swap": any(sw.values()), "l2_kink": any(x in KINK_PASS for x in kv)}
    passed = [t for t in use if tests.get(t)]
    g = {"tests": tests, "used": list(use), "passed": passed, "gated_in": bool(passed), "l2_json": str(l2_path),
         "l2_json_found": Path(l2_path).is_file(), "l2_verdicts": kv, "steering_by_variant": st, "swap_by_variant": sw}
    if not passed:
        g["reason"] = (f"no gate passed ({', '.join(use)}): steering law-like in no variant, swap test passes in no "
                       f"variant, L2 kink verdicts {kv or 'absent'}")
    return g


# ---------------------------------------------------------------- statistics


def _boot_ci(stat, groups, n_boot: int, seed: int = 0):
    """Group-bootstrap 95% CI of stat(row indices) (whole groups resampled); None with < 2 groups or n_boot 0."""
    groups = np.asarray(groups)
    ug = list(dict.fromkeys(groups.tolist()))
    if len(ug) < 2 or not n_boot:
        return None
    rows = [np.flatnonzero(groups == g) for g in ug]
    rng = np.random.default_rng(seed)
    bs = np.array([stat(np.concatenate([rows[k] for k in rng.integers(0, len(ug), len(ug))])) for _ in range(n_boot)],
                  float)
    return [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))] if np.isfinite(bs).any() else None


def loc_pair_scores(obs, req, groups, null: bool = False, moved=None, dth=None, ref: float = float("nan"),
                    max_inv: float = 0.25, n_boot: int = 500, n_shuffle: int = 0) -> dict:
    """Swap / patch pairs of one condition: IIA = share of pairs closer to the law's outcome at the donor's theta
    than to the base's (|obs - req| < |obs|), with a group-bootstrap CI over base groups, and the median fraction
    achieved. Zero-change laws: the theta readout moved (median moved / dtheta, CI) and the invariance ratio
    median |obs| / ``ref`` (CI). ``score`` / ``score_ci95``: IIA, or for zero-change laws the readout moved.
    ``n_shuffle`` > 0: the SHUFFLED-TARGET control instead: the same observed changes scored against the required
    changes (dtheta) of other pairs (``n_shuffle`` seeded permutations, hits averaged per pair): IIA's chance level for
    an edit that moves the readout as much but not toward the donor's outcome."""
    obs, g = np.asarray(obs, float), np.asarray(groups)
    out = {"n_pairs": int(np.isfinite(obs).sum())}
    if not len(obs):
        return {**out, "score": float("nan"), "score_ci95": None}
    perms = [np.random.default_rng(7 + t).permutation(len(obs)) for t in range(int(n_shuffle))] or [np.arange(len(obs))]
    if not null:
        req0 = np.asarray(req, float)
        hits = []
        for pm in perms:
            req = req0[pm]
            ok = np.isfinite(obs) & np.isfinite(req) & (np.abs(req) > 1e-12)
            hits.append(np.where(ok, (np.abs(obs - req) < np.abs(obs)).astype(float), np.nan))
        hit = _nan_guard(np.nanmean, np.stack(hits), 0)
        req = req0
        ok = np.isfinite(obs) & np.isfinite(req) & (np.abs(req) > 1e-12)
        m = lambda ix: _nan_guard(np.nanmean, hit[ix])
        out.update(n_pairs=int(ok.sum()), iia=m(np.arange(len(hit))), iia_ci95=_boot_ci(m, g, n_boot),
                   fraction_achieved=_nan_guard(np.nanmedian, np.where(ok, obs / np.where(ok, req, 1.0), np.nan)),
                   metric="iia")
        out["score"], out["score_ci95"] = out["iia"], out["iia_ci95"]
        return out
    dth0 = np.asarray(dth, float)
    mv = np.asarray(moved if moved is not None else np.full(len(obs), np.nan), float)
    frs = []
    for pm in perms:
        dth = dth0[pm]
        frs.append(np.where(np.abs(dth) > 1e-12, mv / np.where(np.abs(dth) > 1e-12, dth, 1.0), np.nan))
    fr = _nan_guard(np.nanmean, np.stack(frs), 0)
    med = lambda v: (lambda ix: _nan_guard(np.nanmedian, v[ix]))
    inv = np.abs(obs) / ref if np.isfinite(ref) and ref > 0 else np.full(len(obs), np.nan)
    out.update(readout_moved=med(fr)(np.arange(len(fr))), readout_moved_ci95=_boot_ci(med(fr), g, n_boot),
               abs_change=med(np.abs(obs))(np.arange(len(obs))), invariance_ratio=med(inv)(np.arange(len(inv))),
               invariance_ratio_ci95=_boot_ci(med(inv), g, n_boot), reference=float(ref), metric="readout_moved")
    out["invariant"] = bool(np.isfinite(out["invariance_ratio"]) and out["invariance_ratio"] <= max_inv)
    out["score"], out["score_ci95"] = out["readout_moved"], out["readout_moved_ci95"]
    return out


def _lo_hi(rec: dict, key: str = "score") -> tuple[float, float]:
    ci = rec.get(f"{key}_ci95")
    v = float(rec.get(key, np.nan)) if rec.get(key) is not None else float("nan")
    return (float(ci[0]), float(ci[1])) if ci else (v, v)


def beats_controls(rec: dict, controls, key: str = "score", null: bool = False) -> bool:
    """``rec``'s CI lies above every control's CI (lower bound > each upper bound; point values without CIs); needs
    one finite control. Zero-change laws also need the invariance (``invariant``)."""
    lo = _lo_hi(rec, key)[0]
    his = [_lo_hi(c, key)[1] for c in controls if c]
    his = [h for h in his if np.isfinite(h)]
    ok = bool(np.isfinite(lo) and his and all(lo > h for h in his))
    return bool(ok and rec.get("invariant", True)) if null else ok


def knockout_scores(r: dict, theta, theta_c, groups, lc: dict, mode: str, controls=(), tau: float = 0.1,
                    n_boot: int = 500, null: bool = False) -> dict:
    """Law behaviour under knock-outs. ``r`` {condition: [n] readouts over the same clips}, "base" = unablated.
    Dependence = slope of the readout on theta (ladder: within-group, i.e. demeaned per group; across: OLS on the
    residualised theta_c) with a group-bootstrap CI; energy (saturation) on the legal rungs (theta <= fit_max), plus the
    creation rate P(r > 1 + tau), the median eps_hat and the slope on the rungs theta >= 1 (saturation: ~0). Effect of
    a knock-out vs base (paired, same bootstrap draws): slope laws sign(law) (dep_base - dep), saturation / step
    dep_base - dep, zero-change laws |dep| - |dep_base| (invariance broken); energy also the creation-rate increase.
    ``passes``: the effect's CI lies above every control knock-out's (energy: the dependence or the creation effect)."""
    th, raw, g = np.asarray(theta_c, float), np.asarray(theta, float), np.asarray(groups)
    sat = lc["predict"] == "saturation"
    dep_rows = raw <= float(lc.get("fit_max") if lc.get("fit_max") is not None else np.inf) + 1e-9
    above = raw >= 1.0 - 1e-9
    sgn = 1.0 if lc["predict"] in ("saturation", "step") or null else float(np.sign(law_value(lc)) or 1.0)

    def slope(v, ix, rows):
        ix = ix[rows[ix]]
        vv, tt, gg = v[ix], th[ix], g[ix]
        ok = np.isfinite(vv) & np.isfinite(tt)
        vv, tt, gg = vv[ok], tt[ok], gg[ok]
        if len(vv) < 3:
            return float("nan")
        if mode == "ladder":
            for u in set(gg.tolist()):
                s = gg == u
                vv[s], tt[s] = vv[s] - vv[s].mean(), tt[s] - tt[s].mean()
        else:
            vv, tt = vv - vv.mean(), tt - tt.mean()
        den = float((tt ** 2).sum())
        return float((vv * tt).sum() / den) if den > 1e-12 else float("nan")

    def creation(v, ix):
        v = v[ix]
        v = v[np.isfinite(v)]
        return float(np.mean(v > 1 + tau)) if len(v) else float("nan")
    allr = np.arange(len(th))
    base = np.asarray(r["base"], float)
    out = {}
    for c, v in r.items():
        v = np.asarray(v, float)
        e = {"dependence": slope(v, allr, dep_rows), "dependence_ci95": _boot_ci(lambda ix: slope(v, ix, dep_rows), g, n_boot),
             "n_clips": int(np.isfinite(v).sum())}
        if sat:
            e.update(creation_rate=creation(v, allr), median_eps_hat=_nan_guard(np.nanmedian, v),
                     slope_above_1=slope(v, allr, above))
        if c != "base":
            if null:
                eff = lambda ix, v=v: abs(slope(v, ix, dep_rows)) - abs(slope(base, ix, dep_rows))
            else:
                eff = lambda ix, v=v: sgn * (slope(base, ix, dep_rows) - slope(v, ix, dep_rows))
            e["effect"], e["effect_ci95"] = eff(allr), _boot_ci(eff, g, n_boot)
            if sat:
                ce = lambda ix, v=v: creation(v, ix) - creation(base, ix)
                e["creation_effect"], e["creation_effect_ci95"] = ce(allr), _boot_ci(ce, g, n_boot)
        out[c] = e
    ctrl = [out[c] for c in controls if c in out]
    for c in out:
        if c != "base" and c not in controls:
            out[c]["passes"] = bool(beats_controls(out[c], ctrl, "effect") or
                                    (sat and beats_controls(out[c], ctrl, "creation_effect")))
    return out


# ---------------------------------------------------------------- DAS (distributed alignment search)


def theta_decoder(X, y, hidden: int = 64, pcs: int = 32, steps: int = 500, lr: float = 0.01, seed: int = 0,
                  weight_decay: float = 1e-3, val_frac: float = 0.2):
    """The DAS readout (objective ``readout``): a small MLP (tanh, one hidden layer; inputs the top ``pcs`` principal
    components of X, standardised) from cached activations X [n, D] to theta y [n], trained by Adam (full batch, MSE,
    weight decay) with early stopping on a held-out ``val_frac`` of the rows (seeded), then frozen. Returns
    f(torch [m, D]) -> torch [m] (float64), with ``f.train_r2`` and ``f.val_r2``."""
    import torch
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    mu = X.mean(0)
    U = np.linalg.svd(X - mu, full_matrices=False)[2][:pcs].T
    Z = (X - mu) @ U
    sd = Z.std(0) + 1e-8
    ym, ys = float(y.mean()), float(y.std()) + 1e-12
    with torch.random.fork_rng(devices=[]):          # seeded init without reseeding the caller's global torch RNG
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(U.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Zt, yt = torch.as_tensor(Z / sd), torch.as_tensor((y - ym) / ys)
    perm = np.random.default_rng(seed).permutation(len(y))
    nv = int(round(val_frac * len(y))) if len(y) >= 10 else 0
    va, trn = torch.as_tensor(perm[:nv]), torch.as_tensor(perm[nv:])
    mse = lambda rows: float(((net(Zt[rows])[:, 0] - yt[rows]) ** 2).mean())
    best = (np.inf, None, 0)
    for it in range(steps):
        opt.zero_grad()
        loss = ((net(Zt[trn])[:, 0] - yt[trn]) ** 2).mean()
        loss.backward()
        opt.step()
        if nv and (it % 10 == 9 or it == steps - 1):
            with torch.no_grad():
                v = mse(va)
            if v < best[0]:
                best = (v, [p_.detach().clone() for p_ in net.parameters()], it)
    if best[1] is not None:
        with torch.no_grad():
            for p_, b_ in zip(net.parameters(), best[1]):
                p_.copy_(b_)
    for p_ in net.parameters():
        p_.requires_grad_(False)
    mu_t, U_t, sd_t = (torch.as_tensor(a) for a in (mu, U, sd))

    def f(x):
        return net(((x - mu_t) @ U_t) / sd_t)[:, 0] * ys + ym
    with torch.no_grad():
        f.train_r2 = float(1 - mse(trn))
        f.val_r2 = float(1 - mse(va)) if nv else float("nan")
    return f


def das_fit(X, y, pairs, k: int, f, steps: int = 300, lr: float = 0.05, seed: int = 0, restarts: int = 3) -> np.ndarray:
    """Distributed alignment search: an orthonormal W [D, k] (the first k columns of a learned rotation R, QR of a free
    parameter) such that interchanging the k rotated coordinates of a base with a donor's, z' = x_b + W W^T (x_d - x_b),
    makes the frozen readout f output the donor's theta: minimise mean (f(z') - y_d)^2 / var(y) over the training
    pairs by Adam. ``restarts`` initialisations (the first: the top-k principal directions of the pairs' donor - base
    differences; the others random), the lowest final training loss kept (the objective is not convex)."""
    import torch
    X = np.asarray(X, np.float64)
    k = min(int(k), X.shape[1])
    b = torch.as_tensor(X[[i for i, _ in pairs]])
    d = torch.as_tensor(X[[j for _, j in pairs]])
    yd = torch.as_tensor(np.asarray(y, np.float64)[[j for _, j in pairs]])
    ys = float(np.std(y)) + 1e-12
    diff = (d - b).numpy()
    pcs = np.linalg.svd(diff - diff.mean(0), full_matrices=False)[2][:k].T
    best = (np.inf, None)
    for r in range(max(int(restarts), 1)):
        gen = torch.Generator().manual_seed(seed * 1000 + r)
        A0 = torch.randn(X.shape[1], k, generator=gen, dtype=torch.float64)
        if r == 0 and pcs.shape[1] == k:
            A0 = torch.as_tensor(pcs) + 0.01 * A0
        A = A0.requires_grad_(True)
        opt = torch.optim.Adam([A], lr=lr)
        loss_ = lambda: (((f(b + (d - b) @ (W := torch.linalg.qr(A)[0]) @ W.T) - yd) / ys) ** 2).mean()
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_()
            loss.backward()
            opt.step()
        with torch.no_grad():
            fl = float(loss_())
        if fl < best[0]:
            best = (fl, torch.linalg.qr(A.detach())[0].numpy())
    return best[1]


def das_hits(X, y, pairs, W, f) -> np.ndarray:
    """Per pair: the interchanged readout is closer to the donor's theta than to the base's."""
    import torch
    if not len(pairs):
        return np.zeros(0)
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    b, d = X[[i for i, _ in pairs]], X[[j for _, j in pairs]]
    with torch.no_grad():
        p = f(torch.as_tensor(b + (d - b) @ W @ W.T)).numpy()
    return (np.abs(p - y[[j for _, j in pairs]]) < np.abs(p - y[[i for i, _ in pairs]])).astype(float)


def principal_cosines(W, Q) -> list[float]:
    """Cosines of the principal angles between span(W) and span(Q) (orthonormal columns), largest first."""
    if W.shape[1] == 0 or Q.shape[1] == 0:
        return []
    return np.clip(np.linalg.svd(W.T @ Q, compute_uv=False), 0.0, 1.0).tolist()


def _all_pairs(rows, theta, scene, mode: str, min_dt: float, cap: int, rng) -> list[tuple[int, int]]:
    """Ordered (base, donor) pairs among ``rows``: ladder -> same scene, across -> different scenes; |dtheta| >=
    min_dt; a random subset of ``cap``."""
    P = [(int(i), int(j)) for i in rows for j in rows if i != j and abs(theta[j] - theta[i]) >= min_dt
         and ((scene[i] == scene[j]) if mode == "ladder" else (scene[i] != scene[j]))]
    if cap and len(P) > cap:
        P = [P[t] for t in sorted(rng.choice(len(P), cap, replace=False))]
    return P


# ---------------------------------------------------------------- component hooks (encoders and DiTs)


def _comp_modules(blk):
    """(attention output projection, number of heads, MLP) of a transformer block: V-JEPA 2 ``attention.proj`` /
    ``mlp``, VideoMAE ``attn.proj`` (``h`` heads) / ``mlp``, diffusers ``attn1.to_out[0]`` (``heads``) / ``ff`` or
    ``ffn``; None / 0 where absent."""
    import torch
    attn = next((getattr(blk, a) for a in ("attention", "attn", "attn1", "self_attn") if hasattr(blk, a)), None)
    proj, H = None, 0
    if attn is not None:
        for p in ("proj", "to_out", "o_proj", "out_proj"):
            if hasattr(attn, p):
                proj = getattr(attn, p)
                proj = proj[0] if isinstance(proj, (torch.nn.ModuleList, torch.nn.Sequential)) else proj
                break
        H = next((int(getattr(attn, a)) for a in ("num_attention_heads", "heads", "num_heads", "h")
                  if isinstance(getattr(attn, a, None), int)), 0)
    mlp = next((getattr(blk, a) for a in ("mlp", "ff", "ffn", "feed_forward") if hasattr(blk, a)), None)
    return proj, H, mlp


def _out_dim(mod) -> int:
    """Output width of a module (its last Linear's out_features)."""
    import torch
    lin = [x for x in mod.modules() if isinstance(x, torch.nn.Linear)]
    if not lin:
        raise ValueError(f"L4 localize: no Linear in {type(mod).__name__}")
    return int(lin[-1].out_features)


def block_components(m, layer: str) -> list[str]:
    """Component names of block ``layer``: "attn.h<i>" per attention head, "mlp"."""
    proj, H, mlp = _comp_modules(blocks(m, layer)[block_index(layer)])
    return [f"attn.h{h}" for h in range(H if proj is not None else 0)] + (["mlp"] if mlp is not None else [])


@contextmanager
def comp_edit(m, layer: str, comp: str, idx, fn, active=lambda: True):
    """While active, ``fn(v)`` (v [B, n, d]: the component's output at block token positions ``idx``, all video
    tokens for idx None) returns its replacement (None: unchanged, e.g. a capture). comp "attn.h<i>": head i's slice of
    the attention output projection's INPUT (pre-hook); "mlp": the MLP output; "block": the block output. Video
    tokens are the LAST tokens of the component's sequence (CogVideoX concatenates the text first), counted from the
    block's input length."""
    import torch
    blk = blocks(m, layer)[block_index(layer)]
    proj, H, mlp = _comp_modules(blk)
    st = {"n": None}

    def pre_blk(mod, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        st["n"] = int(h.shape[1]) if hasattr(h, "shape") else None

    def apply(t):
        if not active():
            return None
        off = t.shape[1] - (st["n"] or t.shape[1])
        pos = torch.arange(off, t.shape[1], device=t.device) if idx is None else \
            torch.as_tensor(np.asarray(idx) + off, dtype=torch.long, device=t.device)
        if comp.startswith("attn.h"):
            h, dh = int(comp[6:]), t.shape[-1] // H
            sl = slice(h * dh, (h + 1) * dh)
        else:
            sl = slice(None)
        new = fn(t[:, pos, sl])
        if new is None:
            return None
        t = t.clone()
        t[:, pos, sl] = new.to(t.dtype)
        return t
    hs = [blk.register_forward_pre_hook(pre_blk, with_kwargs=True)]
    if comp == "block":
        def bh(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            n = apply(h)
            return None if n is None else ((n, *out[1:]) if isinstance(out, tuple) else n)
        hs.append(blk.register_forward_hook(bh))
    elif comp == "mlp":
        if mlp is None:
            raise ValueError(f"L4 localize: block {layer} has no MLP")
        hs.append(mlp.register_forward_hook(lambda mod, inp, out: apply(out)))
    elif comp.startswith("attn.h"):
        if proj is None or not H:
            raise ValueError(f"L4 localize: block {layer} has no attention output projection")

        def ph(mod, args):
            n = apply(args[0])
            return None if n is None else (n, *args[1:])
        hs.append(proj.register_forward_pre_hook(ph))
    else:
        raise ValueError(f"L4 localize: unknown component {comp!r}")
    try:
        yield
    finally:
        for h_ in hs:
            h_.remove()


def _frames_of(idx, per: int) -> np.ndarray:
    return np.asarray(idx) // int(per)


def _set_frames(target, frames):
    """fn for :func:`comp_edit`: each token-frame's pooled component output (over the edited tokens of that frame) set
    to target[frame] (frames with a NaN or missing target unchanged)."""
    import torch
    tg = np.asarray(target, float)

    def fn(v):
        v = v.clone()
        for f in np.unique(frames):
            if f >= len(tg) or not np.isfinite(tg[f]).all():
                continue
            s = torch.as_tensor(np.flatnonzero(frames == f), device=v.device)
            cur = v[:, s].mean(1, keepdim=True)
            v[:, s] = v[:, s] + (torch.as_tensor(tg[f], device=v.device, dtype=v.dtype)[None, None] - cur)
        return v
    return fn


def _pool_frames(v, frames, T: int) -> np.ndarray:
    """[B, n, d] component output at the edited tokens -> [T, d] per-token-frame mean (row 0; NaN for frames without
    tokens)."""
    a = v[0].detach().float().cpu().numpy()
    out = np.full((T, a.shape[-1]), np.nan)
    for f in np.unique(frames):
        if f < T:
            out[f] = a[frames == f].mean(0)
    return out


class _GradTorch:
    """``m.torch`` stand-in whose ``no_grad()`` keeps gradients (adapters wrap forecast() in ``torch.no_grad()``)."""

    def __init__(self, t):
        self._t = t

    def __getattr__(self, n):
        return getattr(self._t, n)

    def no_grad(self):
        return self._t.enable_grad()


@contextmanager
def _grad_through(m):
    """While active, the adapter's own ``torch.no_grad()`` blocks keep the autograd graph (attribution patching)."""
    own = getattr(m, "torch", None)
    if own is not None:
        m.torch = _GradTorch(own)
    try:
        yield
    finally:
        if own is not None:
            m.torch = own


def fd_grad(fn, z, basis=None, max_dims: int = 4096, h_rel: float = 1e-3):
    """Finite-difference gradient of the scalar readout fn(z) with respect to the pooled forecast features z (any
    shape), along the orthonormal rows of ``basis`` [p, z.size] (default: every coordinate when z.size <= max_dims;
    else None). The readout decodes z with ridge instruments, whose weights lie in the span of their training rows,
    so a basis of the forecast rows' principal directions loses little."""
    z = np.asarray(z, float)
    flat = z.reshape(-1)
    if basis is None:
        if flat.size > max_dims:
            return None
        basis = np.eye(flat.size)
    h = h_rel * max(float(np.sqrt(np.mean(flat ** 2))), 1e-12)
    f0 = fn(z)
    if not np.isfinite(f0):
        return None
    g = np.zeros(flat.size)
    for b in np.asarray(basis, float):
        f1 = fn((flat + h * b).reshape(z.shape))
        if np.isfinite(f1):
            g += (f1 - f0) / h * b
    return g.reshape(z.shape)


def _pool_torch(acts: dict, space: str):
    """The torch counterpart of :func:`_pool_forecast` for one readout space ("final", "<layer>", "...|t")."""
    sp, per_t = space.split("|")[0], space.endswith("|t")
    for key, F in acts.items():
        if ("final" if str(key) == "-1" else str(key)) == sp:
            F = F.float()
            return F.reshape(len(F), -1, F.shape[-1]).mean(1) if per_t else F.reshape(-1, F.shape[-1]).mean(0)
    raise KeyError(f"L4 localize: readout space {space!r} not among the forecast keys {list(acts)}")


@contextmanager
def edit_hooks(m, edits, groups: dict, per: int, T: int, diffusion: bool = False, cached=(), caps: dict | None = None):
    """Install a localization run's edits (module comment above) on adapter ``m``; ``groups`` {name: flat token
    indices}, ``per`` tokens per token-frame, ``T`` token-frames. DiT layer keys "snr<dB>/L<b>" act only at sampling
    steps whose nearest cached SNR (``cached``) is that SNR (the step SNR from each ``m._pass`` timestep, as
    :func:`steer_steps`). ``caps`` {layer: list}: the block output pooled over the site tokens after the edits, per
    active call. ("grad", layer, comp, e) adds the tensor e[frame] (requires grad) to the component's output."""
    import torch
    from contextlib import ExitStack
    st = {"snr": None}
    gated = diffusion and any(snr_of(e[1]) is not None for e in edits)
    gated = gated or (diffusion and any(snr_of(l) is not None for l in (caps or {})))
    act = lambda layer: (lambda: snr_of(layer) is None or not diffusion or st["snr"] == snr_of(layer))
    with ExitStack() as es:
        if gated:
            own, orig = "_pass" in vars(m), m._pass

            def _pass(z_t, t, *a, **kw):
                st["snr"] = nearest_snr(step_snr_db(m, t), cached)
                return orig(z_t, t, *a, **kw)
            m._pass = _pass
            es.callback(lambda: (setattr(m, "_pass", orig) if own else delattr(m, "_pass")))
        for e in edits:
            kind, layer = e[0], e[1]
            if kind == "shift":
                idx, vec = np.asarray(groups[e[2]]), torch.as_tensor(np.asarray(e[3], np.float64))
                es.enter_context(comp_edit(m, layer, "block", idx, lambda v, vec=vec: v + vec.to(v.device, v.dtype),
                                           act(layer)))
            elif kind == "set":
                idx = np.asarray(groups["site"])
                es.enter_context(comp_edit(m, layer, e[2], idx, _set_frames(e[3], _frames_of(idx, per)), act(layer)))
            elif kind == "grad":
                idx = np.asarray(groups["site"])
                fr = torch.as_tensor(_frames_of(idx, per), dtype=torch.long)
                es.enter_context(comp_edit(m, layer, e[2], idx, lambda v, p=e[3], fr=fr: v + p[fr.to(p.device)][None].to(v.dtype),
                                           act(layer)))
            elif kind not in ("live", "pset"):
                raise ValueError(f"L4 localize: unknown edit {kind!r}")
        for layer, out in (caps or {}).items():
            idx = np.asarray(groups["site"])
            es.enter_context(comp_edit(m, layer, "block", idx, lambda v, out=out: out.append(
                v[0].detach().float().mean(0).cpu().numpy()) or None, act(layer)))
        yield


def pred_real_capture(m, x, k: int, T: int, layer: str, comps, n_masks: int) -> dict:
    """Encoders' post-event sites: every component's output at the post-event token positions in the model's own
    P-pred-real pass (masked real post tokens, the rest of the clip visible; the pass the post site's curve is fitted
    on). One capture per mask pass, assembled by position: pass p's targets are the post tokens of spatial group p
    (``spatial_mask_groups``, pred_real's own partition); their sequence index is the position itself (V-JEPA 2's
    position-sorted predictor) or, for VideoMAE's decoder (visible tokens then decode tokens, each in clip order),
    its rank in that concatenation. {comp: [T_post * per, d]} (comp "block": the block output)."""
    import torch
    from contextlib import ExitStack
    from physlaw.models.base import spatial_mask_groups
    per = int(np.prod(m.token_grid.grid_hw))
    kt = k // int(m.token_grid.tubelet)
    n_ctx, n_all = kt * per, T * per
    caps = {c: [] for c in comps}
    with ExitStack() as es, torch.no_grad():
        for c in comps:
            es.enter_context(comp_edit(m, layer, c, None, lambda v, c=c: caps[c].append(v[0].float().cpu().numpy()) or None))
        m.pred_real(x, k, n_masks=n_masks, layers=[])
    out = {}
    for c, passes in caps.items():
        full = np.full((n_all - n_ctx, passes[0].shape[-1]), np.nan)
        for p, grp_ in enumerate(spatial_mask_groups(per, n_masks, 0)):
            tgt = np.sort(np.concatenate([n_ctx + t * per + np.asarray(grp_) for t in range(T - kt)]))
            ctx = np.setdiff1d(np.arange(n_all), tgt)
            order = np.concatenate([ctx, tgt]) if hasattr(m, "_decoder") else np.arange(n_all)
            seq = np.empty(n_all, int)
            seq[order] = np.arange(n_all)
            full[tgt - n_ctx] = passes[p][seq[tgt]]
        out[c] = full
    return out


def token_groups(m, c0, fr, k: int, windows=("pre",), names=LOC_DEFAULTS["token_groups"]) -> dict:
    """Token groups of a clip for the token-localization step, as flat token indices: per window w (``windows``: the
    token-frames the hooked pass holds, "pre" = context token-frames, "post" = the rest) the object ROIs "roi_A.w",
    "roi_B.w", the contact region "contact.w" (tokens/pool.contact_mask: both ROIs in the last pre-contact token-frame,
    dilated), "background.w" (none of these), and the whole window "w"; empty groups dropped."""
    from physlaw.tokens.pool import contact_mask
    from physlaw.tokens.project import project_masks, roi_object_ids, token_frame_phase
    kt = m.n_context_tokens(k) if hasattr(m, "n_context_tokens") else k // m.token_grid.tubelet
    ob = project_masks(c0.masks, fr, m.token_grid, roi_object_ids(c0.labels))
    ph = token_frame_phase(fr, m.token_grid, int(c0.labels["physlaw"]["first_post_contact_frame"]))
    ra, rb = np.asarray(ob["A"].roi, bool), np.asarray(ob["B"].roi, bool)
    reg = {"roi_A": ra, "roi_B": rb, "contact": contact_mask(ra, rb, ph)}
    reg["background"] = ~(ra | rb | reg["contact"])
    tf = np.arange(len(ra))[:, None, None]
    win = {"pre": np.broadcast_to(tf < kt, ra.shape), "post": np.broadcast_to(tf >= kt, ra.shape)}
    out = {}
    for w in windows:
        for r_, msk in reg.items():
            if r_ in names:
                out[f"{r_}.{w}"] = np.flatnonzero((msk & win[w]).reshape(-1))
        if w in names:
            out[w] = np.flatnonzero(np.asarray(win[w]).reshape(-1))
    return {g: v for g, v in out.items() if len(v)}


class LocReadout:
    """How a localization run is read: ``value(i, z) -> {"r", "theta", "post_v"}`` from the pooled forecast feature of
    ``space`` (encoders: the R1 instrument; "theta" the model's own theta readout), or ``video(i, vid)`` from the
    generated video (DiTs: the track-B readout; no theta readout)."""

    def __init__(self, space=None, value=None, video=None):
        self.space, self.value, self.video = space, value, video


def _json_safe(d: dict) -> dict:
    out = {}
    for k_, v in d.items():
        if isinstance(v, dict):
            out[k_] = {str(a): np.asarray(b, float).tolist() for a, b in v.items()}
        elif v is None or isinstance(v, (str, bool)):
            out[k_] = v
        else:
            a = np.asarray(v, float)
            out[k_] = a.tolist() if a.ndim else float(a)
    return out


class ModelRunner:
    """Localization interventions on a loaded adapter ``m``. ``inputs(i)`` -> {"x", "k", "T", "groups" {name: flat
    token indices, incl. "site"}}; ``readout``: a :class:`LocReadout`. Encoders rerun forecast() (pre-event sites: the
    context-only encoder pass; post-event sites: the predictor / decoder targets) and read the forecast; DiTs sample
    with the pinned settings and read the generated video, every edit gated to its SNR's steps. Every run is cached as
    json in ``work`` (resumable; ``force`` recomputes). ``cached_pool(layer, group)`` (DiTs): the stage-4 activations of
    a token group ([n, D] or None)."""

    def __init__(self, m, inputs, readout: LocReadout, positions: str, work, diffusion: bool = False, cached_snrs=(),
                 cached_pool=None, n_masks: int = 4, force: bool = False, max_fd_dims: int = 4096, h_rel: float = 1e-3,
                 log=print):
        self.m, self.inputs, self.readout, self.positions = m, inputs, readout, positions
        self.work, self.diffusion, self.cached = Path(work), diffusion, [float(s) for s in cached_snrs]
        self.cached_pool, self.n_masks, self.force, self.log = cached_pool, int(n_masks), force, log
        self.max_fd_dims, self.h_rel = int(max_fd_dims), float(h_rel)
        self.kind = "dit" if diffusion else "encoder"
        self.live = diffusion
        self.per = int(np.prod(m.token_grid.grid_hw))
        self._memo, self._last, self._acts, self._z, self._gnames = {}, None, {}, {}, set()
        self.no_grad_reason = ("DiT sampling is not differentiated (cost): exact patching on the capped component list"
                               if diffusion else None)

    def _in(self, i):
        if self._last is None or self._last[0] != i:
            self._last = (i, self.inputs(i))
            self._gnames |= {g for g in self._last[1]["groups"] if g != "site"}
        return self._last[1]

    def _file(self, key):
        import re
        return self.work / (re.sub(r"[^A-Za-z0-9_.+=-]", "_", key) + ".json")

    def run(self, key: str, i: int, edits) -> dict:
        """One run of clip i under ``edits`` (cached; the cache key also hashes the edit payloads, so a changed curve,
        donor or DAS subspace is never served from a stale file)."""
        key = f"{key}|{_edit_sig(edits)}" if edits else key
        if key in self._memo:
            return self._memo[key]
        f = self._file(key)
        if f.exists() and not self.force:
            res = json.loads(f.read_text())
        else:
            res = _json_safe(self._forward(i, list(edits), want_z=not edits))
            self.work.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res))
        res = {**res, "site": {k_: np.asarray(v, float) for k_, v in (res.get("site") or {}).items()}}
        if not edits and res.get("z") is not None:
            self._z[i] = np.asarray(res["z"], float)
        self._memo[key] = res
        return res

    def _sample(self, x, k):
        if self.diffusion:
            return self.m.forecast(x, k, seed=0, return_video=True)
        return self.m.forecast(x, k), None

    def _read(self, i, acts, vid) -> dict:
        if self.readout.video is not None and vid is not None:
            return dict(self.readout.video(i, vid))
        z = _pool_forecast(acts)[self.readout.space]
        return {**self.readout.value(i, z), "z": z}

    def _forward(self, i, edits, want_z: bool = False) -> dict:
        import torch
        from contextlib import ExitStack
        inp = self._in(i)
        caps = {e[1]: [] for e in edits}
        slog = []
        with ExitStack() as es, torch.no_grad():
            es.enter_context(edit_hooks(self.m, edits, inp["groups"], self.per, inp["T"], self.diffusion, self.cached, caps))
            for e in edits:
                gate = (lambda t, s=snr_of(e[1]): nearest_snr(step_snr_db(self.m, t), self.cached) == s) \
                    if snr_of(e[1]) is not None else None
                if e[0] == "live":
                    don = self.inputs(int(e[2]))
                    es.enter_context(patch_live(self.m, e[1], inp["groups"]["site"], don["groups"]["site"], inp["x"],
                                                don["x"], inp["k"], slog, e[3], gate=gate))
                elif e[0] == "pset":             # DiT component patch: the donor's live value, paired pass
                    don = self.inputs(int(e[3]))
                    es.enter_context(patch_live_component(self.m, e[1], e[2], inp["groups"]["site"],
                                                          don["groups"]["site"], inp["x"], don["x"], inp["k"], slog,
                                                          gate, ctx_only=self.positions != "post"))
            acts, vid = self._sample(inp["x"], inp["k"])
        res = self._read(i, acts, vid)
        if not want_z:
            res.pop("z", None)
        res["site"] = {l: np.mean(v, 0) for l, v in caps.items() if v}
        return res

    # ---- token groups

    def _post_pred(self, layer) -> bool:
        return not self.diffusion and self.positions == "post" and str(layer).startswith("pred/") and self.n_masks > 1

    def groups(self, layer) -> list[str]:
        """Token groups whose activations exist at this layer (DiTs: the cached pools)."""
        names = sorted(self._gnames)
        if self.diffusion:
            return [g for g in names if self.cached_pool is not None and self.cached_pool(layer, _pool_name(g)) is not None]
        return names

    def acts(self, layer, group, rows):
        """[n, D] pooled activations of token group ``group`` at ``layer`` for the clips ``rows`` (NaN elsewhere), in
        the pass the hook acts on (post-event window at encoders' predictor blocks: the P-pred-real activations, the
        post site's own space). Cached per layer in work/_acts_<layer>.npz."""
        if self.diffusion:
            X = self.cached_pool(layer, _pool_name(group)) if self.cached_pool is not None else None
            return None if X is None else np.asarray(X, np.float64)
        import torch
        rows = sorted(int(r) for r in rows)
        f = self._file(f"_acts|{layer}").with_suffix(".npz")
        memo = self._acts.setdefault(layer, {})
        if not memo and f.exists() and not self.force:
            z = np.load(f)
            memo.update({str(g): {int(r): v for r, v in zip(z["rows"], z[f"g{t}"])} for t, g in enumerate(z["groups"])})
        todo = [r for r in rows if any(r not in v for v in memo.values()) or not memo]
        for r in todo:
            inp = self._in(r)
            gs = {g: v for g, v in inp["groups"].items() if g != "site"}
            got = {}
            with torch.no_grad():
                cap = {g: [] for g in gs}
                from contextlib import ExitStack
                with ExitStack() as es:
                    for g, idx in gs.items():
                        es.enter_context(comp_edit(self.m, layer, "block", idx, lambda v, g=g: cap[g].append(
                            v[0].float().mean(0).cpu().numpy()) or None))
                    self._sample(inp["x"], inp["k"])
                got = {g: np.mean(v, 0) for g, v in cap.items() if v}
                if self._post_pred(layer):
                    kt = inp["k"] // int(self.m.token_grid.tubelet)
                    pr = self.m.pred_real(inp["x"], inp["k"], n_masks=self.n_masks, layers=[block_index(layer)])
                    P = pr[PRED_LAYER.format(block_index(layer))]
                    P = P.reshape(-1, P.shape[-1]).float().cpu().numpy()
                    for g, idx in gs.items():
                        post = np.asarray(idx) - kt * self.per
                        if g.endswith("post") and (post >= 0).all():
                            got[g] = P[post].mean(0)
            for g in self._gnames | set(gs):
                memo.setdefault(g, {})[r] = got.get(g)
        D = next((v.shape[-1] for d in memo.values() for v in d.values() if v is not None), None)
        if todo and D is not None:
            gl = sorted(memo)
            rr = sorted(set().union(*[set(memo[g]) for g in gl]))
            nanv = np.full(D, np.nan)
            self.work.mkdir(parents=True, exist_ok=True)
            np.savez(f, rows=np.array(rr), groups=np.array(gl),
                     **{f"g{t}": np.stack([memo[g].get(r) if memo[g].get(r) is not None else nanv for r in rr])
                        for t, g in enumerate(gl)})
        if group not in memo or D is None:
            return None
        n = max(max(memo[group]) + 1, max(rows) + 1)
        X = np.full((n, D), np.nan)
        for r, v in memo[group].items():
            if v is not None:
                X[r] = v
        return X

    # ---- components

    def components(self, layer) -> list[str]:
        return block_components(self.m, layer)

    def comp_acts(self, layer, rows, comps) -> dict:
        """{comp: {row: [T, d] per-token-frame pooled output over the site tokens}} in the pass the edits act on
        (post-event sites at encoders' predictor blocks: P-pred-real; DiTs: the mean over that SNR's sampling steps);
        cached in work/_comp_<layer>_<row>.npz."""
        import torch
        from contextlib import ExitStack
        out = {c: {} for c in comps}
        for r in sorted(int(x) for x in rows):
            f = self._file(f"_comp|{layer}|{r}").with_suffix(".npz")
            if f.exists() and not self.force:
                z = np.load(f)
                have = [str(c) for c in z["comps"]]
                if all(c in have for c in comps):
                    for c in comps:
                        out[c][r] = z[f"c{have.index(c)}"]
                    continue
            inp = self._in(r)
            idx = np.asarray(inp["groups"]["site"])
            fr = _frames_of(idx, self.per)
            got = {}
            if self._post_pred(layer):
                full = pred_real_capture(self.m, inp["x"], inp["k"], inp["T"], layer, comps, self.n_masks)
                kt = inp["k"] // int(self.m.token_grid.tubelet)
                for c in comps:
                    a = full[c][idx - kt * self.per]
                    got[c] = np.full((inp["T"], a.shape[-1]), np.nan)
                    for t in np.unique(fr):
                        got[c][t] = a[fr == t].mean(0)
            else:
                cap = {c: [] for c in comps}
                with ExitStack() as es, torch.no_grad():
                    st = {"snr": None}
                    if self.diffusion and snr_of(layer) is not None:
                        own, orig = "_pass" in vars(self.m), self.m._pass

                        def _pass(z_t, t, *a, **kw):
                            st["snr"] = nearest_snr(step_snr_db(self.m, t), self.cached)
                            return orig(z_t, t, *a, **kw)
                        self.m._pass = _pass
                        es.callback(lambda: (setattr(self.m, "_pass", orig) if own else delattr(self.m, "_pass")))
                    act = lambda: snr_of(layer) is None or not self.diffusion or st["snr"] == snr_of(layer)
                    for c in comps:
                        es.enter_context(comp_edit(self.m, layer, c, idx, lambda v, c=c: cap[c].append(
                            _pool_frames(v, fr, inp["T"])) or None, act))
                    self._sample(inp["x"], inp["k"])
                got = {c: _nan_guard(np.nanmean, np.stack(v), 0) for c, v in cap.items() if v}
            for c in comps:
                out[c][r] = got[c]
            self.work.mkdir(parents=True, exist_ok=True)
            np.savez(f, comps=np.array(comps), **{f"c{t}": got[c] for t, c in enumerate(comps)})
        return out

    def _lin(self, i, target: str = "r"):
        """(readout gradient in forecast-feature space at clip i's unedited forecast, that forecast's features), by
        :func:`fd_grad` (above ``max_fd_dims`` features: along the principal directions of the base forecasts seen
        so far); None when unavailable."""
        if self.diffusion or self.readout.value is None:
            return None
        b = self.run(f"base|{i}", i, [])
        if b.get("z") is None:
            return None
        z0 = np.asarray(b["z"], float)
        basis = None
        if z0.size > self.max_fd_dims and len(self._z) >= 2:
            Z = np.stack([v.reshape(-1) for v in self._z.values() if v.size == z0.size])
            basis = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][: min(len(Z) - 1, self.max_fd_dims)]
        gz = fd_grad(lambda z: float(self.readout.value(i, z).get(target, np.nan)), z0, basis, self.max_fd_dims, self.h_rel)
        if gz is None or not np.isfinite(gz).all():
            self.no_grad_reason = "readout gradient unavailable (non-finite readout or too many features)"
            return None
        return gz, z0

    def das_train(self, layer, X, pairs, req, k: int, steps: int = 300, lr: float = 0.05, batch: int = 8, seed: int = 0,
                  target: str = "r"):
        """DAS through the model (encoders): W [D, k] (QR of a free parameter) trained so that injecting the
        interchange W W^T (x_donor - x_base) of cached site vectors at the site tokens of block ``layer`` changes the
        forecast's (linearised) readout by the law's required change ``req`` per pair: loss mean ((g_z . (z' - z_base)
        - req) / sd(req))^2, g_z = :meth:`_lin`, back-propagated through forecast() (its no_grad lifted); a random
        ``batch`` of pairs per Adam step. None for DiTs or without a readout gradient."""
        if self.diffusion or self.readout.value is None or not len(pairs):
            return None
        import torch
        from contextlib import ExitStack
        bases = sorted({int(i) for i, _ in pairs})
        lin = {i: self._lin(i, target) for i in bases}
        if any(v is None for v in lin.values()):
            return None
        ins = {i: self.inputs(i) for i in bases}
        X = np.asarray(X, np.float64)
        gen = torch.Generator().manual_seed(seed)
        A = torch.randn(X.shape[1], min(int(k), X.shape[1]), generator=gen, dtype=torch.float64).requires_grad_(True)
        opt = torch.optim.Adam([A], lr=lr)
        rng = np.random.default_rng(seed)
        req = np.asarray(req, float)
        sd = float(np.std(req)) + 1e-12
        for _ in range(int(steps)):
            opt.zero_grad()
            pick = rng.choice(len(pairs), min(int(batch), len(pairs)), replace=False)
            for t in pick:
                i, j = int(pairs[t][0]), int(pairs[t][1])
                inp, (gz, z0) = ins[i], lin[i]
                W = torch.linalg.qr(A)[0]
                d = W @ (W.T @ torch.as_tensor(X[j] - X[i]))
                with ExitStack() as es, _grad_through(self.m), torch.enable_grad():
                    es.enter_context(comp_edit(self.m, layer, "block", np.asarray(inp["groups"]["site"]),
                                               lambda v, d=d: v + d.to(v.device, v.dtype)))
                    acts, _ = self._sample(inp["x"], inp["k"])
                    zt = _pool_torch(acts, self.readout.space)
                    dr = (torch.as_tensor(gz, dtype=zt.dtype, device=zt.device) *
                          (zt - torch.as_tensor(z0, dtype=zt.dtype, device=zt.device))).sum()
                    if not dr.requires_grad:
                        self.no_grad_reason = "no autograd path from the site to the forecast"
                        return None
                    (((dr.double().cpu() - float(req[t])) / sd) ** 2 / len(pick)).backward()
            opt.step()
        return torch.linalg.qr(A.detach())[0].numpy()

    def grad(self, layer, i, comps, target: str = "r"):
        """Attribution patching: d readout / d (per-token-frame pooled output of each component over the site tokens)
        at the unedited forecast of clip i: the readout's gradient in forecast-feature space by finite differences
        (:func:`fd_grad`; above ``max_fd_dims`` features along the principal directions of the base forecasts seen
        so far), then one backward pass through the model. {comp: [T, d]}; None for DiTs or when no gradient flows."""
        lz = self._lin(i, target)
        if lz is None:
            return None
        import torch
        gz = lz[0]
        inp = self._in(i)
        D = {c: None for c in comps}
        params = {}
        from contextlib import ExitStack
        with ExitStack() as es, _grad_through(self.m), torch.enable_grad():
            for c in comps:
                proj, H, mlp = _comp_modules(blocks(self.m, layer)[block_index(layer)])
                ref = proj if c.startswith("attn") else mlp
                w = next(ref.parameters())
                dim = (int(ref.in_features) // H) if c.startswith("attn") else _out_dim(ref)
                params[c] = torch.zeros(inp["T"], dim, dtype=torch.float32, device=w.device, requires_grad=True)
            es.enter_context(edit_hooks(self.m, [("grad", layer, c, p) for c, p in params.items()], inp["groups"],
                                        self.per, inp["T"]))
            acts, _ = self._sample(inp["x"], inp["k"])
            zt = _pool_torch(acts, self.readout.space)
            L = (torch.as_tensor(gz, dtype=zt.dtype, device=zt.device) * zt).sum()
            if not L.requires_grad:
                self.no_grad_reason = "no autograd path from the components to the forecast"
                return None
            L.backward()
        for c, p in params.items():
            D[c] = None if p.grad is None else p.grad.detach().cpu().numpy().astype(float)
        return D if all(v is not None for v in D.values()) else None


def _edit_sig(edits) -> str:
    """Short hash of a run's edits (kinds, layers, groups / components and the payload values, float32)."""
    import hashlib
    h = hashlib.sha1()
    for e in edits:
        h.update("|".join(str(x) for x in e[:3]).encode())
        if len(e) > 3 and e[3] is not None and not hasattr(e[3], "requires_grad"):
            h.update(np.ascontiguousarray(np.asarray(e[3], np.float32)).tobytes())
    return h.hexdigest()[:12]


def _pool_name(group: str) -> str:
    """Token-group name -> stage-4 pool name (DiTs' cached activations): "pre" / "post" -> frame.pre / frame.post."""
    return {"pre": "frame.pre", "post": "frame.post"}.get(group, group)


# ---------------------------------------------------------------- the routine


def _cap_layers(keys, lcfg, diffusion: bool) -> list[str]:
    """Step 1's layers within the caps: encoders up to ``max_layers`` (evenly spaced); DiTs up to ``max_layers``
    blocks x ``max_snrs`` SNRs of the cached (SNR, layer) spaces."""
    even = lambda xs, n: [xs[i] for i in sorted(set(np.linspace(0, len(xs) - 1, n).round().astype(int)))] \
        if n and len(xs) > n else list(xs)
    if diffusion:
        sp = [k_ for k_ in keys if snr_of(k_) is not None]
        bl = even(sorted({block_index(k_) for k_ in sp}), int(lcfg["max_layers"] or 0))
        sn = even(sorted({snr_of(k_) for k_ in sp}), int(lcfg["max_snrs"] or 0))
        return sorted([k_ for k_ in sp if block_index(k_) in bl and snr_of(k_) in sn], key=lambda k_: (snr_of(k_), block_index(k_)))
    ks = sorted(keys, key=lambda k_: (str(k_).startswith("pred/"), block_index(k_)))
    return even(ks, int(lcfg["max_layers"] or 0))


def localize(runner, *, lc: dict, lcfg: dict, theta, theta_c, scene, grp, fit_rows, clip_of: dict, sites: dict,
             nuis=None, size=None, passive=None, untrained=None, patch=None, ref: float = float("nan"),
             max_inv: float = 0.25, min_dt: float = 0.0, k_folds: int = 5, diffusion: bool = False, log=print) -> dict:
    """WHERE the law is used, in four steps (``localize`` config block; caps bound every sweep):

    1. layers (DiTs: every cached (SNR, layer) space): the swap test at each, IIA (zero-change laws: the theta readout
       moved, with the invariance ratio) per layer with group-bootstrap CIs, beside the same sweep for the controls
       (norm-matched random direction; ``shuffled``: the law swap's own changes scored against other pairs' required
       changes, IIA's chance level for an edit that moves the readout as much; the span of a control quantity that is
       not the law's parameter, ``size``, fitted beyond theta; passive propagation for post-event sites; the untrained
       model's curve when given); the peak; ``flagged`` = layers whose CI lies above every control's. DiT momentum also
       the live paired-pass collision patch per block x SNR.
    2. tokens: at the top flagged layers, the swap restricted to each token group (object ROIs, contact, background,
       pre / post token-frames, whichever the pass holds) with its random and shuffled controls.
    3. components: per attention head and MLP at the top layer(s): (a) attribution patching (first-order: d readout /
       d component output . (donor - base), :meth:`ModelRunner.grad`) ranks every component; exact activation patching
       (donor's per-token-frame pooled output into the base) on the top-k and on random control components; (b)
       knock-out by MEAN ablation (the mean over other groups' clips, per token-frame) re-scored on the law's behaviour
       (:func:`knockout_scores`); the flagged layers' curve span is knocked out the same way (control: a random
       subspace). The off-distribution distance (:func:`mahalanobis`) of every patched / ablated site vector.
    4. learned subspace (DAS) at the top layer: :func:`das_fit` per k on training groups, IIA on held-out groups (the
       frozen theta readout, :func:`theta_decoder`) and through the model on held-out pairs; principal angles to the
       fitted curve's span; the smallest k reaching the best IIA within ``tol``.
    Verdict (agreement rule): a layer / component is "localized" only if its swap / patch IIA AND its knock-out effect
    both pass their controls; disagreements are listed."""
    theta, theta_c, scene, grp = (np.asarray(a) for a in (theta, theta_c, scene, grp))
    theta, theta_c = theta.astype(float), theta_c.astype(float)
    mode, null = lc["mode"], null_prediction(lc)
    nb = int(lcfg["bootstrap"])
    scenes = sorted(clip_of)[: int(lcfg["max_groups"] or 0) or None]
    sset = set(scenes)
    loc_rows = np.array([int(r) for r in fit_rows if scene[r] in sset], int)
    folds_te = grouped_kfold(np.array(scenes), max(min(int(k_folds), len(scenes)), 2), 0) if len(scenes) > 1 else []
    rng = np.random.default_rng(0)
    pool = list(loc_rows) if mode == "ladder" else [clip_of[t] for t in scenes]
    folds = []
    for te in folds_te:
        test = sorted(scenes[i] for i in te)
        tg = set(grp[np.isin(scene, test)].tolist())
        tr = loc_rows[~np.isin(grp[loc_rows], list(tg))]
        P = [(clip_of[s], j) for s in test for j in
             pick_donors(clip_of[s], pool, theta, scene, int(lcfg["n_donors"]), min_dt, rng, mode)]
        folds.append({"train": tr, "test": test, "pairs": P})
    fold_of = {s: fi for fi, F in enumerate(folds) for s in F["test"]}
    pairs = [(fi, i, j) for fi, F in enumerate(folds) for i, j in F["pairs"]]
    cap_ko = int(lcfg["max_ko_clips"] or 0) or None
    ko_rows = ([int(r) for s in scenes for r in np.where(scene == s)[0]] if mode == "ladder"
               else [clip_of[s] for s in scenes])[:cap_ko]
    out = {"scenes": scenes, "n_pairs": len(pairs), "ko_clips": len(ko_rows), "null_prediction": null,
           "caps": {k_: lcfg[k_] for k_ in ("max_groups", "n_donors", "max_layers", "max_snrs", "top_layers", "max_heads",
                                             "top_k", "n_control_components", "max_ko_clips")},
           "controls": ["random", "shuffled", *(["size"] if size is not None else []),
                        *(["passive"] if passive is not None else []), *(["untrained"] if untrained else [])]}
    if not pairs:
        out["skipped"] = "no swap pairs (too few scenes or donors within the caps)"
        return out
    base = lambda i: runner.run(f"base|{i}", i, [])
    nuis_ = (lambda rows: None) if nuis is None else (lambda rows: np.asarray(nuis)[rows])
    if null and not (np.isfinite(ref) and ref > 0):     # invariance scale: the readout's sd over the unedited forecasts
        ref = float(np.nanstd([base(i)["r"] for i in ko_rows]))
        out["invariance_reference"] = {"value": ref, "kind": "sd of the unedited readout over the knock-out clips"}
    elif null:
        out["invariance_reference"] = {"value": float(ref), "kind": "the main swap test's reference"}

    def rec(i, j, o):
        b = base(i)
        return {"g": grp[i], "obs": o["r"] - b["r"], "req": pair_required(lc, theta, theta_c, i, j),
                "moved": o.get("theta", np.nan) - b.get("theta", np.nan), "dth": theta_c[j] - theta_c[i]}

    def score(R, n_shuffle=0):
        if not R:
            return {"n_pairs": 0, "score": float("nan"), "score_ci95": None}
        a = {k_: [r_[k_] for r_ in R] for k_ in ("g", "obs", "req", "moved", "dth")}
        return loc_pair_scores(a["obs"], a["req"], a["g"], null, a["moved"], a["dth"], ref, max_inv, nb, n_shuffle)
    NSH = 20

    def site_after(o, l, fallback):
        v = (o.get("site") or {}).get(l)
        return fallback if v is None else np.asarray(v, float)

    def fit_q(X, rows, th, m_=mode, nz=True):
        """Span of the curve of ``th`` at the rows; nz False: the control quantity's curve, fitted across scenes BEYOND
        the law's theta (theta_c as its nuisance), so a sample correlation with theta cannot hand it the law's axis."""
        rows = rows[np.isfinite(X[rows]).all(1)]
        nz_ = nuis_(rows) if nz else theta_c[rows][:, None]
        return curve_basis(fit_manifold(X[rows], th[rows], scene[rows], m_, nz_)), rows

    # ---------------------------------------------------------------- 1. layers
    layers = _cap_layers(list(sites), lcfg, diffusion)
    by_layer, md_of, real_of = {}, {}, {}
    for l in layers:
        X = np.asarray(sites[l], float)
        R = {c: [] for c in ("law", "random", "size", "passive")}
        dist, real_d, mds = {}, [], {}
        try:
            for fi, F in enumerate(folds):
                Q, tr = fit_q(X, F["train"], theta)
                Qs = fit_q(X, F["train"], np.asarray(size, float), "across", False)[0] if size is not None else None
                mds[fi] = md = mahalanobis(X[tr])
                te_rows = loc_rows[np.isin(scene[loc_rows], F["test"])]
                real_d += md(X[te_rows]).tolist()
                r_ = np.random.default_rng(1000 + fi)
                for i, j in F["pairs"]:
                    dl = {"law": swap_delta(X[i], X[j], Q)}
                    dl["random"] = _random_like(dl["law"], r_)
                    if Qs is not None and Qs.shape[1]:
                        dl["size"] = swap_delta(X[i], X[j], Qs)
                    for c, d in dl.items():
                        o = runner.run(f"layer|{l}|{c}|{i}|{j}", i, [("shift", l, "site", d)])
                        R[c].append(rec(i, j, o))
                        dist.setdefault(c, {}).setdefault(0.0, []).append(float(md(site_after(o, l, X[i] + d))[0]))
                    pf = passive(l) if passive is not None else None
                    if pf is not None:
                        R["passive"].append({**rec(i, j, {"r": pf(i, X[i] + dl["law"]), "theta": np.nan}),
                                             "obs": pf(i, X[i] + dl["law"]) - pf(i, X[i])})
        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            by_layer[l] = {"skipped": repr(e)}
            continue
        md_of[l], real_of[l] = mds, real_d
        sc = {c: score(v) for c, v in R.items() if v}
        sc["shuffled"] = score(R["law"], NSH)
        if untrained and l in untrained:
            sc["untrained"] = untrained[l]
        sc["passes"] = beats_controls(sc["law"], [sc[c] for c in LOC_CONTROLS if c in sc], null=null)
        sc["off_distribution"] = off_distribution(dist, np.array(real_d))
        by_layer[l] = sc
    ok_l = [l for l in layers if "law" in by_layer.get(l, {})]
    val = lambda l: by_layer[l]["law"]["score"]
    peak = max((l for l in ok_l if np.isfinite(val(l))), key=val, default=None)
    flagged = [l for l in ok_l if by_layer[l]["passes"]]
    out["layers"] = {"metric": "readout_moved" if null else "iia", "order": layers, "by_layer": by_layer, "peak": peak,
                     "flagged": flagged,
                     "curves": {c: [by_layer.get(l, {}).get(c, {}).get("score", float("nan")) for l in layers]
                                for c in ("law", *LOC_CONTROLS) if any(c in by_layer.get(l, {}) for l in layers)}}
    if patch is not None and runner.live:          # DiT momentum: the live paired-pass patch per block x SNR
        lp = {}
        for l in layers:
            rows_, obs_ = {"patch": [], "patch_random": []}, {"patch": [], "patch_random": []}
            for fi, i, j in [(fi, i, j) for fi, F in enumerate(folds) for i in [clip_of[s] for s in F["test"]]
                             for j in patch["donors"].get(i, [])]:
                b = base(i)
                for c, seed in (("patch", None), ("patch_random", 7919 + i)):
                    o = runner.run(f"live|{l}|{c}|{i}|{j}", i, [("live", l, j, seed)])
                    rows_[c].append((i, j))
                    obs_[c].append(np.asarray(o.get("post_v", np.full((2, 2), np.nan)), float)[1][:2] -
                                   np.asarray(b.get("post_v", np.full((2, 2), np.nan)), float)[1][:2])
            sc = {c: patch["score"](rows_[c], obs_[c]) for c in rows_ if rows_[c]}
            if sc:
                sc["passes"] = beats_controls(sc["patch"], [sc["patch_random"]])
                lp[l] = sc
        out["layers"]["live_patch"] = lp
        out["layers"]["live_patch_flagged"] = [l for l, v in lp.items() if v.get("passes")]

    ranked = sorted(flagged, key=val, reverse=True)
    top = ranked[: int(lcfg["top_layers"])]
    if not top:
        if not lcfg.get("run_unflagged") or peak is None:
            out["steps_2_4"] = "skipped: no layer's swap score lies above every control's (nothing to localize)"
            out["verdict"] = _loc_verdict(out, {}, {})
            return out
        top, out["from_peak_only"] = [peak], True
    need = sorted(set(loc_rows.tolist()) | {i for _, i, _ in pairs} | {j for _, _, j in pairs} | set(ko_rows))

    # ---------------------------------------------------------------- 2. tokens
    tokens = {}
    for l in top:
        tokens[l] = {}
        for gname in runner.groups(l):
            Xg = runner.acts(l, gname, need)
            if Xg is None:
                tokens[l][gname] = {"skipped": "no activations for this group at this layer"}
                continue
            R = {"law": [], "random": []}
            try:
                for fi, F in enumerate(folds):
                    Q, _ = fit_q(Xg, F["train"], theta)
                    r_ = np.random.default_rng(2000 + fi)
                    for i, j in F["pairs"]:
                        if not (np.isfinite(Xg[i]).all() and np.isfinite(Xg[j]).all()):
                            continue
                        d = swap_delta(Xg[i], Xg[j], Q)
                        for c, dd in (("law", d), ("random", _random_like(d, r_))):
                            R[c].append(rec(i, j, runner.run(f"tok|{l}|{gname}|{c}|{i}|{j}", i, [("shift", l, gname, dd)])))
            except (ValueError, np.linalg.LinAlgError) as e:
                tokens[l][gname] = {"skipped": repr(e)}
                continue
            sc = {c: score(v) for c, v in R.items()}
            sc["shuffled"] = score(R["law"], NSH)
            sc["passes"] = beats_controls(sc["law"], [sc["random"], sc["shuffled"]], null=null)
            tokens[l][gname] = sc
        okg = [g for g, v in tokens[l].items() if "law" in v and np.isfinite(v["law"]["score"])]
        tokens[l] = {"by_group": tokens[l], "best": max(okg, key=lambda g: tokens[l][g]["law"]["score"], default=None),
                     "passing": [g for g in okg if tokens[l][g]["passes"]]}
    out["tokens"] = tokens

    # ---------------------------------------------------------------- 3. components (+ the flagged layers' knock-out)
    def ko_targets(A_c, i):
        oth = [j for j in A_c if grp[j] != grp[i] and A_c[j] is not None]
        if not oth:
            return None
        T = max(len(A_c[j]) for j in oth)
        S = np.full((len(oth), T, np.asarray(A_c[oth[0]]).shape[-1]), np.nan)
        for t_, j in enumerate(oth):
            S[t_, : len(A_c[j])] = A_c[j]
        return _nan_guard(np.nanmean, S, 0)
    ko_th, ko_thc, ko_g = theta[ko_rows], theta_c[ko_rows], grp[ko_rows]
    base_ko = [base(i)["r"] for i in ko_rows]
    layer_ko = {}
    for l in top:
        X = np.asarray(sites[l], float)
        r_ = {"base": base_ko, "law_span": [], "random_span": []}
        dist = {}
        rr = np.random.default_rng(3000)
        for i in ko_rows:
            F = folds[fold_of.get(scene[i], 0)]
            Q, _ = fit_q(X, F["train"], theta)
            oth = [j for j in need if grp[j] != grp[i] and np.isfinite(X[j]).all()]
            mu = X[oth].mean(0)
            Qr = np.linalg.qr(rr.standard_normal((X.shape[1], max(Q.shape[1], 1))))[0]
            md = md_of.get(l, {}).get(fold_of.get(scene[i], 0)) or mahalanobis(X[F["train"]])
            for c, B in (("law_span", Q), ("random_span", Qr)):
                d = B @ (B.T @ (mu - X[i]))
                o = runner.run(f"kol|{l}|{c}|{i}", i, [("shift", l, "site", d)])
                r_[c].append(o["r"])
                dist.setdefault(c, {}).setdefault(0.0, []).append(float(md(site_after(o, l, X[i] + d))[0]))
        ks = knockout_scores(r_, ko_th, ko_thc, ko_g, lc, mode, ("random_span",), float(lcfg["creation_tau"]), nb, null)
        layer_ko[l] = {**ks, "off_distribution": off_distribution(dist, np.array(real_of.get(l, [])))}
    out["knockout_layers"] = layer_ko

    comps_out = {}
    for l in top[: int(lcfg["component_layers"])]:
        comps = list(runner.components(l))
        if not comps:
            comps_out[l] = {"skipped": "no attention heads / MLP found in this block"}
            continue
        heads = [c for c in comps if c.startswith("attn")]
        rest = [c for c in comps if not c.startswith("attn")]
        mh = int(lcfg["max_heads"] or 0)
        if mh and len(heads) > mh:
            heads = [heads[t] for t in sorted(set(np.linspace(0, len(heads) - 1, mh).round().astype(int)))]
        comps = heads + rest
        bases = sorted({i for _, i, _ in pairs})
        A = runner.comp_acts(l, sorted(set(need) & ((set() if runner.live else set(bases) | {j for _, _, j in pairs})
                                                    | set(ko_rows))), comps)    # DiT patches: paired pass, not A
        # (a) attribution pre-screen
        attr = {"available": False}
        grads = {i: runner.grad(l, i, comps, "theta" if null else "r") for i in bases}
        if all(g_ is not None for g_ in grads.values()):
            est = {c: [] for c in comps}
            for _, i, j in pairs:
                for c in comps:
                    dlt = np.nan_to_num(np.asarray(A[c][j], float) - np.asarray(A[c][i], float))
                    G = np.asarray(grads[i][c], float)
                    T = min(len(G), len(dlt))
                    est[c].append(float((G[:T] * dlt[:T]).sum()))
            obs_est = lambda c: [{"g": grp[i], "obs": e_, "req": pair_required(lc, theta, theta_c, i, j), "moved": e_,
                                  "dth": theta_c[j] - theta_c[i]} for (_, i, j), e_ in zip(pairs, est[c])]
            attr = {"available": True, "by_component": {c: score(obs_est(c)) for c in comps},
                    "method": "first-order: d readout / d pooled component output (finite-difference readout gradient, "
                              "one backward pass) . (donor - base)"}
            rank = sorted(comps, key=lambda c: -np.nan_to_num(attr["by_component"][c]["score"], nan=-np.inf))
        else:
            attr["reason"] = getattr(runner, "no_grad_reason", None) or "no gradient"
            rank = comps
        k_top = int(lcfg["top_k"])
        exact = rank[:k_top] if attr["available"] else rank
        rest_c = [c for c in comps if c not in exact]
        rc = np.random.default_rng(4000)
        ctrl = [rest_c[t] for t in sorted(rc.choice(len(rest_c), min(int(lcfg["n_control_components"]), len(rest_c)),
                                                    replace=False))] if rest_c else []
        if not attr["available"]:                  # no pre-screen: every capped component is exact; controls = random
            ctrl = []
        P, Rc, dist = {}, {}, {}
        for c in exact + ctrl:
            Rc[c] = []
            for fi, i, j in pairs:
                o = runner.run(f"cp|{l}|{c}|{i}|{j}", i, [("pset", l, c, j)] if runner.live else [("set", l, c, A[c][j])])
                Rc[c].append(rec(i, j, o))
                md = md_of.get(l, {}).get(fi)
                if md is not None:
                    dist.setdefault(c, {}).setdefault(0.0, []).append(float(md(site_after(o, l, np.asarray(sites[l][i], float)))[0]))
            P[c] = score(Rc[c])
        for c in exact:                            # controls: random other components; none -> every other candidate
            others = ctrl or [c2 for c2 in exact if c2 != c]
            ref_c = score([r_ for c2 in others for r_ in Rc[c2]]) if others else None
            P[c]["control"] = None if ref_c is None else {k_: ref_c.get(k_) for k_ in ("score", "score_ci95", "n_pairs")}
            P[c]["control_components"] = others
            P[c]["shuffled"] = score(Rc[c], NSH)
            P[c]["passes"] = beats_controls(P[c], ([ref_c] if ref_c else []) + [P[c]["shuffled"]], null=null)
        # (b) knock-out by mean ablation
        r_ = {"base": base_ko}
        kdist = {}
        for c in exact + ctrl:
            vals = []
            for i in ko_rows:
                tgt = ko_targets(A[c], i)
                o = runner.run(f"ko|{l}|{c}|{i}", i, [("set", l, c, tgt)] if tgt is not None else [])
                vals.append(o["r"])
                md = md_of.get(l, {}).get(fold_of.get(scene[i], 0))
                if md is not None:
                    kdist.setdefault(c, {}).setdefault(0.0, []).append(float(md(site_after(o, l, np.asarray(sites[l][i], float)))[0]))
            r_[c] = vals
        KO = knockout_scores(r_, ko_th, ko_thc, ko_g, lc, mode, tuple(ctrl), float(lcfg["creation_tau"]), nb, null)
        if not ctrl:                               # leave-one-out: each candidate against every other candidate
            sat = lc["predict"] == "saturation"
            for c in exact:
                oth = [KO[c2] for c2 in exact if c2 != c]
                KO[c]["passes"] = bool(beats_controls(KO[c], oth, "effect") or (sat and beats_controls(KO[c], oth, "creation_effect")))
        rho = None
        if attr["available"] and len(exact) >= 3:
            a_ = [attr["by_component"][c]["score"] for c in exact]
            e_ = [P[c]["score"] for c in exact]
            if np.isfinite(a_).all() and np.isfinite(e_).all() and np.ptp(a_) > 0 and np.ptp(e_) > 0:
                rho = float(np.corrcoef(np.argsort(np.argsort(a_)), np.argsort(np.argsort(e_)))[0, 1])
        comps_out[l] = {"components": comps, "attribution": attr, "ranked": rank, "exact": exact, "controls": ctrl,
                        "patch": P, "knockout": KO, "attribution_vs_exact_spearman": rho,
                        "off_distribution": {"patch": off_distribution(dist, np.array(real_of.get(l, []))) if dist else None,
                                             "knockout": off_distribution(kdist, np.array(real_of.get(l, []))) if kdist else None}}
    out["components"] = comps_out

    # ---------------------------------------------------------------- 4. DAS
    out["das"] = _loc_das(runner, lcfg["das"], top[0], np.asarray(sites[top[0]], float), theta, theta_c, scene, grp,
                          loc_rows, mode, nuis_, min_dt, lc, null, nb, log)
    out["verdict"] = _loc_verdict(out, layer_ko, comps_out)
    return out


def _loc_das(runner, dc, l, X, theta, theta_c, scene, grp, rows, mode, nuis_, min_dt, lc, null, nb, log) -> dict:
    """Step 4 at layer ``l`` (:func:`localize`). Objective ``model`` (encoders, whenever the runner can differentiate:
    :meth:`ModelRunner.das_train`, the interchange injected into the model and its linearised readout matched to the
    law's required change; zero-change laws: the model's own theta readout matched to the donor's theta) scored by
    exact interventions on held-out groups' pairs; objective ``readout`` (DiTs, or no gradient: :func:`das_fit` with
    the frozen :func:`theta_decoder` on cached activations, ladder: centred per scene) scored by the decoder on held-out
    pairs, plus exact interventions on ``max_model_pairs`` of them. ``iia`` per k is the objective's held-out IIA."""
    rows = rows[np.isfinite(X[rows]).all(1)]
    Xc = X.copy()
    if mode == "ladder":
        for s in set(scene[rows].tolist()):
            r_ = rows[scene[rows] == s]
            Xc[r_] = X[r_] - X[r_].mean(0)
    sc = sorted(set(scene[rows].tolist()))
    if len(sc) < 2:
        return {"skipped": "fewer than 2 scenes", "layer": l}
    folds = grouped_kfold(np.array(sc), max(min(int(dc["folds"]), len(sc)), 2), 0)
    ks = [int(k) for k in dc["ks"] if int(k) <= X.shape[1]]
    res = {k: {"hits": [], "g": [], "model": [], "cos": [], "Q_rank": []} for k in ks}
    rng = np.random.default_rng(5000)
    want = str(dc.get("objective", "auto"))
    objective, r2, r2v = None, [], []
    req_of = lambda i, j: (theta_c[j] - theta_c[i]) if null else pair_required(lc, theta, theta_c, i, j)

    def model_rec(key, i, j, W):
        o = runner.run(key, i, [("shift", l, "site", W @ (W.T @ (X[j] - X[i])))])
        b = runner.run(f"base|{i}", i, [])
        return {"g": grp[i], "obs": o["r"] - b["r"], "req": pair_required(lc, theta, theta_c, i, j),
                "moved": o.get("theta", np.nan) - b.get("theta", np.nan), "dth": theta_c[j] - theta_c[i]}
    for fi, te in enumerate(folds):
        test = [sc[i] for i in te]
        tg = set(grp[np.isin(scene, test)].tolist())
        tr = rows[~np.isin(grp[rows], list(tg))]
        ter = rows[np.isin(scene[rows], test)]
        if len(tr) < 4 or len(ter) < 2:
            continue
        ptr = _all_pairs(tr, theta, scene, mode, min_dt, int(dc["max_pairs"]), rng)
        pte = _all_pairs(ter, theta, scene, mode, min_dt, int(dc["max_pairs"]), rng)
        if not ptr or not pte:
            continue
        Q = curve_basis(fit_manifold(X[tr], theta[tr], scene[tr], mode, nuis_(tr)))
        f = None
        for k in ks:
            W = None
            if want in ("auto", "model") and hasattr(runner, "das_train"):
                tb = sorted({i for i, _ in ptr})[: int(dc.get("max_train_bases", 24))]
                pm = [(i, j) for i, j in ptr if i in set(tb)]
                W = runner.das_train(l, X, pm, [req_of(i, j) for i, j in pm], k, int(dc["steps"]), float(dc["lr"]),
                                     int(dc.get("batch", 8)), seed=fi, target="theta" if null else "r")
            if W is not None:
                objective = "model"
                for i, j in pte[: int(dc.get("max_eval_pairs", 48))]:
                    res[k]["model"].append(model_rec(f"das|{l}|{k}|{fi}|{i}|{j}", i, j, W))
            else:
                if want == "model":
                    return {"skipped": "objective model: the runner cannot differentiate", "layer": l}
                objective = "readout"
                if f is None:
                    f = theta_decoder(Xc[tr], theta_c[tr], int(dc["hidden"]), int(dc["pcs"]), int(dc["decoder_steps"]),
                                      float(dc["decoder_lr"]), seed=fi, weight_decay=float(dc.get("decoder_wd", 0.01)))
                    r2v.append(f.val_r2)
                    r2.append(f.train_r2)
                W = das_fit(Xc, theta_c, ptr, k, f, int(dc["steps"]), float(dc["lr"]), seed=fi)
                h = das_hits(Xc, theta_c, pte, W, f)
                res[k]["hits"] += h.tolist()
                res[k]["g"] += [grp[i] for i, _ in pte]
                for i, j in pte[: int(dc["max_model_pairs"])]:
                    res[k]["model"].append(model_rec(f"das|{l}|{k}|{fi}|{i}|{j}", i, j, W))
            res[k]["cos"].append(principal_cosines(W, Q))
            res[k]["Q_rank"].append(int(Q.shape[1]))
    by_k = {}
    for k, v in res.items():
        M = v["model"]
        if not M and not v["hits"]:
            continue
        msc = loc_pair_scores([x["obs"] for x in M], [x["req"] for x in M], [x["g"] for x in M], null,
                              [x["moved"] for x in M], [x["dth"] for x in M], n_boot=nb) if M else None
        e = {"model": msc, "curve_rank": int(np.median(v["Q_rank"])) if v["Q_rank"] else 0}
        cos = [c for c in v["cos"] if c]
        e["principal_cosines_to_curve"] = (np.median(np.array([c[: min(len(x) for x in cos)] for c in cos]), 0).tolist()
                                           if cos else [])
        if v["hits"]:
            h, g = np.asarray(v["hits"], float), np.asarray(v["g"])
            m = lambda ix: float(np.mean(h[ix]))
            e.update(iia_readout=m(np.arange(len(h))), iia_readout_ci95=_boot_ci(m, g, nb), n_pairs=len(h))
        if objective == "model":
            e["iia"], e["iia_ci95"], e["n_pairs"] = msc["score"], msc.get("score_ci95"), msc["n_pairs"]
        else:
            e["iia"], e["iia_ci95"] = e.get("iia_readout", float("nan")), e.get("iia_readout_ci95")
        by_k[str(k)] = e
    if not by_k:
        return {"skipped": "no training / held-out pairs", "layer": l}
    best = max(v["iia"] for v in by_k.values() if np.isfinite(v["iia"])) if any(
        np.isfinite(v["iia"]) for v in by_k.values()) else float("nan")
    k_needed = min((int(k) for k, v in by_k.items() if np.isfinite(v["iia"]) and v["iia"] >= best - float(dc["tol"])),
                   default=None)
    crank = max(v["curve_rank"] for v in by_k.values())
    return {"layer": l, "objective": objective, "by_k": by_k, "k_needed": k_needed, "curve_rank": crank,
            "more_dims_than_curve": bool(k_needed is not None and k_needed > max(crank, 1)),
            "iia_by_k": [by_k[str(k)]["iia"] for k in ks if str(k) in by_k],
            "decoder_train_r2": float(np.mean(r2)) if r2 else None,
            "decoder_val_r2": float(np.nanmean(r2v)) if r2v and np.isfinite(r2v).any() else None,
            "readout": ("the model: interchange injected at the site, linearised readout (training), exact readout "
                        "(held-out)") if objective == "model" else
                       "frozen MLP theta decoder on the layer's cached activations (training groups)"}


def _loc_verdict(out: dict, layer_ko: dict, comps_out: dict) -> dict:
    """Agreement rule: localized = swap (patch) IIA passes its controls AND the knock-out effect passes its controls."""
    agree = {}
    bl = (out.get("layers") or {}).get("by_layer", {})
    for l, ko in layer_ko.items():
        sw = bool(bl.get(l, {}).get("passes"))
        kp = bool((ko.get("law_span") or {}).get("passes"))
        agree[f"layer:{l}"] = {"swap": sw, "knockout": kp, "localized": sw and kp}
    for l, co in comps_out.items():
        for c in co.get("exact", []):
            pp = bool(co["patch"].get(c, {}).get("passes"))
            kp = bool(co["knockout"].get(c, {}).get("passes"))
            agree[f"component:{l}/{c}"] = {"swap": pp, "knockout": kp, "localized": pp and kp}
    return {"rule": "localized only if the swap / patch IIA AND the knock-out effect both pass their controls",
            "localized_layers": [k_[6:] for k_, v in agree.items() if k_.startswith("layer:") and v["localized"]],
            "localized_components": [k_[10:] for k_, v in agree.items() if k_.startswith("component:") and v["localized"]],
            "agreement": agree,
            "disagreements": {k_: ("swap passes, knock-out does not" if v["swap"] else "knock-out passes, swap does not")
                              for k_, v in agree.items() if v["swap"] != v["knockout"]},
            "flagged_layers": (out.get("layers") or {}).get("flagged", [])}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def table(res: dict) -> str:
    f = lambda x, s=".3f": "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, s)
    c = lambda ci: f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
    rows = []
    if res.get("stand_in"):
        rows += [f"**Stand-in run:** {res['stand_in']}", ""]
    rows += [f"## L4 manifold steering: {res['law']} ({res['family']}), theta = {res['theta']} ({res['mode']}), pool "
             f"{res['pool']}; law prediction: {res['predict']}", "",
             f"Precondition (theta shuffled {'within scene' if res['mode'] == 'ladder' else 'across scenes'}); "
             f"layers passing: {res['layers_passing']}", "",
             "| layer | scenes | retention (null median, 1/sqrt S) | p retention | LOSO R² line (null median) | p line |",
             "|---|---|---|---|---|---|"]
    for l, p in res["precondition"].items():
        if p:
            rows.append(f"| {l} | {p['n_scenes']} | {f(p.get('retention'))} ({f(p.get('retention_null_median'))}, "
                        f"{f(p.get('retention_iid_reference'))}) | {f(p.get('p_retention'))} | {f(p['loso_r2']['line'], '+.3f')} "
                        f"({f(p['loso_r2_line_null_median'], '+.3f')}) | {f(p['p_loso_line'])} |")
    if "skipped" in res:
        return "\n".join(rows + ["", f"Steering skipped: {res['skipped']}"])
    rows += ["", f"Injection layer {res['layer']} at {res['positions']} tokens; dtheta {res['deltas']}; {res['n_scenes']} scenes; "
             f"curve degree per fold {[m['deg'] if 'deg' in m else {k_: v_['deg'] for k_, v_ in m['by_snr'].items()} for m in res['manifolds']]}", ""]
    inj = res.get("injection")
    if inj:
        sch = (inj.get("steps") or {}).get("law") or []
        rows += [f"DiT per-SNR curves at block L{inj['block']}: fitted {[snr_name(d) for d in inj['fitted_snr_db']]} of cached "
                 f"{[snr_name(d) for d in inj['cached_snr_db']]}; per sampling step the nearest cached SNR's curve "
                 f"(law run): {[r['curve'] for r in sch] or 'not recorded'}", ""]
    for v, d in res["variants"].items():
        if "boundary" in d:
            rows += [f"### {v} (law: the tip indicator steps at theta + dtheta = {d['boundary']:g})", "",
                     "| condition | accuracy [95% CI] | step [95% CI] | switch | crossing scenes | law step - this [95% CI] |",
                     "|---|---|---|---|---|---|"]
            for cn, s in d.items():
                if isinstance(s, dict) and "step" in s:
                    dd = d.get("law", {}).get(f"minus_{cn}")
                    rows.append(f"| {cn} | {f(s['accuracy'], '.2f')}{c(s['accuracy_ci95'])} | {f(s['step'], '+.2f')}"
                                f"{c(s['step_ci95'])} | {f(s['switch'], '.2f')} | {s['n_crossing']} | "
                                f"{(f(dd['median'], '+.2f') + c(dd['ci95'])) if dd else '-'} |")
        elif "law_value" in d:
            rows += [f"### {v} (law slope {d['law_value']:+.3f})", "", "| condition | slope [95% CI] | scenes | law - this [95% CI] |",
                     "|---|---|---|---|"]
            for cn, s in d.items():
                if isinstance(s, dict) and "slope" in s and "n_scenes" in s:
                    dd = d.get("law", {}).get(f"minus_{cn}")
                    rows.append(f"| {cn} | {f(s['slope'], '+.3f')}{c(s['slope_ci95'])} | {s['n_scenes']} | "
                                f"{(f(dd['median'], '+.3f') + c(dd['ci95'])) if dd else '-'} |")
        else:
            rows += [f"### {v} (law: saturation at 1)", "", "| condition | value at 0 | saturation [95% CI] | rise | plateau slope | "
                     "law - this [95% CI] |", "|---|---|---|---|---|---|"]
            for cn, s in d.items():
                if isinstance(s, dict) and "saturation" in s:
                    dd = d.get("law", {}).get(f"minus_{cn}")
                    rows.append(f"| {cn} | {f(s['value_at_0'])} | {f(s['saturation'])}{c(s['saturation_ci95'])} | "
                                f"{f(s['rise'], '+.3f')} | {f(s['plateau_slope'], '+.3f')} | "
                                f"{(f(dd['median'], '+.3f') + c(dd['ci95'])) if dd else '-'} |")
        rows += [f"Verdict: {d.get('verdict')}", ""]
        rv = d.get("readout_validity")
        if rv:
            rows += [f"Readout validity (real clips, readout on the true theta across scenes): slope "
                     f"{f(rv.get('slope'), '+.3f')}{c(rv.get('ci95'))} (law {rv['law_slope']:+g}; n {rv['n']}/{rv['n_clips']}), "
                     f"covers the law: {rv.get('covers_law')}", ""]
        te = d.get("theta_readout")
        if te:
            rows += [f"Push effect on the model's theta readout (fraction of the real-video change; real slope "
                     f"{f(te['real_slope'], '+.3f')}): " + ", ".join(
                         f"{cn} {f(e['fraction_of_real'], '+.2f')}" for cn, e in te["by_condition"].items()), ""]
        rel = [(cn, s) for cn, s in d.items() if isinstance(s, dict) and "per_scene" in s]
        if rel:
            rows += ["Reliability (per-scene quantiles q5 / q50 / q95; share moving the wrong way):", "",
                     "| condition | q5 | q50 | q95 | wrong way |", "|---|---|---|---|---|"]
            rows += [f"| {cn} | {f(s['per_scene']['q5'], '+.3f')} | {f(s['per_scene']['q50'], '+.3f')} | "
                     f"{f(s['per_scene']['q95'], '+.3f')} | {f(s['wrong_way'], '.2f')} |" for cn, s in rel]
            rows.append("")
        for arm, a in (d.get("pairs") or {}).items():
            sel = a.get("selectivity") or {}
            sr = a.get("slope_ratio") or {}
            if "invariance_ratio" in a:
                rows.append(f"Pairs {arm} (invariance; the law predicts no change): n {a.get('n_pairs')}, |change| "
                            f"{f(a.get('abs_change'))}, invariance ratio {f(a.get('invariance_ratio'))} (reference "
                            f"{a['reference_kind']} = {f(a.get('reference'))}), theta readout moved "
                            f"{f(a.get('readout_moved_fraction'), '+.2f')} of the donor difference, result "
                            f"{a.get('result')}")
                continue
            rows.append(f"Pairs {arm}: n {a.get('n_pairs')}, IIA {f(a.get('iia'), '.2f')}, fraction achieved "
                        f"{f(a.get('fraction_achieved'), '+.2f')}, error ratio {f(a.get('error_ratio'))} (p sign-flip "
                        f"{f(a.get('p_sign_flip'))}), |change| {f(a.get('abs_change'))}, selectivity "
                        f"{f(sel.get('ratio'), '.2f')}" + (f", slope ratio high/low k {f(sr.get('observed'), '.2f')} "
                                                            f"{c(sr.get('observed_ci95'))} (physics {f(sr.get('physics'), '.2f')}, "
                                                            f"leak 1)" if sr else "")
                        + (f"; closer to required than to none {f(a['closer_n']['mean'], '+.3f')}{c(a['closer_n']['ci95'])} "
                           f"(p {f(a['closer_n']['p_sign_flip'])})" if a.get("closer_n") else "")
                        + (f"; verdict {a['verdict']}" + (f" ({', '.join(a['reasons'])})" if a.get("reasons") else "")
                           if "verdict" in a else ""))
        rows.append("")
    od = res.get("off_distribution")
    if od:
        rows += [f"Off-distribution (Mahalanobis distance of the steered pooled vector to the training rows; real held-out "
                 f"median {f(od['real_median'], '.1f')}, q95 {f(od['real_q95'], '.1f')}):", "",
                 "| condition | key | median | share > real q95 |", "|---|---|---|---|"]
        rows += [f"| {c} | {k_} | {f(v['median'], '.1f')} | {f(v['frac_above_real_q95'], '.2f')} |"
                 for c, byk in od["by_condition"].items() for k_, v in byk.items()]
    if res.get("localization"):
        rows += ["", loc_table(res["localization"])]
    return "\n".join(rows)


def loc_table(loc: dict) -> str:
    """md section for l4.json ``localization``."""
    f = lambda x, s=".2f": "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, s)
    c = lambda ci: f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    sc = lambda d: (f(d.get("score")) + c(d.get("score_ci95"))) if isinstance(d, dict) and "score" in d else "-"
    g = loc.get("gate") or {}
    rows = ["## L4 localization (where the law is used)", "", f"Gate: passed {g.get('passed')} of {g.get('used')} "
            f"(tests {g.get('tests')}; L2 verdicts {g.get('l2_verdicts') or 'none'})", ""]
    if loc.get("skipped"):
        return "\n".join(rows + [f"Skipped: {loc['skipped']}"])
    L = loc.get("layers") or {}
    rows += [f"Step 1, layers ({L.get('metric')} per layer, group-bootstrap CI; controls {loc.get('controls')}, control "
             f"quantity {loc.get('control_quantity')}): peak {L.get('peak')}, flagged (CI above every control's) "
             f"{L.get('flagged')}", "", "| layer | law | " + " | ".join(LOC_CONTROLS) + " | passes |",
             "|---|---|" + "---|" * len(LOC_CONTROLS) + "---|"]
    for l in L.get("order", []):
        d = (L.get("by_layer") or {}).get(l, {})
        rows.append(f"| {l} | " + " | ".join(sc(d.get(k_)) for k_ in ("law", *LOC_CONTROLS))
                    + f" | {d.get('passes', d.get('skipped', '-'))} |")
    for l, lp in (L.get("live_patch") or {}).items():
        rows.append(f"Live patch {l}: {sc(lp.get('patch'))} vs random {sc(lp.get('patch_random'))}, passes {lp.get('passes')}")
    if loc.get("steps_2_4"):
        rows += ["", f"Steps 2-4: {loc['steps_2_4']}"]
    for l, t in (loc.get("tokens") or {}).items():
        rows += ["", f"Step 2, tokens at {l}: best {t.get('best')}, passing {t.get('passing')}", "",
                 "| group | law | random | passes |", "|---|---|---|---|"]
        rows += [f"| {gn} | {sc(v.get('law'))} | {sc(v.get('random'))} | {v.get('passes', v.get('skipped'))} |"
                 for gn, v in (t.get("by_group") or {}).items()]
    for l, ko in (loc.get("knockout_layers") or {}).items():
        rows.append(f"Layer {l} knock-out (curve span mean-ablated): effect {f((ko.get('law_span') or {}).get('effect'))}"
                    f"{c((ko.get('law_span') or {}).get('effect_ci95'))} vs random subspace "
                    f"{f((ko.get('random_span') or {}).get('effect'))}, passes {(ko.get('law_span') or {}).get('passes')}")
    for l, co in (loc.get("components") or {}).items():
        if co.get("skipped"):
            rows += ["", f"Step 3, components at {l}: {co['skipped']}"]
            continue
        at = co.get("attribution") or {}
        rows += ["", f"Step 3, components at {l} (attribution {'on' if at.get('available') else 'off: ' + str(at.get('reason'))}; "
                 f"exact {co.get('exact')}, controls {co.get('controls') or 'leave-one-out'}; attribution vs exact "
                 f"Spearman {f(co.get('attribution_vs_exact_spearman'))})", "",
                 "| component | attribution | patch | patch passes | knock-out effect | knock-out passes |",
                 "|---|---|---|---|---|---|"]
        for cn in co.get("exact", []) + co.get("controls", []):
            p_, k_ = co["patch"].get(cn, {}), co["knockout"].get(cn, {})
            rows.append(f"| {cn} | {sc((at.get('by_component') or {}).get(cn))} | {sc(p_)} | {p_.get('passes', '-')} | "
                        f"{f(k_.get('effect'))}{c(k_.get('effect_ci95'))} | {k_.get('passes', '-')} |")
    da = loc.get("das") or {}
    if da.get("by_k"):
        rows += ["", f"Step 4, DAS at {da.get('layer')}: k needed {da.get('k_needed')} (curve rank {da.get('curve_rank')}; "
                 f"more dims than the curve: {da.get('more_dims_than_curve')})", "",
                 f"| k | held-out IIA ({da.get('objective')} objective) | IIA through the model | principal cosines to the "
                 "curve |", "|---|---|---|---|"]
        for k_, v in da["by_k"].items():
            rows.append(f"| {k_} | {f(v['iia'])}{c(v.get('iia_ci95'))} | {sc(v.get('model'))} | "
                        f"{[round(x, 2) for x in v.get('principal_cosines_to_curve', [])]} |")
    v = loc.get("verdict") or {}
    rows += ["", f"Verdict ({v.get('rule')}): localized layers {v.get('localized_layers')}, components "
             f"{v.get('localized_components')}; disagreements {v.get('disagreements') or 'none'}"]
    return "\n".join(rows)
