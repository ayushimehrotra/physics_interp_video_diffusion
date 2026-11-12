"""L0 states (spec §5): can per-object states be decoded from pooled activations?

Targets, per object o in {A, B} and phase in {pre, post}:
  velocity (2-D world xy, one block), speed, direction (cos, sin), log_mass, radius,
  momentum (m·v, 2-D) and momentum_partial (m·v residualised on [1, v, m, log m]; see momentum_partial()).
Grid per target: layers x {roi_<o>.<phase>, roi_pair.<phase>, frame.<phase>}.
Score: out-of-fold R^2, grouped 5-fold by twin group, 12 fold seeds, nested alpha
(physlaw.probes.grid, identical to ridge.fit_probe).

Baselines, through the same pipeline (no kinematic observers, no permutation null: user decision 2026-11-11):
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

Selectivity (Hewitt & Liang control tasks, probes.ridge): every target's selected cell is refit on a scene control
(whole twin groups' values swapped across groups) and an object control (one fixed random value per object identity
class: name, shape, material, colour, size tertile); ``selectivity`` {scene, object} = real − control out-of-fold R²
with a group-bootstrap CI, beside every target and the same-instrument R².

Selection rule (spec §5): the reported best cell is chosen on half of the groups and scored on the
other half from the cross-fitted predictions (probes.select.heldout_select), with a group-bootstrap CI of the
selected cell's out-of-fold R² (``r2_ci95``); the in-sample max over the grid is kept as ``best_cell`` / ``r2``.
One-twin-per-group refit (spec §5.1): with < ``one_twin_refit_below`` groups, post-event targets are
refit at the selected cell on one twin per group (each twin type in turn, averaged): pre targets repeat
across a group's twins, post ones do not, which inflates post R² at small n.
Velocity protocol check (spec §5.1; ``velocity_protocols.rigidbench.enabled``): RigidBench's
unregularised linear probe trained with Adam on the own-ball ROI pools, per layer, selected by the rule.

Objects labelled ``aux`` (F9 rods, F11 prop) are never targets, never counted as objects and never enter the
tracker (:func:`object_names`). With one non-aux object the ``roi_pair`` pool kind is not searched (it is roi_A, or
roi_A plus an aux body's ROI: stage 4 pools segmentation ids 1 / 2).

Single-object families (one body, ``A``; no ``B``): :func:`targets` switches on the labels.
  * F8 (Galileo's ramp; :func:`targets_f8`): released from rest at height h on ramp A, descends, crosses the pad
    (speed s -> eps s), climbs ramp C to apex eps^2 h. ``A.pre.velocity`` / ``A.post.velocity`` = the block's
    velocity (x, z plane) just before / just after the pad (labels ``event_state``, io.law_inputs), ``.speed``;
    ``A.pre.height`` = release height, ``A.post.height`` = apex height on ramp C (inside the clean window), both
    above the floor.
  * F7 (free fall; :func:`targets_f7`; projectile, drop and bounce groups in one family): the A rows of the
    collision targets (3-D velocity, constant columns dropped) plus ``A.pre.height`` / ``A.post.height`` = the ball's
    highest point above the floor contact height (``constants.z_floor``) over the phase's in-frame frames (post:
    inside the clean window): the drop height (drop, bounce: the release height) and the rebound apex (bounce:
    eps^2 x the drop height); projectile: the highest point seen before / after the branch. The second drop ball (B)
    exists in drop groups only and is not a target.
  * F6 (slide / pair / glide; :func:`targets_f6`): the A rows of the collision targets plus ``A.pre.acceleration`` /
    ``A.post.acceleration`` = the phase-mean deceleration along the motion while moving (pre: the clip's own friction
    rate; post: after the invisible branch). Pair clips' block B is not a target (the family mixes one- and
    two-block groups).
  * other single-object families: the A rows of the collision targets (3-D velocity, constant columns dropped)
    Pools: roi_A / frame.
  * F10 (rolling, two cylinders; :func:`targets_f10`): velocity, speed and angular velocity w about the rolling
    axis per cylinder (phase means; pre is at rest and dropped), and ``<o>.pre.inertia_factor`` = kappa (solid vs
    hollow, which must be visible before the release).
  * F9 (pendulum; :func:`targets_f9`): per bob (pair: A and B; swing: A) and phase, ``angle`` = [mean, RMS] of the
    bob's angle from the downward vertical through its pivot and ``angular_velocity`` = [mean, RMS] of w_y (RMS w /
    RMS angle = 2 pi / T for a harmonic swing: the period's ingredients), ``log_mass``.
  * F11 (support; :func:`targets_f11`): the block's (A) phase-mean CoM velocity, speed, acceleration (x-z plane)
    and w_y (the tipping rotation); pre is at rest and dropped.
  * F12 (spin): the two-body targets (puck A, bar B, top-down) plus ``angular_velocity`` = the phase-mean w_z
    (the bar's spin after the contact; puck and pre-contact spins are zero and dropped).
  Degenerate target blocks (zero variance across clips) are dropped (single-object families, F10; F1-F4 unchanged).

Same instrument (spec §5.0.3 R1; ``same_instrument`` in the result): per object, ONE velocity probe trained on the
real pre AND post rows jointly (physlaw.probes.ridge.fit_instrument: inverse-calibrated, cross-fitted by group), at the
layers the held-out rule selected for the pre and post velocity targets, roi_own and frame pools. Reported: its
R² on pre and post rows, and, where the family has an exchange event (collision, bounce, pad; :func:`event_inputs`),
eps_hat = laws.epsilon of the decoded velocities against the true eps on the real clips: OLS slope, intercept,
R², median |error|. This is the measurability gate L1 reads. Its targets are the labelled velocities AT the event
instant (collisions: pre / post_velocity; bounce / pad: ``event_state``, exact simulator states entering / leaving the
event), never window averages (audit H3); the phase-mean post targets above are decodability targets only.

Mass before the event (audit H1; ``mass_pre_event``, :func:`mass_pre_event`): the pre-event log_mass targets (and F10's
inertia_factor) with held-out R², CI and selectivity; ``decodable`` needs R² CI > 0, every selectivity > 0 and (in
the table / leaderboard, which see the untrained run) R² above the untrained model's. The leaderboard flags the
mass-independence scores of a model / family without a decodable target as "uninformative".

Note (F1, legacy appearance arm): density is 1.5 or 6.0 and tied to the material, so mass is
visible through size AND material; the tracker (size only) bounds the size part.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from physlaw.probes.grid import grid_probe
from physlaw.probes.ridge import CONTROLS, adam_linear_probe, control_targets, r2_score, selectivity
from physlaw.probes.select import heldout_select
from physlaw.tokens.project import roi_object_ids

TARGETS = ("velocity", "speed", "direction", "log_mass", "radius", "momentum", "momentum_partial", "height",
           "angular_velocity", "angle", "acceleration", "inertia_factor")
MASS_TARGETS = ("log_mass", "inertia_factor")   # H1: must be decodable from PRE-event frames (mass_pre_event)
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


def complete_groups(recs: list[dict], data_dir: Path | None = None) -> list[dict]:
    """Keep only groups whose cache holds the group's full twin set. The full set is the group's own
    ``<data_dir>/<group>/_GROUP_COMPLETE`` twin list (sub-families have different twin sets: F6 slide / pair / glide,
    F7 projectile / drop / bounce, F9 pair / swing); groups without that record (or no data_dir) must hold as many
    twins as the largest such group."""
    twins_of: dict[str, set] = {}
    for r in recs:
        twins_of.setdefault(r["group"], set()).add(r["twin"])
    want = {}
    for g in twins_of:
        f = Path(data_dir) / g / "_GROUP_COMPLETE" if data_dir is not None else None
        if f is not None and f.exists():
            want[g] = set(json.loads(f.read_text())["twins"])
    rest = [len(s) for g, s in twins_of.items() if g not in want]
    full = max(rest) if rest else 0
    keep = {g for g, s in twins_of.items() if (want[g] <= s if g in want else len(s) == full)}
    return [r for r in recs if r["group"] in keep]


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


def momentum_partial(v: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """Partial momentum: p = m·v [n, 2] minus its least-squares fit on [1, v_x, v_y, m, log m] across clips.

    Fitted on the labels only (no features, so no leakage), separately per (object, phase) by the caller.
    It keeps the part of the product m·v that no linear combination of v and m explains, so a probe's R² on
    it is the partial R² of momentum beyond v and m: does the representation encode the product, not just its
    factors? A linear probe on features that linearly encode v and m (and log m) scores ≈ 0 on it.
    Returns None when the residual is degenerate (≈ 0 variance, e.g. mass constant for this object, where m·v
    is linear in v): R² on it would be meaningless, and the block is dropped.
    """
    v, m = np.asarray(v, float), np.asarray(m, float)
    p = m[:, None] * v
    D = np.column_stack([np.ones(len(m)), v, m, np.log(m)])
    coef, *_ = np.linalg.lstsq(D, p, rcond=None)
    r = p - D @ coef
    sst = ((p - p.mean(0)) ** 2).sum()
    if sst <= 0 or (r ** 2).sum() <= 1e-10 * sst:
        return None
    return r


def _family(labels: list[dict]) -> str:
    fams = {l["physlaw"].get("family", "f1") for l in labels}
    if len(fams) != 1:
        raise ValueError(f"L0 runs one family at a time, got {sorted(fams)}")
    return fams.pop()


def object_names(labels: dict) -> list[str]:
    """The clip's objects without the ``aux`` bodies (F9 rods, F11 prop), in label order (= io.law_inputs')."""
    return [n for n, o in labels["physlaw"]["objects"].items() if not o.get("aux")]


def _single_object(labels: list[dict]) -> bool:
    return any(len(object_names(l)) == 1 for l in labels)


def pool_kinds(labels: list[dict]) -> tuple[str, ...]:
    """POOL_KINDS, without roi_pair when the clips hold one non-aux object."""
    return tuple(k for k in POOL_KINDS if k != "roi_pair") if _single_object(labels) else POOL_KINDS


def _drop_degenerate(out: dict) -> dict:
    keep = {}
    for k_, y in out.items():
        y = np.asarray(y, float).reshape(len(y), -1)
        if ((y - y.mean(0)) ** 2).sum() > 1e-12 * max(1.0, (y ** 2).sum()):
            keep[k_] = out[k_]
    return keep


def object_class_ids(labels: list[dict], o: str) -> np.ndarray:
    """Identity class of object o per clip for the object control task (probes.ridge.control_targets): the object's
    name, shape, density (material), colour and size tertile across the clips. NaN-free strings."""
    obs = [l["physlaw"].get("objects", {}).get(o, {}) for l in labels]
    size = np.array([float(ob.get("radius", ob.get("size", 0.0))) for ob in obs])
    cut = np.quantile(size, [1 / 3, 2 / 3]) if len(size) else []
    out = []
    for ob, s_ in zip(obs, size):
        col = ob.get("color_rgb")
        out.append("|".join(str(x) for x in (o, ob.get("shape", ""), ob.get("density", ob.get("material", "")),
                                             "" if col is None else tuple(round(float(c), 2) for c in col),
                                             int(np.searchsorted(cut, s_)))))
    return np.array(out)


def event_inputs(labels: dict):
    """(v_pre [n_obj, 2], v_post [n_obj, 2], ctx) of a clip's exchange event for laws.epsilon / laws.momentum.stats,
    or None if it has none. Collisions (F1-F4, top-down): physlaw pre/post_velocity (x, y), ctx event "collision",
    contact normal n, masses m. Bounce (F7) and pad (F8), side view: the labels' event_state (x, z), ctx event
    ``physlaw.event_kind`` and the direction of gravity g = (0, -1) (a direction only: no magnitude enters)."""
    pl = labels["physlaw"]
    objs = object_names(labels)
    m = np.array([pl["objects"][o]["mass"] for o in objs], float)
    kind = pl.get("event_kind") or ("collision" if "contact_normal" in pl else None)
    if kind == "collision":
        n = np.asarray(pl["contact_normal"], float)[:2]
        return (np.array([pl["pre_velocity"][o][:2] for o in objs], float),
                np.array([pl["post_velocity"][o][:2] for o in objs], float),
                {"event": "collision", "n": n / np.linalg.norm(n), "m": m})
    if kind in ("bounce", "pad") and "event_state" in pl:
        st = pl["event_state"]
        rows = [list(st["pre"].get("objects", objs)).index(o) for o in objs]
        return (np.asarray(st["pre"]["v"], float)[rows][:, [0, 2]], np.asarray(st["post"]["v"], float)[rows][:, [0, 2]],
                {"event": kind, "g": np.array([0.0, -1.0]), "m": m})
    return None


def targets_f8(labels: list[dict]) -> dict[str, np.ndarray]:
    """F8 (Galileo's ramp) targets of the block (object A): see the module docstring."""
    cols = {k_: [] for k_ in ("pre_v", "post_v", "pre_h", "post_h")}
    for l in labels:
        pl, pf = l["physlaw"], l["per_frame"]
        fps = l["scene_constants"]["frame_rate"]
        z = np.array([f["A"]["position"][2] for f in pf], float)
        z0 = float(pl.get("constants", {}).get("z_floor", z.min()))
        k = pl["first_post_contact_frame"]
        end = int(np.floor(float(pl.get("clean_until_s", len(pf) / fps)) * fps))
        st = pl["event_state"]
        cols["pre_v"].append(np.asarray(st["pre"]["v"], float)[0, [0, 2]])
        cols["post_v"].append(np.asarray(st["post"]["v"], float)[0, [0, 2]])
        cols["pre_h"].append(z[0] - z0)
        cols["post_h"].append(float(z[k: max(end, k + 1)].max()) - z0)
    out = {}
    for ph in PHASES:
        v = np.array(cols[f"{ph}_v"], float)
        out[f"A.{ph}.velocity"] = v
        out[f"A.{ph}.speed"] = np.linalg.norm(v, axis=1)
        out[f"A.{ph}.height"] = np.array(cols[f"{ph}_h"], float)
    return _drop_degenerate(out)


def _targets_single(labels: list[dict]) -> dict[str, np.ndarray]:
    """Single-object families other than F8: the A rows of the collision targets, 3-D velocity."""
    out = {}
    m = np.array([l["physlaw"]["objects"]["A"]["mass"] for l in labels], float)
    for ph in PHASES:
        v = np.array([l["physlaw"][f"{ph}_velocity"]["A"] for l in labels], float)
        v = v[:, v.var(0) > 1e-12] if (v.var(0) > 1e-12).any() else v
        s = np.linalg.norm(v, axis=1)
        out[f"A.{ph}.velocity"] = v
        out[f"A.{ph}.speed"] = s
        out[f"A.{ph}.direction"] = v / np.maximum(s, 1e-9)[:, None]
        out[f"A.{ph}.log_mass"] = np.log(m)
        out[f"A.{ph}.radius"] = np.array([l["physlaw"]["objects"]["A"]["radius"] for l in labels])
        out[f"A.{ph}.momentum"] = m[:, None] * v
        pp = momentum_partial(v, m)
        if pp is not None:
            out[f"A.{ph}.momentum_partial"] = pp
    return _drop_degenerate(out)


def targets_f7(labels: list[dict]) -> dict[str, np.ndarray]:
    """F7 (free fall) targets of ball A: :func:`_targets_single` plus the phase's highest point above the floor
    contact height (see the module docstring)."""
    out = _targets_single(labels)
    for ph in PHASES:
        out[f"A.{ph}.height"] = np.array(
            [max(l["per_frame"][f]["A"]["position"][2] for f in phase_frames(l, "A")[ph])
             - float(l["physlaw"]["constants"]["z_floor"]) for l in labels], float)
    return _drop_degenerate(out)


def targets_f6(labels: list[dict]) -> dict[str, np.ndarray]:
    """F6 (slide / pair / glide in one family) targets of block / puck A: :func:`_targets_single` plus
    ``A.<phase>.acceleration`` = the phase-mean deceleration along the motion, -d|v|/dt over consecutive in-frame
    frames while A moves (pre: the slide's own friction rate; post: after the branch, inside the clean window; 0 for
    a glide, negative where a reverse twin speeds up again)."""
    out = _targets_single(labels)
    for ph in PHASES:
        col = []
        for l in labels:
            pf, fps = l["per_frame"], l["scene_constants"]["frame_rate"]
            fr = phase_frames(l, "A")[ph]
            sp = np.array([np.linalg.norm(pf[f]["A"]["velocity"]) for f in fr], float)
            pairs = [i for i in range(len(fr) - 1) if fr[i + 1] == fr[i] + 1 and sp[i] > 1e-9 and sp[i + 1] > 1e-9]
            col.append(float(np.mean([(sp[i] - sp[i + 1]) * fps for i in pairs])) if pairs else 0.0)
        out[f"A.{ph}.acceleration"] = np.array(col, float)
    return _drop_degenerate(out)


def phase_frames(l: dict, o: str) -> dict[str, list[int]]:
    """{phase: frames} of object o: pre = before the event, post = from the event to the end of the clean window,
    in-frame frames only (the event frame if none)."""
    pl, pf = l["physlaw"], l["per_frame"]
    fps = l["scene_constants"]["frame_rate"]
    k = pl["first_post_contact_frame"]
    end = min(len(pf), int(np.floor(float(pl.get("clean_until_s", len(pf) / fps)) * fps)) or len(pf))
    return {ph: [f for f in fr if pf[f][o].get("in_frame", True)] or [min(k, len(pf) - 1)]
            for ph, fr in (("pre", range(0, k)), ("post", range(k, max(end, k + 1))))}


def _varying(v: np.ndarray) -> np.ndarray:
    """The columns of v [n, d] that vary across clips (all of them when none does)."""
    return v[:, v.var(0) > 1e-12] if (v.var(0) > 1e-12).any() else v


def targets_f10(labels: list[dict]) -> dict[str, np.ndarray]:
    """F10 (rolling) targets per cylinder o in {A, B} and phase: mean over the phase's in-frame frames (inside the
    clean window) of the velocity (in-plane components that vary), speed and the angular velocity about the rolling
    axis (world y, signed; slip = |w| r / |v| is laws.rolling's rotation statistic). Released from rest, so the pre
    targets are degenerate and dropped; ``<o>.pre.inertia_factor`` = kappa = I / (m r^2) of each cylinder (H1)."""
    out = {}
    for o in OBJECTS:
        for ph in PHASES:
            rows = [(np.mean([l["per_frame"][f][o]["velocity"] for f in fr], 0),
                     np.mean([l["per_frame"][f][o].get("angular_velocity", [0.0, 0.0, 0.0])[1] for f in fr]))
                    for l in labels for fr in [phase_frames(l, o)[ph]]]
            v = _varying(np.array([r[0] for r in rows], float))
            out[f"{o}.{ph}.velocity"] = v
            out[f"{o}.{ph}.speed"] = np.linalg.norm(v, axis=1)
            out[f"{o}.{ph}.angular_velocity"] = np.array([r[1] for r in rows], float)
        # H1: solid vs hollow (kappa = I / (m r^2)) must be visible before the release for the rolling law to mean
        # anything (a model blind to kappa predicts equal accelerations)
        kap = [((l["physlaw"].get("constants") or {}).get("inertia_factors") or {}).get(o) for l in labels]
        if all(k_ is not None for k_ in kap):
            out[f"{o}.pre.inertia_factor"] = np.array(kap, float)
    return _drop_degenerate(out)


def mass_pre_event(tres: dict, untrained: dict | None = None) -> dict:
    """H1: the mass-independence laws (gravity / friction mass_ratio, oscillation mass_period_ratio) and rolling are
    only evidence when the model encodes mass (solid vs hollow: kappa) BEFORE the event; a model blind to mass
    passes "mass does not matter" trivially. Per pre-event mass target (:data:`MASS_TARGETS`, ``<o>.pre.<target>``):
    held-out R², its CI, selectivity {scene, object} and ``decodable`` = R² CI above 0 AND every selectivity > 0 AND
    (with ``untrained``, that model's L0 result) above the untrained model's held-out R². ``decodable`` of the family
    = any target decodable (None: no such target, e.g. mass constant across the clips)."""
    out = {}
    for b, t in tres.items():
        o, ph, q = b.split(".")
        if ph != "pre" or q not in MASS_TARGETS or "heldout" not in t:
            continue
        r2, ci = float(t["heldout"]["score"]), t.get("r2_ci95") or [float("nan")] * 2
        sel = {c: float(v["selectivity"]) for c, v in (t.get("selectivity") or {}).items()}
        u = ((untrained or {}).get("targets", {}).get(b, {}).get("heldout") or {}).get("score")
        ok = bool(np.isfinite(ci[0]) and ci[0] > 0 and sel and all(v > 0 for v in sel.values())
                  and (u is None or r2 > float(u)))
        out[b] = {"r2": r2, "r2_ci95": ci, "selectivity": sel, "untrained_r2": u, "decodable": ok}
    return {"targets": out, "decodable": (any(v["decodable"] for v in out.values()) if out else None),
            "rule": "held-out R² CI > 0, selectivity (scene, object) > 0, above the untrained model when given"}


def targets_f9(labels: list[dict]) -> dict[str, np.ndarray]:
    """F9 (pendulum) targets per bob (``constants.bobs``, aux rods skipped) and phase: ``angle`` = [mean, RMS] of the
    angle from the downward vertical through the bob's pivot (atan2(dx, -dz), rad), ``angular_velocity`` = [mean, RMS]
    of w_y, ``log_mass``."""
    out = {}
    for o in [b for b in labels[0]["physlaw"]["constants"].get("bobs", object_names(labels[0]))
              if b in object_names(labels[0])]:
        for ph in PHASES:
            ang, w, m = [], [], []
            for l in labels:
                pf, piv = l["per_frame"], np.asarray(l["physlaw"]["constants"]["pivots"][o], float)
                fr = phase_frames(l, o)[ph]
                rel = np.array([pf[f][o]["position"] for f in fr], float) - piv
                th = np.arctan2(rel[:, 0], -rel[:, 2])
                wy = np.array([pf[f][o].get("angular_velocity", [0.0, 0.0, 0.0])[1] for f in fr], float)
                ang.append([th.mean(), np.sqrt((th ** 2).mean())])
                w.append([wy.mean(), np.sqrt((wy ** 2).mean())])
                m.append(np.log(l["physlaw"]["objects"][o]["mass"]))
            out[f"{o}.{ph}.angle"] = np.array(ang, float)
            out[f"{o}.{ph}.angular_velocity"] = np.array(w, float)
            out[f"{o}.{ph}.log_mass"] = np.array(m, float)
    return _drop_degenerate(out)


def targets_f11(labels: list[dict]) -> dict[str, np.ndarray]:
    """F11 (support) targets of the block (A; the prop is aux) per phase: phase means of the CoM velocity (x, z),
    speed, CoM acceleration (x, z; the velocity change over the phase's frames / their span) and w_y (the tipping
    rotation)."""
    out = {}
    for ph in PHASES:
        rows = []
        for l in labels:
            pf, fps = l["per_frame"], l["scene_constants"]["frame_rate"]
            fr = phase_frames(l, "A")[ph]
            v = np.array([pf[f]["A"]["velocity"] for f in fr], float)[:, [0, 2]]
            acc = (v[-1] - v[0]) * fps / (fr[-1] - fr[0]) if len(fr) > 1 else np.zeros(2)     # mean over the phase
            rows.append((v.mean(0), acc, np.mean([pf[f]["A"].get("angular_velocity", [0.0] * 3)[1] for f in fr])))
        v = np.array([r[0] for r in rows], float)
        out[f"A.{ph}.velocity"] = v
        out[f"A.{ph}.speed"] = np.linalg.norm(v, axis=1)
        out[f"A.{ph}.acceleration"] = np.array([r[1] for r in rows], float)
        out[f"A.{ph}.angular_velocity"] = np.array([r[2] for r in rows], float)
    return _drop_degenerate(out)


def spin_targets(labels: list[dict]) -> dict[str, np.ndarray]:
    """F12: per object and phase, the phase-mean w_z (top-down spin; the bar's after the contact)."""
    out = {f"{o}.{ph}.angular_velocity": np.array(
        [np.mean([l["per_frame"][f][o].get("angular_velocity", [0.0] * 3)[2] for f in phase_frames(l, o)[ph]])
         for l in labels], float) for o in OBJECTS for ph in PHASES}
    return _drop_degenerate(out)


def targets(labels: list[dict]) -> dict[str, np.ndarray]:
    fam = _family(labels)
    if fam in ("f9", "f10", "f11"):
        return {"f9": targets_f9, "f10": targets_f10, "f11": targets_f11}[fam](labels)
    if fam == "f7":
        return targets_f7(labels)
    if fam == "f6":
        return targets_f6(labels)
    if _single_object(labels):
        return targets_f8(labels) if fam == "f8" else _targets_single(labels)
    out = {}
    for o in OBJECTS:
        m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels], float)
        for ph in PHASES:
            v = np.array([l["physlaw"][f"{ph}_velocity"][o] for l in labels], float)[:, :2]
            s = np.linalg.norm(v, axis=1)
            out[f"{o}.{ph}.velocity"] = v
            out[f"{o}.{ph}.speed"] = s
            out[f"{o}.{ph}.direction"] = v / np.maximum(s, 1e-9)[:, None]
            out[f"{o}.{ph}.log_mass"] = np.log(m)
            out[f"{o}.{ph}.radius"] = np.array([l["physlaw"]["objects"][o]["radius"] for l in labels])
            out[f"{o}.{ph}.momentum"] = m[:, None] * v
            pp = momentum_partial(v, m)
            if pp is not None:
                out[f"{o}.{ph}.momentum_partial"] = pp
    return {**out, **spin_targets(labels)} if fam == "f12" else out


def _mask_feat_one(args):
    clip_dir, frames, spec, fpc, names, ids = args
    from physlaw.tokens.pool import pool_weights
    from physlaw.tokens.project import project_masks, token_frame_phase
    inst = np.load(Path(clip_dir) / "masks.npz")["instance"]
    obj = project_masks(inst, list(frames), spec, ids)     # the same ROIs as stage 4 (aux objects never pooled)
    phases = token_frame_phase(list(frames), spec, fpc)
    ph = np.array(phases)
    out = {}
    for w in PHASES:     # tracker: the (non-aux) objects' token-space centre, radius, presence per token-frame
        cols = []
        for o in names:
            c = obj[o].center[ph == w]
            present = ~np.isnan(c[:, 0])
            cols += [np.nan_to_num(c[:, 0]), np.nan_to_num(c[:, 1]), obj[o].radius[ph == w], present.astype(float)]
        out[f"geom.{w}"] = np.concatenate(cols)
    W = pool_weights(obj, phases)     # mask floor: the pooling weights of each ROI pool, flattened
    for name, wt in W.items():
        if name.startswith(("roi_", "contact")) and not name.endswith(".all"):
            out[f"w.{name}"] = wt.reshape(-1).astype(np.float32)
    return out


def mask_features(cache_dir: Path, data_dir: Path, recs: list[dict], labels: list[dict], spec,
                  workers: int = 32) -> dict[str, np.ndarray]:
    """{'geom.pre'|'geom.post': [n, F], 'w.<pool>': [n, T*gh*gw]}, cached in <cache_dir>/_maskfeat_v2.npz."""
    f = cache_dir / "_maskfeat_v2.npz"
    clips = [r["clip"] for r in recs]
    if f.exists():
        z = np.load(f)
        if list(z["clips"]) == clips and "w.contact.post" in z.files:   # rebuild caches made before the contact pool
            return {k: z[k] for k in z.files if k != "clips"}
    jobs = [(str(data_dir / r["clip"]), np.load(cache_dir / r["file"])["frames"], spec,
             l["physlaw"]["first_post_contact_frame"], [o for o in OBJECTS if o in object_names(l)], roi_object_ids(l))
            for r, l in zip(recs, labels)]
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
                           fold_seeds=_G["fold_seeds"])


def _features(xfun):
    kind = xfun[0]
    if kind == "act":
        _, pi, li = xfun
        return _G["acts"][:, pi, li]
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


def r2_boot_ci(oof: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_boot: int = 500, seed: int = 0):
    """Group-bootstrap 95% CI of a cell's out-of-fold R² (mean over fold seeds; oof [seeds, n, k])."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        r = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        bs.append(np.mean([r2_score(Y[r], o.reshape(len(Y), -1)[r]) for o in oof]))
    return [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]


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
    probe = cfg["probe"]
    recs = complete_groups(load_index(cache_dir), data_dir)
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
              alphas=tuple(float(x) for x in probe["alphas"]), k=int(probe["k_folds"]), fold_seeds=int(probe["fold_seeds"]))

    # one job per distinct feature matrix, carrying every target block that searches it
    kinds = pool_kinds(labels)
    jobs: dict[tuple, list[str]] = {}
    for b in Y:
        o, ph, _ = b.split(".")
        for kind in kinds:
            pi = pools.index(_pool_name(kind, o, ph))
            for li in range(len(layers)):
                jobs.setdefault(("act", pi, li), []).append(b)
            for lk in lat:
                jobs.setdefault(("lat", pi, lk), []).append(b)
        jobs.setdefault(("mask", f"geom.{ph}"), []).append(b)
        for kind in [k_ for k_ in kinds if k_.startswith("roi")]:
            jobs.setdefault(("mask", f"w.{_pool_name(kind, o, ph)}"), []).append(b)
    log(f"[L0] {len(jobs)} feature cells x their targets; {workers} workers")

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
                        "selection_splits": n_splits}}
    for b in Y:
        o, ph, _ = b.split(".")
        cells = {}
        for kind in kinds:
            pname = _pool_name(kind, o, ph)
            pi = pools.index(pname)
            for li, l in enumerate(layers):
                cells[f"{_lname(l)}/{pname}"] = res[("act", pi, li)][b]
        best_cv = max(cells, key=lambda c: cells[c]["score_mean"])
        curve = {kind: [cells[f"{_lname(l)}/{_pool_name(kind, o, ph)}"]["score_mean"] for l in layers] for kind in kinds}
        geom = res[("mask", f"geom.{ph}")][b]
        bpool = best_cv.rsplit("/", 1)[1]          # cell = "<layer key>/<pool>"; diffusion keys contain "/"
        floor = res[("mask", f"w.{bpool}")][b] if ("mask", f"w.{bpool}") in res else None
        out["targets"][b] = {
            "best_cell": best_cv, "r2": cells[best_cv]["score_mean"], "r2_sd": cells[best_cv]["score_sd"],
            "curve": curve,
            # floor for the best cell's own pool (frame pools have uniform weights: floor 0)
            "mask_floor": {"pool": bpool, "r2": floor["score_mean"] if floor else 0.0,
                           "r2_sd": floor["score_sd"] if floor else 0.0},
            "mask_floor_by_pool": {k[1][2:]: res[k][b]["score_mean"] for k in res
                                   if k[0] == "mask" and k[1].startswith("w.") and b in res[k]},
            "tracker": {"r2": geom["score_mean"], "r2_sd": geom["score_sd"]},
            "heldout": _select(cells, Y[b], groups, n_splits),
        }
        out["targets"][b]["r2_ci95"] = r2_boot_ci(cells[out["targets"][b]["heldout"]["selected"]]["oof"], Y[b], groups)
        if lat:
            lcells = {f"{lk}/{_pool_name(kind, o, ph)}": res[("lat", pools.index(_pool_name(kind, o, ph)), lk)][b]
                      for kind in kinds for lk in lat}
            out["targets"][b]["latent_baseline"] = _select(lcells, Y[b], groups, n_splits)
    # selectivity control tasks (Hewitt & Liang; probes.ridge): the selected cell refit on scene / object controls
    ids = {o: object_class_ids(labels, o) for o in {b.split(".")[0] for b in out["targets"]}}
    sjobs = []
    for b, t in out["targets"].items():
        o = b.split(".")[0]
        for name, Yc in control_targets(Y[b], groups, ids[o], seed=0).items():
            Y[f"{b}|{name}"] = Yc
        lk, pname = t["heldout"]["selected"].rsplit("/", 1)
        sjobs.append((("act", pools.index(pname), [_lname(l) for l in layers].index(lk)), b,
                      [f"{b}|{c}" for c in CONTROLS]))
    with ProcessPoolExecutor(workers) as ex:            # fork after Y holds the control blocks
        for (xf, b), r in ex.map(_run_cell, [((xf, b), xf, v) for xf, b, v in sjobs], chunksize=1):
            P = res[xf][b]["oof"].mean(0)
            out["targets"][b]["selectivity"] = {c: selectivity(Y[b], P, Y[f"{b}|{c}"], r[f"{b}|{c}"]["oof"].mean(0),
                                                               groups) for c in CONTROLS}
    log(f"[L0] selectivity controls: {len(sjobs)} cells")
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

    out["mass_pre_event"] = mass_pre_event(out["targets"])
    si = cfg.get("same_instrument") or {}
    if si.get("enabled", True):
        out["same_instrument"] = same_instrument(acts, pools, layers, labels, groups, out["targets"], probe,
                                                 int(si.get("fold_seeds", 1)), tuple(si.get("pools", ("roi_own", "frame"))))
        log(f"[L0] same instrument: {out['same_instrument'].get('epsilon')}")

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


def eps_gate(eps_hat: np.ndarray, eps_true: np.ndarray) -> dict:
    """OLS of decoded eps on true eps over real clips (the R1 measurability gate): slope, intercept, R², median |err|."""
    ok = np.isfinite(eps_hat) & np.isfinite(eps_true)
    x, y = eps_true[ok], eps_hat[ok]
    if ok.sum() < 3 or np.ptp(x) < 1e-12:
        return {"n": int(ok.sum()), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "median_abs_err": float(np.median(np.abs(y - x))) if ok.any() else float("nan")}
    b, a = np.polyfit(x, y, 1)
    return {"n": int(ok.sum()), "slope": float(b), "intercept": float(a), "r2": r2_score(y, a + b * x) if np.ptp(y) > 0 else float("nan"),
            "median_abs_err": float(np.median(np.abs(y - x)))}


def same_instrument(acts, pools, layers, labels, groups, tres: dict, probe: dict, seeds: int = 1,
                    kinds=("roi_own", "frame")) -> dict:
    """R1 velocity instrument per object (see the module docstring); cells: the layers selected for the object's pre
    and post velocity targets (``tres`` = result["targets"]) x ``kinds``."""
    from physlaw.eval.l3_invariance import epsilons
    from physlaw.probes.ridge import apply_instrument, instrument_folds, instrument_selectivity
    objs = [o for o in OBJECTS if f"{o}.pre.velocity" in tres and f"{o}.post.velocity" in tres]
    if not objs:
        return {"skipped": "no pre and post velocity targets"}
    ev = [event_inputs(l) for l in labels]
    lk = [_lname(l) for l in layers]
    k, alphas = int(probe["k_folds"]), tuple(float(a) for a in probe["alphas"])
    out, dec = {}, {}
    for o in objs:
        Vp = np.array([l["physlaw"]["pre_velocity"][o] for l in labels], float)
        Vq = np.array([l["physlaw"]["post_velocity"][o] for l in labels], float)
        if ev[0] is not None:          # the event's own velocities (F7 bounce / F8 pad: event_state, x-z plane)
            oi = object_names(labels[0]).index(o)
            Vp, Vq = np.stack([e[0][oi] for e in ev]), np.stack([e[1][oi] for e in ev])
        else:
            Vp, Vq = Vp[:, :2], Vq[:, :2]
        Ls = sorted({lk.index(tres[f"{o}.{ph}.velocity"]["heldout"]["selected"].rsplit("/", 1)[0]) for ph in PHASES})
        Xc = {f"{lk[li]}/{kind}": (acts[:, pools.index(_pool_name(kind, o, "pre")), li],
                                   acts[:, pools.index(_pool_name(kind, o, "post")), li])
              for kind in kinds for li in Ls}
        folds = instrument_folds(Xc, (Vp, Vq), groups, None, k, seeds, alphas)
        dp = apply_instrument(folds, {c: x[0] for c, x in Xc.items()}, groups)
        dq = apply_instrument(folds, {c: x[1] for c, x in Xc.items()}, groups)
        dec[o] = (dp, dq)
        out[o] = {"cells": sorted({c for *_, c, _ in folds}), "n_cells": len(Xc),
                  "r2_pre": r2_score(Vp, dp), "r2_post": r2_score(Vq, dq),
                  "selectivity": instrument_selectivity(Xc, (Vp, Vq), groups, object_class_ids(labels, o), k, seeds,
                                                        alphas)}
    if all(e is not None for e in ev) and len(objs) == len(object_names(labels[0])):
        ctxs = [e[2] for e in ev]
        vp, vq = np.stack([dec[o][0] for o in objs], 1), np.stack([dec[o][1] for o in objs], 1)
        out["epsilon"] = eps_gate(epsilons(vp, vq, ctxs),
                                  epsilons(np.stack([e[0] for e in ev]), np.stack([e[1] for e in ev]), ctxs))
    return out


def _sel(sv) -> str:
    """'scene / object' selectivity (real − control R²) with CIs, for the tables."""
    if not sv:
        return ""
    f = lambda d: f"{d['selectivity']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
    return " / ".join(f(sv[c]) for c in ("scene", "object") if c in sv)


def table(res: dict, untrained: dict | None = None) -> str:
    pr = res.get("protocol", {})
    n_var = {b: t["heldout"]["n_variants"] for b, t in res["targets"].items() if "heldout" in t}
    rows = [f"n_groups = {res['n_groups']}, {pr.get('k_folds', 5)}-fold x {pr.get('fold_seeds', '?')} seeds "
            f"(~{pr.get('test_groups_per_fold', res['n_groups'] / 5):.0f} test groups per fold), "
            "variants searched per target: "
            f"{sorted(set(n_var.values())) or '?'} (layers x pools). Held-out R²: cell selected on half the groups, "
            f"scored on the other half ({pr.get('selection_splits', '?')} random splits x 2); CI: group bootstrap of the "
            "selected cell's out-of-fold R²; in-sample max R² for reference.", "",
            "| target | held-out R² (selected cell, picks) | R² 95% CI | in-sample best cell: R² (sd) | untrained held-out R² | "
            "one-twin refit R² (mask floor) | mask floor (same pool) | latent baseline | tracker | selectivity scene / "
            "object [CI] |",
            "|---|---|---|---|---|---|---|---|---|---|"]
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
        ci = t.get("r2_ci95")
        rows.append(f"| {b} | {hs} | {f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else ''} | {t['best_cell']}: {t['r2']:.3f} "
                    f"({t['r2_sd']:.3f}) | {'' if u is None else f'{u:.3f}'} | {os_} | {t['mask_floor']['r2']:.3f} | "
                    f"{lbs} | {t['tracker']['r2']:.3f} | {_sel(t.get('selectivity'))} |")
    vp = {b: t["velocity_protocol"] for b, t in res["targets"].items() if "velocity_protocol" in t}
    if vp:
        rows += ["", "Velocity protocol check (spec §5.1), own-ball ROI pool, layer selected by the held-out rule:", "",
                 "| target | ridge (this benchmark) held-out R² (layer) | RigidBench (Adam, unregularised) held-out R² (layer) |",
                 "|---|---|---|"]
        for b, v in vp.items():
            r, a = v["ridge_same_pool"], v["rigidbench"]
            rows.append(f"| {b} | {r['score']:.3f} ({r['selected']}) | {a['score']:.3f} ({a['selected']}) |")
    mp = mass_pre_event(res["targets"], untrained) if "targets" in res else None
    if mp and mp["targets"]:
        rows += ["", "Mass / inertia from PRE-event frames (H1: the mass-independence laws and rolling read as "
                 "'uninformative' when no target is decodable; rule: " + mp["rule"] + "):", "",
                 "| target | held-out R² [CI] | untrained R² | selectivity scene / object | decodable |", "|---|---|---|---|---|"]
        for b, v in mp["targets"].items():
            u = v["untrained_r2"]
            rows.append(f"| {b} | {v['r2']:.3f} [{v['r2_ci95'][0]:.3f}, {v['r2_ci95'][1]:.3f}] | "
                        f"{'' if u is None else f'{u:.3f}'} | "
                        f"{' / '.join(format(x, '+.2f') for x in v['selectivity'].values())} | "
                        f"{'yes' if v['decodable'] else 'NO'} |")
    si = res.get("same_instrument")
    if si and "skipped" not in si:
        rows += ["", "Same instrument (R1): one velocity probe per object on real pre + post rows, inverse-calibrated:", "",
                 "| object | R² pre | R² post | cells | selectivity scene / object [CI] |", "|---|---|---|---|---|"]
        rows += [f"| {o} | {v['r2_pre']:.3f} | {v['r2_post']:.3f} | {v['cells']} | {_sel(v.get('selectivity'))} |"
                 for o, v in si.items() if o != "epsilon"]
        e = si.get("epsilon")
        if e:
            rows.append(f"\neps_hat vs eps (real clips, n {e['n']}): slope {e['slope']:.3f}, intercept {e['intercept']:.3f}, "
                        f"R² {e['r2']:.3f}, median abs err {e['median_abs_err']:.3f}")
    return "\n".join(rows)
