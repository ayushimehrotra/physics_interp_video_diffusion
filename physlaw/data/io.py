"""Reading rendered clips (any venv with numpy + imageio[ffmpeg])."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Clip:
    path: Path
    labels: dict
    frames: np.ndarray | None = None     # uint8 [T, H, W, 3], master frames
    masks: np.ndarray | None = None      # uint8 [T, H, W], 0 floor, 1 A, 2 B

    @property
    def clip_id(self) -> str:
        pl = self.labels["physlaw"]
        return f"{pl['twin_group_id']}/{pl['twin']}"

    @property
    def pl(self) -> dict:
        return self.labels["physlaw"]


def load_clip(path: str | Path, frames: bool = True, masks: bool = True) -> Clip:
    path = Path(path)
    labels = json.loads((path / "labels.json").read_text())
    c = Clip(path, labels)
    if frames:
        import imageio.v3 as iio
        c.frames = iio.imread(path / "video.mp4", plugin="FFMPEG")
    if masks:
        c.masks = np.load(path / "masks.npz")["instance"]
    return c


def list_clips(data_dir: str | Path, twins: list[str] | None = None) -> list[Path]:
    """Every completed twin clip under a family dir, ordered by group then twin index."""
    out = []
    for g in sorted(Path(data_dir).glob("*/_GROUP_COMPLETE")):
        rec = json.loads(g.read_text())
        for t in rec["twins"]:
            if (twins is None or t in twins) and (g.parent / t / "_COMPLETE").exists():
                out.append(g.parent / t)
    return out


COLLISION_FAMILIES = ("f1", "f2", "f3", "f4")
ACC_FAMILIES = ("f6", "f7", "f10", "f11")
ACC_WINDOW_S = 0.1          # one-sided fit window for accelerations (labels and trackers use the same estimator)


def fit_acceleration(t: np.ndarray, p: np.ndarray, t_ref: float) -> np.ndarray:
    """Least-squares quadratic p(t) = c0 + c1 (t - t_ref) + c2 (t - t_ref)^2 per coordinate; returns a = 2 c2.
    t [T], p [T, ...] (T >= 3)."""
    t = np.asarray(t, np.float64) - t_ref
    A = np.stack([np.ones_like(t), t, t * t], 1)
    P = np.asarray(p, np.float64)
    c = np.linalg.lstsq(A, P.reshape(len(t), -1), rcond=None)[0]
    return 2.0 * c[2].reshape(P.shape[1:])


def contact_positions(p: np.ndarray, v: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Two sphere centres p [2, d] moved with their velocities v [2, d] to the first instant at which
    |p_B - p_A| = r_A + r_B (unchanged if they never touch on that line)."""
    dp, dv, R = p[1] - p[0], v[1] - v[0], float(r[0] + r[1])
    a, b, c = dv @ dv, dp @ dv, dp @ dp - R * R
    disc = b * b - a * c
    tau = (-b - np.sqrt(disc)) / a if a > 0 and disc >= 0 else 0.0
    return p + v * tau


def zero_crossings(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Times at which x(t) changes sign, from per-frame values x and their rates v (cubic Hermite root inside each
    sign-changing frame interval, so the crossing time is exact to O(dt^4) on a smooth path)."""
    out = []
    for i in np.nonzero(np.signbit(x[:-1]) != np.signbit(x[1:]))[0]:
        h = t[i + 1] - t[i]
        c = np.polynomial.Polynomial([x[i], h * v[i], -3 * x[i] - 2 * h * v[i] + 3 * x[i + 1] - h * v[i + 1],
                                      2 * x[i] + h * v[i] - 2 * x[i + 1] + h * v[i + 1]])
        r = [z.real for z in c.roots() if abs(z.imag) < 1e-9 and -1e-9 <= z.real <= 1 + 1e-9]
        out.append(t[i] + h * (min(r, key=lambda z: abs(z - 0.5)) if r else x[i] / (x[i] - x[i + 1])))
    return np.asarray(out)


def pendulum_periods(labels: dict, names: list) -> tuple[np.ndarray, np.ndarray]:
    """F9: (T_pre, T_post) per bob, twice the interval between successive passes through the lowest point
    (horizontal offset from the bob's pivot changes sign), the last interval wholly before the event and the first
    wholly after it. NaN where the clip has no such interval (the swing sub-family has no full one before)."""
    pl, fps = labels["physlaw"], labels["scene_constants"]["frame_rate"]
    te, piv = pl["event_time_s"], pl["constants"]["pivots"]
    pf = labels["per_frame"]
    t = np.arange(len(pf)) / fps
    T0, T1 = np.full(len(names), np.nan), np.full(len(names), np.nan)
    for i, n in enumerate(names):
        x = np.array([f[n]["position"][0] for f in pf]) - piv[n][0]
        c = zero_crossings(t, x, np.array([f[n]["velocity"][0] for f in pf]))
        before = [(a, b) for a, b in zip(c[:-1], c[1:]) if b <= te + 1e-9]
        after = [(a, b) for a, b in zip(c[:-1], c[1:]) if a >= te - 1e-9]
        if before:
            T0[i] = 2.0 * (before[-1][1] - before[-1][0])
        if after:
            T1[i] = 2.0 * (after[0][1] - after[0][0])
    return T0, T1


from physlaw.eval.laws.incline import tip_window  # noqa: E402  F11: "phi" over the tip window after the release

STOP_MARGIN_S = 0.1         # F6 friction windows end / start this long before / after the legal stop


def friction_windows(labels: dict, margin: float = STOP_MARGIN_S) -> dict | None:
    """F6 slide / pair windows (s, absolute; None for other clips): ``te`` the branch; moving window [te, move_end],
    move_end = min(legal stop ``constants.t_stop_s``, this twin's own stop ``t_stop_twin_s``) - margin (every block of
    the twin still moving); stop window [stop_start, end], stop_start = legal stop + margin, end = clean_until_s (the
    clip end). The legal stop is shared by the group (same pre-event frames), so every twin is read on the same
    stop window; a forecast is scored on the same windows."""
    pl = labels["physlaw"]
    c = pl.get("constants") or {}
    if pl.get("family") != "f6" or "t_stop_s" not in c:
        return None
    fps, n = labels["scene_constants"]["frame_rate"], len(labels.get("per_frame") or [])
    t_stop = float(c["t_stop_s"])
    return {"te": float(pl["event_time_s"]), "t_stop": t_stop,
            "move_end": min(t_stop, float(c.get("t_stop_twin_s", np.inf))) - margin,
            "stop_start": t_stop + margin, "end": float(pl.get("clean_until_s", (n - 1) / fps))}


def _rate(per_frame, names, frames, key: str, fps: float) -> np.ndarray:
    """Least-squares slope per object of the per-frame vector ``key`` over ``frames`` ([n, 3]; NaN if < 2 frames)."""
    frames = [f for f in frames if 0 <= f < len(per_frame)]
    if len(frames) < 2:
        return np.full((len(names), 3), np.nan)
    t = np.array(frames, np.float64) / fps
    X = np.array([[per_frame[f][n].get(key, [0.0, 0.0, 0.0]) for n in names] for f in frames], np.float64)
    tc = t - t.mean()
    return np.tensordot(tc, X - X.mean(0), axes=(0, 0)) / float(tc @ tc)


def rotation_angle(per_frame, names, f0: int, f1: int) -> np.ndarray:
    """Per object, the rotation angle (rad) between frames f0 and f1: 2 acos |q0 . q1| from the labelled
    quaternions, else the trapezoid integral of |angular velocity|."""
    out = np.zeros(len(names))
    for i, n in enumerate(names):
        q0, q1 = per_frame[f0][n].get("quaternion"), per_frame[f1][n].get("quaternion")
        if q0 is not None and q1 is not None:
            d = abs(float(np.dot(q0, q1))) / (np.linalg.norm(q0) * np.linalg.norm(q1))
            out[i] = 2.0 * np.arccos(min(d, 1.0))
        else:
            w = np.array([np.linalg.norm(per_frame[f][n].get("angular_velocity", [0.0, 0.0, 0.0]))
                          for f in range(f0, f1 + 1)])
            dt = per_frame[1]["time_s"] - per_frame[0]["time_s"] if len(per_frame) > 1 else 0.0
            out[i] = 0.5 * float((w[:-1] + w[1:]).sum()) * dt
    return out


def _f11_g_clip(labels: dict, names: list, te: float) -> float:
    """F11: g measured in the clip (the statics law uses no metric g): the block's free-fall acceleration after
    it leaves the ledge (labels ``edit.t_separate_s``) when a fit window of ACC_WINDOW_S fits before
    clean_until_s, else the prop's descent divided by its known ratio (``constants.prop_acc_over_g``), else NaN."""
    pl, fps, per_frame = labels["physlaw"], labels["scene_constants"]["frame_rate"], labels["per_frame"]
    w = max(int(round(ACC_WINDOW_S * fps)), 3)
    t_sep = (pl.get("edit") or {}).get("t_separate_s")
    clean = float(pl.get("clean_until_s", len(per_frame) / fps))

    def fall(name, f0):
        fr = [f for f in range(f0, f0 + w) if f < len(per_frame)]
        if len(fr) < 3 or name not in per_frame[0]:
            return float("nan")
        P = np.array([per_frame[f][name]["position"] for f in fr], np.float64)
        return float(np.linalg.norm(fit_acceleration(np.array(fr) / fps, P, fr[0] / fps)))

    if t_sep is not None and t_sep + ACC_WINDOW_S <= clean:
        return fall(names[0], int(np.floor(t_sep * fps)) + 1)
    ratio = pl["constants"].get("prop_acc_over_g")
    prop = [n for n, o in pl["objects"].items() if o.get("aux")]
    if ratio and prop:
        return fall(prop[0], int(np.floor(te * fps)) + 1) / float(ratio)
    return float("nan")


def world_view(labels: dict) -> dict:
    """The labels with their WORLD-frame states as the primary ones. A moving-camera clip (F4 camera boost,
    ``physlaw.frame == "camera"``) labels its primary states (per_frame position / velocity, pre / post_velocity,
    event_state, pre / post_state) in the camera's rest frame -- what the clip shows -- and keeps the world states
    under ``*_world``; this swaps them (a shallow copy; momentum / kinetic_energy per frame are dropped, being
    frame-dependent). Other labels come back unchanged."""
    pl = labels["physlaw"]
    if pl.get("frame") != "camera":
        return labels
    names = list(pl["objects"])
    pf = []
    for fr in labels.get("per_frame") or []:
        e = dict(fr)
        for n in names:
            o = {k: v for k, v in fr[n].items() if k not in ("momentum", "kinetic_energy", "speed")}
            o["position"], o["velocity"] = fr[n]["position_world"], fr[n]["velocity_world"]
            e[n] = o
        pf.append(e)
    pl2 = dict(pl, frame="world")
    for k in ("pre_velocity", "post_velocity", "event_state", "pre_state", "post_state"):
        if f"{k}_world" in pl:
            pl2[k] = pl[f"{k}_world"]
    return {**labels, "per_frame": pf, "physlaw": pl2}


def law_inputs(labels: dict, dt: float | None = None, dt_pre: float | None = None,
               frame: str | None = None) -> tuple[dict, dict, dict]:
    """``(pre, post, ctx)`` for ``physlaw.eval.laws.<law>.stats(pre, post, ctx)``, from any labels file.

    ``frame``: None = the labels' primary frame, which is the motion the clip shows (``physlaw.frame``: "camera" for
    an F4 camera-boost twin, i.e. the camera's rest frame, in which its balls move exactly as in the ball-boost twin;
    "world" otherwise); "world" = the world states (:func:`world_view`; for a camera-boost twin the unboosted
    collision, so the Galilean law is trivially exact there). ctx["frame"] says which. Readouts, levels and model
    comparisons use the default.

    pre / post: {"objects": [names], "p": [n, 3], "v": [n, 3], "w": [n, 3], "t_s"} (world frame, z up), plus
      "a" [n, 3] for the gravity / rolling families (f6, f7, f10): a least-squares quadratic fit of the per-frame
      positions over ACC_WINDOW_S on that side of the event (pre: frames before it; post: frames from the post
      state on), the estimator a tracker would use.
      ``dt=None``: the instants just before and just after the event (labels-2 ``event_state``; for collisions,
      bounces and the F8 pad: entering / leaving it). These are the simulator's exact states AT the event instant
      (closed-form families write them analytically; labels-1 F1 moves at constant velocity on both sides), never
      window averages, so eps from them is exact (audit H3; the decoded L1 estimators extrapolate to the event). ``dt`` (s): post is the labelled frame nearest to
      event + dt (gravity, rolling); ctx["dt"] is the exact gap. ``dt_pre`` (s, >= 0): pre is likewise the
      labelled frame nearest event + dt_pre (on the post side), its "a" fitted forward from it -- two instants of
      one post-event segment, e.g. friction's early / late decelerations (``dt_pre=0, dt=late``).
      The frozen labels-1 F1 render has constant velocities around contact, so its frames serve.
      Exact accelerations ("a") or spins in a labelled event_state (F11) are kept; only missing ones are fitted.
      Objects labelled ``aux`` (F9 rods, F11 prop) are left out everywhere.
      F9: pre["T"] / post["T"] [n] = the pendulum periods before / after the event (:func:`pendulum_periods`);
      swing: pre / post are the apex states around the damper (the release apex and the first apex after it,
      v = 0; labels ``edit.apex_states``), so the energy law compares heights only.
      F11: post["alpha"] [n, 3] = the angular acceleration at the release (exact in the event_state, else fitted to
      the per-frame angular velocities), post["phi"] [n] = the rotation angle (rad) over the tip window
      (laws.incline.tip_window: 2 sqrt(a / g_clip) after the release, or until the block leaves the ledge, whichever
      is first; clipped to clean_until_s; without g_clip: up to clean_until_s), ctx["g_clip"] = g measured in the
      clip (:func:`_f11_g_clip`), ctx["t_separate_s"] = when the block leaves the ledge (tip twins).
      Collision families: p (pre and post) are the centres at the contact instant (:func:`contact_positions`) and
      ctx["n"] = (p_B - p_A) / |p_B - p_A| from them; the labelled ``contact_normal`` (up to 0.66 deg off in F1)
      only when the labelled centres are not near contact (|p_B - p_A| outside 0.5-1.5 x (r_A + r_B)).
    ctx: the eval laws' vocabulary -- event (collision | bounce | pad | swing, else the labels' event_kind:
      branch | release | contact), g (vector [0, 0, -|g|]), g_mag, dt, m [n], r [n] (radius or
      half-extent), k [n] (I / (m r^2)), n, c (contact point, f12), theta, h, e_range (0, 1), force_free (f6
      glide); f6 slide / pair (:func:`friction_windows`): post["a"] = the quadratic fit over the moving window after
      the branch, post["v_stop"] = the mean velocity over the stop window, ctx dt = the moving window's mean time
      after the branch, move_end, stop_start -- plus every ``physlaw.constants`` entry under its own name (f9 swing: p_bottom, kappa; f11:
      overhang_x, aspect_ba; f12: e, contact_point), event_time_s,
      family, subfamily, twin, legal, violated_law, violation_magnitude, and when labelled epsilon (the twin's
      dose), clean_until_s, stop_dose; f4 also ref = (pre, post) of the unboosted twin with the same e.
    """
    if frame not in (None, "world", "camera"):
        raise ValueError(f"frame {frame!r}: None | world | camera")
    if frame == "world":                     # ("camera" = the primary states: a static camera's rest frame is the world)
        labels = world_view(labels)
    pl = labels["physlaw"]
    names = [n for n, o in pl["objects"].items() if not o.get("aux")]
    fps = labels["scene_constants"]["frame_rate"]
    per_frame = labels.get("per_frame") or []
    fam = pl["family"]

    def as_state(st):
        rows = [list(st["objects"]).index(n) for n in names] if "objects" in st else list(range(len(names)))
        return {"objects": names, "t_s": st["t_s"],
                **{k: np.asarray(st[k], dtype=np.float64)[rows] for k in ("p", "v", "w", "a", "alpha") if k in st}}

    def frame(f):
        fr = per_frame[f]
        return {"objects": names, "t_s": f / fps,
                **{k: np.array([fr[n].get(key, [0.0, 0.0, 0.0]) for n in names], dtype=np.float64)
                   for k, key in (("p", "position"), ("v", "velocity"), ("w", "angular_velocity"))}}

    def accel(frames, t_ref):
        frames = [f for f in frames if 0 <= f < len(per_frame)]
        if len(frames) < 3:
            return None
        P = np.array([[per_frame[f][n]["position"] for n in names] for f in frames], dtype=np.float64)
        return fit_acceleration(np.array(frames) / fps, P, t_ref)

    k = pl["first_post_contact_frame"]
    if "event_state" in pl:
        apex = (pl.get("edit") or {}).get("apex_states")
        swing = pl.get("event_kind") == "swing" or (fam == "f9" and pl.get("subfamily") == "swing")
        st = apex if swing and apex else pl["event_state"]          # f9 swing: the apexes around the damper
        pre, post = as_state(st["pre"]), as_state(st["post"])
        c = dict(pl["constants"])
        te = pl["event_time_s"]
    else:                                    # labels-1 (F1, 2026-11-05): constant velocity on both sides
        te = pl["contact_frame_actual"] / fps
        pre, post = frame(k - 1), frame(k)
        c = {"g": 0.0, "fps": fps, "e": pl["e_target"], "e_sim": pl["e_sim"],
             "contact_normal": list(pl["contact_normal"]) + [0.0],
             "masses": {n: pl["objects"][n]["mass"] for n in names},
             "sizes": {n: pl["objects"][n]["radius"] for n in names}}
        for st in (pre, post):               # positions at the event instant
            st["p"] = st["p"] + st["v"] * (te - st["t_s"])
            st["t_s"] = te
        c["legal"] = pl["twin"].startswith("conserve")
        c["violated_law"] = {"mom_violate_e1": "momentum", "mom_violate_e015": "momentum",
                             "energy_create": "energy"}.get(pl["twin"])
    k_post = k_pre = k
    if dt is not None and per_frame:
        k_post = min(max(int(round((te + dt) * fps)), k), len(per_frame) - 1)
        post = frame(k_post)
    if dt_pre is not None and per_frame:
        k_pre = min(max(int(round((te + dt_pre) * fps)), k), len(per_frame) - 1)
        pre = frame(k_pre)
    g_mag = float(c.get("g", 0.0))
    ctx = dict(c)
    r = np.array([c["sizes"][n] for n in names], dtype=np.float64)
    ctx.update(g=np.array([0.0, 0.0, -g_mag]), g_mag=g_mag, dt=float(post["t_s"] - pre["t_s"]),
               m=np.array([c["masses"][n] for n in names], dtype=np.float64), r=r,
               e_range=(0.0, 1.0), a_ref=g_mag, event_time_s=te, family=fam, twin=pl["twin"],
               subfamily=pl.get("subfamily", ""))
    kind = pl.get("event_kind")
    if kind not in ("collision", "bounce", "pad", "swing"):
        if fam in COLLISION_FAMILIES:
            kind = "collision"
        elif fam == "f7" and ctx["subfamily"] == "bounce":
            kind = "bounce"
        elif fam == "f8":
            kind = "pad"
        elif fam == "f9" and ctx["subfamily"] == "swing":
            kind = "swing"
    if kind is not None:
        ctx["event"] = kind
    inertia = {n: pl["objects"][n].get("inertia") for n in names}
    if all(v is not None for v in inertia.values()):
        ctx["k"] = np.array([inertia[n] / (c["masses"][n] * c["sizes"][n] ** 2) for n in names])
    if fam in COLLISION_FAMILIES and len(names) == 2:
        pc = contact_positions(pre["p"], pre["v"], r)
        d = pc[1] - pc[0]
        if 0.5 < np.linalg.norm(pre["p"][1] - pre["p"][0]) / r.sum() < 1.5:     # labelled centres near contact
            ctx["n"] = d / np.linalg.norm(d)
            pre["p"], post["p"] = pc, pc.copy()
    if "n" not in ctx and "contact_normal" in c:
        n = np.asarray(c["contact_normal"], dtype=np.float64)
        ctx["n"] = n / np.linalg.norm(n)
    if "contact_normal" in c:
        ctx["e_ref"] = c.get("e")
    if "ref_event_state" in pl:              # f4: the unboosted twin's (pre, post) for the Galilean law
        ctx["ref"] = (as_state(pl["ref_event_state"]["pre"]), as_state(pl["ref_event_state"]["post"]))
    if fam in ACC_FAMILIES and per_frame:
        w = max(int(round(ACC_WINDOW_S * fps)), 3)
        a_pre = (accel(range(k_pre, k_pre + w), pre["t_s"]) if dt_pre is not None
                 else accel(range(k - w, k), te))
        a_post = accel(range(k_post, k_post + w), post["t_s"])
        if a_pre is not None and a_post is not None:
            pre.setdefault("a", a_pre)
            post.setdefault("a", a_post)
    for src in ("theta_rad", "theta1_rad"):
        if src in c:
            ctx.setdefault("theta", c[src])
    if fam == "f11" and per_frame:
        ctx["g_clip"] = _f11_g_clip(labels, names, te)
        if "alpha" not in post:
            post["alpha"] = _rate(per_frame, names, range(k_post, k_post + max(int(round(ACC_WINDOW_S * fps)), 3)),
                                  "angular_velocity", fps)
        clean = float(pl.get("clean_until_s", len(per_frame) / fps))
        t_sep = (pl.get("edit") or {}).get("t_separate_s")
        if t_sep is not None:
            ctx["t_separate_s"] = float(t_sep)
        t_end = min(te + tip_window(ctx["g_clip"], c["half_length"], None if t_sep is None else t_sep - te,
                                    clean - te), clean)
        post["phi"] = rotation_angle(per_frame, names, k - 1, min(int(np.floor(t_end * fps)), len(per_frame) - 1))
    if fam == "f6" and ctx["subfamily"] == "glide":
        ctx["force_free"] = True
    fw = friction_windows(labels) if dt is None and dt_pre is None and per_frame else None
    if fw is not None:                       # f6 slide / pair: decel while moving after the branch; the stop window
        tt = np.arange(len(per_frame)) / fps
        mv = [f for f in range(k, len(per_frame)) if tt[f] <= fw["move_end"] + 1e-9]
        a_mv = accel(mv, tt[mv].mean()) if len(mv) >= 3 else None
        if a_mv is not None:
            post["a"] = a_mv
            ctx["dt"] = float(tt[mv].mean() - te)
        post["v_stop"] = _rate(per_frame, names, [f for f in range(len(per_frame))
                                                  if fw["stop_start"] - 1e-9 <= tt[f] <= fw["end"] + 1e-9], "position", fps)
        ctx.update({key: fw[key] for key in ("move_end", "stop_start")})
    if fam == "f9" and per_frame and "pivots" in c:
        pre["T"], post["T"] = pendulum_periods(labels, names)
    if "contact_point" in c:
        ctx["c"] = np.asarray(c["contact_point"], dtype=np.float64)
    for key in ("legal", "violated_law", "violation_magnitude", "epsilon", "clean_until_s", "stop_dose", "boost_kind"):
        if key in pl:
            ctx[key] = pl[key]
    ctx["frame"] = pl.get("frame", "world")       # static camera: the world is the camera's rest frame
    return pre, post, ctx
