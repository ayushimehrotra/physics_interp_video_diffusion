#!/usr/bin/env python3
"""Render one twin group (spec §2): one physical state, every twin rendered from it.

Runs in the Kubric venv (bpy + pybullet, no torch). Normally driven by
``scripts/01_generate_data.py``, which runs many groups in parallel; can also be
run directly for one group:

    KUBRIC_USE_GPU=1 /root/envs/kubric_venv/bin/python physlaw/data/generate.py \\
        --config configs/data/f1_collisions.yaml --seed 100000 --out data/f1

Two scene sources, one renderer:

* **Collision families (f1-f4)**: the vendored two-sphere scene (``kubric_scene``).
  Pass 1 (physics only) samples, simulates at e = 1, validates, builds every twin
  analytically (``twins.py``) and rejects draws that leave the frame or violate too
  little. Pass 2 rebuilds the accepted scene with the renderer attached. Twins are the
  simulated trajectory re-keyframed at the exact contact time (F4: Galilean-boosted
  as a whole, F3: per-twin masses).
* **Closed-form families (f6-f10)**: ``physlaw/data/families/``. A family samples a
  ``Scene`` carrying every twin's full trajectory; the same initial condition is run
  in PyBullet as a check (``physics_check``); the Kubric scene is built here.

Rendering (:func:`render_twins`): the first twin of each ``pre_segment`` is rendered in
full; every later twin of that segment renders only frames >= first_post and copies
the earlier frames, so twins are byte-identical before the event BY CONSTRUCTION (the
tests check it on the decoded mp4 too). Each twin writes video.mp4, masks.npz,
labels.json, _COMPLETE. Selected seeds (``nuisance_frac``) also render a nuisance
variant of the whole group into ``<group>_nuisance/`` (``nuisance.py``).
``legal_only`` (PC1 training set) keeps only the legal twins.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from physlaw.data import kubric_scene as KS  # noqa: E402
from physlaw.data import twins as TW  # noqa: E402
from physlaw.data import fast_post as FP  # noqa: E402
from physlaw.data import families as FAM  # noqa: E402
from physlaw.data import nuisance as NZ  # noqa: E402
from physlaw.data.splits import split_for_group  # noqa: E402
from physlaw.config import load_config, config_hash  # noqa: E402

logger = logging.getLogger("physlaw.generate")
SCHEMA_VERSION = "physlaw-labels-2"
COLLISION_FAMILIES = ("f1", "f2", "f3", "f4")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def group_id(family: str, seed: int) -> str:
    return f"{family}_{seed:06d}"


def contact_frame_for(seed: int, lo: int, hi: int) -> int:
    return int(np.random.default_rng([seed, 4242]).integers(lo, hi + 1))


def subfamily_for(cfg, seed: int) -> str:
    """Sub-family of a seed: consecutive blocks of ``cfg.subfamilies`` sizes, cycled past their sum."""
    subs = cfg.get("subfamilies") or {}
    if not subs:
        return ""
    j = (seed - cfg.seed_offset) % sum(subs.values())
    for name, n in subs.items():
        if j < n:
            return name
        j -= n
    raise AssertionError


def master_fmt(cfg) -> dict:
    m = cfg.master
    return {"fps": int(m.fps), "num_frames": int(m.num_frames), "resolution": [int(m.width), int(m.height)],
            "sensor_width_mm": float(m.get("sensor_width_mm", 36.0 * 832 / 720)),
            "aspect": float(m.get("aspect", m.width / m.height)),
            "event_frame_range": list(m.contact_frame_range), "samples_per_pixel": int(m.samples_per_pixel)}


def inframe_until(animation, spheres, p, start: int) -> dict[str, int]:
    """First frame >= start at which a sphere's silhouette leaves the image (NUM_FRAMES if never)."""
    half_w, half_h = KS.half_extents()
    out = {}
    for tag in ("a", "b"):
        pos = np.asarray(animation[spheres[tag]]["position"], dtype=np.float64)
        r = p[f"radius_{tag}"] * KS.SILHOUETTE_SAFETY + KS.FRAME_MARGIN
        inside = (np.abs(pos[:, 0]) + r <= half_w) & (np.abs(pos[:, 1]) + r <= half_h)
        bad = np.where(~inside[start:])[0]
        out[tag.upper()] = int(start + bad[0]) if bad.size else KS.NUM_FRAMES
    return out


LOCK_DIR = Path(os.environ.get("PHYSLAW_LOCK_DIR", "/tmp/physlaw_locks"))


def acquire_group_lock(gid: str) -> Path | None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / gid
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return path
        except FileExistsError:
            try:
                owner = int(path.read_text() or 0)
                os.kill(owner, 0)            # raises if the owner is gone
                return None                  # live owner: someone else is rendering it
            except (ProcessLookupError, ValueError):
                path.unlink(missing_ok=True)  # stale lock: take it over
    return None


def release_simulator(sim) -> None:
    """Disconnect a Kubric PyBullet view AND mark its client dead.

    Kubric's _BulletClient.__del__ disconnects whatever id it stored. pybullet reuses
    client ids, so a plain ``pb.disconnect(id)`` leaves a stale object whose eventual
    __del__ disconnects the NEXT scene's live client (seen as "Not connected to physics
    server" mid-group). Resetting the stored id to -1 makes that __del__ a no-op.
    """
    import pybullet as pb
    client = sim._physics_client
    if client.client >= 0:
        try:
            pb.disconnect(physicsClientId=client.client)
        finally:
            client._client = -1


def _vec3(v2) -> np.ndarray:
    return np.array([float(v2[0]), float(v2[1]), 0.0])


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_mp4_causal(path: Path, rgb: np.ndarray, fps: int, crf: int) -> None:
    """h264 with a strictly causal encoder: no B-frames, no lookahead, no mb-tree, 1 thread.

    Frame k's bitstream then depends only on frames <= k, so twins that share their
    first N raw frames also share their first N decoded frames exactly.
    """
    T, H, W, _ = rgb.shape
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-bf", "0", "-threads", "1",
           "-x264-params", "rc-lookahead=0:mbtree=0:scenecut=0:sync-lookahead=0:threads=1",
           "-movflags", "+faststart", str(path)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(),
                          capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-2000:]}")


def write_twin(tdir: Path, rgb, seg, labels: dict, name: str, crf: int, fps: int, object_ids) -> None:
    if rgb is not None:
        write_mp4_causal(tdir / "video.mp4", rgb, fps, crf)
        np.savez_compressed(tdir / "masks.npz", instance=seg, object_ids=np.array(object_ids, dtype=np.uint8))
    (tdir / "labels.json").write_text(json.dumps(labels, indent=1, default=_json_default))
    (tdir / "_COMPLETE").write_text(name + "\n")


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


# --------------------------------------------------------------------------- #
# collision families (f1-f4): physics in Bullet, twins from twins.py
# --------------------------------------------------------------------------- #


def build_twins(family: str, cfg, p, geo, masses, animation, spheres, seed: int, attempt: int):
    pre = KS._pre_state(p, geo, masses, animation, spheres)
    post = KS._post_velocities(animation, spheres)
    e_sim = KS._measured_restitution(animation, spheres, geo)
    t = cfg.twins
    if family == "f1":
        rng = np.random.default_rng([seed, attempt, 7919])
        s_honest = pre.m_a / (pre.m_a + pre.m_b)
        sign = TW.ke_reducing_sign(pre) if t.get("split_sign", "ke_reducing") == "ke_reducing" else None
        delta = TW.choose_split_delta(rng, s_honest, tuple(t.split_delta_range), t.split_margin, sign=sign)
        if delta is None:
            return None, pre, e_sim, "split_delta_does_not_fit"
        tw = TW.f1_twins(pre, post["a"], post["b"], e_sim, delta, e_low=t.e_low, e_create=t.e_create)
        for name in t.get("require_energy_physical", []):
            if not tw[name].energy_physical:
                return None, pre, e_sim, f"{name}_creates_energy"
    elif family == "f2":
        tw = TW.f2_twins(pre, post["a"], post["b"], e_sim, tuple(t.levels))
    elif family == "f3":
        tw = TW.f3_twins(pre, float(t.e), tuple(t.ratios))
    elif family == "f4":
        rng = np.random.default_rng([seed, attempt, 4441])
        psi = float(rng.uniform(0, 2 * np.pi))
        d = np.array([np.cos(psi), np.sin(psi)])
        boosts = {"u0": np.zeros(2), "usmall": d * rng.uniform(*t.u_small), "ularge": d * rng.uniform(*t.u_large)}
        tw = TW.f4_twins(pre, post["a"], post["b"], e_sim, boosts, e_low=t.e_low)
    else:
        raise ValueError(family)
    return tw, pre, e_sim, None


def _law_of(family: str, name: str, twin, pre) -> tuple[bool, str | None, float]:
    """(legal, violated_law, violation_magnitude) of a collision twin."""
    if family == "f1" and name.startswith("mom_violate"):
        return False, "momentum", TW.momentum_residual(pre, twin.v_a, twin.v_b)
    if family == "f1" and name == "energy_create":
        return False, "energy", TW.kinetic_energy(pre, twin.v_a, twin.v_b) / TW.ke_pre(pre)
    return True, None, 0.0


def collision_scene(cfg, p, geo, masses, animation, spheres, contact: float, tw: dict, pre, fmt) -> FAM.Scene:
    """Express the simulated group as a families.Scene: twin tracks re-keyframed from Bullet's run."""
    family = cfg.family
    T, fps = KS.NUM_FRAMES, KS.FRAME_RATE
    cam = FAM.top_camera(fmt, height=KS.CAMERA_DISTANCE, z_plane=KS.CENTER_HEIGHT)
    bodies = [FAM.Body(T_, "sphere", float(p[f"radius_{t_}"]), float(p[f"density_{t_}"]),
                       {k: (tuple(v) if k == "color" else v) for k, v in p[f"appearance_{t_}"].items()}, KS.SEG_ID[T_])
              for T_, t_ in (("A", "a"), ("B", "b"))]
    times = np.arange(T) / fps
    twins = {}
    for name, twin in tw.items():
        anim = {spheres[x]: {k: [np.asarray(v, float).copy() for v in animation[spheres[x]][k]]
                             for k in ("position", "velocity", "quaternion", "angular_velocity")} for x in "ab"}
        if twin.boost is not None:
            u = _vec3(twin.boost)
            for x in "ab":
                a = anim[spheres[x]]
                a["position"] = [pp + u * (tt - contact / fps) for pp, tt in zip(a["position"], times)]
                a["velocity"] = [vv + u for vv in a["velocity"]]
        KS.rekeyframe_positions_only(spheres, anim, contact, {"a": _vec3(twin.v_a), "b": _vec3(twin.v_b)})
        tracks = {T_: FAM.Track(*(np.array(anim[spheres[t_]][k]) for k in ("position", "velocity", "quaternion",
                                                                           "angular_velocity")))
                  for T_, t_ in (("A", "a"), ("B", "b"))}
        legal, law, mag = _law_of(family, name, twin, twin.pre(pre))
        twins[name] = FAM.Twin(name, tracks, pre_segment=twin.pre_segment, legal=legal, violated_law=law,
                               violation_magnitude=float(mag),
                               flags={"energy_physical": bool(twin.energy_physical),
                                      "momentum_conserved": bool(twin.momentum_conserved)},
                               constants={"e": float(twin.e_target),
                                          "boost_u": (_vec3(twin.boost) if twin.boost is not None else np.zeros(3)).tolist()},
                               record=twin.record,
                               masses=({"A": twin.masses[0], "B": twin.masses[1]} if twin.masses else None))
    sc = FAM.Scene(family, "", bodies, [], cam, 0.0, contact / fps, "contact", twins, fps,
                   constants={"g": 0.0, "contact_normal": _vec3(pre.n).tolist(), "e_sim": None},
                   inframe={"start": 0 if family == "f4" else int(math.ceil(contact)),
                            "stop": int(math.ceil(contact)) + int(round(cfg.acceptance.min_post_inframe_s * fps))},
                   first_post_frame=int(math.ceil(contact)))
    return sc


def feasibility(cfg, tw, pre, p, animation, spheres, contact_frame: float) -> str | None:
    """F1-F3: every twin keeps both spheres in frame long enough; violations are large enough."""
    first_post = int(math.ceil(contact_frame))
    need = first_post + int(round(cfg.acceptance.min_post_inframe_s * KS.FRAME_RATE))
    anim = {k: copy.deepcopy(v) for k, v in animation.items()}   # keys are Kubric objects
    for name, twin in tw.items():
        KS.rekeyframe_positions_only(spheres, anim, contact_frame, {"a": _vec3(twin.v_a), "b": _vec3(twin.v_b)})
        until = inframe_until(anim, spheres, p, first_post)
        if min(until.values()) < need:
            return f"{name}_leaves_frame_at_{min(until.values())}_needs_{need}"
        if not twin.momentum_conserved:
            r = TW.momentum_residual(pre, twin.v_a, twin.v_b)
            if r < cfg.acceptance.min_violation_residual:
                return f"{name}_violation_residual_{r:.3g}_below_threshold"
    return None


def sample_collision(cfg, seed: int, scratch: str, fmt: dict):
    """Pass 1: draw scenes until one survives physics and twin checks. Returns (attempt, p, rejections)."""
    rejections = []
    for attempt in range(cfg.acceptance.max_attempts):
        rng = np.random.default_rng([seed, attempt])
        p, reason = KS.sample_params(rng, violate=False, arm=cfg.arm, restitution=cfg.twins.e_sim,
                                     material_pair=cfg.material_pair)
        if p is None:
            rejections.append({"attempt": attempt, "reason": reason}); continue
        if cfg.family == "f3":
            # identical look (latent_mass arm) and the simulated pair at ratio 1: mass only via density
            p["density_a"] = p["density_b"] = float(np.sqrt(p["density_a"] * p["density_b"]))
        geo, masses = KS.solve_geometry(p), KS.derive_masses(p)
        reason = KS.check_frame0(p, geo)
        if reason:
            rejections.append({"attempt": attempt, "reason": reason}); continue
        (_, spheres, floor, _, sim, _, animation, collisions) = KS.build_and_simulate(p, geo, masses, scratch, render=False)
        pair_ev, floor_ev = KS.analyse_collisions(collisions, spheres, floor)
        reason, meas = KS.check_simulation(p, geo, masses, animation, spheres, pair_ev, floor_ev)
        if not reason:
            tw, pre, e_sim, reason = build_twins(cfg.family, cfg, p, geo, masses, animation, spheres, seed, attempt)
            if not reason and cfg.family == "f4":    # boosts move the whole clip: check every frame
                sc = collision_scene(cfg, p, geo, masses, animation, spheres, meas["contact_frame"], tw, pre, fmt)
                reason = sc.inframe_reason(margin_px=KS.FRAME_MARGIN * sc.camera.px_per_m(KS.CAMERA_DISTANCE))
            elif not reason:
                reason = feasibility(cfg, tw, pre, p, animation, spheres, meas["contact_frame"])
        release_simulator(sim)
        if reason:
            rejections.append({"attempt": attempt, "reason": reason}); continue
        return attempt, p, rejections
    raise RuntimeError(f"no valid scene after {cfg.acceptance.max_attempts} attempts: {rejections[-5:]}")


# --------------------------------------------------------------------------- #
# closed-form families (f6-f10)
# --------------------------------------------------------------------------- #


def physics_check(sc: FAM.Scene, step_rate: int = 960) -> dict:
    """Run the check twin's initial condition in PyBullet; max position gap to the closed form (m).

    960 Hz rather than Kubric's 240: semi-implicit Euler lags a ballistic path by g dt t / 2
    (2 cm per second of flight at 240 Hz), which would swamp real geometry errors."""
    import pybullet as pb
    si = sc.sim_init
    if not si:
        return {"skipped": True}
    tw = sc.twins[si["twin"]]
    fps = sc.fps
    f0 = int(math.ceil(si["t0"] * fps - 1e-9))
    f1 = int(math.floor(si["until"] * fps))
    cid = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, -sc.gravity, physicsClientId=cid)
        pb.setPhysicsEngineParameter(fixedTimeStep=1.0 / step_rate, numSubSteps=1,
                                     splitImpulsePenetrationThreshold=0.0, physicsClientId=cid)
        mu = float(tw.constants.get("mu", 0.0))
        statics = [s for s in sc.statics if tw.statics is None or s.name in tw.statics]
        # the rendered band sits 0.5 mm proud of the floor (no z-fighting); flush for the physics
        boxes = [(s.half, (s.position[0], s.position[1], -s.half[2]) if s.name == "band" else s.position,
                  s.quaternion, mu if s.name in ("band", "slab") else s.friction) for s in statics]
        # Coplanar floor + band split the contacts and halve the friction, and the check starts
        # with the body wholly on the band: the band alone is the support there.
        if sc.look.get("floor", True) and not any(s.name == "band" for s in statics):
            boxes.append(((40.0, 40.0, 0.5), (0.0, 0.0, -0.5), (1, 0, 0, 0), 0.0))
        e_floor = float(tw.constants.get("e_floor", 0.0))
        for half, pos, q, fr in boxes:
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=cid)
            bid = pb.createMultiBody(0, col, -1, pos, (q[1], q[2], q[3], q[0]), physicsClientId=cid)
            pb.changeDynamics(bid, -1, lateralFriction=fr, restitution=1.0, physicsClientId=cid)
        ids = {}
        m = sc.masses(tw)
        for b in sc.bodies:
            tr = tw.tracks[b.name]
            if b.shape == "sphere":
                col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=b.size, physicsClientId=cid)
            elif b.shape == "cube":
                col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[b.size] * 3, physicsClientId=cid)
            else:
                col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=b.size, height=2 * b.length, physicsClientId=cid)
            q = tr.q[f0]
            bid = pb.createMultiBody(m[b.name], col, -1, tr.p[f0].tolist(), (q[1], q[2], q[3], q[0]), physicsClientId=cid)
            pb.changeDynamics(bid, -1, linearDamping=0, angularDamping=0, lateralFriction=1.0,
                              restitution=e_floor, rollingFriction=0, spinningFriction=0, physicsClientId=cid)
            pb.resetBaseVelocity(bid, tr.v[f0].tolist(), tr.w[f0].tolist(), physicsClientId=cid)
            ids[b.name] = bid
        n_sub = step_rate // fps
        err = 0.0
        for f in range(f0, f1 + 1):
            for b in sc.bodies:
                pos, _ = pb.getBasePositionAndOrientation(ids[b.name], physicsClientId=cid)
                err = max(err, float(np.linalg.norm(np.asarray(pos) - tw.tracks[b.name].p[f])))
            for _ in range(n_sub):
                pb.stepSimulation(physicsClientId=cid)
        return {"twin": tw.name, "from_frame": f0, "to_frame": f1, "max_err_m": err, "step_rate": step_rate}
    finally:
        pb.disconnect(physicsClientId=cid)


def sample_family(cfg, seed: int, fmt: dict):
    """Draw closed-form scenes until one passes the frame and physics checks. Returns (Scene, attempt, rejections)."""
    mod = FAM.family_module(cfg.family)
    sub = subfamily_for(cfg, seed)
    rejections = []
    for attempt in range(cfg.acceptance.max_attempts):
        rng = np.random.default_rng([seed, attempt])
        sc = mod.sample(rng, cfg, fmt, sub)
        if isinstance(sc, str):
            rejections.append({"attempt": attempt, "reason": sc}); continue
        sc.check_pre_identity()
        reason = sc.inframe_reason()
        if not reason and cfg.acceptance.get("physics_check_tol"):
            sc.physics = physics_check(sc)
            if sc.physics.get("max_err_m", 0.0) > cfg.acceptance.physics_check_tol:
                reason = f"physics_check_err_{sc.physics['max_err_m']:.3g}"
        if reason:
            rejections.append({"attempt": attempt, "reason": reason}); continue
        return sc, attempt, rejections
    raise RuntimeError(f"no valid scene after {cfg.acceptance.max_attempts} attempts: {rejections[-5:]}")


def _material(look: dict):
    import kubric as kb
    return kb.PrincipledBSDFMaterial(color=kb.Color(*look["color"], 1.0), metallic=look["metallic"],
                                     roughness=look["roughness"], specular=look["specular"], ior=look["ior"])


def build_family_kubric(sc: FAM.Scene, scratch: str, spp: int, denoise: bool = True):
    """Kubric scene + Blender renderer for a closed-form Scene (no simulator: everything is keyframed)."""
    import kubric as kb
    from kubric.renderer import Blender
    W, H = sc.camera.resolution
    scene = kb.Scene(frame_start=0, frame_end=sc.twins[next(iter(sc.twins))].tracks[sc.bodies[0].name].p.shape[0] - 1,
                     frame_rate=sc.fps, step_rate=sc.fps * 5, resolution=(W, H), gravity=(0.0, 0.0, -sc.gravity),
                     background=kb.Color(0.30, 0.30, 0.32, 1.0), ambient_illumination=kb.Color(0.04, 0.04, 0.04, 1.0))
    renderer = Blender(scene, scratch, use_denoising=denoise, adaptive_sampling=True, samples_per_pixel=spp)
    renderer.blender_scene.view_settings.view_transform = "Standard"
    camera = kb.PerspectiveCamera(name="camera", focal_length=sc.camera.focal_mm, sensor_width=sc.camera.sensor_w_mm,
                                  position=sc.camera.position, quaternion=sc.camera.blender_quaternion())
    scene += camera
    scene.camera = camera
    top = abs(sc.camera.rotation()[2][2]) > 0.9
    scene += kb.DirectionalLight(name="sun", position=(5.0, -4.0, 14.0) if top else (-6.0, -12.0, 14.0),
                                 look_at=(0.0, 0.0, 0.0), intensity=3.0)
    floor = None
    if sc.look.get("floor", True):
        floor = kb.Cube(name="floor", scale=(40.0, 40.0, 0.5), position=(0.0, 0.0, -0.5),
                        material=_material(FAM.LOOKS["ramp"] | {"color": (0.42, 0.42, 0.44), "roughness": 0.8}),
                        static=True, background=True, segmentation_id=0)
        scene += floor
    statics = {}
    for st in sc.statics:
        o = kb.Cube(name=st.name, scale=st.half, position=st.position, quaternion=st.quaternion,
                    material=_material(st.look), static=True, background=True, segmentation_id=0)
        scene += o
        statics[st.name] = o
    objs = {}
    for b in sc.bodies:
        kw = dict(name=f"body_{b.name}", material=_material(b.look), mass=b.mass, segmentation_id=b.seg_id)
        if b.shape == "sphere":
            o = kb.Sphere(scale=b.size, **kw)
        elif b.shape == "cube":
            o = kb.Cube(scale=(b.size,) * 3, **kw)
        else:
            raise NotImplementedError(f"shape {b.shape}")
        scene += o
        objs[b.name] = o
    return {"scene": scene, "renderer": renderer, "camera": camera, "objs": objs, "statics": statics, "floor": floor}


# --------------------------------------------------------------------------- #
# rendering + labels, shared by both sources
# --------------------------------------------------------------------------- #


def keyframe_twin(sc: FAM.Scene, tw: FAM.Twin, kb_objs: dict, kb_statics: dict) -> None:
    for name, o in kb_objs.items():
        tr = tw.tracks[name]
        for f in range(tr.p.shape[0]):
            o.position = tuple(tr.p[f])
            o.quaternion = tuple(tr.q[f])
            o.keyframe_insert("position", f)
            o.keyframe_insert("quaternion", f)
    for st in sc.statics:
        shown = tw.statics is None or st.name in tw.statics
        o = kb_statics[st.name]
        o.position = st.position if shown else (st.position[0], st.position[1], st.position[2] - 200.0)
        o.keyframe_insert("position", 0)


def apply_look(h: dict, sc: FAM.Scene, look: dict) -> FAM.Camera:
    """Set a nuisance look on the live Kubric scene; returns the camera the labels must use."""
    import kubric as kb
    cam = NZ.perturb_camera(sc.camera, look)
    h["camera"].position = cam.position
    h["camera"].quaternion = cam.blender_quaternion()
    sun = next(a for a in h["scene"].assets if a.name == "sun")
    sun.position = NZ.sun_position(look)
    sun.look_at((0.0, 0.0, 0.0))
    sun.intensity = look["sun"]["intensity"]
    a = look["ambient"]
    h["scene"].ambient_illumination = kb.Color(a, a, a, 1.0)
    if h.get("floor") is not None:
        floor_texture(h["floor"], h["renderer"], look["floor"])
    for name, o in h["objs"].items():
        o.material.color = kb.Color(*look["bodies"][name], 1.0)
    return cam


def floor_texture(floor, renderer, fl: dict) -> None:
    floor.material.color = __import__("kubric").Color(*fl["color"], 1.0)
    if not fl["checker"]:
        return
    mat = floor.material.linked_objects[renderer]
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexChecker")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    # object coords span [-1, 1] over the 80 m floor: scale = squares per 2 units
    tex.inputs["Scale"].default_value = 80.0 / fl["checker_scale"]
    c = np.asarray(fl["color"])
    tex.inputs["Color1"].default_value = (*c, 1.0)
    tex.inputs["Color2"].default_value = (*(c * (1 - fl["checker_contrast"])), 1.0)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])


def state_at(sc: FAM.Scene, tw: FAM.Twin, f: int) -> dict:
    names = [b.name for b in sc.bodies]
    return {"frame": f, "t_s": f / sc.fps, "objects": names,
            "p": [tw.tracks[n].p[f].tolist() for n in names], "v": [tw.tracks[n].v[f].tolist() for n in names],
            "w": [tw.tracks[n].w[f].tolist() for n in names]}


def law_fields(sc: FAM.Scene, tw: FAM.Twin, cam: FAM.Camera) -> dict:
    """The labels-2 fields every family carries (documented in schema/labels.schema.json)."""
    k = sc.first_post
    m = sc.masses(tw)
    consts = {**sc.constants, **tw.constants, "g": float(sc.gravity), "fps": sc.fps,
              "masses": {b.name: float(m[b.name]) for b in sc.bodies},
              "sizes": {b.name: float(b.size) for b in sc.bodies}}
    return {
        "subfamily": sc.subfamily, "event_kind": sc.event_kind, "event_time_s": float(sc.t_event),
        "event_frame": sc.event_frame, "first_post_contact_frame": k, "pre_segment": tw.pre_segment,
        "legal": bool(tw.legal), "violated_law": tw.violated_law, "violation_magnitude": float(tw.violation_magnitude),
        "flags": tw.flags, "energy_physical": bool(tw.flags.get("energy_physical", True)),
        "constants": consts, "pre_state": state_at(sc, tw, k - 1), "post_state": state_at(sc, tw, k),
        "pre_velocity": {n: tw.tracks[n].v[k - 1].tolist() for n in tw.tracks},
        "post_velocity": {n: tw.tracks[n].v[k].tolist() for n in tw.tracks},
        "camera": cam.block(),
        "objects": {b.name: {"shape": b.shape, "size": b.size, "radius": b.size, "density": float(m[b.name] / b.volume),
                             "volume": b.volume, "mass": float(m[b.name]), "inertia": b.inertia / b.mass * m[b.name],
                             "segmentation_id": b.seg_id, "color_rgb": list(b.look["color"])} for b in sc.bodies},
    }


def per_frame_labels(sc: FAM.Scene, tw: FAM.Twin, cam: FAM.Camera, seg_stats) -> list:
    m = sc.masses(tw)
    W, H = cam.resolution
    out = []
    T = tw.tracks[sc.bodies[0].name].p.shape[0]
    for f in range(T):
        e = {"frame": f, "time_s": f / sc.fps}
        for b in sc.bodies:
            tr = tw.tracks[b.name]
            u, v, z = cam.project(tr.p[f])
            vel, w = tr.v[f], tr.w[f]
            I = b.inertia / b.mass * m[b.name]
            o = {"position": tr.p[f].tolist(), "velocity": vel.tolist(), "angular_velocity": w.tolist(),
                 "quaternion": tr.q[f].tolist(), "pixel": [float(u), float(v)],
                 "in_frame": bool(0 <= u < W and 0 <= v < H and z > 0), "speed": float(np.linalg.norm(vel)),
                 "momentum": (m[b.name] * vel).tolist(),
                 "kinetic_energy": float(0.5 * m[b.name] * vel @ vel + 0.5 * I * w @ w),
                 "potential_energy": float(m[b.name] * sc.gravity * tr.p[f][2])}
            if seg_stats is not None:
                o["segmentation_pixel_count"] = seg_stats[b.name]["count"][f]
                o["segmentation_centroid_pixel"] = seg_stats[b.name]["centroid"][f]
            e[b.name] = o
        out.append(e)
    return out


def seg_stats_for(seg, bodies) -> dict:
    out = {}
    for b in bodies:
        counts, cents = [], []
        for f in range(seg.shape[0]):
            ys, xs = np.nonzero(seg[f] == b.seg_id)
            counts.append(int(xs.size))
            cents.append([float(xs.mean()), float(ys.mean())] if xs.size else None)
        out[b.name] = {"count": counts, "centroid": cents}
    return out


def render_twins(cfg, sc: FAM.Scene, h: dict, gdir: Path, cam: FAM.Camera, render: bool, scratch: str, pool,
                 common: dict, legacy=None, nuisance: int = 0) -> dict:
    """Render (or only label, when not ``render``) every twin of ``sc`` into ``gdir``."""
    from concurrent.futures import ThreadPoolExecutor
    k = sc.first_post
    T = tw0_T = next(iter(sc.twins.values())).tracks[sc.bodies[0].name].p.shape[0]
    pre_ref, summaries = {}, {}
    writer, writes = ThreadPoolExecutor(2), []
    for i, (name, tw) in enumerate(sc.twins.items()):
        if render:
            keyframe_twin(sc, tw, h["objs"], h["statics"])
        tdir = gdir / name
        tdir.mkdir(parents=True, exist_ok=True)
        rt0 = time.time()
        rgb = seg = frame_sha = stats = None
        if render:
            full = tw.pre_segment not in pre_ref
            frames = list(range(T)) if full else list(range(k, T))
            rgb, seg = FP.render_frames(h["renderer"], h["scene"], frames, Path(scratch), pool)
            if rgb.shape[0] != len(frames):
                raise RuntimeError(f"renderer returned {rgb.shape[0]} frames, asked for {len(frames)}")
            if full:
                pre_ref[tw.pre_segment] = (rgb[:k].copy(), seg[:k].copy())
            else:
                rgb = np.concatenate([pre_ref[tw.pre_segment][0], rgb])
                seg = np.concatenate([pre_ref[tw.pre_segment][1], seg])
            assert rgb.shape[0] == tw0_T
            stats = seg_stats_for(seg, sc.bodies)
            frame_sha = [hashlib.sha256(f.tobytes()).hexdigest()[:16] for f in rgb]
        render_s = time.time() - rt0
        fields = law_fields(sc, tw, cam)
        labels = legacy(name, tw, stats) if legacy else {
            "schema_version": 2, "scene_id": f"{common['twin_group_id']}/{name}", "seed": common["seed"],
            "sampled_parameters": sc.params, "objects": fields["objects"],
            "scene_constants": {"num_frames": T, "frame_rate": sc.fps, "resolution": list(cam.resolution),
                                "gravity": [0.0, 0.0, -sc.gravity], "samples_per_pixel": common["master_format"]["samples_per_pixel"],
                                "render_device": common["render_device"], "camera_position": list(cam.position)}}
        labels["per_frame"] = per_frame_labels(sc, tw, cam, stats)
        labels["sampling_attempts"] = common["sampling_attempts"]
        pl = labels.setdefault("physlaw", {})
        if legacy:   # F1 layout: planar 2-vectors of the twin (twins.py); identical numbers
            fields.pop("pre_velocity"), fields.pop("post_velocity")
        pl.update({"schema_version": SCHEMA_VERSION, "family": sc.family, "twin": name, "twin_index": i,
                   "twin_group_id": common["twin_group_id"], "base_group_id": common["base_group_id"],
                   "split": common["split"], "seed": common["seed"], "nuisance": nuisance,
                   "nuisance_look": common.get("look"), **fields,
                   "edit": tw.record, "frame_sha256_16": frame_sha, "master_format": common["master_format"],
                   "render_seconds": render_s, "render_device": common["render_device"],
                   "physics_check": common.get("physics_check"), "config_hash": common["config_hash"]})
        summaries[name] = {"legal": tw.legal, "violated_law": tw.violated_law,
                           "violation_magnitude": tw.violation_magnitude, **tw.flags}
        writes.append(writer.submit(write_twin, tdir, rgb, seg, labels, name, cfg.master.crf, sc.fps,
                                    [b.seg_id for b in sc.bodies]))
        logger.info("%s/%s rendered (%.1fs)", gdir.name, name, render_s)
    for fu in writes:
        fu.result()
    return summaries


def legal_only(sc: FAM.Scene) -> None:
    sc.twins = {n: t for n, t in sc.twins.items() if t.legal}


# --------------------------------------------------------------------------- #
# one group
# --------------------------------------------------------------------------- #


def run_group(cfg, seed: int, out_root: Path, render: bool = True, force: bool = False) -> dict:
    import kubric as kb

    family = cfg.family
    gid = group_id(family, seed)
    gdir = out_root / gid
    if (gdir / "_GROUP_COMPLETE").exists() and not force:
        return {"group": gid, "status": "skipped_exists"}
    # Claim the group with an atomic lock on LOCAL disk (/tmp), so several drivers can share
    # one output directory without one deleting a group another is rendering. A lock whose
    # owner pid is gone is stale and is taken over.
    lock = acquire_group_lock(gid)
    if lock is None:
        return {"group": gid, "status": "skipped_locked"}
    ndir = out_root / f"{gid}_nuisance"
    for d in (gdir, ndir):
        if d.exists():
            shutil.rmtree(d)
    gdir.mkdir(parents=True)

    m = cfg.master
    fmt = master_fmt(cfg)
    t0 = time.time()
    scratch = tempfile.mkdtemp(prefix="physlaw_kubric_")
    pool, sim2 = None, None
    try:
        if family in COLLISION_FAMILIES:
            contact_target = contact_frame_for(seed, *m.contact_frame_range)
            mf = KS.configure(num_frames=m.num_frames, frame_rate=m.fps, resolution=(m.width, m.height),
                              contact_frame=contact_target, samples_per_pixel=m.samples_per_pixel,
                              contact_tolerance_frames=m.contact_tolerance_frames,
                              sensor_width_mm=m.get("sensor_width_mm"), camera_distance=cfg.get("camera_distance"),
                              aspect=m.get("aspect"), denoise=m.get("denoise", True))
            attempt, p, rejections = sample_collision(cfg, seed, scratch, fmt)
            geo, masses = KS.solve_geometry(p), KS.derive_masses(p)
            # Pass 2 with the renderer. Keep `sim2` referenced until the group is done: Kubric's
            # PyBullet.__del__ disconnects the physics client (see release_simulator).
            (kscene, spheres, floor, camera, sim2, renderer, animation, collisions) = KS.build_and_simulate(
                p, geo, masses, scratch, render=render)
            pair_ev, floor_ev = KS.analyse_collisions(collisions, spheres, floor)
            reason, meas = KS.check_simulation(p, geo, masses, animation, spheres, pair_ev, floor_ev)
            if reason:
                raise RuntimeError(f"replay of accepted scene failed: {reason}")
            tw, pre, e_sim, _ = build_twins(family, cfg, p, geo, masses, animation, spheres, seed, attempt)
            contact = float(meas["contact_frame"])
            sc = collision_scene(cfg, p, geo, masses, animation, spheres, contact, tw, pre, fmt)
            sc.constants["e_sim"] = float(e_sim)
            h = {"scene": kscene, "renderer": renderer, "camera": camera, "floor": floor,
                 "objs": {"A": spheres["a"], "B": spheres["b"]}, "statics": {}}
            physics = {"simulator": "pybullet (kubric)", "momentum_rel_error": meas["momentum_rel_error"], "e_sim": e_sim}
            honest = {k_: copy.deepcopy(v) for k_, v in animation.items()}

            def legacy(name, ftw, stats, contact=contact, pre=pre, tw=tw):
                """The F1 label layout (KS.build_labels), with the twin written into the animation."""
                anim = {k_: copy.deepcopy(v) for k_, v in honest.items()}
                for T_, t_ in (("A", "a"), ("B", "b")):
                    anim[spheres[t_]]["position"] = list(ftw.tracks[T_].p)
                    anim[spheres[t_]]["velocity"] = list(ftw.tracks[T_].v)
                seg_ab = None if stats is None else {t_: stats[T_] for T_, t_ in (("A", "a"), ("B", "b"))}
                L = KS.build_labels(p, geo, masses, kscene, spheres, camera, anim, pair_ev, meas, seed,
                                    f"{gid}/{name}", seg_ab, gpu_desc)
                twin = tw[name]
                pre_t = twin.pre(pre)
                L["violates_momentum"] = not twin.momentum_conserved
                L["momentum_conserved"] = twin.momentum_conserved
                L["restitution"] = float(twin.e_target)
                L["kubric_version"] = kb.__version__
                L["physlaw"] = {
                    "contact_frame": contact_target, "contact_frame_actual": contact,
                    "contact_normal": pre.n.tolist(), "e_sim": float(e_sim), "e_target": float(twin.e_target),
                    "momentum_conserved": twin.momentum_conserved,
                    "summary": {**TW.summarize(pre_t, twin), "com_frame_angle": TW.com_frame_angle(pre_t, twin.v_a, twin.v_b)},
                    "inframe_until": inframe_until({spheres[t_]: {"position": ftw.tracks[T_].p}
                                                    for T_, t_ in (("A", "a"), ("B", "b"))}, spheres, p, sc.first_post),
                }
                L["physlaw"]["pre_velocity"] = {"A": pre_t.v_a.tolist(), "B": pre_t.v_b.tolist()}
                L["physlaw"]["post_velocity"] = {"A": twin.v_a.tolist(), "B": twin.v_b.tolist()}
                return L
        else:
            mf = fmt
            sc, attempt, rejections = sample_family(cfg, seed, fmt)
            physics = getattr(sc, "physics", None)
            h = build_family_kubric(sc, scratch, int(m.samples_per_pixel), bool(m.get("denoise", True))) if render else None
            legacy = None
            if render:
                renderer = h["renderer"]
        if cfg.get("legal_only"):
            legal_only(sc)
        gpu_desc = "n/a (physics only)"
        if render:
            gpu_desc = KS.configure_gpu(renderer)
            if gpu_desc.startswith("OPTIX"):
                renderer.blender_scene.cycles.denoising_use_gpu = True
            FP.minimal_passes(renderer)
            bs = renderer.blender_scene
            # Measured on the H100 (docs/STATUS.md "Render cost"): persistent data keeps the BVH and
            # scene data between frames, 0.87 -> 0.32 s/frame solo, output identical to Kubric's.
            bs.render.use_persistent_data = bool(m.get("persistent_data", True))
            if m.get("render_threads"):
                bs.render.threads_mode = "FIXED"
                bs.render.threads = int(m.render_threads)
            W, H = sc.camera.resolution
            bs.render.pixel_aspect_x = float(sc.camera.aspect * H / W)   # 1 except the squashed PC1 set
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(int(m.get("decode_workers", 3)), mp_context=mp.get_context("spawn"))

        split = split_for_group(gid, cfg.splits.train, cfg.splits.val)
        common = {"twin_group_id": gid, "base_group_id": gid, "split": split, "seed": seed,
                  "master_format": mf, "render_device": gpu_desc, "config_hash": config_hash(cfg),
                  "physics_check": physics,
                  "sampling_attempts": {"accepted_attempt": attempt, "rejections": rejections}}
        summaries = render_twins(cfg, sc, h, gdir, sc.camera, render, scratch, pool, common, legacy)

        rec = {"group": gid, "status": "ok", "seed": seed, "family": family, "subfamily": sc.subfamily,
               "split": split, "twins": list(sc.twins), "event_frame": sc.event_frame,
               "first_post_contact_frame": sc.first_post, "accepted_attempt": attempt,
               "n_rejections": len(rejections), "render_device": gpu_desc, "physics_check": physics,
               "summaries": summaries}
        if family in COLLISION_FAMILIES:
            rec["contact_frame"] = contact_target

        if render and float(cfg.get("nuisance_frac", 0.0)) > 0 and NZ.selected(seed, float(cfg.nuisance_frac)):
            rec["nuisance"] = render_nuisance(cfg, sc, h, ndir, scratch, pool, common, legacy, seed)
        rec["seconds"] = time.time() - t0
        (gdir / "_GROUP_COMPLETE").write_text(json.dumps(rec, indent=1, default=_json_default))
        return rec
    except Exception as exc:  # recorded, the driver decides whether to retry
        (gdir / "_FAILED").write_text(traceback.format_exc())
        return {"group": gid, "status": "failed", "error": repr(exc)[:500], "seconds": time.time() - t0}
    finally:
        if sim2 is not None:
            release_simulator(sim2)
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        shutil.rmtree(scratch, ignore_errors=True)


def render_nuisance(cfg, sc, h, ndir: Path, scratch, pool, common, legacy, seed) -> dict:
    """Re-render every twin under one randomised look whose camera keeps every body in frame."""
    rng = np.random.default_rng([seed, 31337])
    identical = len({tuple(b.look["color"]) for b in sc.bodies}) == 1 and len(sc.bodies) > 1
    for attempt in range(20):
        look = NZ.sample(rng, [b.name for b in sc.bodies], identical)
        cam = NZ.perturb_camera(sc.camera, look)
        probe = copy.copy(sc)
        probe.camera = cam
        if probe.inframe_reason() is None:
            break
    else:
        return {"status": "skipped_no_look_keeps_bodies_in_frame"}
    ndir.mkdir(parents=True, exist_ok=True)
    cam = apply_look(h, sc, look)
    gid_n = ndir.name
    common_n = {**common, "twin_group_id": gid_n, "look": look}
    if legacy:
        legacy_n = lambda name, tw, stats: {**legacy(name, tw, stats), "scene_id": f"{gid_n}/{name}"}
    else:
        legacy_n = None
    probe = copy.copy(sc)
    probe.camera = cam
    summaries = render_twins(cfg, probe, h, ndir, cam, True, scratch, pool, common_n, legacy_n, nuisance=1)
    rec = {"group": gid_n, "status": "ok", "base_group_id": common["base_group_id"], "nuisance": 1,
           "split": common["split"], "twins": list(sc.twins), "first_post_contact_frame": sc.first_post,
           "look": look, "render_device": common["render_device"], "summaries": summaries}
    (ndir / "_GROUP_COMPLETE").write_text(json.dumps(rec, indent=1, default=_json_default))
    return {"status": "ok", "group": gid_n, "look_attempt": attempt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, key.path=value")
    a = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    cfg = load_config(a.config, a.set)
    rec = run_group(cfg, a.seed, a.out, render=not a.no_render, force=a.force)
    print("RESULT " + json.dumps(rec, default=_json_default))
    sys.exit(0 if rec["status"] in ("ok", "skipped_exists") else 1)


if __name__ == "__main__":
    main()
