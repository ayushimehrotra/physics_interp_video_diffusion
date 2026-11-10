#!/usr/bin/env python3
"""Render one twin group (spec §2): one honest simulation, every twin rendered from it.

Runs in the Kubric venv (bpy + pybullet, no torch). Normally driven by
``scripts/01_generate_data.py``, which runs many groups in parallel; can also be
run directly for one group:

    KUBRIC_USE_GPU=1 /root/envs/kubric_venv/bin/python physlaw/data/generate.py \\
        --config configs/data/f1_collisions.yaml --seed 100000 --out data/f1

Flow per group
--------------
1. Contact frame for this seed is drawn from ``master.contact_frame_range``.
2. Pass 1 (physics only): sample a scene with e = 1, simulate, validate (the old
   generator's checks), build all twins analytically, and reject the draw unless
   every twin keeps both spheres fully in frame for ``min_post_inframe_s`` after
   contact and every momentum-violating twin's residual exceeds
   ``min_violation_residual``. Resample on rejection.
3. Pass 2: rebuild the accepted scene with the renderer attached, re-validate.
4. Twin 0 is rendered in full. Every later twin is re-keyframed along its own
   post-contact velocities and only frames >= first_post_contact_frame are
   rendered; the earlier frames are copied from twin 0. Twins are therefore
   byte-identical before contact BY CONSTRUCTION (and the test suite checks it on
   the decoded mp4 too).
5. Each twin writes video.mp4, masks.npz, labels.json, _COMPLETE.
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
from physlaw.data.splits import split_for_group  # noqa: E402
from physlaw.config import load_config, config_hash  # noqa: E402

logger = logging.getLogger("physlaw.generate")
SCHEMA_VERSION = "physlaw-labels-1"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def group_id(family: str, seed: int) -> str:
    return f"{family}_{seed:06d}"


def contact_frame_for(seed: int, lo: int, hi: int) -> int:
    return int(np.random.default_rng([seed, 4242]).integers(lo, hi + 1))


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
    else:
        raise NotImplementedError(f"family {family!r}: spec milestone 8 (F3-F6 generators) not built yet")
    return tw, pre, e_sim, None


def feasibility(cfg, tw, pre, p, animation, spheres, contact_frame: float) -> str | None:
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


def camera_block() -> dict:
    """Pinhole intrinsics and OpenCV-convention extrinsics of the fixed overhead camera."""
    W, H = KS.RESOLUTION
    f_px = KS.FOCAL_LENGTH / KS.SENSOR_WIDTH * W
    C = np.array([0.0, 0.0, KS.CENTER_HEIGHT + KS.CAMERA_DISTANCE])
    # Blender camera with identity rotation looks down -Z with image-up = +Y.
    # OpenCV camera frame: x right (+X), y down (-Y), z forward (-Z).
    R = np.diag([1.0, -1.0, -1.0])
    t = -R @ C
    return {
        "model": "pinhole", "convention": "opencv (x right, y down, z forward); pixel origin top-left",
        "K": [[f_px, 0.0, W / 2.0], [0.0, f_px, H / 2.0], [0.0, 0.0, 1.0]],
        "world_to_camera": [[*R[0], t[0]], [*R[1], t[1]], [*R[2], t[2]], [0.0, 0.0, 0.0, 1.0]],
        "camera_position_world": C.tolist(),
        "focal_length_mm": KS.FOCAL_LENGTH, "sensor_width_mm": KS.SENSOR_WIDTH,
        "resolution": [W, H],
    }


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


def write_twin(tdir: Path, rgb, seg, labels: dict, name: str, crf: int) -> None:
    if rgb is not None:
        write_mp4_causal(tdir / "video.mp4", rgb, KS.FRAME_RATE, crf)
        np.savez_compressed(tdir / "masks.npz", instance=seg,
                            object_ids=np.array([KS.SEG_ID["A"], KS.SEG_ID["B"]], dtype=np.uint8))
    (tdir / "labels.json").write_text(json.dumps(labels, indent=1, default=float))
    (tdir / "_COMPLETE").write_text(name + "\n")


# --------------------------------------------------------------------------- #
# one group
# --------------------------------------------------------------------------- #


def run_group(cfg, seed: int, out_root: Path, render: bool = True, force: bool = False) -> dict:
    import kubric as kb
    import pybullet as pb

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
    if gdir.exists():
        shutil.rmtree(gdir)
    gdir.mkdir(parents=True)

    m = cfg.master
    contact_target = contact_frame_for(seed, *m.contact_frame_range)
    fmt = KS.configure(num_frames=m.num_frames, frame_rate=m.fps, resolution=(m.width, m.height),
                       contact_frame=contact_target, samples_per_pixel=m.samples_per_pixel,
                       contact_tolerance_frames=m.contact_tolerance_frames)
    t0 = time.time()
    scratch = tempfile.mkdtemp(prefix="physlaw_kubric_")
    pool = None
    rejections = []
    try:
        accepted = None
        for attempt in range(cfg.acceptance.max_attempts):
            rng = np.random.default_rng([seed, attempt])
            p, reason = KS.sample_params(rng, violate=False, arm=cfg.arm, restitution=cfg.twins.e_sim,
                                         material_pair=cfg.material_pair)
            if p is None:
                rejections.append({"attempt": attempt, "reason": reason}); continue
            geo, masses = KS.solve_geometry(p), KS.derive_masses(p)
            reason = KS.check_frame0(p, geo)
            if reason:
                rejections.append({"attempt": attempt, "reason": reason}); continue
            (_, spheres, floor, _, sim, _, animation, collisions) = KS.build_and_simulate(
                p, geo, masses, scratch, render=False)
            pair_ev, floor_ev = KS.analyse_collisions(collisions, spheres, floor)
            reason, meas = KS.check_simulation(p, geo, masses, animation, spheres, pair_ev, floor_ev)
            if not reason:
                tw, pre, e_sim, reason = build_twins(family, cfg, p, geo, masses, animation, spheres, seed, attempt)
                if not reason:
                    reason = feasibility(cfg, tw, pre, p, animation, spheres, meas["contact_frame"])
            release_simulator(sim)
            if reason:
                rejections.append({"attempt": attempt, "reason": reason}); continue
            accepted = (attempt, p)
            break
        if accepted is None:
            raise RuntimeError(f"no valid scene after {cfg.acceptance.max_attempts} attempts: {rejections[-5:]}")
        attempt, p = accepted
        geo, masses = KS.solve_geometry(p), KS.derive_masses(p)

        # ---- pass 2: with renderer
        # Keep `sim2` referenced until the group is done: Kubric's PyBullet.__del__
        # disconnects the physics client, and re-keyframing a sphere pushes the new
        # position to that client. Discarding it as `_` let the cyclic GC disconnect it
        # at a random point mid-group ("Not connected to physics server", ~1 in 7 groups).
        (scene, spheres, floor, camera, sim2, renderer, animation, collisions) = KS.build_and_simulate(
            p, geo, masses, scratch, render=render)
        pair_ev, floor_ev = KS.analyse_collisions(collisions, spheres, floor)
        reason, meas = KS.check_simulation(p, geo, masses, animation, spheres, pair_ev, floor_ev)
        if reason:
            raise RuntimeError(f"replay of accepted scene failed: {reason}")
        tw, pre, e_sim, _ = build_twins(family, cfg, p, geo, masses, animation, spheres, seed, attempt)
        contact = float(meas["contact_frame"])
        first_post = int(math.ceil(contact))
        gpu_desc = KS.configure_gpu(renderer) if render else "n/a (physics only)"
        if render and gpu_desc.startswith("OPTIX"):
            # Measured on the H100: 2.6 -> 2.3 s/frame. Samples barely matter (16 vs 64 spp:
            # 2.29 vs 2.30 s/frame); the rest is per-frame Blender/Kubric overhead, so
            # throughput comes from running groups in parallel (scripts/01 --workers).
            renderer.blender_scene.cycles.denoising_use_gpu = True
        if render:
            FP.minimal_passes(renderer)
            # Measured on the H100 (docs/STATUS.md "Render cost"): persistent data keeps the BVH and
            # scene data between frames, 0.87 -> 0.32 s/frame solo, output identical to Kubric's.
            renderer.blender_scene.render.use_persistent_data = bool(cfg.master.get("persistent_data", True))
            if cfg.master.get("render_threads"):
                renderer.blender_scene.render.threads_mode = "FIXED"
                renderer.blender_scene.render.threads = int(cfg.master.render_threads)
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(int(cfg.master.get("decode_workers", 3)),
                                       mp_context=mp.get_context("spawn"))

        split = split_for_group(gid, cfg.splits.train, cfg.splits.val)
        cam = camera_block()
        # Keys of `animation` are Kubric objects: copy the values only, never the keys.
        honest_anim = {k: copy.deepcopy(v) for k, v in animation.items()}
        pre_rgb = pre_seg = None
        summaries = {}
        from concurrent.futures import ThreadPoolExecutor
        writer, writes = ThreadPoolExecutor(2), []
        for i, (name, twin) in enumerate(tw.items()):
            for tag in ("a", "b"):          # restore the honest trajectory, then edit
                animation[spheres[tag]] = copy.deepcopy(honest_anim[spheres[tag]])
            KS.rekeyframe_exact(spheres, animation, contact,{"a": _vec3(twin.v_a), "b": _vec3(twin.v_b)})
            tdir = gdir / name
            tdir.mkdir(parents=True, exist_ok=True)
            rt0 = time.time()
            seg_stats, frame_sha = None, None
            if render:
                frames = list(range(KS.NUM_FRAMES)) if i == 0 else list(range(first_post, KS.NUM_FRAMES))
                # Our own render loop (fast_post): each frame is decoded in a process pool as
                # soon as Blender writes it, instead of Kubric's serial postprocess of every
                # layer after the last frame (which left the GPU idle half the time).
                rgb, seg = FP.render_frames(renderer, scene, frames, Path(scratch), pool)
                if rgb.shape[0] != len(frames):
                    raise RuntimeError(f"renderer returned {rgb.shape[0]} frames, asked for {len(frames)}")
                if i == 0:
                    pre_rgb, pre_seg = rgb[:first_post].copy(), seg[:first_post].copy()
                else:
                    rgb, seg = np.concatenate([pre_rgb, rgb]), np.concatenate([pre_seg, seg])
                assert rgb.shape[0] == KS.NUM_FRAMES
                seg_stats = KS.segmentation_stats(seg[..., None])
                frame_sha = [hashlib.sha256(f.tobytes()).hexdigest()[:16] for f in rgb]
            render_s = time.time() - rt0

            labels = KS.build_labels(p, geo, masses, scene, spheres, camera, animation, pair_ev, meas,
                                     seed, f"{gid}/{name}", seg_stats, gpu_desc)
            summ = TW.summarize(pre, twin)
            summaries[name] = summ
            labels["sampling_attempts"] = {"accepted_attempt": attempt, "rejections": rejections}
            labels["kubric_version"] = kb.__version__
            # Old generator fields that describe the honest simulation, overwritten per twin:
            labels["violates_momentum"] = not twin.momentum_conserved
            labels["momentum_conserved"] = twin.momentum_conserved
            labels["restitution"] = float(twin.e_target)
            labels["physlaw"] = {
                "schema_version": SCHEMA_VERSION,
                "family": family, "twin": name, "twin_index": i, "twin_group_id": gid,
                "split": split, "seed": seed, "nuisance": 0,
                "contact_frame": contact_target,
                "contact_frame_actual": contact,
                "first_post_contact_frame": first_post,
                "contact_normal": pre.n.tolist(),
                "camera": cam,
                "objects": {T: {"mass": float(masses[f"mass_{t}"]), "radius": float(p[f"radius_{t}"]),
                                "density": float(p[f"density_{t}"]),
                                "segmentation_id": KS.SEG_ID[T]} for T, t in (("A", "a"), ("B", "b"))},
                "pre_velocity": {"A": pre.v_a.tolist(), "B": pre.v_b.tolist()},
                "post_velocity": {"A": twin.v_a.tolist(), "B": twin.v_b.tolist()},
                "e_sim": float(e_sim), "e_target": float(twin.e_target),
                "momentum_conserved": twin.momentum_conserved,
                "energy_physical": twin.energy_physical,
                "edit": twin.record, "summary": summ,
                "inframe_until": inframe_until(animation, spheres, p, first_post),
                "frame_sha256_16": frame_sha,
                "master_format": fmt,
                "render_seconds": render_s,
                "config_hash": config_hash(cfg),
            }
            # Encode + write in a background thread so the next twin starts rendering at once
            # (the causal single-threaded x264 encode of 289 frames is CPU time the GPU would
            # otherwise sit through). _COMPLETE is written last, after every file is on disk.
            writes.append(writer.submit(write_twin, tdir, rgb if render else None, seg if render else None,
                                        labels, name, cfg.master.crf))
            logger.info("%s/%s rendered (%.1fs)", gid, name, render_s)
        for fu in writes:
            fu.result()

        rec = {"group": gid, "status": "ok", "seed": seed, "family": family, "split": split,
               "twins": list(tw), "contact_frame": contact_target, "first_post_contact_frame": first_post,
               "accepted_attempt": attempt, "n_rejections": len(rejections),
               "seconds": time.time() - t0, "render_device": gpu_desc, "summaries": summaries}
        (gdir / "_GROUP_COMPLETE").write_text(json.dumps(rec, indent=1, default=float))
        release_simulator(sim2)
        return rec
    except Exception as exc:  # recorded, the driver decides whether to retry
        (gdir / "_FAILED").write_text(traceback.format_exc())
        return {"group": gid, "status": "failed", "error": repr(exc)[:500], "seconds": time.time() - t0}
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        shutil.rmtree(scratch, ignore_errors=True)


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
    print("RESULT " + json.dumps(rec, default=float))
    sys.exit(0 if rec["status"] in ("ok", "skipped_exists") else 1)


if __name__ == "__main__":
    main()
