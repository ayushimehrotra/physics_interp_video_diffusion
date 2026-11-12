"""Closed-form scene families on top of the F1 machinery (spec §2.2, §2.3): F6 to F12.

Pure numpy (+ scipy for F8/F9 path integration). Each family module exposes

    SUBFAMILIES: dict[str, int]              default group count per sub-family
    sample(rng, cfg, fmt, sub) -> Scene | str   a Scene, or a rejection reason

and :func:`generate.run_group` renders whatever Scene comes back. A Scene carries
every twin's full per-frame trajectory, so rendering is pure keyframing.

Twin rules (spec §2.2), generalised from F1:
  * one pre-event state per twin group: every twin of one ``pre_segment`` has
    bit-identical trajectories before ``first_post`` (checked by
    :meth:`Scene.check_pre_identity`), so the renderer copies those frames;
  * outcomes after the event are closed form from that state, re-keyframed at the
    exact event time ``t_event`` (which need not be on a frame);
  * each violating twin breaks one law, named in ``violated_law``, with its factor
    in ``violation_magnitude``; side effects on other laws are flags (``flags``);
  * legal twins come in pairs where the physics allows a second legal outcome.

Units: metres, seconds, kilograms; ``G = 9.81``. World axes: z up. Side-view scenes
move in the x-z plane (camera looking along +y); top-down scenes in x-y.

Why the gravity families are closed form end to end (deviation from "simulated
pre-event state", recorded in docs/RESUME.md): Bullet's semi-implicit Euler at
240 Hz offsets a ballistic trajectory by g*dt*t/2 (0.05 m after 2.5 s) and its
box-on-box contacts jitter, so a simulated pre-event segment would fail the
1e-6 law tests for reasons unrelated to the twins. The same initial condition is
still run in PyBullet (``generate.physics_check``) and the gap is recorded per
group (``physics_check.max_err_m``) and bounded.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

G = 9.81
# F9 (pendulum) needs no hinge constraint (Kubric's PyBullet bodies refuse every constraint): the bob and the
# rod are keyframed from the exact (elliptic) pendulum motion, like every other closed-form family.
FAMILY_MODULES = {"f6": "f6_friction", "f7": "f7_freefall", "f8": "f8_ramp", "f9": "f9_pendulum",
                  "f10": "f10_rolling", "f11": "f11_support", "f12": "f12_spin"}


def family_module(family: str):
    return importlib.import_module(f"physlaw.data.families.{FAMILY_MODULES[family]}")


# --------------------------------------------------------------------------- #
# scene description
# --------------------------------------------------------------------------- #


@dataclass
class Body:
    """A dynamic object. ``size`` is the radius (sphere, cylinder) or the half-extent (cube; box: half[0])."""
    name: str                     # "A", "B" (label key)
    shape: str                    # sphere | cube | cylinder | tube | box
    size: float
    density: float
    look: dict                    # color (rgb), metallic, roughness, specular, ior
    seg_id: int
    length: float = 0.0           # cylinders: half-length along y
    inner: float = 0.9            # tube: inner radius / outer radius (F10 samples it per group)
    half: tuple | None = None     # box: half-extents along the body's x, y, z
    spin_axis: int = 1            # box: the body axis (0 x, 1 y, 2 z) that `inertia` is about
    aux: bool = False             # visual-only attachment (pendulum rod, release prop): labelled aux, left out of
    #                               io.law_inputs and of the in-frame check

    @property
    def volume(self) -> float:
        if self.shape == "sphere":
            return 4.0 / 3.0 * math.pi * self.size ** 3
        if self.shape == "cube":
            return (2.0 * self.size) ** 3
        if self.shape == "cylinder":
            return math.pi * self.size ** 2 * 2.0 * self.length
        if self.shape == "tube":       # wall from inner * r to r
            return math.pi * self.size ** 2 * (1 - self.inner ** 2) * 2.0 * self.length
        if self.shape == "box":
            return 8.0 * float(np.prod(self.half))
        raise ValueError(self.shape)

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def inertia(self) -> float:
        """Moment of inertia about a centroidal axis (the rolling axis for cylinders; thin-walled tube; box:
        ``spin_axis``)."""
        if self.shape == "box":
            h = [x for i, x in enumerate(self.half) if i != self.spin_axis]
            return self.mass * (h[0] ** 2 + h[1] ** 2) / 3.0
        k = {"sphere": 0.4 * self.size ** 2, "cylinder": 0.5 * self.size ** 2, "tube": 0.5 * (1 + self.inner ** 2) * self.size ** 2,
             "cube": (2.0 * self.size) ** 2 / 6.0}[self.shape]
        return k * self.mass


@dataclass
class Static:
    """A static box (ramps, ledges, bands, hinge posts). ``half`` are half-extents.

    ``profile`` (F8 ramps): instead a prism, the surface polyline [(x, z), ...] (x increasing) extruded over
    y in [-half[1], half[1]] and closed below at z = -2 half[2]; rendered as ONE mesh (:func:`profile_mesh_obj`: a
    smooth swept surface, no overlapping slabs), simulated as one box under each polyline segment
    (:func:`static_boxes`); ``position`` / ``quaternion`` stay the identity."""
    name: str
    half: tuple
    position: tuple
    quaternion: tuple = (1.0, 0.0, 0.0, 0.0)    # wxyz
    look: dict = field(default_factory=dict)
    friction: float = 0.0
    profile: list | None = None


def static_boxes(st: Static) -> list:
    """[(half, position, quaternion wxyz)] of the boxes that make up a static (physics)."""
    if st.profile is None:
        return [(tuple(st.half), tuple(st.position), tuple(st.quaternion))]
    out = []
    for (xa, za), (xb, zb) in zip(st.profile[:-1], st.profile[1:]):
        L = math.hypot(xb - xa, zb - za)
        if L < 1e-9:
            continue
        phi = math.atan2(-(zb - za), xb - xa)                  # descent angle of the segment (x increasing)
        n = np.array([math.sin(phi), 0.0, math.cos(phi)])
        mid = np.array([0.5 * (xa + xb), 0.0, 0.5 * (za + zb)]) - st.half[2] * n
        out.append(((0.5 * L, st.half[1], st.half[2]), tuple(mid), quat_y(phi)))
    return out


def profile_mesh_obj(st: Static, path) -> None:
    """Write the extruded ``st.profile`` prism as an .obj (Blender axes as f10_rolling.mesh_obj: x, y, z)."""
    xz = [tuple(map(float, q)) for q in st.profile]
    y, zb = float(st.half[1]), -2.0 * float(st.half[2])
    V, F = [], []
    for x, z in xz:                                          # per profile point: front top, back top, front/back bottom
        V += [(x, -y, z), (x, y, z), (x, -y, zb), (x, y, zb)]
    for i in range(len(xz) - 1):
        a, b = 4 * i, 4 * (i + 1)
        F += [(a, b, b + 1, a + 1),                          # surface strip
              (a, a + 2, b + 2, b),                          # front face (y = -half[1]): coplanar quads, no overlap
              (a + 1, b + 1, b + 3, a + 3),                  # back face
              (a + 2, a + 3, b + 3, b + 2)]                  # bottom
    n = 4 * (len(xz) - 1)
    F += [(0, 1, 3, 2), (n, n + 2, n + 3, n + 1)]            # end caps
    with open(path, "w") as f:
        f.writelines(f"v {x:.6f} {yy:.6f} {z:.6f}\n" for x, yy, z in V)
        f.writelines("f " + " ".join(str(i + 1) for i in face) + "\n" for face in F)


@dataclass
class Track:
    """Per-frame state of one body: position, velocity, quaternion (wxyz), angular velocity."""
    p: np.ndarray                 # [T, 3]
    v: np.ndarray                 # [T, 3]
    q: np.ndarray                 # [T, 4]
    w: np.ndarray                 # [T, 3]

    @staticmethod
    def still(T: int, p, q=(1.0, 0.0, 0.0, 0.0)) -> "Track":
        return Track(np.tile(np.asarray(p, float), (T, 1)), np.zeros((T, 3)),
                     np.tile(np.asarray(q, float), (T, 1)), np.zeros((T, 3)))

    def copy(self) -> "Track":
        return Track(self.p.copy(), self.v.copy(), self.q.copy(), self.w.copy())


@dataclass
class Twin:
    name: str
    tracks: dict                  # body name -> Track
    pre_segment: str = "0"        # twins sharing it are identical before first_post
    legal: bool = True
    violated_law: str | None = None
    violation_magnitude: float = 0.0
    flags: dict = field(default_factory=dict)     # e.g. energy_physical
    statics: list | None = None   # names of the statics visible in this twin (None: all)
    constants: dict = field(default_factory=dict)  # per-twin legal-world constants (e.g. mu, e_floor)
    record: dict = field(default_factory=dict)    # what was done, for the labels' `edit`
    masses: dict | None = None    # per-twin body masses where they differ (F3-like)
    # energy-ladder twins (F7 bounce, F8; collisions get theirs from :func:`collision_energy_labels`):
    epsilon: float | None = None          # generalised restitution of the exchange event (spec §5.0.1)
    clean_until_s: float | None = None    # first secondary event after the exchange (spec §5.0.2); also
    #                                       the end of this twin's in-frame window (Scene.inframe_reason)
    event_state: dict | None = None       # exact {"pre", "post"} states when not the event instant's limits
    # moving camera (F4 camera boost): [T, 3] world translation of the camera from Scene.camera (the camera at
    # ``camera_t_ref_s``); the camera moves at constant velocity ``camera_velocity`` and never turns
    camera_shift: np.ndarray | None = None
    camera_velocity: np.ndarray | None = None
    camera_t_ref_s: float | None = None

    def camera_frame(self) -> "Twin":
        """This twin in the camera's rest frame (world axes, origin moving with the camera, = the world at
        ``camera_t_ref_s``): positions - camera_shift, velocities - camera_velocity. Scene.camera projects these to the
        rendered pixels at every frame. A static camera: the twin itself."""
        if self.camera_shift is None:
            return self
        u = np.asarray(self.camera_velocity, float)
        tracks = {n: Track(t.p - self.camera_shift, t.v - u, t.q.copy(), t.w.copy()) for n, t in self.tracks.items()}
        es = None
        if self.event_state is not None:                 # exact states at their own instants t_s
            sh = lambda t_s: u * (float(t_s) - float(self.camera_t_ref_s))
            es = {k: {**st, "p": (np.asarray(st["p"], float) - sh(st["t_s"])).tolist(),
                      "v": (np.asarray(st["v"], float) - u).tolist()} for k, st in self.event_state.items()}
        return Twin(**{**self.__dict__, "tracks": tracks, "event_state": es, "camera_shift": None,
                       "camera_velocity": None, "camera_t_ref_s": None})


@dataclass
class Camera:
    """Pinhole camera; OpenCV convention for labels, Blender quaternion for Kubric."""
    position: tuple
    look_at: tuple
    up: tuple = (0.0, 0.0, 1.0)
    focal_mm: float = 50.0
    sensor_w_mm: float = 41.6
    resolution: tuple = (832, 480)
    aspect: float = 832 / 480     # logical frame aspect; != W/H means non-square pixels (PC1 256x256)

    def rotation(self) -> np.ndarray:
        """World -> camera rotation, rows = (right, down, forward)."""
        C, T = np.asarray(self.position, float), np.asarray(self.look_at, float)
        f = T - C
        f /= np.linalg.norm(f)
        up = np.asarray(self.up, float)
        if abs(np.dot(f, up / np.linalg.norm(up))) > 0.999:
            up = np.array([0.0, 1.0, 0.0])      # straight down: image-up = +y (F1 convention)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        d = np.cross(f, r)
        return np.stack([r, d, f])

    def K(self) -> np.ndarray:
        W, H = self.resolution
        fx = self.focal_mm / self.sensor_w_mm * W
        fy = fx * (H / W) * self.aspect
        return np.array([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])

    def world_to_camera(self) -> np.ndarray:
        R = self.rotation()
        t = -R @ np.asarray(self.position, float)
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, t
        return E

    def project(self, X: np.ndarray) -> np.ndarray:
        """[..., 3] world -> [..., 3] (u, v, depth)."""
        X = np.asarray(X, float)
        E = self.world_to_camera()
        Xc = X @ E[:3, :3].T + E[:3, 3]
        uv = Xc @ self.K().T
        return np.concatenate([uv[..., :2] / uv[..., 2:3], Xc[..., 2:3]], axis=-1)

    def blender_quaternion(self) -> tuple:
        """wxyz of a Blender camera (looks along local -z, image-up local +y)."""
        R = self.rotation()
        M = np.stack([R[0], -R[1], -R[2]], axis=1)   # columns: cam x, y, z in world
        return tuple(float(x) for x in mat_to_quat(M))

    def px_per_m(self, depth: float) -> float:
        return self.K()[0, 0] / depth

    def block(self) -> dict:
        E = self.world_to_camera()
        return {"model": "pinhole", "convention": "opencv (x right, y down, z forward); pixel origin top-left",
                "K": self.K().tolist(), "world_to_camera": E.tolist(),
                "camera_position_world": list(map(float, self.position)),
                "look_at_world": list(map(float, self.look_at)),
                "quaternion_wxyz_blender": list(self.blender_quaternion()),
                "focal_length_mm": self.focal_mm, "sensor_width_mm": self.sensor_w_mm,
                "resolution": list(self.resolution), "aspect": self.aspect}


def side_camera(fmt: dict, zc: float = 4.3, dist: float = 20.0, x0: float = 0.0) -> Camera:
    """Horizontal camera looking along +y at the x-z plane y = 0 (50 px/m at 832x480)."""
    return Camera((x0, -dist, zc), (x0, 0.0, zc), (0.0, 0.0, 1.0), sensor_w_mm=fmt["sensor_width_mm"],
                  resolution=tuple(fmt["resolution"]), aspect=fmt["aspect"])


def framed_side_camera(fmt: dict, x_lo: float, x_hi: float, z_lo: float, z_hi: float, fill: float = 0.85) -> Camera:
    """Side camera (as :func:`side_camera`) centred on the box [x_lo, x_hi] x [z_lo, z_hi] of the plane y = 0, at the
    distance where the box fills ``fill`` of the binding image dimension (the rest is an even margin)."""
    hw1, hh1 = visible_box(side_camera(fmt, dist=1.0), 1.0)          # half extents per metre of depth
    dist = max(0.5 * (x_hi - x_lo) / hw1, 0.5 * (z_hi - z_lo) / hh1) / fill
    return side_camera(fmt, zc=0.5 * (z_lo + z_hi), dist=dist, x0=0.5 * (x_lo + x_hi))


def top_camera(fmt: dict, height: float = 20.0, z_plane: float = 0.0) -> Camera:
    return Camera((0.0, 0.0, z_plane + height), (0.0, 0.0, z_plane), (0.0, 1.0, 0.0),
                  sensor_w_mm=fmt["sensor_width_mm"], resolution=tuple(fmt["resolution"]),
                  aspect=fmt["aspect"])


def visible_box(cam: Camera, depth: float) -> tuple[float, float]:
    """Half width/height in metres of the view at ``depth`` from the camera."""
    K = cam.K()
    W, H = cam.resolution
    return 0.5 * W / K[0, 0] * depth, 0.5 * H / K[1, 1] * depth


def mat_to_quat(M: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion wxyz (Shepperd)."""
    m = np.asarray(M, float)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def quat_y(angle: float) -> tuple:
    """wxyz rotation by ``angle`` about +y: a body tilted to slope DOWN to the right (+x) by angle."""
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def quat_y_array(angles: np.ndarray) -> np.ndarray:
    a = np.asarray(angles, float)
    return np.stack([np.cos(a / 2), np.zeros_like(a), np.sin(a / 2), np.zeros_like(a)], -1)


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #


@dataclass
class Scene:
    family: str
    subfamily: str
    bodies: list
    statics: list
    camera: Camera
    gravity: float                # |g| along -z (0 for planar collision scenes)
    t_event: float                # exact event time (s)
    event_kind: str               # collision | bounce | pad (energy exchange events) | branch | release | contact
    twins: dict                   # name -> Twin (ordered; the first twin of each pre_segment renders in full)
    fps: int
    constants: dict = field(default_factory=dict)   # legal-world scene constants shared by all twins
    stop_dose: dict | None = None                   # {body: dose} over the group's epsilon ladder (stop_dose())
    look: dict = field(default_factory=dict)        # floor / light / background settings
    sim_init: dict = field(default_factory=dict)    # PyBullet check: {"t0": s, "twin": name, "until": s}
    inframe: dict = field(default_factory=dict)     # {"start": frame, "stop": frame}: bodies fully visible
    params: dict = field(default_factory=dict)      # sampled parameters, copied into labels
    first_post_frame: int | None = None             # override (collision families keep ceil(contact))

    @property
    def first_post(self) -> int:
        """First frame after the event: frame floor(t*fps) shows the shared state."""
        if self.first_post_frame is not None:
            return self.first_post_frame
        return int(math.floor(self.t_event * self.fps + 1e-9)) + 1

    @property
    def event_frame(self) -> int:
        return self.first_post - 1

    def body(self, name: str) -> Body:
        return next(b for b in self.bodies if b.name == name)

    def masses(self, twin: Twin) -> dict:
        return twin.masses or {b.name: b.mass for b in self.bodies}

    def check_pre_identity(self) -> None:
        k = self.first_post
        first: dict[str, Twin] = {}
        for tw in self.twins.values():
            ref = first.setdefault(tw.pre_segment, tw)
            for n in tw.tracks:
                a, b = tw.tracks[n], ref.tracks[n]
                for key in ("p", "v", "q", "w"):
                    if not np.array_equal(getattr(a, key)[:k], getattr(b, key)[:k]):
                        raise AssertionError(f"{tw.name}/{n}.{key} differs from {ref.name} before frame {k}")
                if tw.statics != ref.statics:
                    raise AssertionError(f"{tw.name}: statics differ inside pre_segment {tw.pre_segment}")
            sa, sb = tw.camera_shift, ref.camera_shift
            if (sa is None) != (sb is None) or (sa is not None and not np.array_equal(sa[:k], sb[:k])):
                raise AssertionError(f"{tw.name}: camera motion differs from {ref.name} before frame {k}")

    def inframe_reason(self, margin_px: float = 4.0) -> str | None:
        """Every body's silhouette inside the image over [inframe.start, inframe.stop) in every twin; a twin
        with ``clean_until_s`` uses its own stop instead: every frame up to that instant."""
        lo = self.inframe.get("start", self.first_post)
        hi = self.inframe.get("stop", self.first_post + self.fps)
        for tw in self.twins.values():
            stop = hi if tw.clean_until_s is None else int(math.floor(tw.clean_until_s * self.fps + 1e-9)) + 1
            cf = tw.camera_frame()                          # a moving camera: positions relative to it
            for b in (b for b in self.bodies if not b.aux):
                bad = outside(self.camera, b, cf.tracks[b.name].p[lo:stop], margin_px)
                if bad.any():
                    return f"{tw.name}_{b.name}_out_of_frame_at_{lo + int(np.argmax(bad))}"
        return None


def outside(cam: Camera, b: "Body", p: np.ndarray, margin_px: float = 4.0) -> np.ndarray:
    """[n] bool: the body's silhouette (bounding sphere x 1.05 + margin) is not wholly inside the image."""
    W, H = cam.resolution
    uvz = cam.project(p)
    rad = bounding_radius(b) * cam.K()[0, 0] / uvz[:, 2] * 1.05 + margin_px
    rad_v = rad * cam.K()[1, 1] / cam.K()[0, 0]
    return (uvz[:, 0] - rad < 0) | (uvz[:, 0] + rad > W) | (uvz[:, 1] - rad_v < 0) | (uvz[:, 1] + rad_v > H)


def set_clean_until(sc: "Scene", t_secondary: dict | None = None, margin_px: float = 4.0) -> str | None:
    """Set ``clean_until_s`` of every twin whose first secondary event after the event is an object leaving the
    frame or ``t_secondary[twin]`` (an exact time, e.g. the first floor contact), whichever comes first: the time
    of the last frame at which every non-aux body is wholly in frame, capped at ``t_secondary``. A body that never
    leaves gives the clip's last frame, so a twin with no secondary event in the clip ends at the clip end. Returns
    a rejection reason when a twin leaves the frame before ``inframe.stop`` (the in-frame span the sampler
    requires; :meth:`Scene.inframe_reason` checks only up to ``clean_until_s`` once it is set)."""
    k, fps = sc.first_post, sc.fps
    need = sc.inframe.get("stop", k + fps)
    for tw in sc.twins.values():
        T = next(iter(tw.tracks.values())).p.shape[0]
        stop = T
        cf = tw.camera_frame()
        for b in (b for b in sc.bodies if not b.aux):
            out = outside(sc.camera, b, cf.tracks[b.name].p[k:], margin_px)
            if out.any():
                stop = min(stop, k + int(np.argmax(out)))
        if stop < min(need, T):
            return f"{tw.name}_leaves_frame_at_{stop}"
        tw.clean_until_s = min((stop - 1) / fps, float((t_secondary or {}).get(tw.name, math.inf)))
    return None


# --------------------------------------------------------------------------- #
# energy-ladder labels (spec §5.0.1-5.0.2): epsilon, clean_until_s, stop_dose
# --------------------------------------------------------------------------- #


EPS_LADDER = tuple(round(0.5 + 0.05 * i, 2) for i in range(21))   # 0.5, 0.55, ..., 1.5 (spec §5.0.2)


def eps_ladder(cfg, key: str) -> list:
    """The epsilon ladder ``cfg.twins.<key>`` (default :data:`EPS_LADDER`), sorted."""
    tw = (cfg.get("twins") if hasattr(cfg, "get") else None) or {}
    return sorted(float(e) for e in (tw.get(key) or EPS_LADDER))


def eps_name(e: float) -> str:
    """Twin name of ladder rung e: 0.5 -> "eps0.5", 1.0 -> "eps1.0", 1.25 -> "eps1.25"."""
    s = f"{float(e):.3f}".rstrip("0")
    return f"eps{s}0" if s.endswith(".") else f"eps{s}"


def stop_dose(ladder: dict) -> dict:
    """{body: dose} at which |v'(eps)| is minimal along an epsilon ladder ``{eps: {body: post velocity}}``.

    Post-event velocities are affine in eps (v' = v0 + eps d), so the two ladder ends fix the line and the
    minimiser is eps* = -v0.d / |d|^2. NaN when eps* is not strictly inside the ladder (the minimum is at a
    ladder end), or when v' does not depend on eps."""
    e_lo, e_hi = min(ladder), max(ladder)
    out = {}
    for n in ladder[e_lo]:
        a, b = np.asarray(ladder[e_lo][n], float), np.asarray(ladder[e_hi][n], float)
        d = (b - a) / (e_hi - e_lo)
        v0 = a - e_lo * d
        dd = float(d @ d)
        x = -float(v0 @ d) / dd if dd > 1e-24 else math.nan
        tol = 1e-9 * (e_hi - e_lo)
        out[n] = x if (math.isfinite(x) and e_lo + tol < x < e_hi - tol) else math.nan
    return out


def collision_energy_labels(labels: dict) -> dict:
    """``event_kind``, ``epsilon``, ``clean_until_s`` of a two-sphere collision clip (f1-f4), from its labels.

    Works on labels-1 (the frozen F1 render) and labels-2: needs only physlaw.pre_velocity / post_velocity
    (the twin's, planar), contact_normal (A -> B) and inframe_until (first frame at which a sphere's
    silhouette leaves the image, num_frames if never). epsilon = -(v_A' - v_B').n / (v_A - v_B).n.
    clean_until_s = time of the last frame before a sphere leaves the image, else of the last frame:
    frictionless spheres on a plane with no walls have no other secondary event (they separate at
    eps u_n >= 0 along n and never meet again)."""
    pl = labels["physlaw"]
    n = np.asarray(pl["contact_normal"], float)[:2]
    pv, qv = pl["pre_velocity"], pl["post_velocity"]
    rel = lambda v: np.asarray(v["A"], float)[:2] - np.asarray(v["B"], float)[:2]
    T, fps = labels["scene_constants"]["num_frames"], labels["scene_constants"]["frame_rate"]
    f_out = min(min(pl["inframe_until"].values()), T)
    return {"event_kind": "collision", "epsilon": float(-(rel(qv) @ n) / (rel(pv) @ n)),
            "clean_until_s": (f_out - 1) / fps}


def bounding_radius(b: Body) -> float:
    if b.shape == "cube":
        return b.size * math.sqrt(3.0)
    if b.shape in ("cylinder", "tube"):
        return math.hypot(b.size, b.length)
    if b.shape == "box":
        return float(np.linalg.norm(b.half))
    return b.size


def frame_times(fmt: dict) -> np.ndarray:
    return np.arange(fmt["num_frames"]) / fmt["fps"]


def event_time(rng: np.random.Generator, fmt: dict, aligned: bool) -> float:
    """Event time in the master's event window: on a frame (branch, release) or between frames (contact)."""
    lo, hi = fmt["event_frame_range"]
    k = int(rng.integers(lo, hi + 1))
    return k / fmt["fps"] if aligned else (k - 1 + float(rng.uniform(0.05, 0.95))) / fmt["fps"]


# --------------------------------------------------------------------------- #
# closed-form motion
# --------------------------------------------------------------------------- #


def ballistic(t: np.ndarray, t0: float, p0, v0, g: float, z_floor: float,
              e_seq: Callable[[int], float] | float, rest_speed: float = 0.05):
    """Positions/velocities at times ``t`` (all >= t0) of a point under gravity -g z with floor bounces.

    The floor is z = z_floor (the body's centre height when touching it). Bounce i
    reverses v_z with restitution ``e_seq(i)``; below ``rest_speed`` the body rests.
    Horizontal velocity is untouched (frictionless floor). Returns (p [n,3], v [n,3],
    bounce_times list).
    """
    e_of = (lambda i: float(e_seq)) if not callable(e_seq) else e_seq
    t = np.asarray(t, float)
    p = np.zeros((t.size, 3))
    v = np.zeros((t.size, 3))
    tc, pc, vc = float(t0), np.asarray(p0, float).copy(), np.asarray(v0, float).copy()
    bounces, resting = [], False
    order = np.argsort(t)
    for idx in order:
        ti = t[idx]
        while not resting:
            h = pc[2] - z_floor
            disc = vc[2] ** 2 + 2.0 * g * h
            tau = (vc[2] + math.sqrt(max(disc, 0.0))) / g if g > 0 else math.inf
            if tc + tau > ti:
                break
            pc = pc + vc * tau
            pc[2] = z_floor
            vz_in = vc[2] - g * tau
            tc += tau
            e = e_of(len(bounces))
            bounces.append(tc)
            vc = np.array([vc[0], vc[1], -e * vz_in])
            if abs(vc[2]) < rest_speed:
                vc[2] = 0.0
                resting = True
        dt = ti - tc
        if resting:
            p[idx] = pc + np.array([vc[0], vc[1], 0.0]) * dt
            v[idx] = [vc[0], vc[1], 0.0]
        else:
            p[idx] = pc + vc * dt - np.array([0.0, 0.0, 0.5 * g * dt * dt])
            v[idx] = vc - np.array([0.0, 0.0, g * dt])
    return p, v, bounces


def splice(pre: Track, post_p, post_v, k: int, post_q=None, post_w=None) -> Track:
    """A track equal to ``pre`` before frame k and to the given arrays from k on."""
    tr = pre.copy()
    tr.p[k:], tr.v[k:] = post_p, post_v
    if post_q is not None:
        tr.q[k:] = post_q
    if post_w is not None:
        tr.w[k:] = post_w
    return tr


def energy(scene: Scene, twin: Twin, frame: int, g: float | None = None) -> float:
    """Total mechanical energy (translational + rotational KE + m g z) at a frame."""
    g = scene.gravity if g is None else g
    m = scene.masses(twin)
    E = 0.0
    for b in scene.bodies:
        tr = twin.tracks[b.name]
        v, w = tr.v[frame], tr.w[frame]
        I = b.inertia / b.mass * m[b.name]
        E += 0.5 * m[b.name] * float(v @ v) + 0.5 * I * float(w @ w) + m[b.name] * g * float(tr.p[frame, 2])
    return E


# --------------------------------------------------------------------------- #
# looks shared by the families
# --------------------------------------------------------------------------- #

LOOKS = {
    "rubber_red": {"color": (0.80, 0.10, 0.10), "metallic": 0.0, "roughness": 0.8, "specular": 0.3, "ior": 1.45},
    "metal": {"color": (0.72, 0.74, 0.78), "metallic": 0.9, "roughness": 0.25, "specular": 0.9, "ior": 2.5},
    "plastic_blue": {"color": (0.12, 0.30, 0.75), "metallic": 0.0, "roughness": 0.5, "specular": 0.4, "ior": 1.45},
    "plastic_yellow": {"color": (0.85, 0.65, 0.10), "metallic": 0.0, "roughness": 0.5, "specular": 0.4, "ior": 1.45},
    "wood": {"color": (0.55, 0.38, 0.22), "metallic": 0.0, "roughness": 0.7, "specular": 0.2, "ior": 1.45},
    "ramp": {"color": (0.62, 0.62, 0.66), "metallic": 0.0, "roughness": 0.6, "specular": 0.2, "ior": 1.45},
    "pad": {"color": (0.18, 0.50, 0.28), "metallic": 0.0, "roughness": 0.95, "specular": 0.1, "ior": 1.45},
    "neutral": {"color": (0.22, 0.22, 0.22), "metallic": 0.0, "roughness": 0.6, "specular": 0.4, "ior": 1.45},
    "ice": {"color": (0.78, 0.86, 0.92), "metallic": 0.0, "roughness": 0.15, "specular": 0.6, "ior": 1.31},
}
BODY_LOOKS = ("rubber_red", "plastic_blue", "plastic_yellow", "metal", "wood")


def pick_look(rng: np.random.Generator, exclude=()) -> str:
    names = [n for n in BODY_LOOKS if n not in exclude]
    return str(names[int(rng.integers(len(names)))])


# --------------------------------------------------------------------------- #
# acceptance (spec §2.4): one twin's physics, from a Scene or from labels.json
# --------------------------------------------------------------------------- #

REL = 1e-6


@dataclass
class View:
    family: str
    sub: str
    twin: str
    legal: bool
    violated_law: str | None
    magnitude: float
    p: dict                       # body -> [T, 3]
    v: dict
    constants: dict
    record: dict
    fps: int
    t_event: float
    k: int                        # first post-event frame
    masses: dict
    w: dict = field(default_factory=dict)

    @staticmethod
    def from_scene(sc: "Scene", tw: Twin) -> "View":
        return View(sc.family, sc.subfamily, tw.name, tw.legal, tw.violated_law, tw.violation_magnitude,
                    {n: t.p for n, t in tw.tracks.items()}, {n: t.v for n, t in tw.tracks.items()},
                    {**sc.constants, **tw.constants, "g": sc.gravity, "sizes": {b.name: b.size for b in sc.bodies}},
                    tw.record, sc.fps, sc.t_event, sc.first_post, sc.masses(tw),
                    {n: t.w for n, t in tw.tracks.items()})

    @staticmethod
    def from_labels(L: dict) -> "View":
        pl = L["physlaw"]
        names = list(pl["objects"])
        arr = lambda key, n: np.array([f[n][key] for f in L["per_frame"]], float)
        return View(pl["family"], pl.get("subfamily", ""), pl["twin"], pl["legal"], pl["violated_law"],
                    pl["violation_magnitude"], {n: arr("position", n) for n in names},
                    {n: arr("velocity", n) for n in names}, pl["constants"], pl.get("edit") or {},
                    pl["constants"]["fps"], pl["event_time_s"], pl["first_post_contact_frame"],
                    {n: pl["objects"][n]["mass"] for n in names}, {n: arr("angular_velocity", n) for n in names})

    def accel(self, n: str, f0: int, f1: int) -> np.ndarray:
        """Finite-difference acceleration over frame pairs (f, f+1), f in [f0, f1)."""
        v = self.v[n]
        return (v[f0 + 1:f1 + 1] - v[f0:f1]) * self.fps


def close(a, b, rel=REL, scale=None) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = scale if scale is not None else max(float(np.max(np.abs(b))), 1e-12)
    return bool(np.max(np.abs(a - b)) <= rel * s)


def acceptance(view: View) -> list:
    """Spec §2.4 law checks for one twin; returns the failures (empty = pass)."""
    return family_module(view.family).acceptance(view)
