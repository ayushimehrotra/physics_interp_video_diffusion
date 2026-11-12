"""L0 targets on synthetic labels (no cache): p = m v, the partial (beyond v and m) target, F8 / F9 / F10 / F11 / F12
targets (aux bodies skipped), the R1 same instrument."""
import json

import numpy as np

from physlaw.eval.l0_states import OBJECTS, PHASES, momentum_partial, targets
from physlaw.probes.grid import grid_probe


def _labels(n=240, seed=0, const_mass_B=False):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        objs, pre, post = {}, {}, {}
        for o in OBJECTS:
            m = 1.0 if (const_mass_B and o == "B") else float(rng.uniform(0.3, 3.0))
            objs[o] = {"mass": m, "radius": float(rng.uniform(0.2, 0.5))}
            pre[o] = list(rng.normal(size=3))
            post[o] = list(rng.normal(size=3))
        out.append({"physlaw": {"objects": objs, "pre_velocity": pre, "post_velocity": post}})
    return out


def test_momentum_is_m_times_v():
    labels = _labels()
    Y = targets(labels)
    for o in OBJECTS:
        m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels])
        for ph in PHASES:
            v = np.array([l["physlaw"][f"{ph}_velocity"][o][:2] for l in labels])
            assert Y[f"{o}.{ph}.momentum"].shape == (len(labels), 2)
            np.testing.assert_array_equal(Y[f"{o}.{ph}.momentum"], m[:, None] * v)


def test_partial_uncorrelated_with_factors_and_nonzero():
    labels = _labels()
    Y = targets(labels)
    for o in OBJECTS:
        m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels])
        for ph in PHASES:
            v = Y[f"{o}.{ph}.velocity"]
            r = Y[f"{o}.{ph}.momentum_partial"]
            assert r.shape == (len(labels), 2)
            for f in (v[:, 0], v[:, 1], m, np.log(m)):
                for j in range(2):
                    assert abs(np.corrcoef(f, r[:, j])[0, 1]) < 1e-8
            # a sizeable share of m v's variance is beyond v and m
            p = Y[f"{o}.{ph}.momentum"]
            assert r.var(0).sum() > 0.05 * p.var(0).sum()


def test_partial_dropped_when_mass_constant():
    Y = targets(_labels(const_mass_B=True))
    assert "B.pre.momentum_partial" not in Y and "B.pre.momentum" in Y
    assert "A.pre.momentum_partial" in Y
    v = np.random.default_rng(1).normal(size=(50, 2))
    assert momentum_partial(v, np.full(50, 2.0)) is None


def test_probe_partial_r2_needs_the_product():
    labels = _labels(n=300)
    Y = targets(labels)
    o, ph = "A", "post"
    m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels])
    v = Y[f"{o}.{ph}.velocity"]
    groups = np.repeat(np.arange(len(labels) // 2), 2)
    b = f"{o}.{ph}.momentum_partial"
    factors = np.column_stack([v, m, np.log(m)])
    product = np.column_stack([factors, m[:, None] * v])
    kw = dict(alphas=(1e-6, 1e-3, 1e-1), k=5, fold_seeds=2)
    r_fac = grid_probe(factors, {b: Y[b]}, groups, **kw)[b]["score_mean"]
    r_prod = grid_probe(product, {b: Y[b]}, groups, **kw)[b]["score_mean"]
    assert r_fac < 0.05, r_fac
    assert r_prod > 0.95, r_prod


# ---------------------------------------------------------------- F8 Galileo's ramp, F10 rolling, the R1 gate

def _f8_labels(eps, h, n_frames=40, k=15, fps=10):
    """Block released from rest at height h, descends (frames < k), pad at frame k (speed s -> eps s), climbs to
    eps^2 h; x-z positions only (what targets_f8 reads)."""
    s = np.sqrt(2 * 9.81 * h)
    z = np.r_[np.linspace(h, 0, k), np.zeros(5), np.linspace(0, eps ** 2 * h, n_frames - k - 5)]
    return {"physlaw": {"family": "f8", "objects": {"A": {"mass": 1.0, "radius": 0.2}}, "first_post_contact_frame": k,
                        "event_kind": "pad", "clean_until_s": n_frames / fps, "constants": {"h": h, "z_floor": 0.0},
                        "event_state": {"pre": {"v": [[s, 0.0, 0.0]]}, "post": {"v": [[eps * s, 0.0, 0.0]]}}},
            "scene_constants": {"frame_rate": fps},
            "per_frame": [{"A": {"position": [0.0, 0.0, float(zz)]}} for zz in z]}


def test_f8_galileo_targets_and_event_inputs():
    from physlaw.eval.l0_states import event_inputs
    labs = [_f8_labels(e, h) for e in (0.5, 1.0, 1.5) for h in (1.0, 2.0)]
    Y = targets(labs)
    assert {"A.pre.velocity", "A.post.velocity", "A.pre.speed", "A.post.speed", "A.pre.height", "A.post.height"} <= set(Y)
    assert np.allclose(Y["A.post.speed"] / Y["A.pre.speed"], [0.5, 0.5, 1.0, 1.0, 1.5, 1.5])
    assert np.allclose(Y["A.post.height"] / Y["A.pre.height"], [0.25, 0.25, 1.0, 1.0, 2.25, 2.25])
    vp, vq, ctx = event_inputs(labs[0])
    assert ctx["event"] == "pad" and vp.shape == (1, 2)
    from physlaw.eval import laws
    assert laws.epsilon({"v": vp}, {"v": vq}, ctx) == np.float64(0.5)


def test_f10_angular_velocity_targets():
    rng = np.random.default_rng(0)
    labs = []
    for i in range(6):
        th, r = rng.uniform(0.2, 0.5), 0.3
        pf = []
        for f in range(20):
            row = {}
            for o, k in (("A", 0.5), ("B", 0.905)):
                v = 0.0 if f < 5 else 9.81 * np.sin(th) / (1 + k) * (f - 5) / 10
                row[o] = {"velocity": [v * np.cos(th), 0.0, -v * np.sin(th)], "angular_velocity": [0.0, v / r, 0.0]}
            pf.append(row)
        labs.append({"physlaw": {"family": "f10", "first_post_contact_frame": 5,
                                 "objects": {o: {"mass": 1.0, "radius": r} for o in OBJECTS},
                                 "constants": {"inertia_factors": {"A": 0.5 + 0.05 * i, "B": 0.905}}},
                     "scene_constants": {"frame_rate": 10}, "per_frame": pf})
    Y = targets(labs)
    assert "A.post.angular_velocity" in Y and "A.pre.angular_velocity" not in Y        # at rest before: dropped
    assert "A.pre.inertia_factor" in Y and "B.pre.inertia_factor" not in Y   # H1: kappa before the release (B constant)
    assert np.allclose(Y["A.post.angular_velocity"] * 0.3, Y["A.post.speed"])            # rolling: |w| r = |v|


def test_eps_gate_slope_and_r2():
    from physlaw.eval.l0_states import eps_gate
    e = np.linspace(0.5, 1.5, 11)
    g = eps_gate(0.1 + 0.8 * e, e)
    assert abs(g["slope"] - 0.8) < 1e-9 and abs(g["intercept"] - 0.1) < 1e-9 and abs(g["r2"] - 1.0) < 1e-9 and g["n"] == 11


def test_same_instrument_recovers_eps_on_real_clips():
    """One velocity probe per ball on pre + post rows; eps_hat (laws.epsilon of decoded velocities) tracks eps."""
    from physlaw.eval.l0_states import same_instrument
    rng = np.random.default_rng(0)
    D, labs, rows = 10, [], []
    W = rng.normal(size=(2, D))
    pools = ["roi_A.pre", "roi_A.post", "roi_B.pre", "roi_B.post", "frame.pre", "frame.post"]
    for g in range(40):
        for e in (0.25, 0.75, 1.25):
            va, vb = np.array([rng.uniform(1, 2), 0.0]), np.array([0.0, 0.0])
            pa, pb = va - (1 + e) / 2 * va, (1 + e) / 2 * va                   # equal masses, head-on along x
            labs.append({"physlaw": {"objects": {o: {"mass": 1.0} for o in "AB"}, "contact_normal": [1.0, 0.0],
                                     "pre_velocity": {"A": va.tolist(), "B": vb.tolist()},
                                     "post_velocity": {"A": pa.tolist(), "B": pb.tolist()}}})
            f = lambda v: v @ W + 0.01 * rng.normal(size=D)
            rows.append([f(va), f(pa), f(vb), f(pb), f(va + vb), f(pa + pb)])
    acts = np.array(rows)[:, :, None]
    groups = np.repeat(np.arange(40), 3)
    tres = {f"{o}.{ph}.velocity": {"heldout": {"selected": f"L0/roi_{o}.{ph}"}} for o in "AB" for ph in PHASES}
    out = same_instrument(acts, pools, [0], labs, groups, tres, {"k_folds": 3, "alphas": [1e-2, 1.0]}, kinds=("roi_own",))
    assert out["A"]["r2_post"] > 0.95 and out["epsilon"]["slope"] > 0.9 and out["epsilon"]["r2"] > 0.9


def _family_labels(fam, sub, n=3):
    """labels.json content (no frames, no masks) of every twin of the first n accepted scenes of a closed-form family."""
    from physlaw.data import families as FAM
    from physlaw.data import generate as GEN
    from physlaw.config import load_config
    cfg = load_config(f"configs/data/{dict(f7='f7_freefall', f9='f9_pendulum', f11='f11_support', f12='f12_spin')[fam]}.yaml")
    fmt, mod, out, got = GEN.master_fmt(cfg), FAM.family_module(fam), [], 0
    for seed in range(60):
        sc = mod.sample(np.random.default_rng([seed, 0]), cfg, fmt, sub)
        if isinstance(sc, str) or sc.inframe_reason():
            continue
        for name, tw in sc.twins.items():
            L = {"per_frame": GEN.per_frame_labels(sc, tw, sc.camera, None),
                 "scene_constants": {"num_frames": len(tw.tracks[sc.bodies[0].name].p), "frame_rate": sc.fps},
                 "physlaw": {"family": sc.family, "twin": name, "edit": tw.record, **GEN.law_fields(sc, tw, sc.camera)}}
            out.append(json.loads(json.dumps(L, default=GEN._json_default)))
        got += 1
        if got == n:
            return out
    raise AssertionError("sampler rejects almost everything")


def test_f9_pendulum_targets_skip_the_rods():
    from physlaw.eval.l0_states import object_names, pool_kinds
    labs = _family_labels("f9", "pair")
    Y = targets(labs)
    assert object_names(labs[0]) == ["A", "B"] and pool_kinds(labs) == ("roi_own", "roi_pair", "frame")
    assert {f"{o}.{ph}.{t}" for o in "AB" for ph in PHASES for t in ("angle", "angular_velocity")} <= set(Y)
    assert not any(k.startswith("R") for k in Y)
    amp = np.array([l["physlaw"]["constants"]["amplitude_rad"] for l in labs])
    assert np.allclose(Y["A.pre.angle"][:, 1], amp / np.sqrt(2), rtol=0.15)               # RMS of a near-harmonic swing
    legal = np.array([l["physlaw"]["twin"] == "legal" for l in labs])
    T = np.array([l["physlaw"]["constants"]["period_legal_s"] for l in labs])
    w_over_th = Y["A.post.angular_velocity"][:, 1] / Y["A.post.angle"][:, 1]              # RMS w / RMS angle = 2 pi / T
    assert np.allclose(w_over_th[legal], 2 * np.pi / T[legal], rtol=0.1)
    sw = _family_labels("f9", "swing", 2)
    assert object_names(sw[0]) == ["A"] and pool_kinds(sw) == ("roi_own", "frame")         # rod R is aux
    assert {k.split(".")[0] for k in targets(sw)} == {"A"}


def test_f7_height_targets():
    """F7 (projectile, drop and bounce groups in one family): A's highest point above the floor contact height per
    phase. Bounce: pre = the drop height, post = the rebound apex eps^2 x the drop height (inside the clean window,
    which ends at the second floor contact); drop: pre = the drop height."""
    labs = [l for sub in ("projectile", "drop", "bounce") for l in _family_labels("f7", sub, 2)]
    Y = targets(labs)
    assert {"A.pre.height", "A.post.height", "A.pre.velocity", "A.post.velocity"} <= set(Y)
    assert not any(k.startswith("B.") for k in Y)
    sub = np.array([l["physlaw"]["subfamily"] for l in labs])
    H = np.array([l["physlaw"]["constants"].get("drop_height", np.nan) for l in labs])
    eps = np.array([l["physlaw"].get("epsilon") or np.nan for l in labs])
    b, d = sub == "bounce", sub == "drop"
    assert np.allclose(Y["A.pre.height"][b | d], H[b | d], rtol=1e-9)
    assert np.allclose(Y["A.post.height"][b], eps[b] ** 2 * H[b], rtol=2e-3)              # apex between frames
    assert np.all(Y["A.post.height"][d] < H[d])                                           # mid-fall after the branch


def test_f11_and_f12_targets():
    labs = _family_labels("f11", "support")
    Y = targets(labs)
    assert {"A.post.acceleration", "A.post.angular_velocity", "A.post.speed"} <= set(Y)
    assert not any(k.startswith("P") or k.startswith("A.pre") for k in Y)                 # prop aux; at rest before
    tip = np.array([l["physlaw"]["twin"] == "tip" for l in labs])
    assert np.all(np.abs(Y["A.post.angular_velocity"][tip]) > 0) and np.all(Y["A.post.angular_velocity"][~tip] == 0)
    assert np.all(Y["A.post.acceleration"][~tip] == 0) and np.all(Y["A.post.acceleration"][tip][:, 1] < 0)
    Y12 = targets(_family_labels("f12", "spin", 2))
    assert "B.post.angular_velocity" in Y12 and "A.post.angular_velocity" not in Y12      # the bar spins, the puck not
    assert "B.pre.angular_velocity" not in Y12 and Y12["A.post.velocity"].shape[1] == 2
