"""Energy ladders (spec §5.0.2): F8 Galileo's ramp and the F7 bounce ladder, plus the energy labels of every
exchange family (epsilon, clean_until_s, stop_dose, event_kind) and their schema. CPU only, no rendering."""
import json
import math

import numpy as np
import pytest

from physlaw.config import load_config
from physlaw.data import families as FAM
from physlaw.data import generate as GEN
from physlaw.data.families import f7_freefall as F7
from physlaw.data.families import f8_ramp as R

LADDER = [round(0.5 + 0.05 * i, 2) for i in range(21)]      # the configs' ladders: 21 rungs, step 0.05
NAMES = [FAM.eps_name(e) for e in LADDER]                    # eps0.5, eps0.55, ..., eps1.0, eps1.05, ..., eps1.5
SCHEMA = GEN.REPO / "physlaw" / "data" / "schema" / "labels.schema.json"


def _scenes(mod, cfg_path, sub, n=4, seed0=0):
    cfg = load_config(cfg_path)
    fmt = GEN.master_fmt(cfg)
    out = []
    for seed in range(seed0, seed0 + 40):
        sc = mod.sample(np.random.default_rng([seed, 0]), cfg, fmt, sub)
        if not isinstance(sc, str) and sc.inframe_reason() is None:
            out.append(sc)
        if len(out) == n:
            break
    assert len(out) == n, "sampler rejects almost everything"
    return out


@pytest.fixture(scope="module")
def f8():
    return _scenes(R, "configs/data/f8_ramp.yaml", "ramp")


@pytest.fixture(scope="module")
def f7b():
    return _scenes(F7, "configs/data/f7_freefall.yaml", "bounce")


def labels_of(sc, name):
    """labels.json content of one twin as generate.render_twins writes it (no frames, no masks)."""
    tw = sc.twins[name]
    T = tw.tracks[sc.bodies[0].name].p.shape[0]
    L = {"per_frame": GEN.per_frame_labels(sc, tw, sc.camera, None),
         "objects": {b.name: {} for b in sc.bodies},
         "scene_constants": {"num_frames": T, "frame_rate": sc.fps},
         "physlaw": {"schema_version": GEN.SCHEMA_VERSION, "family": sc.family, "twin": name, "twin_index": 0,
                     "twin_group_id": "g", "base_group_id": "g", "split": "train", "seed": 0, "nuisance": 0,
                     "master_format": {}, "config_hash": "x", "render_device": "n/a", "edit": tw.record,
                     **GEN.law_fields(sc, tw, sc.camera)}}
    return json.loads(json.dumps(L, default=GEN._json_default))


def test_config_ladders():
    # 21 rungs, step 0.05 (user decision 2026-11-11), also the samplers' default FAM.EPS_LADDER
    assert load_config("configs/data/f8_ramp.yaml").twins.epsilon_ladder == LADDER
    assert load_config("configs/data/f7_freefall.yaml").twins.bounce_ladder == LADDER
    assert load_config("configs/data/f9_pendulum.yaml").twins.swing_ladder == LADDER
    assert list(FAM.EPS_LADDER) == LADDER
    assert NAMES[:3] == ["eps0.5", "eps0.55", "eps0.6"] and NAMES[10] == "eps1.0" and NAMES[-1] == "eps1.5"
    assert [FAM.eps_name(e) for e in (0.5, 0.55, 1.0, 1.05, 1.25)] == ["eps0.5", "eps0.55", "eps1.0", "eps1.05", "eps1.25"]


def test_f8_ladder_closed_form(f8):
    for sc in f8:
        assert list(sc.twins) == NAMES and sc.event_kind == "pad"
        sc.check_pre_identity()                               # every pre-event frame byte-identical
        assert len({tw.pre_segment for tw in sc.twins.values()}) == 1
        h, g, k = sc.constants["h"], sc.gravity, sc.first_post
        s_in = math.sqrt(2 * g * h)
        assert 111 <= sc.fps * sc.t_event <= 128                # master event window (frames 112-128)
        ref = sc.twins["eps1.0"]
        for e, name in zip(LADDER, NAMES):
            tw = sc.twins[name]
            for key in "pvqw":
                assert np.array_equal(getattr(tw.tracks["A"], key)[:k], getattr(ref.tracks["A"], key)[:k])
            r = tw.record
            assert tw.epsilon == e and tw.legal == (e <= 1) and tw.violated_law == (None if e <= 1 else "energy")
            assert r["pad_entry_speed"] == pytest.approx(s_in, rel=1e-9)
            assert r["pad_exit_speed"] == pytest.approx(e * r["pad_entry_speed"], rel=1e-6)
            assert r["apex_height"] == pytest.approx(e * e * h, rel=1e-6)          # energy check, c = 1
            assert tw.clean_until_s == r["t_apex_s"]
            es = tw.event_state
            assert es["pre"]["t_s"] == pytest.approx(sc.t_event, abs=1e-9)
            assert np.linalg.norm(es["post"]["v"][0]) == pytest.approx(e * s_in, rel=1e-9)
            assert FAM.acceptance(FAM.View.from_scene(sc, tw)) == []
            # the block is in frame from frame 0 through its apex (inframe_reason honours clean_until_s)
            f_apex = int(math.floor(tw.clean_until_s * sc.fps))
            assert not FAM.outside(sc.camera, sc.bodies[0], tw.tracks["A"].p[:f_apex + 1]).any()
        assert sc.twins["eps1.5"].clean_until_s < (sc.twins["eps1.0"].tracks["A"].p.shape[0] - 1) / sc.fps
        assert math.isnan(sc.stop_dose["A"])                  # |v'| = eps s is smallest at eps = 0


def test_f8_block_never_enters_the_ramp(f8):
    """The block's bottom corners never go below the rendered surface (ramp meshes' profiles, else the floor) by more
    than the mesh's fillet chords' sagitta (256 chords per fillet: < 0.02 mm), and at least one corner touches it, in every frame up to the apex."""
    for sc in f8:
        s = sc.constants["half_size"]
        prof = [np.array(st.profile) for st in sc.statics if st.profile is not None]
        assert len(prof) == 2
        surf_z = lambda x: np.max([np.interp(x, q[:, 0], q[:, 1], left=0.0, right=0.0) for q in prof] + [0 * x], 0)
        for name in ("eps0.5", "eps1.0", "eps1.5"):
            tw = sc.twins[name]
            stop = int(math.floor(tw.clean_until_s * sc.fps)) + 1
            tr = tw.tracks["A"]
            th = 2 * np.arctan2(tr.q[:stop, 2], tr.q[:stop, 0])
            ex = np.stack([np.cos(th), -np.sin(th)], -1)
            ez = np.stack([np.sin(th), np.cos(th)], -1)
            c = tr.p[:stop][:, [0, 2]]
            gaps = np.stack([(c + sg * s * ex - s * ez)[:, 1] - surf_z((c + sg * s * ex - s * ez)[:, 0])
                             for sg in (-1, 1)], -1)
            assert gaps.min() > -5e-5, (name, gaps.min())
            assert np.abs(gaps).min(1).max() < 5e-5, (name, np.abs(gaps).min(1).max())


def test_f8_pad_is_identical_in_every_twin(f8):
    sc = f8[0]
    assert sum(st.name == "pad" for st in sc.statics) == 1 and all(tw.statics is None for tw in sc.twins.values())


def test_f7_bounce_ladder(f7b):
    for sc in f7b:
        assert list(sc.twins) == NAMES and sc.event_kind == "bounce"
        sc.check_pre_identity()
        H, g = sc.constants["drop_height"], sc.gravity
        for e, name in zip(LADDER, NAMES):
            tw = sc.twins[name]
            assert tw.epsilon == e and tw.legal == (e <= 1)
            pre, post = GEN.event_limits(sc, tw)["pre"], GEN.event_limits(sc, tw)["post"]
            assert -post["v"][0][2] / pre["v"][0][2] == pytest.approx(e, rel=1e-6)
            assert tw.record["apex_height"] == pytest.approx(e * e * H, rel=1e-12)
            b = tw.record["bounce_times_s"]
            assert tw.clean_until_s == pytest.approx(b[1])         # second floor contact
            # apex of the rebound (between the bounces) e^2 H
            t = np.arange(tw.tracks["A"].p.shape[0]) / sc.fps
            m = (t > b[0]) & (t < b[1])
            z = tw.tracks["A"].p[m, 2] - sc.constants["z_floor"]
            assert z.max() <= e * e * H + 1e-9 and z.max() > e * e * H * 0.98
            assert FAM.acceptance(FAM.View.from_scene(sc, tw)) == []
        assert math.isnan(sc.stop_dose["A"])


def test_f7_projectile_and_drop_unchanged():
    cfg = load_config("configs/data/f7_freefall.yaml")
    fmt = GEN.master_fmt(cfg)
    for sub, names in (("projectile", ["legal", "legal_e04", "gravity_x15", "vx_x06"]),
                       ("drop", ["legal", "legal_e04", "heavy_x13"])):
        sc = next(s for s in (F7.sample(np.random.default_rng([i, 0]), cfg, fmt, sub) for i in range(20))
                  if not isinstance(s, str))
        assert list(sc.twins) == names and sc.event_kind == "branch"


def test_stop_dose_analytic():
    # 1-D head-on collision, B at rest: v_A' = v (m_A - e m_B) / M stops at e = m_A / m_B
    m_a, m_b, v = 1.0, 2.5, 1.3
    post = lambda e: {"A": [v * (m_a - e * m_b) / (m_a + m_b), 0.0, 0.0], "B": [v * m_a * (1 + e) / (m_a + m_b), 0, 0]}
    sd = FAM.stop_dose({e: post(e) for e in (0.0, 0.7, 1.5)})
    assert sd["A"] == pytest.approx(m_a / m_b, rel=1e-12)
    assert math.isnan(sd["B"])                           # |v_B'| grows with e: minimum at the ladder end
    # oblique: v'(e) = v0 + e d, minimiser -v0.d/|d|^2
    v0, d = np.array([0.3, -1.0, 0.2]), np.array([-0.4, 1.1, 0.0])
    sd = FAM.stop_dose({0.5: {"X": v0 + 0.5 * d}, 1.5: {"X": v0 + 1.5 * d}})
    assert sd["X"] == pytest.approx(-(v0 @ d) / (d @ d), rel=1e-12)
    assert math.isnan(FAM.stop_dose({0.0: post(0.0), 0.3: post(0.3)})["A"])   # minimiser past the ladder end


def test_labels_carry_energy_fields_and_validate(f8, f7b):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text())
    for sc in (f8[0], f7b[0]):
        for name in ("eps0.5", "eps1.0", "eps1.5"):
            L = labels_of(sc, name)
            pl = L["physlaw"]
            jsonschema.validate(L, schema)
            assert pl["event_kind"] == sc.event_kind and pl["epsilon"] == float(name[3:])
            assert pl["clean_until_s"] == sc.twins[name].clean_until_s and math.isnan(pl["stop_dose"]["A"])
            assert pl["legal"] == (pl["epsilon"] <= 1)
            assert FAM.acceptance(FAM.View.from_labels(L)) == []
    pl = labels_of(f8[0], "eps1.2")["physlaw"]     # F8 event_state: entering / leaving the pad
    rec = f8[0].twins["eps1.2"].record
    assert pl["event_state"]["post"]["t_s"] == pytest.approx(rec["t_pad_exit_s"])
    assert np.linalg.norm(pl["event_state"]["post"]["v"][0]) / np.linalg.norm(pl["event_state"]["pre"]["v"][0]) \
        == pytest.approx(1.2, rel=1e-9)
    bad = labels_of(f8[0], "eps1.0")
    del bad["physlaw"]["clean_until_s"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_collision_energy_labels_from_f1_layout():
    """FAM.collision_energy_labels on a labels-1-shaped dict (what the frozen F1 render carries)."""
    n = np.array([0.6, 0.8])
    va, vb = np.array([1.0, 0.5]), np.array([-0.2, 0.1])
    u = (va - vb) @ n
    e = 0.15
    m_a = m_b = 1.0
    dv = (1 + e) * u * m_b / (m_a + m_b)
    L = {"scene_constants": {"num_frames": 289, "frame_rate": 48},
         "physlaw": {"contact_normal": n.tolist(), "pre_velocity": {"A": va.tolist(), "B": vb.tolist()},
                     "post_velocity": {"A": (va - dv * n).tolist(), "B": (vb + dv * n).tolist()},
                     "inframe_until": {"A": 289, "B": 200}}}
    out = FAM.collision_energy_labels(L)
    assert out["event_kind"] == "collision" and out["epsilon"] == pytest.approx(e, rel=1e-12)
    assert out["clean_until_s"] == pytest.approx(199 / 48)
    L["physlaw"]["inframe_until"]["B"] = 289
    assert FAM.collision_energy_labels(L)["clean_until_s"] == pytest.approx(288 / 48)
