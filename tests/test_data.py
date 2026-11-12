"""Spec §2.4 data acceptance tests, run against a rendered family directory (any family).

    PHYSLAW_DATA=data/f1 pytest tests/test_data.py

Skipped when PHYSLAW_DATA is unset. scripts/02_validate_data.py runs this and
also writes <data>/_validation/report.json with the measured numbers. Nuisance
variants (<group>_nuisance) are groups of their own and pass the same tests.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = os.environ.get("PHYSLAW_DATA")
pytestmark = pytest.mark.skipif(not ROOT, reason="set PHYSLAW_DATA to a rendered family dir")

SCHEMA = Path(__file__).resolve().parents[1] / "physlaw" / "data" / "schema" / "labels.schema.json"
VIOLATION_MIN = float(os.environ.get("PHYSLAW_VIOLATION_MIN", "0.05"))
COLLISION = ("f1", "f2", "f3", "f4")
RGB_TOL = float(os.environ.get("PHYSLAW_RGB_TOL", "0.04"))   # |mean RGB A - B| / 255, identical-look bodies


def groups():
    if not ROOT:
        return []
    return sorted(p for p in Path(ROOT).iterdir() if (p / "_GROUP_COMPLETE").exists())


def twins(g: Path):
    return sorted((p for p in g.iterdir() if (p / "_COMPLETE").exists()),
                  key=lambda p: json.loads((p / "labels.json").read_text())["physlaw"]["twin_index"])


def labels(t: Path) -> dict:
    return json.loads((t / "labels.json").read_text())


def family(g: Path) -> str:
    return labels(twins(g)[0])["physlaw"]["family"]


def segments(g: Path) -> dict:
    """pre_segment -> twin dirs (labels-1 has one segment)."""
    out = defaultdict(list)
    for t in twins(g):
        out[labels(t)["physlaw"].get("pre_segment", "0")].append(t)
    return out


def collision_only(g: Path):
    if family(g) not in COLLISION:
        pytest.skip("collision families only")


def decode(path: Path) -> np.ndarray:
    import imageio.v3 as iio
    return iio.imread(path, plugin="FFMPEG") if hasattr(iio, "imread") else None


@pytest.fixture(scope="module", params=groups(), ids=lambda p: p.name)
def group(request):
    return request.param


def test_groups_exist():
    assert groups(), f"no completed groups under {ROOT}"


def test_labels_validate_against_schema(group):
    import jsonschema
    schema = json.loads(SCHEMA.read_text())
    for t in twins(group):
        jsonschema.validate(labels(t), schema)


def test_twins_share_group_and_split(group):
    ls = [labels(t)["physlaw"] for t in twins(group)]
    assert len({l["twin_group_id"] for l in ls}) == 1
    assert len({l["split"] for l in ls}) == 1
    assert len({l["first_post_contact_frame"] for l in ls}) == 1


def test_pre_contact_frames_identical_raw(group):
    """Raw rendered RGB (hashes recorded before encoding) identical before the event, within a pre_segment."""
    for seg in segments(group).values():
        ls = [labels(t)["physlaw"] for t in seg]
        if ls[0]["frame_sha256_16"] is None:
            pytest.skip("physics-only group")
        k = ls[0]["first_post_contact_frame"]
        for l in ls[1:]:
            assert l["frame_sha256_16"][:k] == ls[0]["frame_sha256_16"][:k], l["twin"]
            assert l["frame_sha256_16"][k:] != ls[0]["frame_sha256_16"][k:], f"{l['twin']} post frames identical to twin 0"


def test_pre_contact_frames_identical_decoded(group):
    """And identical after h264 encode + decode (the encoder is causal by construction)."""
    for seg in segments(group).values():
        if not (seg[0] / "video.mp4").exists():
            pytest.skip("physics-only group")
        k = labels(seg[0])["physlaw"]["first_post_contact_frame"]
        ref = decode(seg[0] / "video.mp4")[:k]
        for t in seg[1:]:
            v = decode(t / "video.mp4")
            assert v.shape[0] == labels(t)["scene_constants"]["num_frames"]
            assert np.array_equal(v[:k], ref), f"{t.name}: max |diff| {np.abs(v[:k].astype(int) - ref).max()}"


def test_pre_event_states_identical(group):
    """Labelled trajectories identical before the event within a pre_segment (all families)."""
    for seg in segments(group).values():
        k = labels(seg[0])["physlaw"]["first_post_contact_frame"]
        ref = labels(seg[0])["per_frame"][:k]
        for t in seg[1:]:
            for a, b in zip(labels(t)["per_frame"][:k], ref):
                for obj in labels(t)["physlaw"]["objects"]:
                    assert a[obj]["position"] == b[obj]["position"], (t.name, a["frame"], obj)


def test_momentum_residuals(group):
    collision_only(group)
    for t in twins(group):
        l = labels(t)["physlaw"]
        r = l["summary"]["momentum_residual"]
        if l["momentum_conserved"]:
            assert r < 1e-6, (t.name, r)
        else:
            assert r > VIOLATION_MIN, (t.name, r)


def test_momentum_from_per_frame_labels(group):
    """Residual recomputed from the per-frame trajectory, not the summary."""
    collision_only(group)
    for t in twins(group):
        L = labels(t)
        pl = L["physlaw"]
        mA, mB = pl["objects"]["A"]["mass"], pl["objects"]["B"]["mass"]
        k = pl["first_post_contact_frame"]
        fr = L["per_frame"]
        pre = mA * np.array(fr[1]["A"]["velocity"]) + mB * np.array(fr[1]["B"]["velocity"])
        post = mA * np.array(fr[-1]["A"]["velocity"]) + mB * np.array(fr[-1]["B"]["velocity"])
        r = np.linalg.norm(post - pre) / np.linalg.norm(pre)
        assert (r < 1e-6) == pl["momentum_conserved"], (t.name, r)
        # positions follow the velocities after contact
        dt = 1.0 / L["scene_constants"]["frame_rate"]
        pa = np.array([f["A"]["position"] for f in fr[k:]])
        assert np.allclose(np.diff(pa, axis=0) / dt, np.array(fr[-1]["A"]["velocity"]), atol=1e-6)


def test_energy_create_creates_energy(group):
    for t in twins(group):
        l = labels(t)["physlaw"]
        if l["twin"] == "energy_create":
            assert l["summary"]["ke_ratio"] > 1.0
            assert l["summary"]["momentum_residual"] < 1e-6


def test_restitution_realised(group):
    collision_only(group)
    for t in twins(group):
        l = labels(t)["physlaw"]
        assert l["summary"]["restitution_realised"] == pytest.approx(l["e_target"], abs=1e-6), t.name


def test_contact_frame_in_range(group):
    for t in twins(group):
        L = labels(t)
        l = L["physlaw"]
        if l["family"] in COLLISION:
            assert abs(l["contact_frame_actual"] - l["contact_frame"]) <= l["master_format"]["contact_tolerance_frames"]
        else:   # event window of the master format (frames 112-128 at 48 fps)
            lo, hi = l["master_format"]["event_frame_range"]
            assert lo - 1 <= l["event_time_s"] * l["constants"]["fps"] <= hi + 1e-9, l["event_time_s"]


def test_family_laws(group):
    """Spec §2.4: legal twins exact to 1e-6, each violation breaks its target law by its factor and the
    other laws hold (families.<fk>.acceptance, recomputed from the per-frame labels)."""
    if family(group) in COLLISION:
        pytest.skip("collision families: momentum/restitution tests above")
    from physlaw.data import families as FAM
    for t in twins(group):
        assert FAM.acceptance(FAM.View.from_labels(labels(t))) == [], t.name


def test_law_inputs(group):
    """io.law_inputs gives (pre, post, ctx) at the event for every twin (the eval laws' input):
    positions continuous through the event, legal collisions / bounces exact at the event instant."""
    from physlaw.data.io import law_inputs
    for t in twins(group):
        L = labels(t)
        pre, post, ctx = law_inputs(L)
        n = sum(not o.get("aux") for o in L["physlaw"]["objects"].values())     # aux rods / props are dropped
        assert pre["v"].shape == post["v"].shape == (n, 3) and ctx["m"].shape == (n,)
        if L["physlaw"].get("event_kind") != "swing":
            assert pre["t_s"] == pytest.approx(ctx["event_time_s"])
        if L["physlaw"].get("event_kind") == "pad":    # f8: post = the block leaving the pad
            assert post["t_s"] == pytest.approx(L["physlaw"]["edit"]["t_pad_exit_s"])
            assert np.linalg.norm(post["v"]) / np.linalg.norm(pre["v"]) == pytest.approx(L["physlaw"]["epsilon"], rel=1e-6)
        elif L["physlaw"].get("event_kind") == "swing":    # f9 swing: the release apex and the next apex (v = 0)
            from physlaw.eval.laws import epsilon
            assert pre["t_s"] == pytest.approx(ctx["release_time_s"])
            assert post["t_s"] == pytest.approx(L["physlaw"]["edit"]["t_apex_after_s"])
            assert epsilon(pre, post, ctx) == pytest.approx(L["physlaw"]["epsilon"], rel=1e-6)
            continue
        else:
            assert post["t_s"] == pytest.approx(ctx["event_time_s"])
            assert np.allclose(pre["p"], post["p"], atol=1e-9), t.name
        if ctx.get("subfamily") == "bounce":
            e = -post["v"][0, 2] / pre["v"][0, 2]
            assert e == pytest.approx(L["physlaw"]["epsilon"], rel=1e-6), t.name
        _, post2, ctx2 = law_inputs(L, dt=0.25)
        assert ctx2["dt"] == pytest.approx(0.25, abs=0.6 / ctx["fps"])


def test_new_law_statistics(group):
    """F6 glide / slide / pair, F9, F11, F12 (labels -> io.law_inputs -> physlaw.eval.laws): legal twins at the legal value of
    their law's statistics, each violator off it in its own statistic, every other statistic legal."""
    from physlaw.data.io import law_inputs
    from physlaw.eval import laws as L
    pl0 = labels(twins(group)[0])["physlaw"]
    fam, sub = pl0["family"], pl0.get("subfamily", "")              # labels-1 F1 has no sub-family
    if not (fam in ("f9", "f11", "f12") or fam == "f6"):
        pytest.skip("no new-law statistics")
    for t in twins(group):
        lab = labels(t)
        pl = lab["physlaw"]
        if fam == "f6" and sub in ("slide", "pair"):
            pre, post, ctx = law_inputs(lab)
            st, law, tw = L.friction.stats(pre, post, ctx), "friction", pl["twin"]
            viol = {"reverse": {"stop_hold"}, "viscous": {"decel_const", "stop_hold"},
                    "heavy_x13": {"mass_ratio", "decel_const"}}.get(tw, set())
            if tw.startswith("decel_x"):
                viol = {"decel_const"} | ({"stop_hold"} if float(tw[7:]) < 1 else set())
        else:
            pre, post, ctx = law_inputs(lab)
            law = {"f6": "momentum", "f12": "angular_momentum", "f11": "statics"}.get(fam) or (
                "oscillation" if sub == "pair" else "energy")
            st = L.LAWS[law].stats(pre, post, ctx)
            viol = {"momentum": {"inertia_turn" if pl["twin"].startswith("turn") else "inertia_speed"},
                    "angular_momentum": {"L_ratio"}, "statics": {"tip_consistency", "alpha_ratio"},
                    "energy": {"epsilon"},
                    "oscillation": {"mass_period_ratio", "period_ratio"} if pl["twin"].startswith("heavy")
                    else {"period_ratio"}}[law]
            if pl["legal"]:
                viol = set()
        for k, (op, v) in L.LAWS[law].legal.items():
            x = st[k]
            if not np.isfinite(x):
                continue
            ok = abs(x - v) < 1e-5 if op == "eq" else x <= v + 1e-6
            assert ok == (k not in viol), (t.name, k, x)


def test_energy_ladder_labels(group):
    """Exchange families (spec §5.0.2): every twin carries event_kind, epsilon and clean_until_s (after the
    event); F2 / F7 bounce / F8 carry one stop_dose per group; F8 ladder twins share h and the release."""
    from physlaw.data import families as FAM
    ls = []
    for t in twins(group):
        L = labels(t)
        if L["physlaw"]["schema_version"] == "physlaw-labels-1":     # frozen F1: derived offline
            L["physlaw"].update(event_time_s=L["physlaw"]["contact_frame_actual"] / L["scene_constants"]["frame_rate"],
                                **FAM.collision_energy_labels(L))
        ls.append(L["physlaw"])
    if ls[0].get("event_kind") not in ("collision", "bounce", "pad"):
        pytest.skip("no energy exchange event")
    for pl in ls:
        assert pl["clean_until_s"] > pl["event_time_s"] and math.isfinite(pl["epsilon"])
        if pl["event_kind"] != "collision":
            assert pl["legal"] == (pl["epsilon"] <= 1.0)
    if ls[0]["family"] in ("f2", "f8") or ls[0]["event_kind"] in ("bounce", "swing"):
        assert len({json.dumps(pl["stop_dose"]) for pl in ls}) == 1
    if ls[0]["family"] == "f8":
        assert len({pl["constants"]["h"] for pl in ls}) == 1 and len({json.dumps(pl["pre_state"]) for pl in ls}) == 1
        for pl in ls:
            es = pl["event_state"]
            ratio = np.linalg.norm(es["post"]["v"][0]) / np.linalg.norm(es["pre"]["v"][0])
            assert ratio == pytest.approx(pl["epsilon"], rel=1e-6)


def test_f4_invariance_across_boosts(group):
    """Restitution and the centre-of-mass-frame scattering angle are boost-invariant to 1e-6."""
    if family(group) != "f4":
        pytest.skip("f4 only")
    by_e = defaultdict(list)
    for t in twins(group):
        s = labels(t)["physlaw"]["summary"]
        by_e[t.name.split("_")[-1]].append((s["restitution_realised"], s["com_frame_angle"]))
    for e, vals in by_e.items():
        assert len(vals) in (3, 5), e            # u0, usmall, ularge (+ cam_usmall, cam_ularge)
        for r, ang in vals[1:]:
            assert abs(r - vals[0][0]) < 1e-6 and abs(ang - vals[0][1]) < 1e-6, (e, vals)


def test_f4_camera_boost_matches_ball_boost(group):
    """F4 camera boosts (cam_<level>_<e>): world states = the u0 twin's; camera-frame (primary) states = the ball-boost
    twin's; projected ball centres equal the ball-boost twin's to < 0.5 px in every frame up to clean_until_s (and
    the rendered masks' centroids agree to < 1 px where both balls are wholly inside); the world positions projected
    through the labelled camera trajectory give the same pixels; boost_kind / frame labels; law_inputs in both frames."""
    from physlaw.data.io import law_inputs
    if family(group) != "f4":
        pytest.skip("f4 only")
    by = {t.name: t for t in twins(group)}
    cams = [n for n in by if n.startswith("cam_")]
    if not cams:
        pytest.skip("no camera-boost twins (rendered before 2026-11-12)")
    for n in cams:
        Lc, Lb, L0 = labels(by[n]), labels(by[n[len("cam_"):]]), labels(by["u0_" + n.split("_")[-1]])
        pc, pb = Lc["physlaw"], Lb["physlaw"]
        assert (pc["boost_kind"], pc["frame"], pb["boost_kind"], pb["frame"]) == ("camera", "camera", "ball", "world")
        assert pc["edit"]["boost"] == pb["edit"]["boost"]
        assert np.allclose(pc["camera_trajectory"]["velocity_world"][:2], -np.asarray(pb["edit"]["boost"]), atol=1e-12)
        fps = Lc["scene_constants"]["frame_rate"]
        stop = int(math.floor(min(pc["clean_until_s"], pb["clean_until_s"]) * fps + 1e-9)) + 1
        assert pc["clean_until_s"] == pytest.approx(pb["clean_until_s"])
        K, E = np.array(pc["camera"]["K"]), np.array(pc["camera"]["world_to_camera"])
        traj = np.array(pc["camera_trajectory"]["position_world"])
        c0 = np.array(pc["camera_trajectory"]["position_at_t_ref"])
        for f in range(stop):
            fc, fb, f0 = Lc["per_frame"][f], Lb["per_frame"][f], L0["per_frame"][f]
            for o in "AB":
                assert np.abs(np.subtract(fc[o]["pixel"], fb[o]["pixel"])).max() < 0.5, (n, f, o)
                assert np.allclose(fc[o]["position"], fb[o]["position"], atol=1e-9), (n, f, o)
                assert np.allclose(fc[o]["velocity"], fb[o]["velocity"], atol=1e-9), (n, f, o)
                assert np.allclose(fc[o]["position_world"], f0[o]["position"], atol=1e-12), (n, f, o)
                x = K @ (E @ np.r_[np.asarray(fc[o]["position_world"]) - (traj[f] - c0), 1.0])[:3]
                assert np.allclose(x[:2] / x[2], fc[o]["pixel"], atol=1e-6), (n, f, o)
        for k_ in ("pre_velocity", "post_velocity"):
            for o in "AB":
                assert np.allclose(pc[k_][o], pb[k_][o], atol=1e-9)
                assert np.allclose(pc[f"{k_}_world"][o][:2], L0["physlaw"][k_][o][:2], atol=1e-9)
        pre, post, ctx = law_inputs(Lc)
        preb, postb, _ = law_inputs(Lb)
        prew, postw, ctxw = law_inputs(Lc, frame="world")
        pre0, post0, _ = law_inputs(L0)
        assert ctx["frame"] == "camera" and ctxw["frame"] == "world" and ctx["boost_kind"] == "camera"
        assert np.allclose(pre["v"], preb["v"], atol=1e-9) and np.allclose(post["v"], postb["v"], atol=1e-9)
        assert np.allclose(prew["v"], pre0["v"], atol=1e-9) and np.allclose(postw["v"], post0["v"], atol=1e-9)
        if (by[n] / "masks.npz").exists():
            mc, mb = np.load(by[n] / "masks.npz")["instance"], np.load(by[n[len("cam_"):]] / "masks.npz")["instance"]
            for f in range(0, stop, 4):
                for o in "AB":
                    sid = pc["objects"][o]["segmentation_id"]
                    yc, xc = np.nonzero(mc[f] == sid)
                    yb, xb = np.nonzero(mb[f] == sid)
                    if not xc.size or not xb.size or min(xc.min(), yc.min(), xb.min(), yb.min()) == 0 or \
                            xc.max() == mc.shape[2] - 1 or xb.max() == mb.shape[2] - 1 or \
                            yc.max() == mc.shape[1] - 1 or yb.max() == mb.shape[1] - 1:
                        continue
                    assert abs(xc.mean() - xb.mean()) < 1.0 and abs(yc.mean() - yb.mean()) < 1.0, (n, f, o)


def test_identical_look_mean_rgb(group):
    """F3 and F7 drop pairs: the two bodies look the same (per-object mean RGB inside the masks)."""
    ls = labels(twins(group)[0])["physlaw"]
    if not (ls["family"] == "f3" or ls.get("subfamily") == "drop"):
        pytest.skip("identical-look families only")
    t = twins(group)[0]
    if not (t / "video.mp4").exists():
        pytest.skip("physics-only group")
    v, m = decode(t / "video.mp4"), np.load(t / "masks.npz")["instance"]
    diffs = []
    for f in range(0, ls["first_post_contact_frame"], 8):
        a, b = m[f] == 1, m[f] == 2
        if a.sum() < 30 or b.sum() < 30:
            continue
        diffs.append(np.abs(v[f][a].mean(0) - v[f][b].mean(0)).max() / 255.0)
    assert diffs, "no frame shows both bodies"
    assert np.median(diffs) < RGB_TOL, f"median |mean RGB A - B| = {np.median(diffs):.3f}"


def test_masks_match_labels(group):
    ts = twins(group)
    if not (ts[0] / "masks.npz").exists():
        pytest.skip("physics-only group")
    for t in ts:
        L = labels(t)
        m = np.load(t / "masks.npz")["instance"]
        assert m.shape[0] == L["scene_constants"]["num_frames"]
        tol = 3.0 * max(1.0, L["physlaw"]["camera"]["resolution"][0] / 832)
        for f in (0, L["physlaw"]["first_post_contact_frame"] + 5, m.shape[0] - 1):
            for obj, o in L["physlaw"]["objects"].items():
                sid = o["segmentation_id"]
                if not L["per_frame"][f][obj]["in_frame"]:
                    continue
                ys, xs = np.nonzero(m[f] == sid)
                assert xs.size > 0, (t.name, f, obj)
                if ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[1] - 1 or xs.max() == m.shape[2] - 1:
                    continue   # partly outside the image: `in_frame` is centre-based, centroid is biased
                px = L["per_frame"][f][obj]["pixel"]
                assert abs(xs.mean() - px[0]) < tol and abs(ys.mean() - px[1]) < tol, (t.name, f, obj, xs.mean(), ys.mean(), px)


def test_camera_projects_to_pixel_labels(group):
    """The labelled camera projects the primary positions to the pixel labels (a moving camera, F4 camera boosts:
    primary positions are camera-frame, so the camera at t_ref serves every frame)."""
    for t in twins(group)[:1] + [t for t in twins(group) if t.name.startswith("cam_")]:
        L = labels(t)
        cam = L["physlaw"]["camera"]
        K, E = np.array(cam["K"]), np.array(cam["world_to_camera"])
        for f in (0, len(L["per_frame"]) // 2):
            X = np.array(L["per_frame"][f]["A"]["position"] + [1.0])
            x = K @ (E @ X)[:3]
            assert np.allclose(x[:2] / x[2], L["per_frame"][f]["A"]["pixel"][:2], atol=0.5), (t.name, f)
