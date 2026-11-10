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


def test_f8_slope_twins_share_h(group):
    if family(group) != "f8":
        pytest.skip("f8 only")
    ls = [labels(t) for t in twins(group)]
    hs = {l["physlaw"]["constants"]["h"] for l in ls}
    z0 = {l["physlaw"]["pre_state"]["p"][0][2] for l in ls}
    assert len(hs) == 1 and len(z0) == 1, (hs, z0)
    for l in ls:
        if l["physlaw"]["legal"]:
            c = l["physlaw"]["constants"]
            assert l["physlaw"]["edit"]["v_bottom"] == pytest.approx(math.sqrt(2 * c["g"] * c["h"]), rel=1e-6)


def test_f4_invariance_across_boosts(group):
    """Restitution and the centre-of-mass-frame scattering angle are boost-invariant to 1e-6."""
    if family(group) != "f4":
        pytest.skip("f4 only")
    by_e = defaultdict(list)
    for t in twins(group):
        s = labels(t)["physlaw"]["summary"]
        by_e[t.name.split("_")[-1]].append((s["restitution_realised"], s["com_frame_angle"]))
    for e, vals in by_e.items():
        assert len(vals) == 3, e
        for r, ang in vals[1:]:
            assert abs(r - vals[0][0]) < 1e-6 and abs(ang - vals[0][1]) < 1e-6, (e, vals)


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
    t = twins(group)[0]
    L = labels(t)
    cam = L["physlaw"]["camera"]
    K, E = np.array(cam["K"]), np.array(cam["world_to_camera"])
    for f in (0, len(L["per_frame"]) // 2):
        X = np.array(L["per_frame"][f]["A"]["position"] + [1.0])
        x = K @ (E @ X)[:3]
        assert np.allclose(x[:2] / x[2], L["per_frame"][f]["A"]["pixel"][:2], atol=0.5)
