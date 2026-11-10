"""Spec §2.3 data acceptance tests, run against a rendered family directory.

    PHYSLAW_DATA=data/f1 pytest tests/test_data.py

Skipped when PHYSLAW_DATA is unset. scripts/02_validate_data.py runs this and
also writes <data>/_validation/report.json with the measured numbers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = os.environ.get("PHYSLAW_DATA")
pytestmark = pytest.mark.skipif(not ROOT, reason="set PHYSLAW_DATA to a rendered family dir")

SCHEMA = Path(__file__).resolve().parents[1] / "physlaw" / "data" / "schema" / "labels.schema.json"
VIOLATION_MIN = float(os.environ.get("PHYSLAW_VIOLATION_MIN", "0.05"))


def groups():
    if not ROOT:
        return []
    return sorted(p for p in Path(ROOT).iterdir() if (p / "_GROUP_COMPLETE").exists())


def twins(g: Path):
    return sorted((p for p in g.iterdir() if (p / "_COMPLETE").exists()),
                  key=lambda p: json.loads((p / "labels.json").read_text())["physlaw"]["twin_index"])


def labels(t: Path) -> dict:
    return json.loads((t / "labels.json").read_text())


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
    """Raw rendered RGB (hashes recorded before encoding) identical before contact."""
    ls = [labels(t)["physlaw"] for t in twins(group)]
    if ls[0]["frame_sha256_16"] is None:
        pytest.skip("physics-only group")
    k = ls[0]["first_post_contact_frame"]
    for l in ls[1:]:
        assert l["frame_sha256_16"][:k] == ls[0]["frame_sha256_16"][:k], l["twin"]
        assert l["frame_sha256_16"][k:] != ls[0]["frame_sha256_16"][k:], f"{l['twin']} post frames identical to twin 0"


def test_pre_contact_frames_identical_decoded(group):
    """And identical after h264 encode + decode (the encoder is causal by construction)."""
    ts = twins(group)
    if not (ts[0] / "video.mp4").exists():
        pytest.skip("physics-only group")
    k = labels(ts[0])["physlaw"]["first_post_contact_frame"]
    ref = decode(ts[0] / "video.mp4")[:k]
    for t in ts[1:]:
        v = decode(t / "video.mp4")
        assert v.shape[0] == labels(t)["scene_constants"]["num_frames"]
        assert np.array_equal(v[:k], ref), f"{t.name}: max |diff| {np.abs(v[:k].astype(int) - ref).max()}"


def test_momentum_residuals(group):
    for t in twins(group):
        l = labels(t)["physlaw"]
        r = l["summary"]["momentum_residual"]
        if l["momentum_conserved"]:
            assert r < 1e-6, (t.name, r)
        else:
            assert r > VIOLATION_MIN, (t.name, r)


def test_momentum_from_per_frame_labels(group):
    """Residual recomputed from the per-frame trajectory, not the summary."""
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
    for t in twins(group):
        l = labels(t)["physlaw"]
        assert l["summary"]["restitution_realised"] == pytest.approx(l["e_target"], abs=1e-6), t.name


def test_contact_frame_in_range(group):
    for t in twins(group):
        L = labels(t)
        l = L["physlaw"]
        assert abs(l["contact_frame_actual"] - l["contact_frame"]) <= l["master_format"]["contact_tolerance_frames"]


def test_masks_match_labels(group):
    ts = twins(group)
    if not (ts[0] / "masks.npz").exists():
        pytest.skip("physics-only group")
    for t in ts:
        L = labels(t)
        m = np.load(t / "masks.npz")["instance"]
        assert m.shape[0] == L["scene_constants"]["num_frames"]
        for f in (0, L["physlaw"]["first_post_contact_frame"] + 5, m.shape[0] - 1):
            for obj, sid in (("A", 1), ("B", 2)):
                if not L["per_frame"][f][obj]["in_frame"]:
                    continue
                ys, xs = np.nonzero(m[f] == sid)
                assert xs.size > 0, (t.name, f, obj)
                if ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[1] - 1 or xs.max() == m.shape[2] - 1:
                    continue   # partly outside the image: `in_frame` is centre-based, centroid is biased
                px = L["per_frame"][f][obj]["pixel"]
                assert abs(xs.mean() - px[0]) < 3.0 and abs(ys.mean() - px[1]) < 3.0, (t.name, f, obj)


def test_camera_projects_to_pixel_labels(group):
    t = twins(group)[0]
    L = labels(t)
    cam = L["physlaw"]["camera"]
    K, E = np.array(cam["K"]), np.array(cam["world_to_camera"])
    for f in (0, len(L["per_frame"]) // 2):
        X = np.array(L["per_frame"][f]["A"]["position"] + [1.0])
        x = K @ (E @ X)[:3]
        assert np.allclose(x[:2] / x[2], L["per_frame"][f]["A"]["pixel"][:2], atol=0.5)
