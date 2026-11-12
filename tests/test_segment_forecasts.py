"""One forecast per (group, pre_segment): stage 4 (scripts/04_cache_activations.py) on a fake adapter, the guard,
and the readers (l1_forecast_laws.forecast_path / _load_spaces, l3_invariance.forecast_spaces)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physlaw.config import load_config  # noqa: E402
from physlaw.eval import l1_forecast_laws as L1  # noqa: E402
from physlaw.eval import l3_invariance as L3  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T, K, TUB, D = 8, 4, 2, 6                     # frames, context frames, tubelet, width
PHASES = ["pre", "pre", "post", "post"]       # token-frames
# F8-like group: three ramp geometries (pre_segments a / b / k), a twice; F1-like group: one segment ("0")
GROUPS = {"f8_000001": {"legal_slope_a": "a", "legal_slope_b": "b", "legal_kink": "k", "energy_x125": "a"},
          "f1_000002": {"conserve_e1": "0", "conserve_e015": "0", "energy_create": "0"}}
PRE_VALUE = {"a": 10.0, "b": 20.0, "k": 30.0, "0": 40.0}          # pre-event frame value of each segment


def _stage4():
    spec = importlib.util.spec_from_file_location("stage4", REPO / "scripts" / "04_cache_activations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip(path, frames=True, masks=True):
    g, t = path.parent.name, path.name
    s = GROUPS[g][t]
    pl = {"twin_group_id": g, "twin": t, "split": "train", "family": g.split("_")[0], "first_post_contact_frame": K,
          "pre_segment": s}
    fr = np.full((T, 4, 4, 3), PRE_VALUE[s], np.float32)
    fr[K:] = 1.0 + list(GROUPS[g]).index(t)                          # post-event frames differ per twin
    return SimpleNamespace(labels={"physlaw": pl, "scene_constants": {"frame_rate": 4}}, pl=pl, frames=fr if frames else None,
                           masks=np.zeros((T, 4, 4)) if masks else None)


class FakeModel:
    """Tokens = mean of the input frames; forecast() = the mean of the CONTEXT frames (identifies the segment)."""
    token_grid = SimpleNamespace(tubelet=TUB, token_frames=lambda n: [list(range(i, i + TUB)) for i in range(0, n, TUB)])
    per_frame = predicts = True
    fps = 4

    def __init__(self, diffusion):
        self.diffusion, self.n_forecasts = diffusion, 0

    def input_frames(self, labels):
        return np.arange(T)

    def context_frames(self, labels):
        return K

    def to_tensor(self, frames):
        return torch.from_numpy(np.asarray(frames, np.float32))

    def encode(self, x):
        tok = x.reshape(T // TUB, TUB, -1).mean((1, 2))[:, None, None, None].expand(T // TUB, 2, 2, D).clone()
        out = {0: tok, -1: tok * 2}
        if self.diffusion:
            out["latent/snr+8"] = tok[..., :3].clone()
        return out

    def surprise(self, x, k, roi):
        n = (T - k) // TUB
        res = {"mse_all": 1.5}
        if self.per_frame:
            res["per_frame"] = {"surprise": {"mse_all": [1.0, 2.0][:n]}, "weight": {"mse_all": [1.0] * n}}
        return res

    def forecast(self, x, k, seed=0, return_video=False):
        self.n_forecasts += 1
        F = x[:k].mean().expand(T // TUB - k // TUB, 2, 2, D).clone()
        if return_video:
            return {"snr0/L1": F}, np.zeros((4, 16, 16, 3), np.uint8)
        return {-1: F}


def _run(tmp_path, monkeypatch, diffusion, extra=(), model=None):
    S4 = _stage4()
    data = tmp_path / "data"
    clips = [data / g / t for g in GROUPS for t in GROUPS[g]]
    yml = tmp_path / "fake.yaml"
    yml.write_text(f"name: fake\nkind: {'diffusion' if diffusion else 'encoder'}\n")
    m = model or FakeModel(diffusion)
    roi = SimpleNamespace(roi=np.zeros((T // TUB, 2, 2), bool))
    monkeypatch.setattr(S4, "list_clips", lambda d: clips)
    monkeypatch.setattr(S4, "load_clip", _clip)
    monkeypatch.setattr(S4, "load_adapter", lambda cfg, need=None: m)
    monkeypatch.setattr(S4, "project_masks", lambda masks, fr, grid, ids: {"A": roi, "B": roi})
    monkeypatch.setattr(S4, "group_object_region", lambda clips, fr, grid, t_end, fps: roi.roi[0])
    monkeypatch.setattr(S4, "token_frame_phase", lambda fr, grid, k: PHASES)
    monkeypatch.setattr(S4, "pool_weights", lambda obj, ph: {
        f"frame.{w}": (np.ones((len(ph), 2, 2)) * np.array([p == w or w == "all" for p in ph])[:, None, None]
                       ).astype(np.float32)
        for w in ("pre", "post", "all")})
    out = tmp_path / ("cache_dit" if diffusion else "cache_enc")
    monkeypatch.setattr(sys, "argv", ["04", "--model", str(yml), "--data", str(data), "--out", str(out), *extra])
    S4.main()
    return out, m


def test_encoder_one_forecast_file_per_segment(tmp_path, monkeypatch):
    out, m = _run(tmp_path, monkeypatch, diffusion=False)
    g8, g1 = list(GROUPS)
    assert sorted(f.name for f in (out / g8).glob("_forecast*.npz")) == ["_forecast_a.npz", "_forecast_b.npz",
                                                                         "_forecast_k.npz"]
    for s in "abk":
        z = np.load(L1.forecast_path(out, g8, s))
        assert np.allclose(z["final"], PRE_VALUE[s]) and str(z["pre_segment"]) == s
    # F1-like group: the one _forecast.npz, exactly as before (no extra keys)
    assert [f.name for f in (out / g1).glob("_forecast*.npz")] == ["_forecast.npz"]
    z = np.load(L1.forecast_path(out, g1, None))
    assert sorted(z.files) == ["context_frames", "final"] and np.allclose(z["final"], PRE_VALUE["0"])
    assert m.n_forecasts == 4            # once per forecast file (skipped when it exists), as before
    # rerun: nothing recomputed
    m2 = FakeModel(False)
    _run(tmp_path, monkeypatch, diffusion=False, model=m2)
    assert m2.n_forecasts == 0


def test_encoder_backfills_segment_forecasts_of_an_old_cache(tmp_path, monkeypatch):
    out, _ = _run(tmp_path, monkeypatch, diffusion=False)
    g8 = list(GROUPS)[0]
    for s in "abk":
        L1.forecast_path(out, g8, s).unlink()
    np.savez(L1.forecast_path(out, g8), final=np.zeros(1), context_frames=K)     # the old shared forecast
    m = FakeModel(False)
    _run(tmp_path, monkeypatch, diffusion=False, model=m)
    assert m.n_forecasts == 3 and all(L1.forecast_path(out, g8, s).exists() for s in "abk")
    assert np.allclose(np.load(L1.forecast_path(out, g8, "b"))["final"], PRE_VALUE["b"])


def test_stage4_per_frame_surprise_and_token_frame_pools(tmp_path, monkeypatch):
    """Record format for the L2 clean window: index per_frame (t_s = last master frame of each post token-frame) and
    the per-token-frame frame pools; an adapter without per-frame values falls back to the clip-level record."""
    out, _ = _run(tmp_path, monkeypatch, diffusion=False)
    recs = [json.loads(l) for l in (out / "_index.jsonl").read_text().splitlines()]
    r = recs[0]
    assert r["surprise"] == {"mse_all": 1.5}
    assert r["per_frame"] == {"t_s": [5 / 4, 7 / 4], "surprise": {"mse_all": [1.0, 2.0]}, "weight": {"mse_all": [1.0, 1.0]}}
    z = np.load(out / r["file"])
    assert z["pooled_frame_t"].shape == (T // TUB, 2, D) and z["token_t_s"].tolist() == [0.25, 0.75, 1.25, 1.75]
    assert z["layers"].tolist() == [-1, 0] and np.allclose(z["pooled_frame_t"][:, 0], 2 * z["pooled_frame_t"][:, 1])
    assert z["pooled_frame_t"][:, 1, 0].tolist() == [10.0, 10.0, 1.0, 1.0]           # pre frames, then this twin's post
    from physlaw.eval import l2_surprise as L2
    assert L2.window_surprise(r, 1.3) == {"mse_all": 1.0}
    m = FakeModel(False)
    m.per_frame = False
    (tmp_path / "old").mkdir()
    out2, _ = _run(tmp_path / "old", monkeypatch, diffusion=False, model=m)
    assert all("per_frame" not in json.loads(l) for l in (out2 / "_index.jsonl").read_text().splitlines())
    S4 = _stage4()
    assert S4.surprise_per_frame({"a": 1.0, "per_frame": {"surprise": {"a": [1.0]}, "weight": {}}}, [0.1, 0.2]) is None


def test_encoder_only_model_caches_encode_alone(tmp_path, monkeypatch):
    """An adapter that does not predict (encoder-only weights, VideoMAE-v2 base): encode() pools, no surprise, no
    forecast files; a rerun recomputes nothing."""
    m = FakeModel(False)
    m.predicts = False
    m.surprise = m.forecast = None                       # never called
    out, _ = _run(tmp_path, monkeypatch, diffusion=False, model=m)
    recs = [json.loads(l) for l in (out / "_index.jsonl").read_text().splitlines()]
    assert len(recs) == 7 and all(r["surprise"] == {} and "per_frame" not in r for r in recs)
    assert not list(out.glob("*/_forecast*.npz")) and "pooled_frame_t" in np.load(out / recs[0]["file"]).files
    _run(tmp_path, monkeypatch, diffusion=False, model=m)
    assert len((out / "_index.jsonl").read_text().splitlines()) == 7


def test_diffusion_pools_each_twin_with_its_own_segment(tmp_path, monkeypatch):
    out, m = _run(tmp_path, monkeypatch, diffusion=True)
    assert m.n_forecasts == 4           # one sampled forecast per (group, segment)
    for g, tw in GROUPS.items():
        for t, s in tw.items():
            z = np.load(out / g / f"{t}.npz")
            assert np.allclose(z["fc_pooled"][:2], PRE_VALUE[s])   # frame.all, frame.post
            # per forecast token-frame (L1 gravity / rolling): rows = token-frames context_tokens..
            assert z["fc_pooled_frame_t"].shape == (T // TUB - int(z["context_tokens"]), 1, D)
            assert np.allclose(z["fc_pooled_frame_t"], PRE_VALUE[s])
            assert (str(z["fc_segment"]) == s) if g.startswith("f8") else "fc_segment" not in z.files
    g8, g1 = list(GROUPS)
    assert sorted(f.name for f in (out / g8).glob("gen_seed0*.mp4")) == [f"gen_seed0_{s}.mp4" for s in "abk"]
    assert [f.name for f in (out / g1).glob("gen_seed0*.mp4")] == ["gen_seed0.mp4"]
    assert not list(out.glob("*/_forecast*.npz"))


def test_diffusion_pass2_redoes_shared_forecasts(tmp_path, monkeypatch):
    out, m = _run(tmp_path, monkeypatch, diffusion=True, extra=["--no-forecast"])
    assert m.n_forecasts == 0
    g8, g1 = list(GROUPS)
    f = out / g8 / "legal_slope_b.npz"     # fc_pooled from the old one-per-group scheme (no fc_segment)
    z = dict(np.load(f))
    np.savez(f, **z, fc_pooled=np.zeros((3, 1, D), np.float16), fc_keys=np.array(["snr0/L1"]), context_tokens=2)
    m = FakeModel(True)
    _run(tmp_path, monkeypatch, diffusion=True, model=m)
    assert m.n_forecasts == 4
    for g, tw in GROUPS.items():
        for t, s in tw.items():
            z = np.load(out / g / f"{t}.npz")
            assert np.allclose(z["fc_pooled"][:2], PRE_VALUE[s]) and np.allclose(z["fc_pooled_frame_t"], PRE_VALUE[s])
    m = FakeModel(True)
    _run(tmp_path, monkeypatch, diffusion=True, model=m)
    assert m.n_forecasts == 0
    f1 = out / g1 / "conserve_e1.npz"       # forecast cached before fc_pooled_frame_t: that group is sampled again
    z = dict(np.load(f1))
    z.pop("fc_pooled_frame_t")
    np.savez(f1, **z)
    m = FakeModel(True)
    _run(tmp_path, monkeypatch, diffusion=True, model=m)
    assert m.n_forecasts == 1 and np.allclose(np.load(f1)["fc_pooled_frame_t"], PRE_VALUE["0"])


def test_guard_refuses_a_forecast_shared_across_segments(tmp_path):
    S4 = _stage4()
    L1.check_shared_forecast("g", ["a", "a"])
    with pytest.raises(RuntimeError, match="different pre-event frames"):
        L1.check_shared_forecast("g", ["a", "b"])
    # diffusion memo: a forecast sampled for segment a is never pooled into a twin of segment b
    memo = {"gid": "g", "seg": {None: ("a", {"snr0/L1": torch.zeros(2, 2, 2, D)})}}
    Wt = torch.ones(1, 16)
    with pytest.raises(RuntimeError, match="different pre-event frames"):
        S4.add_forecast(FakeModel(True), None, K, "g", None, "b", tmp_path, Wt, ["frame.post"], PHASES, 2, memo)
    # encoders: an existing per-segment file written for another segment
    (tmp_path / "g").mkdir()
    f = L1.forecast_path(tmp_path, "g", "a")
    np.savez(f, final=np.zeros(1), context_frames=K, pre_segment=np.array("b"))
    with pytest.raises(RuntimeError, match="different pre-event frames"):
        S4.write_forecast(FakeModel(False), None, K, f, "a", "a", False, {})
    users = {}
    S4.write_forecast(FakeModel(False), torch.zeros(T, 4, 4, 3), K, L1.forecast_path(tmp_path, "g"), None, "0", False, users)
    with pytest.raises(RuntimeError, match="different pre-event frames"):
        S4.write_forecast(FakeModel(False), None, K, L1.forecast_path(tmp_path, "g"), None, "1", False, users)


def test_forecast_path_and_segments():
    assert L1.forecast_path("c", "g", None) == Path("c/g/_forecast.npz")
    assert L1.forecast_path("c", "g", "u0") == Path("c/g/_forecast_u0.npz")
    key = L1.forecast_segments({"g/a": "0", "g/b": "0", "h/x": "a", "h/y": "b", "h/z": "a"})
    assert key == {"g/a": None, "g/b": None, "h/x": "a", "h/y": "b", "h/z": "a"}
    assert L1.clip_segment({"physlaw": {}}) == "0" and L1.clip_segment({"physlaw": {"pre_segment": "k"}}) == "k"


def _labels_dir(tmp_path):
    data = tmp_path / "data"
    for g, tw in GROUPS.items():
        for t in tw:
            (data / g / t).mkdir(parents=True, exist_ok=True)
            (data / g / t / "labels.json").write_text(json.dumps(_clip(data / g / t, False, False).labels))
    return data


def test_readers_find_each_twins_own_forecast(tmp_path, monkeypatch):
    out, _ = _run(tmp_path, monkeypatch, diffusion=False)
    data = _labels_dir(tmp_path)
    recs = [{"clip": f"{g}/{t}", "group": g, "twin": t, "file": f"{g}/{t}.npz"} for g in GROUPS for t in GROUPS[g]]
    clips = [r["clip"] for r in recs]
    seg = {c: GROUPS[c.split("/")[0]][c.split("/")[1]] for c in clips}
    want = np.array([PRE_VALUE[seg[c]] for c in clips])
    mcfg = load_config(tmp_path / "fake.yaml", ["token_grid.tubelet=2"])
    # L1 (seg given): every clip reads its own segment's forecast; the pre-event pool comes along (R1 probes)
    sp = L1._load_spaces(out, recs, clips, mcfg, {}, seg)["final"]
    assert np.allclose(sp["fc"]["frame.post"][:, 0], want) and "frame.pre" in sp["real"]
    # without seg (old call): one forecast per group, the F1-like group still reads _forecast.npz
    r1 = [r for r in recs if r["group"].startswith("f1")]
    sp1 = L1._load_spaces(out, r1, [r["clip"] for r in r1], mcfg, {})["final"]
    assert np.allclose(sp1["fc"]["frame.post"][:, 0], PRE_VALUE["0"])
    # L3 (Galilean F4: each boost level's own forecast): segments from labels.json
    monkeypatch.setattr(L3, "_mask_weights", lambda *a: {})
    real, fc = L3.forecast_spaces(out, data, recs, {"token_grid": {"tubelet": TUB}}, ["frame.post"])["final"]
    assert np.allclose(fc["frame.post"][:, 0], want)
