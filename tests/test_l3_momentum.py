"""L3 momentum invariance under Galilean boosts on synthetic decodes."""
import numpy as np
import pytest

from physlaw.eval import l3_invariance as L3


def _f4(n_scenes=40, seed=0):
    """Scenes x levels {u0, usmall, ularge} x e {1, 0.15}: true pre / post velocities of 1-D-normal collisions."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenes):
        m = rng.uniform(1, 5, 2)
        n = (lambda v: v / np.linalg.norm(v))(rng.normal(size=2))
        vA, vB = 1.5 * n + rng.normal(0, 0.3, 2), rng.normal(0, 0.3, 2)
        un = (vA - vB) @ n
        d = rng.normal(size=2); d /= np.linalg.norm(d)
        for lv, speed in (("u0", 0.0), ("usmall", 1.0), ("ularge", 4.0)):
            u = speed * d
            for e in (1.0, 0.15):
                J = (1 + e) * un * m[0] * m[1] / m.sum()
                post = np.stack([vA - J / m[0] * n, vB + J / m[1] * n]) + u
                rows.append(dict(scene=s, level=lv, e=e, m=m, n=n, pre=np.stack([vA, vB]) + u, post=post, u=u))
    return rows


def _scores(rows, post):
    return L3.momentum_invariance_scores(
        np.stack(post), np.stack([r["pre"] for r in rows]), np.stack([r["m"] for r in rows]),
        np.stack([r["n"] for r in rows]), np.array([r["level"] for r in rows]),
        np.array([f"{r['scene']}/{r['e']}" for r in rows]), np.array([str(r["scene"]) for r in rows]), "u0", n_boot=300)


def test_true_post_velocities_have_zero_eta_at_every_level():
    rows = _f4()
    res = _scores(rows, [r["post"] for r in rows])
    for lv, v in res.items():
        assert v["eta_mean"] == pytest.approx(0.0, abs=1e-12) and v["antiparallel_mean"] == pytest.approx(1.0)
        if lv != "u0":
            assert v["eta_diff"] == pytest.approx(0.0, abs=1e-12) and v["eta_n_paired"] == 80


def test_boost_invariant_decoder_gives_zero_difference():
    rows = _f4()
    rng = np.random.default_rng(1)
    noise = {(r["scene"], r["e"]): rng.normal(0, 0.1, (2, 2)) for r in rows}   # error fixed per scene, not per boost
    res = _scores(rows, [r["post"] + noise[r["scene"], r["e"]] for r in rows])
    assert res["u0"]["eta_mean"] > 0.02                                           # imperfect decoding: eta > 0
    for lv in ("usmall", "ularge"):
        assert res[lv]["eta_diff"] == pytest.approx(0.0, abs=1e-12)
        assert res[lv]["antiparallel_diff"] == pytest.approx(0.0, abs=1e-12)


def test_decoder_whose_error_grows_with_speed_breaks_invariance():
    rows = _f4()
    rng = np.random.default_rng(2)
    # decoded = truth + noise proportional to the clip's speed (u0 clips: their own speed only)
    post = [r["post"] + 0.1 * np.linalg.norm(r["pre"], axis=1)[:, None] * rng.normal(size=(2, 2)) for r in rows]
    res = _scores(rows, post)
    assert res["ularge"]["eta_diff"] > 0.05 and res["ularge"]["eta_diff_ci95"][0] > 0
    assert res["ularge"]["eta_diff"] > res["usmall"]["eta_diff"]
    assert res["ularge"]["antiparallel_diff"] < 0 and res["ularge"]["antiparallel_diff_ci95"][1] < 0


def test_post_velocity_target():
    lab = {"physlaw": {"pre_velocity": {"A": [1.0, 2.0, 0.0], "B": [3.0, 4.0, 0.0]},
                       "post_velocity": {"A": [5.0, 6.0, 0.0], "B": [7.0, 8.0, 0.0]}}}
    assert L3.clip_target(lab, "post_velocity").tolist() == [5.0, 6.0, 7.0, 8.0]
    assert L3.clip_target(lab, "pre_velocity").tolist() == [1.0, 2.0, 3.0, 4.0]
