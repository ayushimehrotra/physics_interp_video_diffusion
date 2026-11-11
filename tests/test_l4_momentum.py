"""L4 momentum metrics on synthetic decodes: the e-free deflection-ratio slope and the residualised direction target."""
import numpy as np
import pytest

from physlaw.eval import l4_causal as L4


def _points(n_groups=30, follow=True, e_range=(0.2, 1.0), seed=0):
    """Per (group, twin): pre state, true q0, and decoded deflections at injected dlog q in {-2..2} * 0.3."""
    rng = np.random.default_rng(seed)
    x, a, b, unit, grp = [], [], [], [], []
    for g in range(n_groups):
        for t in range(3):
            q0 = np.exp(rng.normal(0, 0.5))
            vn = rng.uniform(0.5, 2.0)                 # approach speed along n
            e = rng.uniform(*e_range)                  # restitution varies: the slope must not care
            for dl in np.arange(-2, 3) * 0.3:
                q = q0 * np.exp(dl) if follow else q0
                aa, bb = (1 + e) * vn / (1 + q), (1 + e) * vn * q / (1 + q)   # m_A a = m_B b, a + b = (1 + e) vn
                if not follow:                         # a readout unrelated to the injection
                    aa, bb = rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)
                else:
                    aa, bb = aa * np.exp(0.02 * rng.normal()), bb * np.exp(0.02 * rng.normal())
                x.append(dl); a.append(aa); b.append(bb); unit.append(f"{g}|{t}"); grp.append(g)
    return x, a, b, unit, grp


def test_deflection_slope_recovers_one_when_decodes_follow_momentum():
    r = L4.deflection_slope(*_points(), n_boot=200)
    assert r["slope"] == pytest.approx(1.0, abs=0.02)
    assert r["slope_ci95"][0] < 1.0 < r["slope_ci95"][1] and r["n_excluded"] == 0


def test_deflection_slope_is_zero_for_unrelated_noise():
    r = L4.deflection_slope(*_points(follow=False), n_boot=200)
    assert abs(r["slope"]) < 0.2
    assert r["slope_ci95"][0] < 0 < r["slope_ci95"][1]


def test_deflection_slope_excludes_non_collisions_and_ignores_unit_offsets():
    x, a, b, unit, grp = _points(n_groups=10)
    a, b = np.array(a), np.array(b)
    a[:4] = -0.1                                       # A pulled forward: no real collision
    b[4] = 0.0
    r = L4.deflection_slope(x, a, b, unit, grp, n_boot=0)
    assert r["n_excluded"] == 5 and r["n"] == len(x) and r["slope_ci95"] is None
    assert r["slope"] == pytest.approx(1.0, abs=0.03)
    # a constant multiplicative bias per (group, twin) (e.g. probe shrinkage of one ball) does not move the slope
    bias = np.exp(np.array([hash(u) % 7 for u in unit]) * 0.1)
    assert L4.deflection_slope(x, a, b * bias, unit, grp, n_boot=0)["slope"] == pytest.approx(r["slope"], abs=1e-9)


def test_residualised_target_is_uncorrelated_with_radius_ratio():
    rng = np.random.default_rng(0)
    n = 400
    lr = rng.normal(0, 0.2, n)                         # log(r_A / r_B)
    ld = rng.normal(0, 0.5, n)                         # log density ratio: the "mass beyond size"
    lq = 3 * lr + ld                                   # m ~ rho r^3
    res, coef = L4.residualise(lq, lr)
    assert abs(np.corrcoef(res, lr)[0, 1]) < 1e-10 and abs(res.mean()) < 1e-10
    assert coef[1] == pytest.approx(3.0, abs=0.4)
    assert np.corrcoef(res, ld)[0, 1] > 0.99
    tr = np.arange(200)                                # fitted on training rows only, applied to all
    res_tr, coef_tr = L4.residualise(lq, lr, tr)
    assert abs(np.corrcoef(res_tr[tr], lr[tr])[0, 1]) < 1e-10 and len(res_tr) == n
    # constant radius ratio (F3): only the mean comes off
    r3, c3 = L4.residualise(lq, np.zeros(n))
    assert c3[1] == 0.0 and r3 == pytest.approx(lq - lq.mean())


def test_log_r_and_log_q_from_labels():
    lab = {"physlaw": {"objects": {"A": {"mass": 2.0, "radius": 0.4}, "B": {"mass": 1.0, "radius": 0.5}}}}
    assert L4.log_q(lab) == pytest.approx(np.log(2.0)) and L4.log_r(lab) == pytest.approx(np.log(0.8))


def test_generative_direction_recovers_the_label_axis():
    rng = np.random.default_rng(3)
    n, D = 300, 12
    y = rng.normal(size=n)
    v = np.zeros(D); v[2], v[5] = 2.0, -1.0
    X = np.outer(y, v) + 0.1 * rng.normal(size=(n, D)) + 5.0
    d = L4.fit_generative(X, y, np.repeat(np.arange(100), 3))
    assert abs(d["u"] @ v / np.linalg.norm(v)) > 0.99 and d["gain"] > 0
    assert d["gain"] == pytest.approx(1 / np.linalg.norm(v), rel=0.05)    # target units per unit along u
    assert d["cv_r2"] > 0.95 and d["sd"] == pytest.approx((X @ d["u"]).std())
    assert np.corrcoef(d["predict"](X), y)[0, 1] > 0.99


@pytest.mark.parametrize("source,kind,target", [(None, "generative", "residual"), ("f1", "ridge", "raw")])
def test_run_analysis_end_to_end_without_gpu(tmp_path, monkeypatch, source, kind, target):
    """run() on fake caches and fake cached interventions: the analysis, the passive control and the table."""
    from physlaw.config import load_config
    rng = np.random.default_rng(0)
    D, groups, twins = 8, [f"g{i}" for i in range(8)], ["q05", "q2"]
    recs, labels, feats = [], [], []
    for g in groups:
        ra, rb = rng.uniform(0.3, 0.6, 2)
        for t in twins:
            q = 0.5 if t == "q05" else 2.0
            ma = q * rb ** 3 * rng.uniform(0.8, 1.2) / ra ** 3
            pre = {"A": [1.0, 0.0], "B": [0.0, 0.0]}
            post = {"A": [1 - 2 / (1 + q), 0.0], "B": [2 * q / (1 + q), 0.0]}
            recs.append({"group": g, "twin": t, "clip": f"{g}/{t}"})
            labels.append({"physlaw": {"objects": {"A": {"mass": ma * ra ** 3, "radius": ra}, "B": {"mass": rb ** 3, "radius": rb}},
                                       "contact_normal": [1.0, 0.0], "pre_velocity": pre, "post_velocity": post}})
            feats.append(np.r_[np.log(q), post["A"], post["B"], rng.normal(size=D - 5)])
    F = np.array(feats)
    A = np.stack([F, F + 0.1 * rng.normal(size=F.shape)], 1)[:, None].astype(np.float32)   # [n, P=1, L=2, D]
    monkeypatch.setattr(L4, "load_family", lambda c, d: (recs, labels, A, ["roi_A.pre"], ["3", "-1"]))
    post_pools = {p: F + 0.05 * rng.normal(size=F.shape) for p in L4.POST_POOLS}
    monkeypatch.setattr(L4, "forecast_spaces", lambda *a, **k: {"final": (post_pools, post_pools)})
    work = tmp_path / "work"
    work.mkdir()
    for g in groups:
        rec = {}
        for cond, j in (("base", 0), ("mass", 0), ("random_dir", 0)):
            for a in ([0.0] if cond == "base" else [-1.0, 1.0]):
                for t in twins:
                    i = groups.index(g) * 2 + twins.index(t)
                    for p in L4.POST_POOLS:
                        rec[f"{cond}|{j}|{a:g}|{t}|final|{p}"] = (F[i] + (0 if cond == "base" else 0.1 * a)).astype(np.float32)
        np.savez(work / f"{g}.npz", **rec)
    cfg = load_config("configs/eval/l4_causal.yaml")
    cfg.update(direction_data=source, direction=kind, direction_target=target, controls={"random_dir": 1}, layer="3", alphas_sd=[-1, 0, 1], bootstrap={"n": 20})
    cfg["probe"]["k_folds"] = 3
    mcfg = type("M", (dict,), {"name": "fake"})()
    res = L4.run(tmp_path / "cache", tmp_path / "f3", cfg, mcfg, work, log=lambda *a: None)
    assert res["direction"] == kind and res["direction_target"] == target
    dd = res["directions"][0]
    assert {"generative|raw", "generative|residual", "ridge|raw", "ridge|residual"} == set(dd["cv_r2_by"])
    for v in res["variants"].values():
        for c in ("mass", "random_dir"):
            s = v[c]
            assert np.isfinite(s["deflection_slope"]["slope"]) and s["passive_slope"] is not None
            assert "slope_over_passive" in s and "change_over_passive_B_n" in s
    md = L4.table(res)
    assert "slope log(b/a)" in md and "passive" in md and "generative|residual" in md
