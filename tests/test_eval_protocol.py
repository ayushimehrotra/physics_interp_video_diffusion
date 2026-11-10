"""Shared evaluation protocol (spec §5, §7): held-out selection, Holm, the RigidBench probe, the L2b logistic
probe, the latent-baseline cache layout, and L2's control / dose-response bookkeeping."""
from __future__ import annotations

import json

import numpy as np
import pytest

from physlaw.probes.ridge import adam_linear_probe
from physlaw.probes.select import heldout_select, holm


def test_holm():
    p = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert p["a"] == pytest.approx(0.03) and p["c"] == pytest.approx(0.06) and p["b"] == pytest.approx(0.06)
    assert np.isnan(p["d"])


def test_heldout_select_removes_winners_curse():
    rng = np.random.default_rng(0)
    groups = np.arange(60)
    noise = {f"v{i}": rng.normal(size=60) for i in range(20)}            # 20 pure-noise variants
    noise["real"] = rng.normal(size=60) + 0.8
    score = lambda v, g: float(noise[v][np.isin(groups, g)].mean())
    sel = heldout_select(score, list(noise), groups, n_splits=20)
    assert sel["selected"] == "real" and sel["n_variants"] == 21
    assert sel["n_select_groups"] == sel["n_eval_groups"] == 30
    del noise["real"]
    sel = heldout_select(score, list(noise), groups, n_splits=50)
    in_sample = max(v.mean() for v in noise.values())
    assert in_sample > 0.2 and sel["score"] < in_sample - 0.1          # the in-sample max is optimistic


def test_adam_linear_probe_learns_linear_map():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 20))
    Y = X[:, :2] @ np.array([[1.0, 0.3], [-0.5, 1.0]]) + 0.1 * rng.normal(size=(300, 2))
    out = adam_linear_probe(X, Y, np.repeat(np.arange(60), 5), epochs=200, lr=1e-2, fold_seeds=1)
    assert out["score_mean"] > 0.9
    short = adam_linear_probe(X, Y, np.repeat(np.arange(60), 5), epochs=1, lr=1e-3, fold_seeds=1)
    assert short["score_mean"] < out["score_mean"]                     # 20 epochs at small n can underfit


def test_logistic_probe():
    from scipy.optimize import minimize
    from physlaw.eval.l2b_detection import cv_auc, logistic_fit, swap_perms
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(80, 5))
    y = (Z[:, 0] + 0.5 * rng.normal(size=80) > 0).astype(float)
    beta = logistic_fit(Z, y, 2.0)
    f = lambda b: np.logaddexp(0, -(2 * y - 1) * (Z @ b[:-1] + b[-1])).sum() + 1.0 * (b[:-1] ** 2).sum()
    ref = minimize(f, np.zeros(6), method="BFGS", options={"gtol": 1e-9}).x
    assert np.allclose(beta, ref, atol=1e-4)
    groups = np.repeat(np.arange(40), 2)
    yy = np.tile([1.0, 0.0], 40)
    X = rng.normal(size=(80, 30)) + 1.5 * yy[:, None] * (np.arange(30) == 0)
    aucs, oof, _, nulls = cv_auc(X, yy, groups, (0.1, 10.0), 5, 2, ys=swap_perms(groups, yy, 10))
    assert np.mean(aucs) > 0.8 and len(nulls) == 10
    P = swap_perms(groups, yy, 5)
    assert np.all(P.reshape(5, 40, 2).sum(-1) == 1)                     # one "violating" twin per group


def test_load_pooled_latent_layout(tmp_path):
    from physlaw.eval.l0_states import load_pooled
    recs = []
    for i in range(3):
        f = tmp_path / f"g{i}.npz"
        np.savez(f, pooled=np.full((2, 3, 4), i, np.float16), pools=np.array(["frame.post", "roi_A.post"]),
                 layers=np.array(["snr+0/L0", "snr+0/L3", "snr+8/L0"]),
                 pooled_latent=np.arange(2 * 2 * 5, dtype=np.float16).reshape(2, 2, 5) + i,
                 latent_keys=np.array(["latent/snr+0", "latent/snr+8"]))
        recs.append({"file": f.name})
    A, pools, layers, lat = load_pooled(tmp_path, recs)
    assert A.shape == (3, 2, 3, 4) and layers == ["snr+0/L0", "snr+0/L3", "snr+8/L0"]
    assert sorted(lat) == ["latent/snr+0", "latent/snr+8"] and lat["latent/snr+8"].shape == (3, 2, 5)
    assert lat["latent/snr+8"][2, 1, 0] == 2 + 15
    np.savez(tmp_path / "g0.npz", pooled=np.zeros((2, 3, 4)), pools=np.array(["a", "b"]), layers=np.array([0, 1, 2]))
    assert load_pooled(tmp_path, recs[:1])[3] == {}


def _l2_fixture(control_bias: float):
    """Surprise = motion (post speed) + violation effect. With ``control_bias`` the e = 1 twin is extra surprising
    whether it moves more or less than its partner, which motion cannot explain: the control must fail."""
    rng = np.random.default_rng(3)
    recs, cov, labs = {}, {}, {}
    for g in range(40):
        speeds = {"conserve_e1": 2.0 if not control_bias else rng.choice([0.5, 2.0]), "conserve_e015": 1.0,
                  "mom_violate_e1": 1.8}
        mag = {"conserve_e1": 0.0, "conserve_e015": 0.0, "mom_violate_e1": rng.uniform(0.05, 0.4)}
        for t, sp in speeds.items():
            sp = sp * rng.uniform(0.8, 1.2)
            s = sp * np.exp(rng.normal(0, 0.02) + 2 * mag[t] + (control_bias if t == "conserve_e1" else 0))
            c = f"g{g}/{t}"
            recs[c] = {"group": f"g{g}", "surprise": {"l1_obj": s}}
            cov[c] = {"post_speed_sum": sp, "post_ke": sp ** 2, "copy_last_mse_obj": sp * rng.uniform(0.9, 1.1)}
            labs[c] = {"physlaw": {"summary": {"momentum_residual": mag[t]}, "energy_physical": g % 2 == 0}}
    return recs, cov, labs


@pytest.mark.parametrize("bias,ok", [(0.0, True), (0.3, False)])
def test_l2_control_check_dose_and_holm(bias, ok):
    from physlaw.eval.l2_surprise import motion_adjusted
    recs, cov, labs = _l2_fixture(bias)
    con = [{"viol": "mom_violate_e1", "legal": "conserve_e1", "magnitude": "momentum_residual", "stratify": "energy_physical"}]
    out = motion_adjusted(recs, cov, con, labels=labs.__getitem__, primary="post_speed", n_boot=200, n_splits=5)
    by_c = out["contrasts"]["post_speed"]
    assert by_c["conserve_e1 vs conserve_e015"]["l1_obj"]["control_at_chance"] is ok
    rec = by_c["mom_violate_e1 vs conserve_e1"]["l1_obj"]
    assert "adj_wilcoxon_p_holm" in rec and set(by_c) >= {"mom_violate_e1 vs conserve_e1 [energy_physical]",
                                                          "mom_violate_e1 vs conserve_e1 [not energy_physical]"}
    if ok:
        assert rec["verdict"].startswith(("more", "less")) and rec["dose_spearman"] > 0.5
    else:
        assert rec["verdict"].startswith("inconclusive")
    assert out["summary"]["mom_violate_e1 vs conserve_e1"]["n_variants"] == 1
    json.dumps(out)
