"""L2 F2 group contrasts: speed_matched and family_min (synthetic twin groups, labels-only observers)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physlaw.data import twins as TW
from physlaw.eval import l2_surprise as L

REPO = Path(__file__).resolve().parents[1]


def random_pre(rng):
    while True:
        m_a, m_b = np.exp(rng.uniform(np.log(0.3), np.log(3.0), 2))
        n = rng.normal(size=2); n /= np.linalg.norm(n)
        pre = TW.PreState(m_a=m_a, m_b=m_b, v_a=rng.normal(size=2) + 1.5 * n, v_b=rng.normal(size=2) * 0.5, n=n)
        if pre.approach_speed > 0.3:
            return pre


def synthetic_kin(n_groups=60, levels=TW.F2_LEVELS, seed=0, e_viol=(1.0, 0.5, 0.15)):
    """group -> (PreState, {twin: o4}): the F2 ladder + violators from twins.py on random pre-states."""
    rng = np.random.default_rng(seed)
    kin = {}
    for g in range(n_groups):
        pre = random_pre(rng)
        dv = TW.CP.legal_outcome(pre, 1.0)
        pa, pb = pre.v_a + dv[:2], pre.v_b + dv[2:]
        if len(levels) == 11:
            tw = TW.f2_twins(pre, pa, pb, 1.0, levels)
            outs = {k: np.concatenate([t.v_a - pre.v_a, t.v_b - pre.v_b]) for k, t in tw.items()}
        else:                     # fine ladder (f2_twins names levels to one decimal)
            outs = {f"e{e:.2f}": TW.CP.legal_outcome(pre, e) for e in levels}
        vt = TW.f2_violators(pre, pa, pb, 1.0, e_viol, float(rng.uniform(0.10, 0.35)))[0]
        outs.update({k: np.concatenate([t.v_a - pre.v_a, t.v_b - pre.v_b]) for k, t in vt.items()})
        kin[f"g{g:03d}"] = (pre, outs)
    return kin


def spec_for(levels, **sm):
    ladder = [f"e{e:.1f}" if len(levels) == 11 else f"e{e:.2f}" for e in levels]
    return {"ladder": ladder, "calibration_pairs": [(ladder[-1], ladder[0])], "controls": [(ladder[4], ladder[3])],
            "contrasts": [{"name": "speed_matched", "kind": "speed_matched", **sm},
                          {"name": "family_min", "kind": "family_min"}], "pending": []}


def test_speed_matching_picks_the_ke_closest_twin():
    kin = synthetic_kin(20)
    spec = spec_for(TW.F2_LEVELS, match="ke")
    rows = L.build_rows(kin, spec)["speed_matched"]
    assert rows
    for r in rows:
        st, outs = kin[r["group"]]
        kv = L.kin_scalar(st, outs[r["viol"]], "ke")
        gaps = {t: abs(np.log(L.kin_scalar(st, outs[t], "ke") / kv)) for t in spec["ladder"]}
        assert r["legal"] == [min(gaps, key=gaps.get)]
        assert r["log_gap_ke"] == pytest.approx(min(gaps.values()))
    # hand-made: the legal outcome at e = 0.58 has KE between e0.5 and e0.6, much nearer e0.6; at e = 0.52, e0.5
    st, outs = kin["g000"]
    lad = {t: outs[t] for t in spec["ladder"]}
    assert L.match_partner(st, TW.CP.legal_outcome(st, 0.58), lad, "ke")[0] == "e0.6"
    assert L.match_partner(st, TW.CP.legal_outcome(st, 0.52), lad, "ke")[0] == "e0.5"
    assert L.match_partner(st, TW.CP.legal_outcome(st, 0.52), lad, "ke")[1] > 0


def test_family_min_perfect_observer_wins():
    kin = synthetic_kin(40)
    spec = spec_for(TW.F2_LEVELS, match="ke")
    rows_by = L.build_rows(kin, spec)
    io = L.ideal_group_bands(kin, rows_by, spec)
    fm = io["family_min"]
    assert fm["win_rate"]["perfect"]["all"] == 1.0 and fm["rank_mean"]["perfect"]["all"] == 1.0
    assert fm["chance_win"] == pytest.approx(11 / 12)
    assert io["speed_matched"]["win_rate"]["perfect"]["all"] == 1.0
    # the scorer on a perfect-observer "model": violators are off the legal segment, ladder twins on it
    clips = {f"{g}/{t}": {"d_obj": L.group_observers(st)["perfect"](o) + 1e-3}
             for g, (st, outs) in kin.items() for t, o in outs.items()}
    res = L.group_contrasts(kin, {"ideal": clips}, spec, n_boot=50)
    s = res["contrasts"]["family_min"]["raw"]["ideal"]["d_obj"]["all"]
    assert s["win_rate"] == 1.0 and s["rank_mean"] == 1.0 and s["chance_win"] == pytest.approx(11 / 12)
    assert res["contrasts"]["speed_matched"]["raw"]["ideal"]["d_obj"]["all"]["win_rate"] == 1.0


def test_rank_and_ties():
    a = np.array([5.0, 1.0, 2.0, 5.0, 0.5])
    iv, il = np.array([0, 3]), np.array([[1, 2, 3], [0, 1, 2]])
    win, rank = L.score_rows("family_min", a, iv, il)
    assert win.tolist() == [1.0, 1.0]
    assert rank[0] == pytest.approx((2 + 0.5) / 3) and rank[1] == pytest.approx((2 + 0.5) / 3)
    win, rank = L.score_rows("family_min", np.array([0.5, 1.0, 2.0]), np.array([0]), np.array([[1, 2]]))
    assert win[0] == 0.0 and rank[0] == 0.0
    w, r = L.score_rows("speed_matched", np.array([1.0, 1.0, np.nan]), np.array([0, 0]), np.array([[1], [2]]))
    assert w[0] == 0.5 and np.isnan(w[1]) and r is None


@pytest.mark.parametrize("by", ["speed", "ke"])
def test_scalar_only_observer_is_at_chance_on_speed_matched(by):
    """Fine ladder, bracketed matches, faster / slower balanced: an observer that only knows the expected
    speed (or KE) cannot tell the violator from its matched legal twin."""
    levels = tuple(round(0.01 * i, 2) for i in range(101))
    kin = synthetic_kin(150, levels=levels, seed=1)
    spec = spec_for(levels, match=by, max_log_gap=0.02, require_bracketed=True, balance_faster=True)
    rows_by = L.build_rows(kin, spec)
    io = L.ideal_group_bands(kin, rows_by, spec)["speed_matched"]
    assert io["n"] > 100
    for e in L.IDEAL_E:
        assert io["win_rate"][f"{'speed' if by == 'speed' else 'ke'}_e{e:g}"]["all"] == pytest.approx(0.5, abs=0.1), e
    assert io["win_rate"]["perfect"]["all"] == 1.0
    assert io["win_rate"]["law_e0.5"]["all"] > 0.75


def test_group_contrasts_motion_adjusted_and_md():
    kin = synthetic_kin(40, seed=2)
    spec = spec_for(TW.F2_LEVELS, match="ke", max_log_gap=0.2)
    spec["calibration_pairs"] = [("e1.0", "e0.0"), ("e1.0", "e0.5"), ("e0.8", "e0.2")]
    spec["controls"] = [("e0.4", "e0.3"), ("e0.7", "e0.6")]
    spec["pending"] = [{"name": "energy_kink", "kind": "energy_kink", "twins": spec["ladder"]}]
    rng = np.random.default_rng(0)
    sur, pix, cov = {}, {}, {}
    for g, (st, outs) in kin.items():
        for t, o in outs.items():
            c = f"{g}/{t}"
            sp = L.kin_scalar(st, o, "speed")
            sur[c] = {"l1_obj": sp * rng.uniform(0.9, 1.1) * (1.5 if t.startswith("mom") else 1.0)}
            pix[c] = {"copy_last_mse_obj": sp * rng.uniform(0.9, 1.1), "constvel_px": sp * rng.uniform(0.5, 1.5)}
            cov[c] = {**pix[c], "post_speed_sum": sp, "post_ke": L.kin_scalar(st, o, "ke")}
    res = L.group_contrasts(kin, {"model": sur, "pixel_baselines": pix}, spec, cov=cov, model="model",
                            primary="post_speed", n_boot=100)
    json.dumps(res)
    ma = res["motion_adjusted"]
    assert set(ma["controls"]["post_speed"]["l1_obj"]) == {"e0.4 vs e0.3", "e0.7 vs e0.6"}
    fm = ma["contrasts"]["post_speed"]["family_min"]["l1_obj"]["all"]
    assert fm["adj_rank_mean"] > 0.8 and fm["chance_win"] == pytest.approx(11 / 12)
    g = fm["vs_pixel"]["copy_last"]
    assert g["adj_minus_baseline"] == pytest.approx(fm["adj_win_rate"] - g["baseline_raw_win"])
    assert "adj_rank_minus_baseline" in g
    sm = ma["contrasts"]["post_speed"]["speed_matched"]["l1_obj"]["all"]
    assert sm["adj_win_rate"] > 0.8 and "verdict" in sm and set(sm["vs_pixel"]) == {"copy_last", "const_velocity"}
    m = res["contrasts"]["speed_matched"]["matching"]
    assert m["median_abs_log_ke_ratio"] <= 0.2
    assert res["contrasts"]["energy_kink"]["status"].startswith("pending")
    md = L.group_table(res)
    assert "speed_matched" in md and "family_min" in md and "Held-out legal controls" in md and "speed_e1" in md


def test_f2_config():
    from physlaw.config import load_config
    cfg = load_config(REPO / "configs" / "eval" / "l2_surprise.yaml")
    spec = L.group_spec(cfg, "f2")
    assert spec["ladder"][0] == "e0.0" and spec["ladder"][-1] == "e1.0" and len(spec["ladder"]) == 11
    fit = {tuple(p) for p in spec["calibration_pairs"]}
    assert not fit & {tuple(p) for p in spec["controls"]}               # held-out controls never in the β fit
    for a, b in spec["controls"]:                                         # adjacent ladder levels
        assert abs(spec["ladder"].index(a) - spec["ladder"].index(b)) == 1
    assert {c["kind"] for c in spec["contrasts"]} == {"speed_matched", "family_min"}
    assert [c["kind"] for c in spec["pending"]] == ["energy_kink"]
    assert spec["pending"][0]["twins"][-1] == "e1.5"
    assert L.group_spec(cfg, "f1") is None
    cal, cons = L.family_contrasts(cfg, "f2")
    assert cal == ("e1.0", "e0.0") and all(c["magnitude"] == "eta" for c in cons)
