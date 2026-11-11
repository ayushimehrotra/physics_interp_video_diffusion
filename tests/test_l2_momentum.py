"""L2 momentum measurement: ideal-observer bands, η dose, primary contrast, pixel-baseline gap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physlaw.data import collision_physics as cp

REPO = Path(__file__).resolve().parents[1]
F1 = REPO / "data" / "f1"


def _state(rng, ma, mb):
    n = rng.normal(size=2); n /= np.linalg.norm(n)
    va = rng.normal(size=2) + 2 * n
    vb = rng.normal(size=2) - 2 * n
    return cp.PreState(ma, mb, va, vb, n)


def _mb_error(st, ol):
    """mass-blind prediction − legal outcome (a pure CoM drift)."""
    h = (ol[:2] - ol[2:]) / 2
    return np.concatenate([h, -h]) - ol


def test_ideal_observer_synthetic():
    from physlaw.eval.l2_surprise import ideal_observer_bands
    rng = np.random.default_rng(0)
    items, items_eq = [], []
    for g in range(40):
        st = _state(rng, 1.0, 3.0)
        ol = cp.legal_outcome(st, 0.5)
        e = _mb_error(st, ol)
        # half the groups: viol = legal + 3 e (mass-blind is nearer legal -> wins); half: viol = legal + e (loses)
        items.append((st, ol + (3.0 if g % 2 else 1.0) * e, ol))
        st2 = _state(rng, 2.0, 2.0)
        ol2 = cp.legal_outcome(st2, 0.5)
        items_eq.append((st2, ol2 + np.tile(rng.normal(size=2), 2), ol2))
    io = ideal_observer_bands(items)
    assert io["win_rate"]["perfect"] == 1.0 and io["win_rate"]["law_e0.5"] == 1.0
    assert io["win_rate"]["mass_blind"] == pytest.approx(0.5)
    assert io["n_unequal_mass"] == 40 and io["win_rate_unequal_mass"]["mass_blind"] == pytest.approx(0.5)
    eq = ideal_observer_bands(items_eq)            # equal masses: mass-blind = the legal outcome
    assert eq["win_rate"]["mass_blind"] == 1.0 and eq["n_unequal_mass"] == 0
    json.dumps(io)


def _labels(m, v_pre, v_post, n):
    n = np.asarray(n, float)
    return {"physlaw": {"family": "f1", "objects": {"A": {"mass": m[0], "radius": 0.1}, "B": {"mass": m[1], "radius": 0.1}},
                        "contact_normal": list(n), "pre_velocity": {"A": list(v_pre[0]), "B": list(v_pre[1])},
                        "post_velocity": {"A": list(v_post[0]), "B": list(v_post[1])}, "summary": {}},
            "collision": {"contact_point_world": [0.0, 0.0, 0.0]}}


def test_eta_dose():
    from physlaw.eval.l2_surprise import contrast_magnitude, eta_of
    st = cp.PreState(1.0, 2.0, np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
    v0 = np.stack([st.v_a, st.v_b])
    ol = cp.legal_outcome(st, 1.0).reshape(2, 2)
    legal = _labels([1.0, 2.0], v0, v0 + ol, st.n)
    assert eta_of(legal) == pytest.approx(0.0, abs=1e-12)
    for d, prev in ((0.1, 0.0), (0.4, None)):
        drift = np.array([0.0, d])
        viol = _labels([1.0, 2.0], v0, v0 + ol + drift, st.n)
        q = np.array([1.0, 2.0])[:, None] * (ol + drift)
        want = np.linalg.norm(q.sum(0)) / np.linalg.norm(q, axis=1).sum()
        assert contrast_magnitude("eta", viol, legal) == pytest.approx(want)
    assert contrast_magnitude("eta", _labels([1.0, 2.0], v0, v0 + ol + [0, 0.4], st.n), legal) > \
        contrast_magnitude("eta", _labels([1.0, 2.0], v0, v0 + ol + [0, 0.1], st.n), legal)


def test_primary_first_and_eta_config():
    from physlaw.config import load_config
    from physlaw.eval.l2_surprise import family_contrasts
    cfg = load_config(REPO / "configs" / "eval" / "l2_surprise.yaml")
    _, cons = family_contrasts(cfg, "f1")
    assert cons[0]["viol"] == "mom_violate_e015" and cons[0]["primary"]
    assert all(c["magnitude"] == "eta" for c in cons if c["viol"].startswith("mom_violate"))


def test_pixel_gap_fields():
    from physlaw.eval.l2_surprise import adjusted_table, motion_adjusted
    rng = np.random.default_rng(1)
    recs, cov = {}, {}
    for g in range(30):
        for t, sp in (("conserve_e1", 2.0), ("conserve_e015", 1.0), ("mom_violate_e015", 0.9)):
            sp = sp * rng.uniform(0.8, 1.2)
            c = f"g{g}/{t}"
            recs[c] = {"group": f"g{g}", "surprise": {"l1_obj": sp * rng.uniform(0.9, 1.1) * (1.08 if t.startswith("mom") else 1)}}
            cov[c] = {"post_speed_sum": sp, "post_ke": sp ** 2, "copy_last_mse_obj": sp, "constvel_px": sp * rng.uniform(0.5, 1.5)}
    con = [{"viol": "mom_violate_e015", "legal": "conserve_e015", "primary": True}]
    out = motion_adjusted(recs, cov, con, primary="post_speed", n_boot=200, n_splits=5)
    assert out["primary_contrast"] == "mom_violate_e015 vs conserve_e015"
    rec = out["contrasts"]["post_speed"]["mom_violate_e015 vs conserve_e015"]["l1_obj"]
    vp = rec["vs_pixel"]
    assert set(vp) == {"copy_last", "const_velocity"}
    g = vp["copy_last"]
    assert g["adj_minus_baseline"] == pytest.approx(rec["adj_win_rate"] - g["baseline_raw_win"])
    assert g["adj_minus_baseline_ci95"][0] <= g["adj_minus_baseline"] <= g["adj_minus_baseline_ci95"][1]
    assert rec["adj_win_rate_boot_sd"] > 0
    assert out["summary"]["mom_violate_e015 vs conserve_e015"]["primary"]
    md = adjusted_table(out)
    assert "Primary motion control" in md and "not identified for CoM motion" in md
    json.dumps(out)


@pytest.mark.skipif(not F1.exists(), reason="data/f1 absent")
def test_ideal_observer_f1():
    from physlaw.eval.l2_surprise import ideal_observer
    groups = sorted(p for p in F1.iterdir() if p.is_dir() and (p / "conserve_e1" / "labels.json").exists())
    lab = lambda g, t: json.loads((g / t / "labels.json").read_text())
    want = {"mom_violate_e1": ({"perfect": 1.00, "law_e1": 1.00, "law_e0.5": 0.84, "law_e0": 0.53, "mass_blind": 0.50}, 0.72),
            "mom_violate_e015": ({"perfect": 1.00, "law_e1": 0.95, "law_e0.5": 0.98, "law_e0": 1.00, "mass_blind": 0.50}, 0.81)}
    for viol, (w, slow) in want.items():
        legal = viol.replace("mom_violate", "conserve")
        io = ideal_observer([(lab(g, viol), lab(g, legal)) for g in groups])
        for k, v in w.items():
            assert io["win_rate"][k] == pytest.approx(v, abs=0.006), (viol, k)
        assert io["win_rate_unequal_mass"]["mass_blind"] == pytest.approx(0.41, abs=0.006)
        assert io["viol_slower"] == pytest.approx(slow, abs=0.006)
