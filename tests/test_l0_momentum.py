"""L0 momentum targets on synthetic labels (no cache): p = m v, and the partial (beyond v and m) target."""
import numpy as np

from physlaw.eval.l0_states import OBJECTS, PHASES, momentum_partial, targets
from physlaw.probes.grid import grid_probe


def _labels(n=240, seed=0, const_mass_B=False):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        objs, pre, post = {}, {}, {}
        for o in OBJECTS:
            m = 1.0 if (const_mass_B and o == "B") else float(rng.uniform(0.3, 3.0))
            objs[o] = {"mass": m, "radius": float(rng.uniform(0.2, 0.5))}
            pre[o] = list(rng.normal(size=3))
            post[o] = list(rng.normal(size=3))
        out.append({"physlaw": {"objects": objs, "pre_velocity": pre, "post_velocity": post}})
    return out


def test_momentum_is_m_times_v():
    labels = _labels()
    Y = targets(labels)
    for o in OBJECTS:
        m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels])
        for ph in PHASES:
            v = np.array([l["physlaw"][f"{ph}_velocity"][o][:2] for l in labels])
            assert Y[f"{o}.{ph}.momentum"].shape == (len(labels), 2)
            np.testing.assert_array_equal(Y[f"{o}.{ph}.momentum"], m[:, None] * v)


def test_partial_uncorrelated_with_factors_and_nonzero():
    labels = _labels()
    Y = targets(labels)
    for o in OBJECTS:
        m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels])
        for ph in PHASES:
            v = Y[f"{o}.{ph}.velocity"]
            r = Y[f"{o}.{ph}.momentum_partial"]
            assert r.shape == (len(labels), 2)
            for f in (v[:, 0], v[:, 1], m, np.log(m)):
                for j in range(2):
                    assert abs(np.corrcoef(f, r[:, j])[0, 1]) < 1e-8
            # a sizeable share of m v's variance is beyond v and m
            p = Y[f"{o}.{ph}.momentum"]
            assert r.var(0).sum() > 0.05 * p.var(0).sum()


def test_partial_dropped_when_mass_constant():
    Y = targets(_labels(const_mass_B=True))
    assert "B.pre.momentum_partial" not in Y and "B.pre.momentum" in Y
    assert "A.pre.momentum_partial" in Y
    v = np.random.default_rng(1).normal(size=(50, 2))
    assert momentum_partial(v, np.full(50, 2.0)) is None


def test_probe_partial_r2_needs_the_product():
    labels = _labels(n=300)
    Y = targets(labels)
    o, ph = "A", "post"
    m = np.array([l["physlaw"]["objects"][o]["mass"] for l in labels])
    v = Y[f"{o}.{ph}.velocity"]
    groups = np.repeat(np.arange(len(labels) // 2), 2)
    b = f"{o}.{ph}.momentum_partial"
    factors = np.column_stack([v, m, np.log(m)])
    product = np.column_stack([factors, m[:, None] * v])
    kw = dict(alphas=(1e-6, 1e-3, 1e-1), k=5, fold_seeds=2)
    r_fac = grid_probe(factors, {b: Y[b]}, groups, **kw)[b]["score_mean"]
    r_prod = grid_probe(product, {b: Y[b]}, groups, **kw)[b]["score_mean"]
    assert r_fac < 0.05, r_fac
    assert r_prod > 0.95, r_prod
