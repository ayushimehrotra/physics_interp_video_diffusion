"""grid_probe must reproduce ridge.fit_probe / ridge.permutation_null exactly (same folds, same alphas)."""
import numpy as np

from physlaw.probes.grid import grid_max_p, grid_probe, group_block_perms
from physlaw.probes.ridge import fit_probe, permutation_null


def _data(seed=0, n_groups=20, twins=5, d=40):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), twins)
    t = np.tile(np.arange(twins), n_groups)
    X = rng.normal(size=(len(g), d))
    y1 = X[:, :2] @ [[1.0, 0.5], [0.2, -1.0]] + 0.5 * rng.normal(size=(len(g), 2))
    y2 = X[:, 3] * 0.3 + rng.normal(size=len(g))
    return X, {"vel": y1, "spd": y2}, g, t


def test_matches_fit_probe():
    X, blocks, g, _ = _data()
    out = grid_probe(X, blocks, g, fold_seeds=3)
    for name, y in blocks.items():
        ref = fit_probe(X, y, g, fold_seeds=3)
        assert np.allclose(out[name]["scores"], ref.scores, atol=1e-8)
        assert out[name]["alphas_chosen"] == [float(a) for a in ref.alphas_chosen]


def test_matches_permutation_null():
    X, blocks, g, t = _data(1)
    perms = group_block_perms(g, t, 7, seed=3)
    out = grid_probe(X, blocks, g, fold_seeds=1, perms=perms)
    y = blocks["vel"]
    ref = permutation_null({"c": X}, y, g, n_permutations=1)            # observed part only
    assert np.isclose(out["vel"]["quick"], ref["observed"]["c"], atol=1e-8)
    # null for one draw == quick protocol on the permuted targets
    ref_p = permutation_null({"c": X}, y[perms[2]], g, n_permutations=1)
    assert out["vel"]["alpha_null"] == ref["alphas"]["c"]
    if ref_p["alphas"]["c"] == ref["alphas"]["c"]:
        assert np.isclose(out["vel"]["null"][2], ref_p["observed"]["c"], atol=1e-8)


def test_group_block_perms_keep_twin_slot():
    _, _, g, t = _data()
    p = group_block_perms(g, t, 5)
    for row in p:
        assert (t[row] == t).all()
        # whole groups move together
        for gg in np.unique(g):
            assert len(np.unique(g[row[g == gg]])) == 1


def test_grid_max_p():
    obs = {"a": 0.5, "b": 0.1}
    null = {"a": np.array([0.0, 0.6]), "b": np.array([0.7, 0.0])}
    best, p, nm = grid_max_p(obs, null)
    assert best == "a" and np.allclose(nm, [0.7, 0.6]) and np.isclose(p, 3 / 3)
