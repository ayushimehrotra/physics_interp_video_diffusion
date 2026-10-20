"""Build the generation plan for `conservation_set_v1` -- the 2x2 conservation set.

WHY A PLAN STEP EXISTS. The violating twin's aftermath is not drawn independently:
it is another scene's mass ratio, transplanted. That makes the implied-ratio
distribution identical in the two classes BY CONSTRUCTION (a permutation preserves
a multiset), which is what closes the kinematic leak surface. A permutation is a
global object, so it cannot be decided one scene at a time inside the generator --
hence a planning pass that runs physics only, chooses the permutation, and writes
a manifest the renderer then executes.

DESIGN, and the measurement behind each choice:

  factor 1  elasticity   e = 1.00 (elastic) or 0.15 (inelastic), pinned not sampled.
            The elastic honest cell conserves energy exactly, so it is the positive
            control for the energy axis; the inelastic row loses energy in BOTH
            classes, so detection surviving there is not an energy detector.

  factor 2  conservation honest (lambda = 0) or violating (ratio transplanted).

  matched pairs          every seed is rendered in both classes. Verified 51/51
            seeds give a bit-identical physical scene under violate=True/False,
            so the twins differ only after contact.

  material   80% `mixed`, 20% `same`. Mixed is the main arm: the model's absolute
            error on the mass ratio is the SAME either way (RMSE 0.304 vs 0.308
            VideoMAE, 0.282 vs 0.292 V-JEPA2), but mixed widens the ratio spread
            (sd 0.471 vs 0.338), so the same noise floor buys ~1.4x the margin.
            The `same` arm is the control: it forces the readout through r^3 alone
            and should degrade in proportion to the spread, NOT collapse. If it
            collapses, the model is keying on material rather than on mass.

  permutation strata     within (e-band x material policy). The leak guarantee is
            per cell, so the ratios must be permuted inside a cell, never across.

  energy     any permutation preserves the ratio marginal, so that constraint is
            free; the remaining freedom is spent searching permutations whose
            KE-ratio distribution also matches. Unmatched, violations gain energy
            systematically (E[dKE] >= |dp|^2/2M > 0).
"""
from __future__ import annotations
import argparse, importlib.util, json, logging, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gs", HERE / "generate_scene.py")
gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)

E_ELASTIC, E_INELASTIC = 1.0, 0.15
MIN_REL_VIOLATION = 0.08     # below this the label is unrecoverable by anyone
MAX_REL_VIOLATION = 1.50     # above it the aftermath is gratuitous


def probe_scene(seed: int, e: float, pair: str, scratch: Path):
    """Physics only. Returns the quantities the plan needs, or None if rejected."""
    L = gs.generate(seed, scratch / f"{seed}_{e}_{pair}", f"s{seed}", render=False,
                    violate=False, arm="legacy", restitution=e, material_pair=pair)
    pf = {p["frame"]: p for p in L["per_frame"]}
    v = lambda t, fr: np.array(pf[fr][t]["velocity"][:2], float)
    mA, mB = L["objects"]["A"]["mass"], L["objects"]["B"]["mass"]
    uA, uB = v("A", 18), v("B", 18)
    dvA, dvB = v("A", 22) - uA, v("B", 22) - uB
    return dict(seed=seed, e=e, pair=pair, rho=L["mass_ratio"], mA=mA, mB=mB,
                uA=uA, uB=uB, dvA=dvA, dvB=dvB,
                p_pre=float(np.linalg.norm(mA * uA + mB * uB)),
                ke_pre=float(0.5 * mA * uA @ uA + 0.5 * mB * uB @ uB),
                material=(L["objects"]["A"]["material"], L["objects"]["B"]["material"]))


def outcome(s, rho_p):
    """Analytic aftermath if scene `s` is given implied ratio `rho_p`."""
    a, b = np.linalg.norm(s["dvA"]), np.linalg.norm(s["dvB"])
    total = a + b
    a2, b2 = total / (1 + rho_p), total * rho_p / (1 + rho_p)
    dvA2, dvB2 = (a2 / a) * s["dvA"], (b2 / b) * s["dvB"]
    dp = s["mA"] * dvA2 + s["mB"] * dvB2
    vA, vB = s["uA"] + dvA2, s["uB"] + dvB2
    ke = (0.5 * s["mA"] * vA @ vA + 0.5 * s["mB"] * vB @ vB) / s["ke_pre"]
    return float(np.linalg.norm(dp) / s["p_pre"]), float(ke)


def auc(v, c):
    v, c = np.asarray(v, float), np.asarray(c, float)
    return float(((v[:, None] > c[None, :]).sum()
                  + 0.5 * (v[:, None] == c[None, :]).sum()) / (len(v) * len(c)))


def _matching(adj, rng):
    """One perfect matching of scenes to donors, or None. Kuhn's algorithm with a
    randomised order, so repeated calls explore different matchings."""
    n = len(adj)
    match = [-1] * n          # donor -> scene
    order = list(rng.permutation(n))
    def augment(i, seen):
        for j in rng.permutation(np.flatnonzero(adj[i])):
            if seen[j]:
                continue
            seen[j] = True
            if match[j] == -1 or augment(match[j], seen):
                match[j] = i
                return True
        return False
    for i in order:
        if not augment(i, [False] * n):
            return None
    perm = [0] * n
    for j, i in enumerate(match):
        perm[i] = j
    return np.array(perm)


def admissibility(cell):
    """rel-violation and KE-ratio for every (scene, donor) pair, and which pairs are
    usable: donor j is allowed for scene i when the violation lands in range and j != i."""
    n = len(cell)
    rho = np.array([s["rho"] for s in cell])
    rel_m = np.zeros((n, n)); ke_m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rel_m[i, j], ke_m[i, j] = outcome(cell[i], rho[j])
    adj = (rel_m >= MIN_REL_VIOLATION) & (rel_m <= MAX_REL_VIOLATION)
    np.fill_diagonal(adj, False)
    return rel_m, ke_m, adj


def choose_permutation(cell, rng, tries=400):
    """A derangement of the mass ratios inside one cell.

    Rejection sampling cannot do this: requiring EVERY one of n violations to land
    in the usable size range makes a random derangement admissible with probability
    that vanishes in n. Instead build the admissibility graph -- donor j is allowed
    for scene i when the resulting violation is usable and j != i -- and draw
    perfect matchings from it, which are admissible by construction. Any matching
    is a permutation, so the implied-ratio marginal is still exact; the restarts
    are spent picking one whose KE-ratio distribution also matches.
    """
    n = len(cell)
    ke_h = np.array([outcome(s, s["rho"])[1] for s in cell])
    rel_m, ke_m, adj = admissibility(cell)
    thin = np.flatnonzero(adj.sum(1) == 0)
    if len(thin):
        raise RuntimeError(f"{len(thin)} scenes have no admissible donor in this cell")
    best = None
    for _ in range(tries):
        perm = _matching(adj, rng)
        if perm is None:
            continue
        rel = rel_m[np.arange(n), perm]; ke = ke_m[np.arange(n), perm]
        score = abs(auc(ke, ke_h) - 0.5) + abs(ke.mean() - ke_h.mean()) / ke_h.mean()
        if best is None or score < best[0]:
            best = (score, perm, rel, ke, ke_h)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mixed", type=int, default=120, help="seeds in the main arm")
    ap.add_argument("--n-same", type=int, default=30, help="seeds in the r^3 control arm")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--plan-seed", type=int, default=0)
    ap.add_argument("-o", "--out", type=str, default="momentum_dataset/conservation_set_v1/plan.json")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    rng = np.random.default_rng(args.plan_seed)
    scratch = Path(tempfile.mkdtemp(prefix="plan_"))
    cache_path = Path(args.out).with_name("plan_probe_cache.npz")
    cache = {}
    if cache_path.exists():
        raw = np.load(cache_path, allow_pickle=True)
        cache = raw["cache"].item()
        print(f"  loaded {len(cache)} cached probes from {cache_path}", flush=True)

    # Each seed belongs to ONE cell: (material policy, e band). Half of each arm
    # goes to each e band, so the four cells are balanced.
    want = {("mixed", E_ELASTIC): args.n_mixed // 2, ("mixed", E_INELASTIC): args.n_mixed // 2,
            ("same", E_ELASTIC): args.n_same // 2,   ("same", E_INELASTIC): args.n_same // 2}
    cells = {k: [] for k in want}
    order = [k for k in want for _ in range(want[k])]
    rng.shuffle(order)

    seed = args.seed_offset
    rejected = 0

    def next_scene(target):
        nonlocal seed, rejected
        pair, e = target
        while True:
            key = f"{seed}|{e}|{pair}"
            if key in cache:
                s = cache[key]; seed += 1
                if s is None:
                    rejected += 1; continue
                return s
            try:
                s = probe_scene(seed, e, pair, scratch)
                cache[key] = s; seed += 1
                return s
            except Exception:
                cache[key] = None; rejected += 1; seed += 1

    for target in order:
        cells[target].append(next_scene(target))
        done = sum(len(v) for v in cells.values())
        if done % 10 == 0:
            print(f"  probed {done}/{len(order)} (rejected {rejected})", flush=True)

    # A scene no donor in its cell can violate by a usable amount (every swap lands
    # under MIN_REL_VIOLATION, e.g. a glancing inelastic hit with tiny deflections)
    # has no place in any matching. Drop it -- both twins, before any class is
    # assigned, so the classes stay balanced -- and draw the next seed for that
    # cell. Rare: 1 seed in 500 at n_mixed=400.
    no_donor = []
    for target in cells:
        while True:
            thin = np.flatnonzero(admissibility(cells[target])[2].sum(1) == 0)
            if not len(thin):
                break
            for i in sorted(thin, reverse=True):
                no_donor.append(cells[target].pop(i)["seed"])
            for _ in thin:
                cells[target].append(next_scene(target))
    if no_donor:
        print(f"  replaced {len(no_donor)} seeds with no admissible donor: {no_donor}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, cache=np.array(cache, dtype=object))
    plan, summary = [], {}
    for key, cell in cells.items():
        pair, e = key
        got = choose_permutation(cell, rng)
        if got is None:
            raise RuntimeError(f"no admissible permutation for cell {key}")
        score, perm, rel, ke, ke_h = got
        rho = [s["rho"] for s in cell]
        for i, s in enumerate(cell):
            common = dict(seed=s["seed"], restitution=e, material_pair=pair,
                          cell=f"{'elastic' if e == E_ELASTIC else 'inelastic'}_{pair}",
                          true_rho=s["rho"], materials=list(s["material"]))
            plan.append({**common, "name": f"{common['cell']}_honest_seed{s['seed']}",
                         "violate": False, "implied_ratio": None,
                         "expected_rel_violation": 0.0, "expected_ke_ratio": ke_h[i]})
            plan.append({**common, "name": f"{common['cell']}_violating_seed{s['seed']}",
                         "violate": True, "implied_ratio": rho[perm[i]],
                         "donor_seed": cell[perm[i]]["seed"],
                         "expected_rel_violation": rel[i], "expected_ke_ratio": ke[i]})
        summary[f"{pair}_{e}"] = dict(
            n_pairs=len(cell), rel_violation_median=float(np.median(rel)),
            rel_violation_range=[float(rel.min()), float(rel.max())],
            ke_auc_violating_vs_honest=auc(ke, ke_h),
            ke_mean_honest=float(ke_h.mean()), ke_mean_violating=float(ke.mean()),
            implied_ratio_marginal_matches="exact (permutation)",
            permutation_score=float(score))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"clips": plan, "summary": summary,
               "n_clips": len(plan), "seeds_rejected": rejected,
               "seeds_without_donor": no_donor,
               "config": vars(args)}, open(out, "w"), indent=1)
    print(f"\nwrote {out}  ({len(plan)} clips, {len(plan)//2} pairs, {rejected} seeds rejected)")
    for k, v in summary.items():
        print(f"  {k:18s} pairs {v['n_pairs']:3d}  |dp|/|p| median {v['rel_violation_median']:.2f} "
              f"range [{v['rel_violation_range'][0]:.2f},{v['rel_violation_range'][1]:.2f}]  "
              f"KE AUC {v['ke_auc_violating_vs_honest']:.3f}")


if __name__ == "__main__":
    main()
