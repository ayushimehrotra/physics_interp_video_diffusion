#!/usr/bin/env python
"""VoE on `energy_voe_pilot`: is the IMPOSSIBLE twin (e=k) more surprising than the
LEGAL one (e=1/k)? V-JEPA2's own predictor, no fitting anywhere.

WHY THIS SET AND NOT A PROBE. §46c: `e` is mass-free and visible, so a probe reads the
two classes off separation speed and demonstrates nothing (measured, §46d: AUC 0.825,
below the 1.000 mass-free ceiling). For violation of expectation the visibility is a
REQUIREMENT -- the model cannot react to what it cannot see. The question is whether
`impossible` is treated differently from merely `unusual`, and only the k / 1/k pairing
isolates that: both twins sit the same distance from elastic in log space, one separates
k times too fast and the other k times too slow, and exactly one is impossible.

    more surprised by k than by 1/k  ->  an energy prior
    equally surprised by both        ->  novelty detection

THE CONTEXT COMES FROM THE ANCHOR, which is the one change of substance from
`voe_twins.py`. All THREE clips of a seed share a bit-identical pre-contact segment
(the validator scores twin pre-contact positions at 0.0), so any of them yields the same
expectation -- but taking it from the `e=1` anchor means the prediction belongs to
NEITHER twin, and no asymmetry can enter through the choice of context. The identity
control below verifies the three predictions really do coincide.

THE ANCHOR IS ALSO SCORED AS A FUTURE, which `voe_twins.py` had no analogue for. It
gives the novelty curve its origin: if surprise is symmetric in |log e| the anchor sits
below both twins by the same margin, and that is the null this set was built to be able
to distinguish from an energy prior.

THE §15 TRAP, CARRIED OVER. HF's `VJEPA2Model` runs the encoder on the FULL clip and
only then splits context/target, so its "context" tokens have already seen the
aftermath. This encodes frames 1-20 alone and calls `m.predictor(...)` directly.

CONTROLS
  substrate    the predictor must beat PERSISTENCE (copy the last pre-contact tubelet)
               against the anchor future. If it does not, there is no expectation to
               violate and a null here is uninterpretable.
  identity     predictions from the anchor's, the impossible twin's and the legal
               twin's own pre-frames must agree at cos ~1.0.
  room         cos(T_anchor, T_twin) must be < 1: the futures have to actually differ
               in the tokens being scored, or there is nothing to be surprised by.
  null         nothing is fit, so exchangeability is exactly a sign test -- under H0
               each pair is a fair coin. n = 20 pairs: 15/20 reaches p = 0.021,
               14/20 only 0.058. This run can see a LARGE effect and nothing subtler.

    .venv/bin/python scripts/voe_energy_twins.py --dataset-dir momentum_dataset/energy_voe_pilot
"""
from __future__ import annotations
import argparse, glob, json, re, time
from collections import defaultdict
from math import comb
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics_interp.video import load_frames, load_segmentation, masks_to_latent_grid  # noqa: E402


def dists(P, T, idx):
    """P, T: [tokens, dim]; idx a boolean token selector. The model's own loss shapes."""
    d = P[idx] - T[idx]
    pn = P[idx] / (np.linalg.norm(P[idx], axis=1, keepdims=True) + 1e-8)
    tn = T[idx] / (np.linalg.norm(T[idx], axis=1, keepdims=True) + 1e-8)
    return dict(l1=float(np.abs(d).mean()), mse=float((d ** 2).mean()),
                cos=float((pn * tn).sum(1).mean()))


def sign_test(worse: np.ndarray) -> tuple[int, int, float]:
    k, n = int(worse.sum()), len(worse)
    return k, n, sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="momentum_dataset/energy_voe_pilot")
    ap.add_argument("--repo", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-pre-frames", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--n-seeds", type=int, default=0, help="0 = all")
    ap.add_argument("-o", "--out", default="docs/artifacts/voe_energy_vjepa2.json")
    args = ap.parse_args()

    from transformers import AutoModel
    from physics_interp.models.encoders import VJEPA2Adapter
    dev = args.device

    # ---- triples, keyed off e_target in the labels rather than the directory name --
    tri: dict[str, dict] = defaultdict(dict)
    ks: dict[str, float] = {}
    for p in sorted(glob.glob(f"{args.dataset_dir}/*_seed*/labels.json")):
        d = Path(p).parent
        seed = re.search(r"seed(\d+)$", d.name).group(1)
        e = json.load(open(p))["energy_edit"]["e_target"]
        role = "anchor" if abs(e - 1.0) < 1e-9 else ("impossible" if e > 1 else "legal")
        tri[seed][role] = d
        if role == "impossible":
            ks[seed] = e
    seeds = [s for s in sorted(tri) if len(tri[s]) == 3]
    if args.n_seeds:
        seeds = seeds[: args.n_seeds]
    print(f"{len(seeds)} complete triples in {args.dataset_dir}   "
          f"k = {sorted(set(round(v, 4) for v in ks.values()))}")

    print(f"loading V-JEPA2 on {dev} ...", flush=True)
    m = AutoModel.from_pretrained(args.repo, dtype=torch.float32).eval().to(dev)
    ad = VJEPA2Adapter(pipe=m, repo=args.repo, device=torch.device(dev), dtype=torch.float32)
    per_tub = 256
    n_pre = (args.n_pre_frames // 2) * per_tub
    n_tok = (args.n_frames // 2) * per_tub
    ctx = [torch.arange(n_pre, device=dev)[None]]
    tgt = [torch.arange(n_pre, n_tok, device=dev)[None]]
    layout = SimpleNamespace(frames=args.n_frames // ad.tubelet, height=16, width=16)
    to_cell = lambda t: min(t // ad.tubelet, layout.frames - 1)      # noqa: E731
    pre_idx = list(range(1, args.n_pre_frames + 1))
    full_idx = list(range(1, args.n_frames + 1))
    print(f"context tokens 0-{n_pre}, target {n_pre}-{n_tok} ({n_tok - n_pre} predicted)")

    @torch.no_grad()
    def enc(sdir, idx):
        x = ad.preprocess(load_frames(sdir, num_frames=49)[idx]).to(dev)
        return m(pixel_values_videos=x, skip_predictor=True).last_hidden_state

    @torch.no_grad()
    def pred(hp):
        return m.predictor(encoder_hidden_states=hp, context_mask=ctx,
                           target_mask=tgt).last_hidden_state[0].float().cpu().numpy()

    def ball_tokens(sdir):
        seg = ad.preprocess_masks(load_segmentation(sdir)[full_idx])
        occ = masks_to_latent_grid(seg, layout, ids=(1, 2), frame_to_cell=to_cell)
        return (occ[0] + occ[1]).reshape(-1) > 0

    def cosmean(A, B):
        an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return float((an * bn).sum(1).mean())

    rows, t0 = [], time.perf_counter()
    for i, seed in enumerate(seeds, 1):
        t = tri[seed]
        hp = enc(t["anchor"], pre_idx)                 # the one expectation
        P = pred(hp)
        id_imp = cosmean(P, pred(enc(t["impossible"], pre_idx)))
        id_leg = cosmean(P, pred(enc(t["legal"], pre_idx)))

        T = {r: enc(t[r], full_idx)[0, n_pre:n_tok].float().cpu().numpy()
             for r in ("anchor", "impossible", "legal")}
        last = hp[0, n_pre - per_tub:n_pre].float().cpu().numpy()
        PERS = np.tile(last, ((n_tok - n_pre) // per_tub, 1))

        union = np.zeros(n_tok, bool)
        for r in ("anchor", "impossible", "legal"):
            union |= ball_tokens(t[r])
        union = union[n_pre:n_tok]
        allx = np.ones(n_tok - n_pre, bool)

        row = dict(seed=seed, k=ks[seed], ball_frac=float(union.mean()),
                   id_impossible=id_imp, id_legal=id_leg)
        for tag, idx in (("all", allx), ("balls", union)):
            row[tag] = {r: dists(P, T[r], idx) for r in T}
            row[tag]["pers_anchor"] = dists(PERS, T["anchor"], idx)
            row[tag]["room_impossible"] = dists(T["anchor"], T["impossible"], idx)["cos"]
            row[tag]["room_legal"] = dists(T["anchor"], T["legal"], idx)["cos"]
        rows.append(row)
        el = time.perf_counter() - t0
        print(f"  [{i}/{len(seeds)}] seed {seed}  {el:6.1f}s  "
              f"eta {el / i * (len(seeds) - i) / 60:5.1f} min", flush=True)

    # ---- controls -------------------------------------------------------------
    print(f"\n{'=' * 84}\nCONTROLS")
    ii = np.array([r["id_impossible"] for r in rows])
    il = np.array([r["id_legal"] for r in rows])
    print(f"  identity   prediction from anchor-pre vs impossible-pre  cos "
          f"min {ii.min():.5f} mean {ii.mean():.5f}")
    print(f"             prediction from anchor-pre vs legal-pre       cos "
          f"min {il.min():.5f} mean {il.mean():.5f}   (both must be ~1.0)")
    bf = np.array([r["ball_frac"] for r in rows])
    print(f"  ball tokens {bf.mean():.1%} of the {n_tok - n_pre} target tokens "
          f"[{bf.min():.1%}, {bf.max():.1%}]")
    for tag in ("all", "balls"):
        for met, better in (("cos", "higher"), ("mse", "lower")):
            pa = np.array([r[tag]["pers_anchor"][met] for r in rows])
            da = np.array([r[tag]["anchor"][met] for r in rows])
            beats = (da > pa) if met == "cos" else (da < pa)
            print(f"  substrate[{tag:5s}] {met}: predictor {da.mean():.4f} vs "
                  f"persistence {pa.mean():.4f} ({better} better)  "
                  f"predictor wins {int(beats.sum())}/{len(beats)}")
        ri = np.array([r[tag]["room_impossible"] for r in rows])
        rl = np.array([r[tag]["room_legal"] for r in rows])
        print(f"  room[{tag:5s}]      cos(anchor, impossible) {ri.mean():.4f}   "
              f"cos(anchor, legal) {rl.mean():.4f}   (must be < 1)")

    # ---- the measurement ------------------------------------------------------
    rep = {"n_seeds": len(rows), "n_target_tokens": n_tok - n_pre, "rows": rows,
           "summary": {}}
    for tag in ("all", "balls"):
        print(f"\n{'=' * 84}")
        print(f"VIOLATION OF EXPECTATION -- {'all target tokens' if tag == 'all' else 'BALL-OCCUPIED tokens only'}")
        print(f"{'=' * 84}")
        print(f"  {'metric':<6}{'anchor':>10}{'legal':>10}{'impossible':>12}"
              f"{'imp-leg':>10}{'imp worse':>11}{'sign p':>9}")
        rep["summary"][tag] = {}
        for met, sign in (("cos", -1), ("l1", +1), ("mse", +1)):
            a = np.array([r[tag]["anchor"][met] for r in rows])
            g = np.array([r[tag]["legal"][met] for r in rows])
            b = np.array([r[tag]["impossible"][met] for r in rows])
            k, n, p = sign_test(sign * (b - g) > 0)
            print(f"  {met:<6}{a.mean():>10.4f}{g.mean():>10.4f}{b.mean():>12.4f}"
                  f"{(b - g).mean():>+10.4f}{f'{k}/{n}':>11}{p:>9.4f}")
            # the symmetry check: is each twin surprising RELATIVE TO THE ANCHOR?
            ka, _, pa_ = sign_test(sign * (b - a) > 0)
            kg, _, pg_ = sign_test(sign * (g - a) > 0)
            rep["summary"][tag][met] = {
                "anchor": float(a.mean()), "legal": float(g.mean()),
                "impossible": float(b.mean()), "diff_imp_minus_leg": float((b - g).mean()),
                "n_impossible_worse": k, "n": n, "sign_test_p": float(p),
                "impossible_vs_anchor": {"n_worse": ka, "p": float(pa_)},
                "legal_vs_anchor": {"n_worse": kg, "p": float(pg_)}}
        print("  -- vs the anchor (both twins SHOULD beat it; that is novelty, not physics) --")
        for met in ("cos", "l1", "mse"):
            s = rep["summary"][tag][met]
            print(f"     {met:<5} impossible worse than anchor "
                  f"{s['impossible_vs_anchor']['n_worse']}/{len(rows)} p={s['impossible_vs_anchor']['p']:.4f}"
                  f"   |   legal worse than anchor "
                  f"{s['legal_vs_anchor']['n_worse']}/{len(rows)} p={s['legal_vs_anchor']['p']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
