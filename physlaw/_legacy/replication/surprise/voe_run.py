#!/usr/bin/env python
"""VoE surprise for momentum twins (conservation_set_1000) and the energy ladder
(energy_ladder_v1), V-JEPA2 predictor, nothing fit. Adapted from
scripts/kinetic_energy/voe_energy_twins.py (same encode-pre-alone + m.predictor call,
the §15 trap avoided; same dists(); ball tokens via preprocess_masks/masks_to_latent_grid).

One pass over the data scores every requested model (clips are read once, sequentially,
by one reader thread -- the FUSE mount is fragile). Per group:
  context clip  -> pre frames 1..20 encoded alone -> predictor -> P
  every future  -> full frames 1..48 encoded -> T (target tokens 2560..6144)
  identity: each future's OWN pre frames -> predictor -> cos with P
  substrate: persistence (tile last pre tubelet) vs context clip's own future
  room: cos(T_ctx, T_future)
Token sets: all target tokens; 'balls' = union of ball-occupied tokens across all futures
of the group (common set, so paired); 'own' = the future's own ball tokens.
Per-tubelet distances on all/balls are stored for the trend test.
Writes one JSON line per group per model (resumable).
"""
from __future__ import annotations
import argparse, glob, json, os, queue, re, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

REPO = Path("/workspace/meta/physics_interp_video_diffusion")
sys.path.insert(0, str(REPO / "scripts"))
import _momentum_path  # noqa: F401,E402
from physics_interp.video import load_frames, load_segmentation, masks_to_latent_grid  # noqa: E402
from physics_interp.models.encoders import VJEPA2Adapter  # noqa: E402

N_PRE, N_FR, PER_TUB = 20, 48, 256
PRE_IDX = list(range(0, N_PRE))      # into the frames-1..48 array => frames 1..20
FULL_IDX = list(range(0, N_FR))     # frames 1..48


def groups_momentum(root, drop=("255", "262", "264")):
    out = []
    for cell in ("elastic_mixed", "elastic_same", "inelastic_mixed", "inelastic_same"):
        for h in sorted(glob.glob(f"{root}/{cell}_honest_seed*")):
            seed = re.search(r"seed(\d+)$", h).group(1)
            v = f"{root}/{cell}_violating_seed{seed}"
            if not os.path.isdir(v):
                continue
            if cell == "elastic_mixed" and seed in drop:
                continue
            out.append(dict(gid=f"{cell}_{seed}", cell=cell, seed=seed, ctx="honest",
                            futures={"honest": h, "violating": v}))
    return out


def groups_ladder(root):
    man = json.load(open(f"{root}/manifest.json"))
    by = {}
    for name in man["complete"]:
        s, arm = name.split("_", 1)
        by.setdefault(s, {})[arm] = f"{root}/{name}"
    out = []
    for s in sorted(by):
        if "honest" not in by[s]:
            continue
        out.append(dict(gid=s, seed=s, ctx="honest", futures=by[s]))
    return out


def dists(P, T, idx):
    d = P[idx] - T[idx]
    pn = P[idx] / (np.linalg.norm(P[idx], axis=1, keepdims=True) + 1e-8)
    tn = T[idx] / (np.linalg.norm(T[idx], axis=1, keepdims=True) + 1e-8)
    return dict(l1=float(np.abs(d).mean()), mse=float((d ** 2).mean()),
                cos=float((pn * tn).sum(1).mean()))


def per_tub(P, T, idx):
    n_t = P.shape[0] // PER_TUB
    res = {"l1": [], "mse": [], "cos": [], "n": []}
    for t in range(n_t):
        sl = slice(t * PER_TUB, (t + 1) * PER_TUB)
        ii = idx[sl]
        res["n"].append(int(ii.sum()))
        if ii.sum() == 0:
            for k in ("l1", "mse", "cos"):
                res[k].append(None)
            continue
        dd = dists(P[sl], T[sl], ii)
        for k in ("l1", "mse", "cos"):
            res[k].append(dd[k])
    return res


def cosmean(A, B):
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return float((an * bn).sum(1).mean())


IO_THREADS = int(os.environ.get("VOE_IO_THREADS", "6"))
_POOL = None


def _read_png(path):
    import imageio.v3 as iio
    with open(path, "rb") as f:
        b = f.read()
    return np.asarray(iio.imread(b, extension=".png"))


def load_clip(d):
    """Frames 1..48 (RGB) and segmentation 21..48 only -- the only files the measurement uses
    (frame 0 is dropped by the tubelet layout; ball tokens are read on target cells only).
    Returns frames [48,H,W,3] for frames 1..48 and seg [48,H,W] with frames 1..20 zeroed.
    Files are fetched by a small thread pool: geesefs is latency-bound (~0.4 s/open)."""
    fpaths = [f"{d}/frames/frame_{i:04d}.png" for i in range(1, 49)]
    spaths = [f"{d}/segmentation/segmentation_{i:04d}.png" for i in range(21, 49)]
    imgs = list(_POOL.map(_read_png, fpaths + spaths))
    fr = np.stack([x[..., :3] for x in imgs[:48]])
    sg = [x[..., 0] if x.ndim == 3 else x for x in imgs[48:]]
    seg = np.zeros((48,) + sg[0].shape, sg[0].dtype)
    seg[20:] = np.stack(sg)
    return fr, seg


def reader(groups, q, labels_keys):
    global _POOL
    from concurrent.futures import ThreadPoolExecutor
    _POOL = ThreadPoolExecutor(IO_THREADS)
    for g in groups:
        try:
            data = {}
            for role, d in g["futures"].items():
                fr, seg = load_clip(d)
                lab = json.load(open(f"{d}/labels.json"))
                meta = {k: lab.get(k) for k in labels_keys}
                data[role] = (fr, seg, meta)
            q.put((g, data))
        except Exception as ex:  # noqa: BLE001
            print(f"READ FAIL {g['gid']}: {ex}", flush=True)
            q.put((g, None))
    q.put(None)


class Scorer:
    def __init__(self, repo, dev, bs):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(repo, dtype=torch.float32).eval().to(dev)
        self.ad = VJEPA2Adapter(pipe=self.m, repo=repo, device=torch.device(dev), dtype=torch.float32)
        self.dev, self.bs = dev, bs
        self.n_pre = (N_PRE // 2) * PER_TUB
        self.n_tok = (N_FR // 2) * PER_TUB
        self.layout = SimpleNamespace(frames=N_FR // self.ad.tubelet, height=16, width=16)

    @torch.no_grad()
    def enc(self, frames_list, idx):
        outs = []
        for i in range(0, len(frames_list), self.bs):
            x = torch.cat([self.ad.preprocess(f[idx]) for f in frames_list[i:i + self.bs]]).to(self.dev)
            outs.append(self.m(pixel_values_videos=x, skip_predictor=True).last_hidden_state)
        return torch.cat(outs)

    @torch.no_grad()
    def pred(self, hp):
        outs = []
        for i in range(0, hp.shape[0], self.bs):
            h = hp[i:i + self.bs]
            B = h.shape[0]
            ctx = [torch.arange(self.n_pre, device=self.dev)[None].expand(B, -1)]
            tgt = [torch.arange(self.n_pre, self.n_tok, device=self.dev)[None].expand(B, -1)]
            outs.append(self.m.predictor(encoder_hidden_states=h, context_mask=ctx,
                                         target_mask=tgt).last_hidden_state.float().cpu().numpy())
        return np.concatenate(outs)

    def ball_tokens(self, seg):
        to_cell = lambda t: min(t // self.ad.tubelet, self.layout.frames - 1)  # noqa: E731
        s = self.ad.preprocess_masks(seg[FULL_IDX])
        occ = masks_to_latent_grid(s, self.layout, ids=(1, 2), frame_to_cell=to_cell)
        return ((occ[0] + occ[1]).reshape(-1) > 0)[self.n_pre:self.n_tok]

    def score(self, g, data):
        roles = list(g["futures"])
        ci = roles.index(g["ctx"])
        frames = [data[r][0] for r in roles]
        hp = self.enc(frames, PRE_IDX)                      # every future's own pre frames
        Pall = self.pred(hp)
        P = Pall[ci]
        ident = {r: cosmean(P, Pall[i]) for i, r in enumerate(roles)}
        T = self.enc(frames, FULL_IDX)[:, self.n_pre:self.n_tok].float().cpu().numpy()
        T = {r: T[i] for i, r in enumerate(roles)}
        last = hp[ci, self.n_pre - PER_TUB:self.n_pre].float().cpu().numpy()
        PERS = np.tile(last, ((self.n_tok - self.n_pre) // PER_TUB, 1))
        own = {r: self.ball_tokens(data[r][1]) for r in roles}
        union = np.zeros_like(own[roles[0]])
        for r in roles:
            union |= own[r]
        allx = np.ones_like(union)
        row = dict(gid=g["gid"], seed=g["seed"], cell=g.get("cell"), ctx=g["ctx"],
                   ball_frac=float(union.mean()), identity=ident,
                   meta={r: data[r][2] for r in roles}, futures={})
        Tc = T[g["ctx"]]
        row["substrate"] = {tag: dict(pred=dists(P, Tc, idx), pers=dists(PERS, Tc, idx))
                            for tag, idx in (("all", allx), ("balls", union))}
        for r in roles:
            fr = {}
            for tag, idx in (("all", allx), ("balls", union), ("own", own[r])):
                fr[tag] = dists(P, T[r], idx) if idx.any() else None
            fr["pers"] = {tag: (dists(PERS, T[r], idx) if idx.any() else None)
                          for tag, idx in (("all", allx), ("balls", union))}
            fr["room"] = {tag: (dists(Tc, T[r], idx)["cos"] if idx.any() else None)
                          for tag, idx in (("all", allx), ("balls", union))}
            fr["tub_all"] = per_tub(P, T[r], allx)
            fr["tub_balls"] = per_tub(P, T[r], union)
            row["futures"][r] = fr
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["momentum", "ladder"], required=True)
    ap.add_argument("--models", default="vitl,vitg")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--outdir", default="/root/fast/repl/surprise")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    if args.exp == "momentum":
        groups = groups_momentum(str(REPO / "datasets/conservation_set_1000"))
        keys = ["restitution", "stratum", "b_moving", "violates_momentum", "mass_ratio"]
    else:
        groups = groups_ladder(str(REPO / "datasets/energy_ladder_v1"))
        keys = ["arm", "restitution", "restitution_realised", "energy_created", "stratum", "mass_ratio"]
    # blobs of HF_HOME's vjepa2-vit{l,g}-fpc64-256 snapshots (the FUSE mount lost the symlinks)
    repos = {"vitl": "/dev/shm/surprise/models/vitl", "vitg": "/dev/shm/surprise/models/vitg"}
    models = args.models.split(",")
    outs = {mk: Path(args.outdir) / f"{args.exp}_{mk}.jsonl" for mk in models}
    done = set()
    for mk, p in outs.items():
        s = set(json.loads(l)["gid"] for l in open(p)) if p.exists() else set()
        done = s if not done and mk == models[0] else (done & s)
    groups = [g for g in groups if g["gid"] not in done]
    if args.limit:
        groups = groups[: args.limit]
    print(f"{args.exp}: {len(groups)} groups to score (skipped {len(done)} done)", flush=True)
    scorers = {}
    for mk in models:
        t = time.time()
        scorers[mk] = Scorer(repos[mk], dev, args.bs)
        print(f"loaded {mk} in {time.time() - t:.1f}s  "
              f"gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    q = queue.Queue(maxsize=3)
    th = threading.Thread(target=reader, args=(groups, q, keys), daemon=True)
    th.start()
    fhs = {mk: open(p, "a") for mk, p in outs.items()}
    t0, i, tread = time.time(), 0, 0.0
    while True:
        tw = time.time()
        item = q.get()
        tread += time.time() - tw
        if item is None:
            break
        g, data = item
        i += 1
        if data is None:
            continue
        for mk, sc in scorers.items():
            row = sc.score(g, data)
            row["model"] = mk
            fhs[mk].write(json.dumps(row) + "\n")
            fhs[mk].flush()
        el = time.time() - t0
        if i % 10 == 0 or i == len(groups):
            print(f"  [{i}/{len(groups)}] {g['gid']}  {el:7.1f}s  waiting-on-read {tread:6.1f}s  "
                  f"eta {el / i * (len(groups) - i) / 60:5.1f} min", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
