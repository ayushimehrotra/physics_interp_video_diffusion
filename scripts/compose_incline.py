"""mu direct vs composed from decoded (a, theta), incline_set_v2. Matches fit_probe's protocol."""
import json, glob, sys, numpy as np
sys.path.insert(0, '/workspace/physics_interp_video_diffusion')
from physics_interp.interp import fit_probe

MODEL = sys.argv[1]
OUT = f'outputs_incline_{MODEL}/momentum_probe/activations'
POOLS = ["A.pre","A.post","A.all","B.pre","B.post","B.all","all.pre","all.post","all.all"]
scenes = sorted(glob.glob(f'{OUT}/scene_*.npz'))
lab = {}
for f in scenes:
    sid = f.split('/')[-1][:-4]
    lab[sid] = json.load(open(f'momentum_dataset/incline_set_v2/{sid}/labels.json'))['friction']
sids = sorted(lab)
G = lab[sids[0]]['gravity']
mu   = np.array([[lab[s]['mu']] for s in sids])
th   = np.array([[lab[s]['theta']] for s in sids])
a    = np.array([[lab[s]['accel_measured']] for s in sids])

Z = {s: np.load(f'{OUT}/{s}.npz') for s in sids}
SITES = [k for k in Z[sids[0]].keys() if k.startswith('act.block.')]
SITES.sort(key=lambda k: int(k.split('.')[2]))

def feats(site, pool):
    return np.stack([Z[s][site][0, POOLS.index(pool)].astype(np.float32) for s in sids])

def pred(site, pool, Y):
    """held-out predictions under fit_probe's exact split, plus its R2"""
    X = feats(site, pool)
    r = fit_probe(X, Y, permutation_control=False)
    n = len(sids); rng = np.random.default_rng(0); order = rng.permutation(n)
    tr, te = order[:int(round(0.6*n))], order[int(round(0.6*n)):]
    n_inner = max(4, int(round(0.75*len(tr)))); itr = tr[:n_inner]
    m, sd = X[itr].mean(0), X[itr].std(0)+1e-6
    from physics_interp.interp import ridge_fit
    m2, sd2 = X[tr].mean(0), X[tr].std(0)+1e-6
    W, b = ridge_fit((X[tr]-m2)/sd2, Y[tr], r['alpha'])
    return r['r2'], (X[te]-m2)/sd2 @ W + b, te

def r2(y, p):
    return float(1 - ((y-p)**2).sum() / ((y-y.mean(0))**2).sum())

print(f'=== {MODEL} · incline_set_v2 · n={len(sids)} · fit_probe protocol (60/40, seed 0)\n')
rows = []
for pool in ["A.pre","A.all","all.pre","all.all"]:
    for site in SITES:
        d_mu, p_mu, te = pred(site, pool, mu)
        _,    p_a,  _  = pred(site, pool, a)
        _,    p_th, _  = pred(site, pool, th)
        comp = (np.sin(p_th) - p_a/G) / np.cos(p_th)
        rows.append((pool, site, d_mu, r2(mu[te], comp), r2(a[te], p_a), r2(th[te], p_th)))
print(f'{"pool":<9}{"block":>6}{"mu DIRECT":>11}{"mu COMPOSED":>13}{"a":>8}{"theta":>8}')
best = {}
for pool, site, d, c, ra, rt in rows:
    if pool not in best or c > best[pool][1]: best[pool] = (d, c, site, ra, rt)
for pool in ["A.pre","A.all","all.pre","all.all"]:
    d, c, site, ra, rt = best[pool]
    print(f'{pool:<9}{site.split(".")[2]:>6}{d:>11.3f}{c:>13.3f}{ra:>8.3f}{rt:>8.3f}')
bd = max(r[2] for r in rows); bc = max(r[3] for r in rows)
print(f'\nbest DIRECT anywhere   {bd:.3f}')
print(f'best COMPOSED anywhere {bc:.3f}')
print(f'composed - direct      {bc-bd:+.3f}')
print(f'the bar (acceleration alone, corr -0.645)  0.416')
json.dump([{k: v for k, v in zip(("pool","site","direct","composed","a","theta"), r)} for r in rows],
          open(f'/tmp/claude-0/-workspace/a174e055-0555-41ae-a752-85214cf89308/scratchpad/compose_{MODEL}.json','w'))
