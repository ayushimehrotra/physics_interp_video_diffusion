"""Cross-geometry transfer, BOTH directions, plus the diagnostic cross-target check.

    .venv/bin/python old_experiments/scripts/transfer_geometry.py        # no GPU, ~2 min, existing captures

Written for CLAUDE.md §35c. Reads `outputs/{slide,incline}_<model>` and needs no
render and no re-capture.

  slide_set_v2   flat,   mu = a/g          (g=1.0)   -> corr(mu, a) = +0.9998
  incline_set_v2 tilted, mu = (sin th - a/g)/cos th  -> corr(mu, a) = -0.645

Same coefficient, two different functions of the visible kinematics, and the sign
of the mu<->a relation FLIPS between them. Three questions per direction:

  1. does a mu probe trained on one geometry predict mu on the other?
  2. CONTROL: does a `size` probe cross the same domain gap? (camera 14 vs 18,
     flat vs tilted, different band pattern). If the control crosses and mu does
     not, the failure is not generic domain shift.
  3. DIAGNOSTIC: does the mu probe predict the other set's ACCELERATION? If the
     probe is really an acceleration readout wearing mu's name, this is where it
     shows up.

Per-domain standardisation of X and of the target: a constant offset or scale
difference between two renders must not by itself decide the test.
"""
import json, glob, os, sys, numpy as np
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))  # noqa: E401,E702
import _momentum_path  # noqa: F401,E402
from physics_interp.interp import ridge_fit
POOLS=['A.pre','A.post','A.all','B.pre','B.post','B.all','all.pre','all.post','all.all']
POOL='all.all'          # frame-wide: mask is all-ones, so no mask-geometry difference

def load(o):
    acts,labels={},[]
    for f in sorted(glob.glob(f'{o}/momentum_probe/activations/scene_*.npz')):
        z=np.load(f,allow_pickle=False); labels.append(json.loads(bytes(z['meta']).decode())['labels'])
        for k in z.files:
            if k.startswith('act.'): acts.setdefault(k[4:],[]).append(np.asarray(z[k],np.float64))
    return {k:np.stack(v)[:,0] for k,v in acts.items()},labels

def r2(y,p): return float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())

def lab(labels,name,kind):
    if name=='mu':    return np.array([l['friction']['mu'] for l in labels])
    if name=='size':  return np.array([l['objects']['A']['size'] for l in labels])
    if name=='accel':
        k='decel_measured' if kind=='slide' else 'accel_measured'
        return np.array([abs(l['friction'][k]) for l in labels])
    raise KeyError(name)

for model in ['videomae-base','vjepa2-vitl']:
    A={'slide':load(f'outputs/slide_{model}'),'incline':load(f'outputs/incline_{model}')}
    sites=sorted(set(A['slide'][0])&set(A['incline'][0]),
                 key=lambda s:(1,int(s.split('.')[1])) if s.startswith('block.') else (0,-1))
    mu={k:lab(A[k][1],'mu',k) for k in A}
    lo,hi=max(mu['slide'].min(),mu['incline'].min()),min(mu['slide'].max(),mu['incline'].max())
    keep={k:(mu[k]>=lo)&(mu[k]<=hi) for k in A}
    print(f'\n{"="*84}\n{model}    mu overlap [{lo:.3f}, {hi:.3f}]   '
          f'slide {keep["slide"].sum()} / incline {keep["incline"].sum()} scenes   pool={POOL}')

    for src,dst in [('incline','slide'),('slide','incline')]:
        Xs=A[src][0]; Xd=A[dst][0]; ks,kd=keep[src],keep[dst]
        print(f'\n  --- TRAIN ON {src.upper()}  ->  TEST ON {dst.upper()}  '
              f'(n_train {ks.sum()}, n_test {kd.sum()})')
        for tgt_tr,tgt_te in [('mu','mu'),('size','size'),('mu','accel')]:
            ytr=lab(A[src][1],tgt_tr,src)[ks]; yte=lab(A[dst][1],tgt_te,dst)[kd]
            ytr_z=(ytr-ytr.mean())/ytr.std(); yte_z=(yte-yte.mean())/yte.std()
            rows=[]
            for site in sites:
                Ftr=Xs[site][ks][:,POOLS.index(POOL)]; Fte=Xd[site][kd][:,POOLS.index(POOL)]
                mtr,str_=Ftr.mean(0),Ftr.std(0)+1e-6
                mte,ste=Fte.mean(0),Fte.std(0)+1e-6
                Ztr=(Ftr-mtr)/str_; Zte=(Fte-mte)/ste      # each domain by its OWN moments
                n=len(ytr); tr=np.arange(n)%3!=0; te=~tr
                best=(-9e9,None)
                for al in [1,10,100,1e3,1e4]:
                    W,b=ridge_fit(Ztr[tr],ytr_z[tr,None],al)
                    v=r2(ytr_z[te],(Ztr[te]@W+b).ravel())
                    if v>best[0]: best=(v,al)
                within,al=best
                W,b=ridge_fit(Ztr,ytr_z[:,None],al)
                pred=(Zte@W+b).ravel()
                rows.append((site,within,r2(yte_z,pred),float(np.corrcoef(pred,yte_z)[0,1])))
            bw=max(rows,key=lambda r:r[1]); bt=max(rows,key=lambda r:abs(r[3]))
            nm=lambda s:s.split('.')[1] if 'block' in s else 'emb'
            tag=f'{tgt_tr}->{tgt_te}'
            print(f'    {tag:<12} within-{src} {bw[1]:.3f} @{nm(bw[0]):<4}'
                  f' | transfer R2 {bw[2]:+.3f}  corr {bw[3]:+.3f}'
                  f' | largest |corr| anywhere {bt[3]:+.3f} @{nm(bt[0])}')
