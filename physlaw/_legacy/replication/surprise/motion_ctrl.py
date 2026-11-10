"""Motion-amount control for the ladder: per seed, fit predictor surprise ~ a + b*persistence error
on the 6 LEGAL arms only, extrapolate to impossible arms; sign test of residuals.
Also sign tests of persistence error itself for the matched pairs (physics-free baseline)."""
import json, sys
from math import comb
import numpy as np
ARMS = ["e0.2","e0.3333","e0.5","e0.6667","e0.8","honest","e1.25","e1.5","e2","e3","e5"]
LEGAL, IMP = ARMS[:6], ARMS[6:]
MATCH = [("e1.25","e0.8"),("e1.5","e0.6667"),("e2","e0.5"),("e3","e0.3333"),("e5","e0.2"),("e1.5","e0.5"),("e2","e0.2")]
sp = lambda k,n: sum(comb(n,j) for j in range(k,n+1))/2**n
out = {}
for m in ("vitl","vitg"):
    rows = [json.loads(l) for l in open(f"ladder_rerun/ladder_{m}.jsonl")]
    out[m] = {}
    for tag in ("balls","all"):
        for met, sg in (("mse",1),("l1",1),("cos",-1)):
            res = {a: [] for a in IMP}; slopes=[]; r2=[]
            pm = {}
            for r in rows:
                F = r["futures"]
                x = np.array([sg*F[a]["pers"][tag][met] for a in LEGAL]); y = np.array([sg*F[a][tag][met] for a in LEGAL])
                c = np.polyfit(x, y, 1); slopes.append(c[0])
                r2.append(1 - ((np.polyval(c,x)-y)**2).sum()/((y-y.mean())**2).sum())
                for a in IMP:
                    if a in F:
                        res[a].append(sg*F[a][tag][met] - np.polyval(c, sg*F[a]["pers"][tag][met]))
            allr = np.concatenate([np.array(v) for v in res.values()])
            k = int((allr>0).sum())
            pmt = {}
            for i, l in MATCH:
                ss = [r for r in rows if i in r["futures"]]
                d = np.array([sg*(r["futures"][i]["pers"][tag][met]-r["futures"][l]["pers"][tag][met]) for r in ss])
                pmt[f"{i}_vs_{l}"] = f"{int((d>0).sum())}/{len(d)}"
            out[m][f"{tag}.{met}"] = {"legal_slope_median": float(np.median(slopes)), "legal_r2_median": float(np.median(r2)),
                "resid_per_arm": {a: {"n_above": int((np.array(v)>0).sum()), "n": len(v), "median": float(np.median(v))} for a,v in res.items()},
                "pooled_above": f"{k}/{len(allr)}", "p_one_sided_above": sp(k,len(allr)), "p_one_sided_below": sp(len(allr)-k,len(allr)),
                "persistence_matched_imp_worse": pmt}
json.dump(out, open("ladder_motion_control.json","w"), indent=1)
for m in out:
    for k,v in out[m].items():
        print(m,k,"slope",round(v["legal_slope_median"],3),"r2",round(v["legal_r2_median"],2),"pooled",v["pooled_above"],"p_above",round(v["p_one_sided_above"],4),"p_below",round(v["p_one_sided_below"],4),
              {a:(f'{d["n_above"]}/{d["n"]}',round(d["median"],4)) for a,d in v["resid_per_arm"].items()}, "pers-matched", v["persistence_matched_imp_worse"])
