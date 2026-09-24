"""Paired FEM comparisons across the sweep: per-run table plus b1 provider differences with 95% bootstrap CIs.

    uv run python experiments/paper2/paired.py results/budget
"""
import json, csv, numpy as np
from pathlib import Path
import sys
R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/budget"); man = json.load(open(R/"manifest.json"))
D = {}
for e in man:
    p = R/"kratos"/e["slug"]/"metrics.csv"
    if not p.exists(): continue
    mf = np.full(100,np.nan); fail=0
    for r in csv.DictReader(open(p)):
        if r["status"]=="ok": mf[int(r["sample_index"])]=float(r["mf_resultants_area_mean"])
        else: fail+=1
    z = np.load(R/"samples"/e["file"])["z"]; z = z[:,0] if z.ndim==4 else z
    lap = np.abs(z[:,2:,1:-1]+z[:,:-2,1:-1]+z[:,1:-1,2:]+z[:,1:-1,:-2]-4*z[:,1:-1,1:-1])*100
    zm = z.reshape(len(z),-1).max(1)
    key=(e["block"],e["provider"],e["steps"],e["eta"],e["clip_denoised"],e["guidance_scale"],e["engineer"],e["guide_every"])
    D[key]=dict(mf=mf,fail=fail,rm=lap.mean(),rx=np.median(lap.reshape(len(z),-1).max(1)),cv=zm.std()/zm.mean(),sec=e["seconds"],
                pred=(e["trace"][-1]["mf"] if e["trace"] else np.nan), gmax=max([s["grad"] for s in e["trace"]],default=0))
print(f"{'blk':<4}{'prov':<14}{'K':>5}{'eta':>5}{'clip':>5}{'g':>6}{'eng':>7}{'ev':>3}{'n':>4}{'mf':>7}{'sd':>6}{'P>.9':>6}{'P<.7':>6}{'p5':>6}{'CV':>6}{'rgh':>6}{'rgx':>6}{'pred':>6}{'gmax':>6}")
for k in sorted(D, key=lambda k:(k[0],k[1],k[2],k[3])):
    d=D[k]; m=d["mf"][~np.isnan(d["mf"])]
    print(f"{k[0]:<4}{k[1]:<14}{k[2]:>5}{k[3]:>5g}{'T' if k[4] else 'F':>5}{k[5]:>6g}{k[6]:>7}{k[7]:>3}{len(m):>4}{m.mean():>7.3f}{m.std():>6.3f}{(m>.9).mean():>6.2f}{(m<.7).mean():>6.2f}{np.percentile(m,5):>6.3f}{d['cv']:>6.3f}{d['rm']:>6.1f}{d['rx']:>6.0f}{d['pred']:>6.3f}{d['gmax']:>6.0f}")
rng=np.random.default_rng(0)
def ci(a,b):
    d=(a-b); d=d[~np.isnan(d)]; bs=[rng.choice(d,len(d)).mean() for _ in range(3000)]
    return f"{d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]"
b1={(k[1],k[2],k[3]):v for k,v in D.items() if k[0]=="b1"}
print("\nK    eta | NA-TC                 | NA-TS                 | TS-TC")
for K in (10,20,50,100,250,1000):
    for eta in (0.0,1.0):
        c=[]
        for x,y in (("noise_aware","tweedie_clean"),("noise_aware","tweedie_self"),("tweedie_self","tweedie_clean")):
            c.append(ci(b1[(x,K,eta)]["mf"],b1[(y,K,eta)]["mf"]) if (x,K,eta) in b1 and (y,K,eta) in b1 else " "*21)
        print(f"{K:<5}{eta:<4g}| "+" | ".join(c))
