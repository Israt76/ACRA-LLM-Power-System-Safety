"""DETERMINISTIC per-target injection magnitude (Table 2).
The minimum-meter MILP fixes the meter COUNT, not the injection magnitude, so the
magnitude of a k_app-meter attack is solver-dependent (not reproducible). This script
instead reports a well-defined, reproducible quantity: rho* = the minimum achievable
maximum per-meter RELATIVE injection magnitude for a stealthy approve attack,
   rho* = min_c  max_i |(Hc)_i| / |z_i|   s.t.  approve goal on (f + F c),
an LP (no integer variables). It is obtained by spreading the injection across meters,
so rho* is small; the few-meter attack of Table 1 (k_app meters) instead concentrates
the injection and needs larger per-meter magnitude. Reporting rho* makes Table 2
reproducible and exposes the meter-count vs magnitude trade-off.
"""
import json, numpy as np, pandapower.networks as nw
from scipy.optimize import linprog
import acra_physics as A
CAND={0,6,7,8,9,10,11,12,14}
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; H=gm["H"]; m,n=H.shape
theta=gm["theta"]; sigma=gm["sigma"]; absz=np.maximum(np.abs(H@theta),1e-6)
F=gm["F"]; tg=A.valid_targets(gm, cand_lines=CAND)
rows=[]
print("Deterministic min-L-infinity injection magnitude per target (Table 2):")
print(" target  k_app   rho*=min max|a|/|z|   max|a|/sigma")
for L in tg:
    lid=br[L][4][1]
    ka,_=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m)
    goals=A.goals_approve(gm,L)
    c_obj=np.zeros(n+1); c_obj[-1]=1.0            # minimize t
    Aub=[]; bub=[]
    for i in range(m):
        r=np.zeros(n+1); r[:n]=H[i];  r[-1]=-absz[i]; Aub.append(r); bub.append(0.0)
        r=np.zeros(n+1); r[:n]=-H[i]; r[-1]=-absz[i]; Aub.append(r); bub.append(0.0)
    for (row,sense,rhs) in goals:
        r=np.zeros(n+1)
        if sense=="<=": r[:n]=row;              Aub.append(r); bub.append(rhs)
        else:           r[:n]=-np.array(row);   Aub.append(r); bub.append(-rhs)
    res=linprog(c_obj,A_ub=np.array(Aub),b_ub=np.array(bub),
                bounds=[(-50,50)]*n+[(0,None)],method="highs")
    if not res.success:
        print(f"  L{lid:<3} infeasible"); continue
    c=res.x[:n]; a=H@c
    mz=float(np.max(np.abs(a)/absz)); ms=float(np.max(np.abs(a)/sigma))
    rows.append(dict(line=lid,k_app=int(ka),rho_star=round(mz,2),max_over_sigma=round(ms,0)))
    print(f"  L{lid:<3}   {ka:<2}      {mz:5.2f}               {ms:6.0f}")
with open("../data/ieee14_magnitude_per_target.csv","w") as f:
    f.write("line,k_app,rho_star_max_frac_of_meas,max_in_sigma\n")
    for r in rows: f.write(f"{r['line']},{r['k_app']},{r['rho_star']},{int(r['max_over_sigma'])}\n")
json.dump(rows, open("../data/ieee14_magnitude_minlinf.json","w"), indent=2)
print("\nwrote ../data/ieee14_magnitude_per_target.csv  (reproducible rho*)")
