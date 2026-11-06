"""Magnitude characterization + fraction-of-measurement cap sweep (IEEE-14).
(1) For the unbounded minimum-meter approve attack, report the actual injection magnitudes:
    max |a_i|/|z_i| and max |a_i|/sigma_i. (2) Cap each meter at |a_i| <= rho*|z_i| and sweep rho."""
import numpy as np, json, pandapower.networks as nw
import acra_physics as A

CAND={0,6,7,8,9,10,11,12,14}
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; H=gm["H"]; z=gm["z"]; sig=gm["sigma"]; m_=H.shape[0]
tg=A.valid_targets(gm, cand_lines=CAND)
absz=np.maximum(np.abs(z),1e-6)
print("targets:",[br[L][4][1] for L in tg])

# (1) magnitude of unbounded minimum-meter approve solution
mag=[]
for L in tg:
    c=A.solve_capture_c(gm,A.goals_approve(gm,L),cap=m_)
    if c is None: continue
    a=H@c; used=np.abs(a)>1e-6
    frac=np.abs(a[used])/absz[used]; insig=np.abs(a[used])/sig[used]
    mag.append(dict(line=br[L][4][1], meters=int(used.sum()),
                    max_frac_of_meas=round(float(frac.max()),3), mean_frac_of_meas=round(float(frac.mean()),3),
                    max_in_sigma=round(float(insig.max()),1)))
    print(f"L{br[L][4][1]:<3d} meters={int(used.sum())} max|a|/|z|={frac.max():.2f} mean={frac.mean():.2f} max|a|/sigma={insig.max():.0f}")

# (2) cap |a_i| <= rho*|z_i|, sweep rho (fraction of the measurement itself)
print("\nfraction-of-measurement cap sweep:")
sweep=[]
for rho in [0.05,0.10,0.20,0.50,1.00,None]:
    bnd = (rho*absz) if rho is not None else None
    per=[]
    for L in tg:
        ka,_=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m_,bnd=bnd)
        kt,_=A.solve_min_meters(gm,A.goals_induce(gm,L)+A.goals_approve(gm,L),cap=m_,bnd=bnd)
        per.append((br[L][4][1],ka,kt))
    aj8=sum(1 for _,ka,_ in per if ka is not None and ka<=8)
    at8=sum(1 for _,_,kt in per if kt is not None and kt<=8)
    dks=[kt-ka for _,ka,kt in per if ka is not None and kt is not None]
    sweep.append(dict(rho=rho, A_joint_at_8=aj8, A_tel_at_8=at8,
                      approve_feasible=sum(1 for _,ka,_ in per if ka is not None),
                      mean_delta_k=round(float(np.mean(dks)),2) if dks else None))
    print(f"  rho={str(rho):>5}  approve_feasible={sweep[-1]['approve_feasible']}/9  A_joint@8={aj8}  A_tel@8={at8}  meanDk={sweep[-1]['mean_delta_k']}")

json.dump(dict(targets=[br[L][4][1] for L in tg], magnitudes=mag, frac_cap_sweep=sweep),
          open("exp_magnitude.json","w"), indent=2)
print("saved exp_magnitude.json")
