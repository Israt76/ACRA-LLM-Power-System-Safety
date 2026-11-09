"""FDI-budget ablation + BDD-sensitivity on IEEE-14."""
import numpy as np, json, pandapower.networks as nw
from scipy.stats import chi2
import acra_physics as A

CAND={0,6,7,8,9,10,11,12,14}
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; H=gm["H"]; W=gm["W"]; m_=H.shape[0]; n=H.shape[1]
tg=A.valid_targets(gm, cand_lines=CAND)
print("targets:",[br[L][4][1] for L in tg],"| m =",m_,"dof =",m_-n)

def wls(z_att):
    G=H.T@W@H; xhat=np.linalg.solve(G,H.T@W@z_att); r=z_att-H@xhat; return xhat,float(r.T@W@r)

# ---- per-target min budgets (approve vs induce+approve) ----
budg=[]
for L in tg:
    ka,used_app=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m_)
    kt,_=A.solve_min_meters(gm,A.goals_induce(gm,L)+A.goals_approve(gm,L),cap=m_)
    budg.append(dict(line=br[L][4][1],k_app=ka,k_tel=kt,approve_c_used=used_app))

# ---- FDI-budget ablation: feasible targets vs K ----
ablation=[]
for K in range(1,m_+1):
    aj=sum(1 for b in budg if b['k_app'] is not None and b['k_app']<=K)
    at=sum(1 for b in budg if b['k_tel'] is not None and b['k_tel']<=K)
    ablation.append(dict(K=K,A_joint=aj,A_tel=at))
print("\nFDI-budget ablation (feasible targets vs K):")
for a in ablation[:14]:
    print(f"  K={a['K']:2d}  A_joint={a['A_joint']}  A_tel={a['A_tel']}")

# ---- BDD sensitivity: stealthy approve injection residual vs thresholds, and random-injection detection ----
dof=m_-n
thr={a:float(chi2.ppf(a,dof)) for a in [0.90,0.95,0.99,0.999]}
_,J_clean=wls(gm["z"])
rng=np.random.default_rng(1234)
bdd=[]
for L in tg:
    c_vec=A.solve_capture_c(gm,A.goals_approve(gm,L),cap=m_)
    if c_vec is None:
        continue
    z_att=H@(gm["theta"]+c_vec)
    _,J_att=wls(z_att)
    # random injection of same L2 magnitude on same number of meters
    a_stealth=H@c_vec; k_used=int(np.sum(np.abs(a_stealth)>1e-6))
    flagged_rand=0; TR=200
    for _ in range(TR):
        idx=rng.choice(m_,max(k_used,1),replace=False)
        ar=np.zeros(m_); ar[idx]=rng.normal(0,1,len(idx))*np.linalg.norm(a_stealth)/np.sqrt(len(idx))
        _,Jr=wls(gm["z"]+ar)
        if Jr>thr[0.99]: flagged_rand+=1
    bdd.append(dict(line=br[L][4][1], J_clean=round(J_clean,3), J_stealth=round(J_att,3),
                    passes_all_thresholds=bool(J_att<=thr[0.999]),
                    random_inject_detect_rate_at_0p99=round(flagged_rand/TR,3)))
    print(f"  L{br[L][4][1]:<3d} J_stealth={J_att:.3f} (thr99={thr[0.99]:.1f}) stealth_passes={J_att<=thr[0.999]}  random_detected={flagged_rand}/{TR}")

json.dump(dict(targets=[br[L][4][1] for L in tg], budgets=budg, fdi_budget_ablation=ablation,
               chi2_thresholds=thr, bdd_sensitivity=bdd), open("exp_ablation.json","w"), indent=2)
print("\nsaved exp_ablation.json")
