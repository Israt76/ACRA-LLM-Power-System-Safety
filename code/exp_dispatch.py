"""Generation-dispatch and headroom robustness (IEEE-14).
(A) 3 load levels x 3 generation dispatch patterns = 9 operating scenarios.
(B) line-rating headroom sensitivity at 1.25 / 1.35 / 1.50."""
import numpy as np, json, pandapower.networks as nw
import acra_physics as A

CAND={0,6,7,8,9,10,11,12,14}
def summarize(gm):
    br=gm["br"]; m_=gm["H"].shape[0]
    tg=A.valid_targets(gm, cand_lines=CAND)
    aj=at=0; dks=[]
    for L in tg:
        ka,_=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m_)
        kt,_=A.solve_min_meters(gm,A.goals_induce(gm,L)+A.goals_approve(gm,L),cap=m_)
        if ka is not None and ka<=8: aj+=1
        if kt is not None and kt<=8: at+=1
        if ka is not None and kt is not None: dks.append(kt-ka)
    return len(tg), aj, at, (round(float(np.mean(dks)),2) if dks else None)

# (A) dispatch patterns (multipliers on gen setpoints; slack rebalances)
def pat_uniform(ng): return np.ones(ng)
def pat_ramp(ng):    return np.linspace(0.6,1.4,ng)      # skew output toward later gens
def pat_altern(ng):  return np.where(np.arange(ng)%2==0,1.3,0.7)  # alternating high/low

dispatch={"uniform":pat_uniform,"ramped":pat_ramp,"alternating":pat_altern}
LOADS=[0.95,1.05,1.15]
print("=== (A) load x dispatch (A_joint@8 / A_tel@8 / meanDk) ===")
gridA=[]
for ls in LOADS:
    for name,fn in dispatch.items():
        gm=A.build(nw.case14, load_scale=ls, gen_mod=fn)
        nt,aj,at,dk=summarize(gm)
        gridA.append(dict(load=ls,dispatch=name,n_targets=nt,A_joint_at_8=aj,A_tel_at_8=at,mean_delta_k=dk))
        print(f"  load={ls:.2f}  dispatch={name:11}  targets={nt}  A_joint@8={aj}  A_tel@8={at}  meanDk={dk}")

# (B) headroom sensitivity
print("\n=== (B) headroom sensitivity (A_joint@8 / A_tel@8 / meanDk) ===")
gridB=[]
for hr in [1.25,1.35,1.50]:
    gm=A.build(nw.case14, load_scale=1.05, headroom=hr)
    nt,aj,at,dk=summarize(gm)
    gridB.append(dict(headroom=hr,n_targets=nt,A_joint_at_8=aj,A_tel_at_8=at,mean_delta_k=dk))
    print(f"  headroom={hr}  targets={nt}  A_joint@8={aj}  A_tel@8={at}  meanDk={dk}")

json.dump(dict(load_x_dispatch=gridA, headroom=gridB), open("exp_dispatch.json","w"), indent=2)
print("saved exp_dispatch.json")
