"""Scale-up with clean methodology: predeclared RANDOM target sample (seeded), long solver limit,
and explicit infeasible-vs-timeout status for k_tel. IEEE-39 and IEEE-118."""
import numpy as np, json, pandapower.networks as nw
import acra_physics as A

SEED=1234; TL=60   # long solver limit
def run(name, net_fn, sample_n, tl=TL):
    gm=A.build(net_fn, load_scale=1.05); br=gm["br"]; m_=gm["H"].shape[0]
    line_h=[i for i,bb in enumerate(br) if bb[4][0]=="line"]
    rng=np.random.default_rng(SEED)
    order=list(rng.permutation(line_h))     # predeclared random order
    rows=[]
    for Lh in order:
        if len(rows)>=sample_n: break
        lid=br[Lh][4][1]
        if not A.ac_unsafe_open(gm,lid): continue
        if A.goals_approve(gm,Lh) is None: continue
        ka,sa=A.solve_status(gm,A.goals_approve(gm,Lh),cap=m_,time_limit=tl)
        if ka is None: continue      # approve must be feasible to be a valid target
        kt,st=A.solve_status(gm,A.goals_induce(gm,Lh)+A.goals_approve(gm,Lh),cap=m_,time_limit=tl)
        rows.append(dict(line=lid,k_app=ka,k_tel=kt,k_tel_status=st,
                         delta_k=(kt-ka) if kt is not None else None))
        print(f"  [{name}] L{lid:<4d} k_app={ka} k_tel={kt} ({st}) Dk={rows[-1]['delta_k']}")
    kapps=[r['k_app'] for r in rows]; ktels=[r['k_tel'] for r in rows if r['k_tel'] is not None]
    proven_inf=sum(1 for r in rows if r['k_tel_status']=="infeasible")
    timed=sum(1 for r in rows if r['k_tel_status']=="timeout")
    summ=dict(case=name, meters=m_, n_targets=len(rows), sampling="seeded random (SEED=1234)",
              solver_time_limit_s=tl, k_app_range=[min(kapps),max(kapps)] if kapps else None,
              k_tel_range=[min(ktels),max(ktels)] if ktels else None,
              tel_proven_infeasible=proven_inf, tel_timeout=timed,
              mean_delta_k=round(float(np.mean([r['delta_k'] for r in rows if r['delta_k'] is not None])),2) if ktels else None,
              per_target=rows)
    print(f"  [{name}] meters={m_} targets={len(rows)} k_app={summ['k_app_range']} k_tel={summ['k_tel_range']} "
          f"proven_infeasible={proven_inf} timeout={timed} meanDk={summ['mean_delta_k']}")
    return summ

out={}
print("=== IEEE-39 (random sample, 60s limit, status-tagged) ===")
out['ieee39']=run("IEEE-39", nw.case39, sample_n=10)
print("\n=== IEEE-118 (random sample, 60s limit, status-tagged) ===")
out['ieee118']=run("IEEE-118", nw.case118, sample_n=6)
json.dump(out, open("exp_scaleup2.json","w"), indent=2)
print("\nsaved exp_scaleup2.json")
