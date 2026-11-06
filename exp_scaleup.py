"""Scale-up: budget/reachability separation on IEEE-39 and a sample of IEEE-118 targets (physics only)."""
import numpy as np, json, sys, pandapower.networks as nw
import acra_physics as A

def run_case(name, net_fn, max_targets, cand=None, tl=8):
    gm=A.build(net_fn, load_scale=1.05); br=gm["br"]; m_=gm["H"].shape[0]
    line_ids=[bb[4][1] for bb in br if bb[4][0]=="line"]
    pool = line_ids if cand is None else [l for l in line_ids if l in cand]
    tg=[]
    for lid in pool:
        Lh=next(i for i,bb in enumerate(br) if bb[4]==("line",lid))
        if not A.ac_unsafe_open(gm,lid): continue
        if A.goals_approve(gm,Lh) is None: continue
        k,_=A.solve_min_meters(gm,A.goals_approve(gm,Lh),cap=m_,time_limit=tl)
        if k is None: continue
        tg.append(Lh)
        if len(tg)>=max_targets: break
    rows=[]
    for Lh in tg:
        lid=br[Lh][4][1]
        ka,_=A.solve_min_meters(gm,A.goals_approve(gm,Lh),cap=m_,time_limit=tl)
        kt,_=A.solve_min_meters(gm,A.goals_induce(gm,Lh)+A.goals_approve(gm,Lh),cap=m_,time_limit=tl)
        dk=(kt-ka) if (kt is not None and ka is not None) else None
        rows.append(dict(line=lid,k_app=ka,k_tel=kt,delta_k=dk))
        print(f"  [{name}] L{lid:<4d} k_app={ka} k_tel={kt} Dk={dk}")
    kapps=[r['k_app'] for r in rows if r['k_app'] is not None]
    ktels=[r['k_tel'] for r in rows if r['k_tel'] is not None]
    infeas=sum(1 for r in rows if r['k_tel'] is None)
    summ=dict(case=name, meters=m_, n_targets=len(rows),
              k_app_range=[min(kapps),max(kapps)] if kapps else None,
              k_tel_range=[min(ktels),max(ktels)] if ktels else None,
              tel_only_infeasible=infeas,
              mean_delta_k=round(float(np.mean([r['delta_k'] for r in rows if r['delta_k'] is not None])),2) if ktels else None,
              per_target=rows)
    print(f"  [{name}] meters={m_} targets={len(rows)} k_app={summ['k_app_range']} k_tel={summ['k_tel_range']} "
          f"tel_infeasible={infeas} meanDk={summ['mean_delta_k']}")
    return summ

out={}
print("=== IEEE-39 ===")
out['ieee39']=run_case("IEEE-39", nw.case39, max_targets=12, tl=8)
print("\n=== IEEE-118 (sample) ===")
out['ieee118']=run_case("IEEE-118", nw.case118, max_targets=8, tl=12)
json.dump(out, open("exp_scaleup.json","w"), indent=2)
print("\nsaved exp_scaleup.json")
