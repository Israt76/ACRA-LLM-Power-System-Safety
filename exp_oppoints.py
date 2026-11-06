"""Operating-point robustness: sweep load levels on IEEE-14, recompute k_app/k_tel/Delta k per point."""
import numpy as np, json, pandapower.networks as nw
import acra_physics as A

LOADS=np.round(np.linspace(0.90,1.20,16),3)   # 16 operating points
CAND={0,6,7,8,9,10,11,12,14}
rows=[]
for ls in LOADS:
    gm=A.build(nw.case14, load_scale=float(ls))
    br=gm["br"]
    tg=A.valid_targets(gm, cand_lines=CAND)
    per=[]
    for Lh in tg:
        lid=br[Lh][4][1]
        ga=A.goals_approve(gm,Lh); gi=A.goals_induce(gm,Lh)
        ka,_=A.solve_min_meters(gm,ga,cap=gm["H"].shape[0])
        kt,_=A.solve_min_meters(gm,gi+ga,cap=gm["H"].shape[0])
        per.append(dict(line=lid,k_app=ka,k_tel=kt))
    # summary at K=8
    aj8=sum(1 for p in per if p['k_app'] is not None and p['k_app']<=8)
    at8=sum(1 for p in per if p['k_tel'] is not None and p['k_tel']<=8)
    dks=[p['k_tel']-p['k_app'] for p in per if p['k_tel'] is not None and p['k_app'] is not None]
    tel_infeas=sum(1 for p in per if p['k_tel'] is None)
    kapps=[p['k_app'] for p in per if p['k_app'] is not None]
    rows.append(dict(load=float(ls), n_targets=len(tg),
                     A_joint_at_8=aj8, A_tel_at_8=at8,
                     mean_delta_k=round(float(np.mean(dks)),2) if dks else None,
                     tel_only_infeasible=tel_infeas,
                     mean_k_app=round(float(np.mean(kapps)),2) if kapps else None,
                     per_target=per))
    print(f"load={ls:.3f}  targets={len(tg):2d}  A_joint@8={aj8:2d}  A_tel@8={at8:2d}  "
          f"meanDk={rows[-1]['mean_delta_k']}  tel_infeasible={tel_infeas}  mean_k_app={rows[-1]['mean_k_app']}")

json.dump(rows, open("exp_oppoints.json","w"), indent=2)
# aggregate
aj=[r['A_joint_at_8'] for r in rows]; at=[r['A_tel_at_8'] for r in rows]
print("\n=== SUMMARY over",len(LOADS),"operating points ===")
print(f"A_joint@8: mean {np.mean(aj):.1f}, min {min(aj)}, max {max(aj)}")
print(f"A_tel@8  : mean {np.mean(at):.1f}, min {min(at)}, max {max(at)}  (telemetry-only reach under 8-meter cap)")
print("saved exp_oppoints.json")
