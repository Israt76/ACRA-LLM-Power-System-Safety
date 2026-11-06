"""Recompute the per-target UNBOUNDED minimum-meter APPROVE attack and its injection
magnitude, using exactly the same optimization as Table I (approve goal only, no bound),
so Table III's meter counts equal Table I's k_app by construction. Resolves the L8 conflict."""
import json, numpy as np, pandapower.networks as nw
import acra_physics as A
CAND={0,6,7,8,9,10,11,12,14}
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; H=gm["H"]; m_=H.shape[0]
theta=gm["theta"]; sigma=gm["sigma"]; zmag=np.abs(H@theta)
tg=A.valid_targets(gm, cand_lines=CAND)
rows=[]
for L in tg:
    lid=br[L][4][1]
    goals=A.goals_approve(gm,L)
    k,used=A.solve_min_meters(gm, goals, cap=m_)
    # pin the injection to the SAME k-meter solution by forbidding all other meters,
    # so the magnitude is that of the actual minimum-meter attack (support subset of used)
    comp=[i for i in range(m_) if i not in set(used)]
    c=A.solve_capture_c(gm, goals, cap=m_, protected=comp)
    a=H@c if c is not None else None
    if a is None:
        rows.append(dict(line=lid,meters=k,max_over_z=None,max_over_sigma=None));
        print(f"L{lid}: meters={k} (magnitude n/a)"); continue
    idx=[i for i in range(m_) if abs(a[i])>1e-9]
    over_z=max((abs(a[i])/zmag[i] for i in idx if zmag[i]>1e-9), default=0.0)
    over_s=max((abs(a[i])/sigma[i] for i in idx if sigma[i]>1e-9), default=0.0)
    rows.append(dict(line=lid,meters=int(k),support=len(idx),
                     max_over_z=round(float(over_z),2),max_over_sigma=round(float(over_s),1)))
    print(f"L{lid}: k_app(meters)={k}  support={len(idx)}  max|a|/|z|={over_z:.2f}  max|a|/sigma={over_s:.1f}")
json.dump(rows, open("exp_magnitude_recompute.json","w"), indent=2)
print("saved exp_magnitude_recompute.json")
