"""Full secure-estimation defense sweep on IEEE-14: attack-feasible targets vs #protected meters,
aware (high-leverage) vs naive (random, averaged), and minimal protection that kills the attack."""
import numpy as np, json, pandapower.networks as nw
import acra_physics as A

CAND={0,6,7,8,9,10,11,12,14}; CAP=8; SEED=1234
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; m_=gm["H"].shape[0]
tg=A.valid_targets(gm, cand_lines=CAND)
print("targets:",[br[L][4][1] for L in tg],"| meters m =",m_)

# leverage = frequency a meter is chosen in each target's minimum approve set
tally=np.zeros(m_)
for L in tg:
    _,used=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m_)
    if used:
        for i in used: tally[i]+=1
order_aware=list(np.argsort(-tally))
rng=np.random.default_rng(SEED)

def feasible_count(protected):
    c=0
    for L in tg:
        k,_=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=CAP,protected=set(protected))
        if k is not None: c+=1
    return c

rows=[]
NA=len(tg)
for nprot in range(0, 17):
    aware=feasible_count(order_aware[:nprot])
    # naive: average over 5 random placements
    naive_vals=[]
    for s in range(5):
        r=np.random.default_rng(SEED+s).permutation(m_)
        naive_vals.append(feasible_count(list(r[:nprot])))
    naive=round(float(np.mean(naive_vals)),1)
    rows.append(dict(n_protected=nprot, feasible_aware=aware, feasible_naive=naive, n_targets=NA))
    print(f"protected={nprot:2d}  aware {aware}/{NA}   naive(avg5) {naive}/{NA}")

min_kill=next((r['n_protected'] for r in rows if r['feasible_aware']==0), None)
print(f"\nMINIMAL attack-aware protection that removes ALL targets: {min_kill} meters (of {m_})")
json.dump(dict(meters=m_, targets=[br[L][4][1] for L in tg], leverage=tally.tolist(),
               sweep=rows, min_protection_kill=min_kill), open("exp_secure.json","w"), indent=2)
print("saved exp_secure.json")
