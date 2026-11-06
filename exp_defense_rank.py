"""Defense meter-selection mechanism (IEEE-14): formal ranking + marginal reduction.
For each meter, count how often it appears in the per-target minimum approve-FDI set (leverage),
rank by that frequency, and report the MARGINAL reduction in attackable targets as high-leverage
meters are protected one by one (greedy)."""
import numpy as np, json, pandapower.networks as nw
import acra_physics as A

CAND={0,6,7,8,9,10,11,12,14}; CAP=8
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; m_=gm["H"].shape[0]; nbr=gm["nbr"]
tg=A.valid_targets(gm, cand_lines=CAND)

# meter type/location: measurement rows are [branch flows (nbr)] then [bus injections (n-1 reduced)]
def meter_desc(i):
    if i<nbr:
        typ=br[i][4]; return f"flow/{typ[0]} L{typ[1]}"
    else:
        return f"bus-injection #{i-nbr}"

# leverage frequency over minimum approve sets
freq=np.zeros(m_)
for L in tg:
    _,used=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m_)
    if used:
        for i in used: freq[i]+=1
rank=list(np.argsort(-freq))

def feasible(protected):
    return sum(1 for L in tg if A.solve_min_meters(gm,A.goals_approve(gm,L),cap=CAP,protected=set(protected))[0] is not None)

# greedy marginal reduction following the frequency ranking
protected=[]; prev=feasible([])
print(f"{'rank':>4} {'meter':>5} {'freq':>4} {'description':<18} {'attackable_after':>16} {'marginal_drop':>13}")
table=[]
for r,i in enumerate(rank):
    if freq[i]==0: break
    protected.append(i); now=feasible(protected); drop=prev-now
    table.append(dict(rank=r+1, meter=int(i), freq=int(freq[i]), description=meter_desc(i),
                      attackable_after=now, marginal_drop=drop))
    print(f"{r+1:>4} {i:>5} {int(freq[i]):>4} {meter_desc(i):<18} {now:>16} {drop:>13}")
    prev=now
    if now==0: break

json.dump(dict(targets=[br[L][4][1] for L in tg], n_meters=m_, n_branches=nbr,
               greedy_protection=table), open("exp_defense_rank.json","w"), indent=2)
print("saved exp_defense_rank.json")
