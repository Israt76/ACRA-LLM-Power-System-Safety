import numpy as np, json, pandapower.networks as nw
import acra_physics as A
CAND={0,6,7,8,9,10,11,12,14}; CAP=8
gm=A.build(nw.case14, load_scale=1.05); br=gm['br']; m=gm['H'].shape[0]; nbr=gm['nbr']
TG=A.valid_targets(gm, cand_lines=CAND)
def attackable(protected):
    s=[]
    for L in TG:
        if A.solve_min_meters(gm,A.goals_approve(gm,L),cap=CAP,protected=set(protected),time_limit=5)[0] is not None:
            s.append(br[L][4][1])
    return set(s)
protected=[]; cur=attackable([]); steps=[]
print('marginal-greedy defense:')
while cur:
    best=None; bestrem=-1; bestset=None
    for i in range(m):
        if i in protected: continue
        ns=attackable(protected+[i]); rem=len(cur)-len(ns)
        if rem>bestrem: bestrem=rem; best=i; bestset=ns
    if bestrem<=0:
        print('  no further reduction'); break
    protected.append(best); drop=len(cur)-len(bestset); cur=bestset
    typ=f'flow L{br[best][4][1]}' if best<nbr else f'bus-injection #{best-nbr}'
    steps.append(dict(meter=int(best),type=typ,attackable_after=len(cur),drop=drop))
    print(f'  protect meter {best:2d} ({typ:18}) -> attackable={len(cur)} (drop {drop})')
print('minimal set size:',len(protected),'->',protected)
json.dump(dict(steps=steps,min_set=[int(x) for x in protected],size=len(protected)),open('defgreedy.json','w'),indent=2)
