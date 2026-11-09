"""Recompute key IEEE-14 numbers and check them against the submitted CSVs."""
import csv, sys
sys.path.insert(0,"code")
import acra_physics as A
import pandapower.networks as nw

gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; m_=gm["H"].shape[0]
tg=A.valid_targets(gm, cand_lines={0,6,7,8,9,10,11,12,14})
recomputed={}
for L in tg:
    ka,_=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=m_)
    kt,_=A.solve_min_meters(gm,A.goals_induce(gm,L)+A.goals_approve(gm,L),cap=m_)
    recomputed[br[L][4][1]]=(ka,kt)

ok=True
with open("data/ieee14_budget.csv") as f:
    for row in csv.DictReader(f):
        lid=int(row["line"]); ka=None if row["k_app"] in("","None") else int(row["k_app"])
        kt=None if row["k_tel"] in("","None") else int(row["k_tel"])
        rk=recomputed.get(lid)
        match = rk==(ka,kt)
        ok &= match
        print(f"L{lid:<3d} submitted=({ka},{kt}) recomputed={rk}  {'OK' if match else 'MISMATCH'}")
print("\nALL BUDGET VALUES MATCH" if ok else "\nSOME MISMATCHES (check solver version / seed)")
