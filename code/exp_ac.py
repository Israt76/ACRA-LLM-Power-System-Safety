"""AC-verifier validation on IEEE-14.
The DC-constructed coordinated (approve) injection shifts the estimated bus-angle state by c,
equivalent to an estimated injection shift dP = Bred @ c. We realize that falsified operating point
and run a FULL AC power flow as the verifier (instead of the DC/LODF twin), with the target line opened.
Attack survives an AC verifier if: AC verifier APPROVES on the falsified state (max loading <= 100%)
while the TRUE AC outcome is unsafe (>100%).
"""
import numpy as np, json, copy, pandapower as pp, pandapower.networks as nw
import acra_physics as A

gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; Bred=gm["Bred"]; pos=gm["pos"]; slack=gm["slack"]
CAND={0,6,7,8,9,10,11,12,14}
tg=A.valid_targets(gm, cand_lines=CAND)
# reduced index -> bus label (case14: label==enum, slack excluded)
red=[i for i in range(len(gm["net"].bus)) if i!=slack]

def ac_maxload_open(net, lid):
    n=copy.deepcopy(net)
    n.line.at[lid,"in_service"]=False
    try:
        pp.runpp(n,numba=False)
    except Exception:
        return None
    if not n.converged: return None
    return float(n.res_line.loading_percent.max())

rows=[]
print(f"{'line':>5} {'true_AC_open%':>13} {'falsified_AC_open%':>18} {'AC_verifier_approves':>20} {'attack_survives_AC':>18}")
for L in tg:
    lid=br[L][4][1]
    c=A.solve_capture_c(gm,A.goals_approve(gm,L),cap=gm["H"].shape[0])
    if c is None: continue
    dP=Bred@c  # MW injection shift per reduced bus
    # build falsified net: add sgen(dP) at each non-slack bus (slack absorbs imbalance)
    fnet=copy.deepcopy(gm["net"])
    for r,busr in enumerate(red):
        if abs(dP[r])>1e-6:
            pp.create_sgen(fnet, bus=busr, p_mw=float(dP[r]), q_mvar=0.0)
    true_open = ac_maxload_open(gm["net"], lid)       # true AC outcome
    fal_open  = ac_maxload_open(fnet, lid)            # AC verifier sees this (falsified)
    approves = (fal_open is not None and fal_open<=100.0+1e-6)
    survives = bool(approves and true_open is not None and true_open>100.0)
    rows.append(dict(line=lid, true_AC_open_pct=round(true_open,1) if true_open else None,
                     falsified_AC_open_pct=round(fal_open,1) if fal_open else None,
                     ac_verifier_approves=bool(approves), attack_survives_AC=survives))
    print(f"L{lid:<4d} {str(round(true_open,1) if true_open else None):>13} "
          f"{str(round(fal_open,1) if fal_open else None):>18} {str(approves):>20} {str(survives):>18}")

nsurv=sum(1 for r in rows if r['attack_survives_AC'])
print(f"\nattack survives the AC verifier on {nsurv}/{len(rows)} targets")
json.dump(dict(targets=[br[L][4][1] for L in tg], results=rows, n_survive=nsurv, n_targets=len(rows)),
          open("exp_ac.json","w"), indent=2)
print("saved exp_ac.json")
