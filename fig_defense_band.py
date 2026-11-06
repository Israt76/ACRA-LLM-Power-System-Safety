"""Regenerate fig2_reach_defense.png so that:
 (a) reachability advantage A_joint vs A_tel (from exp_ablation.json), unchanged;
 (b) secure-estimation defense: attack-aware curve = marginal-greedy Algorithm 1
     (defgreedy: 9->6->3->1->0 at meters 24,12,11,0, matching Table IV), and
     naive random protection shown as MEAN with a shaded min-max band over many seeds.
"""
import json, numpy as np, pandapower.networks as nw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import acra_physics as A

plt.rcParams.update({"font.family":"serif","font.size":10})
CAND={0,6,7,8,9,10,11,12,14}; CAP=8
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; m_=gm["H"].shape[0]
tg=A.valid_targets(gm, cand_lines=CAND); NA=len(tg)

def feas(protected):
    c=0
    for L in tg:
        k,_=A.solve_min_meters(gm,A.goals_approve(gm,L),cap=CAP,protected=set(protected),time_limit=5)
        if k is not None: c+=1
    return c

# ---- aware: marginal greedy Algorithm 1 (deterministic) ----
dg=json.load(open("defgreedy.json"))
greedy=dg["min_set"]                       # [24,12,11,0]
aware_x=list(range(0, 9))
aware_y=[NA]+[s["attackable_after"] for s in dg["steps"]]   # [9,6,3,1,0]
aware_y=aware_y+[0]*(len(aware_x)-len(aware_y))             # stays 0 afterwards

# ---- naive: many random placements, mean + min/max band ----
NS=30; maxp=9
naive_mean=[]; naive_lo=[]; naive_hi=[]
for nprot in range(0, maxp):
    vals=[]
    for s in range(NS):
        r=np.random.default_rng(1234+s).permutation(m_)
        vals.append(feas(list(r[:nprot])))
    vals=np.array(vals,float)
    naive_mean.append(vals.mean()); naive_lo.append(vals.min()); naive_hi.append(vals.max())
    print(f"nprot={nprot:2d} aware={aware_y[nprot]} naive mean={vals.mean():.1f} [{vals.min():.0f},{vals.max():.0f}]")

# ---- panel (a) reachability from ablation ----
ab=json.load(open("exp_ablation.json"))
K=[a["K"] for a in ab["fdi_budget_ablation"]]
aj=[a["A_joint"] for a in ab["fdi_budget_ablation"]]
at=[a["A_tel"] for a in ab["fdi_budget_ablation"]]

fig,(a1,a2)=plt.subplots(1,2,figsize=(8.4,3.3))
a1.plot(K,aj,"o-",color="#2b6cb0",label="coordinated  $A_{joint}(K)$")
a1.plot(K,at,"s--",color="#b22",label="telemetry-only  $A_{tel}(K)$")
a1.axvline(8,ls=":",color="#888"); a1.set_xlabel("telemetry budget $K$ (meters)")
a1.set_ylabel("reachable unsafe actions (of 9)"); a1.set_ylim(-0.3,9.5)
a1.legend(fontsize=8); a1.set_title("(a) Reachability advantage")

nx=list(range(0,maxp))
a2.fill_between(nx, naive_lo, naive_hi, color="#bbb", alpha=0.5, label="naive random (min-max, 30 seeds)")
a2.plot(nx, naive_mean, "s--", color="#888", label="naive random (mean)")
a2.plot(aware_x, aware_y, "o-", color="#2b6cb0", label="attack-aware greedy (Alg. 1)")
a2.axhline(0, color="#000", lw=0.6)
a2.set_xlabel("# protected meters"); a2.set_ylabel("attackable targets (of 9)")
a2.set_ylim(-0.3,9.5); a2.legend(fontsize=7.5); a2.set_title("(b) Secure-estimation defense")
fig.suptitle("Fig. 2  Budget reachability and secure-estimation defense (IEEE-14)",fontsize=10)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("fig2_reach_defense.png",dpi=200,bbox_inches="tight"); plt.close(fig)
json.dump(dict(aware_x=aware_x,aware_y=aware_y,naive_x=nx,naive_mean=naive_mean,
               naive_lo=naive_lo,naive_hi=naive_hi,n_seeds=NS,greedy_set=greedy),
          open("fig_defense_band.json","w"), indent=2)
print("wrote fig2_reach_defense.png (aware=greedy Alg.1, naive=mean+band)")
