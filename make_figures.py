import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family":"serif","font.size":10})

# ---------- Figure 1: architecture / threat model ----------
fig,ax=plt.subplots(figsize=(7.2,3.4)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
def box(x,y,w,h,txt,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.12",
                 fc=fc,ec="#222",lw=1.2)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9)
def arrow(x1,y1,x2,y2,color="#222",style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=12,lw=1.3,color=color))
# physical plant + measurements
box(0.2,2.0,1.9,1.0,"Power grid\n(true AC state)","#e8eef7")
box(2.6,2.0,1.9,1.0,"Measurements\n+ WLS / BDD","#e8eef7")
box(5.0,3.1,2.1,1.0,"LLM agent\n(action proposal)","#eaf3ea")
box(5.0,0.9,2.1,1.0,"Digital-twin\nverifier","#eaf3ea")
box(7.8,2.0,1.9,1.0,"Execute / block\nswitching action","#f7efe8")
arrow(2.1,2.5,2.6,2.5);
arrow(4.5,2.6,5.0,3.4); arrow(4.5,2.4,5.0,1.4)   # shared estimate to both
arrow(7.1,3.5,7.9,3.0); arrow(7.1,1.3,7.9,2.0)
ax.text(4.75,2.0,"shared\nstate estimate",ha="center",va="center",fontsize=7.5,color="#444")
# attacker channels
box(5.0,4.4,2.1,0.55,"SEMANTIC attack (forged procedure)","#f6d9d9")
box(2.6,0.35,1.9,0.55,"TELEMETRY FDI  a = Hc","#f6d9d9")
arrow(6.05,4.4,6.05,4.1,color="#b22"); arrow(3.55,0.9,3.55,1.1,color="#b22")
ax.text(5.0,0.05,"coordinated around one target action  (ACRA)",fontsize=8,color="#b22")
ax.set_title("Fig. 1  Coordinated semantic-telemetry attack on an LLM-agent safety shield",fontsize=10)
fig.tight_layout(); fig.savefig("fig1_architecture.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# ---------- Figure 2: reachability + defense ----------
ab=json.load(open("exp_ablation.json")); sec=json.load(open("exp_secure.json"))
K=[a["K"] for a in ab["fdi_budget_ablation"]]; aj=[a["A_joint"] for a in ab["fdi_budget_ablation"]]
at=[a["A_tel"] for a in ab["fdi_budget_ablation"]]
pr=[r["n_protected"] for r in sec["sweep"]]; fa=[r["feasible_aware"] for r in sec["sweep"]]
fn=[r["feasible_naive"] for r in sec["sweep"]]
fig,(a1,a2)=plt.subplots(1,2,figsize=(8.4,3.3))
a1.plot(K,aj,"o-",color="#2b6cb0",label="coordinated  $A_{joint}(K)$")
a1.plot(K,at,"s--",color="#b22",label="telemetry-only  $A_{tel}(K)$")
a1.axvline(8,ls=":",color="#888"); a1.set_xlabel("telemetry budget $K$ (meters)")
a1.set_ylabel("reachable unsafe actions (of 9)"); a1.set_ylim(-0.3,9.5); a1.legend(fontsize=8); a1.set_title("(a) Reachability advantage")
a2.plot(pr,fa,"o-",color="#2b6cb0",label="attack-aware protection")
a2.plot(pr,fn,"s--",color="#888",label="naive (random) protection")
a2.set_xlabel("# protected meters"); a2.set_ylabel("attackable targets (of 9)"); a2.set_ylim(-0.3,9.5)
a2.legend(fontsize=8); a2.set_title("(b) Secure-estimation defense")
fig.suptitle("Fig. 2  Budget reachability and secure-estimation defense (IEEE-14)",fontsize=10)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig2_reach_defense.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("wrote fig1_architecture.png fig2_reach_defense.png")
