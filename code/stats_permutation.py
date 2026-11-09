"""Cluster-level permutation test for the main LLM result (reported in Section VI).
Tests JOINT vs the strongest competing condition, permuting condition labels within
(model x target x wording) clusters. Reads ../data/llm_four_model_raw.jsonl.
Reproduces: observed JOINT-minus-best-other success-rate gap = 0.543, p < 1e-4."""
import json, collections, random, os
HERE=os.path.dirname(__file__)
PATH=os.path.join(HERE,"..","data","llm_four_model_raw.jsonl")
rows=[json.loads(l) for l in open(PATH)]
conds=sorted(set(r["condition"] for r in rows))
by=collections.defaultdict(list)
for r in rows: by[r["condition"]].append(1 if r["attack_success"] else 0)
print("condition success rates:")
for c in conds: print(f"  {c:10s} n={len(by[c]):4d} rate={sum(by[c])/len(by[c]):.3f}")

clusters=collections.defaultdict(list)
for r in rows: clusters[(r["model"],r["target"],r["doc"])].append(r)
def stat(assign):
    s=collections.defaultdict(lambda:[0,0])
    for r in rows:
        c=assign[id(r)]; s[c][0]+=1 if r["attack_success"] else 0; s[c][1]+=1
    jr=s["JOINT"][0]/s["JOINT"][1] if s["JOINT"][1] else 0
    other=max((s[c][0]/s[c][1] if s[c][1] else 0) for c in conds if c!="JOINT")
    return jr-other
obs=stat({id(r):r["condition"] for r in rows})
random.seed(1234); N=10000; ge=0
lab={k:[r["condition"] for r in v] for k,v in clusters.items()}
for _ in range(N):
    assign={}
    for k,v in clusters.items():
        p=lab[k][:]; random.shuffle(p)
        for r,l in zip(v,p): assign[id(r)]=l
    if stat(assign)>=obs-1e-12: ge+=1
print(f"\nobserved JOINT - best-other gap = {obs:.3f}")
print(f"cluster-permutation p = {(ge+1)/(N+1):.5f}  ({N} perms; {ge} >= observed)")
