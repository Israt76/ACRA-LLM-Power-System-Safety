"""Noisy Monte-Carlo BDD experiment (IEEE-14).
Adds Gaussian measurement noise N(0,sigma). Reports, with Wilson 95% CIs over TRIALS draws:
  - clean false-alarm rate (no attack)
  - stealth coordinated-injection detection rate (should match false-alarm; it is in col-space)
  - random same-magnitude injection detection rate (per target -> report distribution)
"""
import numpy as np, json, math, pandapower.networks as nw
from scipy.stats import chi2
import acra_physics as A

CAND={0,6,7,8,9,10,11,12,14}; TRIALS=3000; ALPHA=0.99; SEED=1234
gm=A.build(nw.case14, load_scale=1.05); br=gm["br"]; H=gm["H"]; W=gm["W"]; sig=gm["sigma"]
m_,n=H.shape; dof=m_-n; thr=float(chi2.ppf(ALPHA,dof)); ztrue=gm["z"]
tg=A.valid_targets(gm, cand_lines=CAND)
G=H.T@W@H; Ginv=np.linalg.inv(G)
def J_of(zz):
    xh=Ginv@(H.T@W@zz); r=zz-H@xh; return float(r.T@W@r)
def wilson(k,N,z=1.96):
    if N==0: return (0,0,0)
    p=k/N; d=1+z*z/N; c=(p+z*z/(2*N))/d; h=(z*math.sqrt(p*(1-p)/N+z*z/(4*N*N)))/d
    return (round(p,4),round(max(0,c-h),4),round(min(1,c+h),4))
rng=np.random.default_rng(SEED)
noise=rng.normal(0,1,(TRIALS,m_))*sig  # shared noise draws

# clean false-alarm
fa=sum(1 for t in range(TRIALS) if J_of(ztrue+noise[t])>thr)
print(f"chi2 thr(alpha={ALPHA},dof={dof})={thr:.1f}  |  clean false-alarm: {wilson(fa,TRIALS)}")

per=[]
for L in tg:
    c=A.solve_capture_c(gm,A.goals_approve(gm,L),cap=m_)
    if c is None: continue
    a=H@c; used=np.where(np.abs(a)>1e-6)[0]; k=len(used); L2=np.linalg.norm(a)
    det_stealth=0; det_rand=0
    for t in range(TRIALS):
        if J_of(ztrue+noise[t]+a)>thr: det_stealth+=1
        ar=np.zeros(m_); idx=rng.choice(m_,max(k,1),replace=False)
        v=rng.normal(0,1,len(idx)); v=v/np.linalg.norm(v)*L2; ar[idx]=v
        if J_of(ztrue+noise[t]+ar)>thr: det_rand+=1
    per.append(dict(line=br[L][4][1], meters=k,
                    stealth_detect=wilson(det_stealth,TRIALS), random_detect=wilson(det_rand,TRIALS)))
    print(f"  L{br[L][4][1]:<3d} k={k:2d}  stealth_detect={per[-1]['stealth_detect']}  random_detect={per[-1]['random_detect']}")

rd=[p['random_detect'][0] for p in per]
sd=[p['stealth_detect'][0] for p in per]
print(f"\nstealth detection across targets: min {min(sd):.3f} max {max(sd):.3f} (≈ false-alarm {fa/TRIALS:.3f})")
print(f"random  detection across targets: min {min(rd):.3f} max {max(rd):.3f} mean {np.mean(rd):.3f}")
json.dump(dict(trials=TRIALS, alpha=ALPHA, dof=dof, chi2_threshold=round(thr,2),
               clean_false_alarm=wilson(fa,TRIALS), per_target=per,
               random_detect_range=[round(min(rd),3),round(max(rd),3)],
               stealth_detect_range=[round(min(sd),3),round(max(sd),3)]),
          open("exp_bdd_noise.json","w"), indent=2)
print("saved exp_bdd_noise.json")
