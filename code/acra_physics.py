"""
Grid-agnostic ACRA physics core (pandapower + MILP). No LLM.
Works on any pandapower case (case14, case39, case118).
Exposes: build(net_fn, load_scale), and per-target k_app / k_tel / reachability / secure sweep.
"""
import numpy as np, copy
import pandapower as pp, pandapower.networks as nw
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from scipy.stats import chi2

HEADROOM=1.35; MBIG=1e4; MARGIN=0.97

def build(net_fn, load_scale=1.05, headroom=None, gen_mod=None):
    hr=HEADROOM if headroom is None else headroom
    net=net_fn(); pp.runpp(net,numba=False)
    i=net.res_line.i_ka.values.copy(); fl=max(1e-3,np.median(i)*0.25)
    net.line["max_i_ka"]=np.maximum(i,fl)*hr
    net=copy.deepcopy(net); net.load["p_mw"]*=load_scale; net.load["q_mvar"]*=load_scale
    if gen_mod is not None and len(net.gen)>0:   # redistribute generation dispatch
        net.gen["p_mw"]=net.gen["p_mw"].values*gen_mod(len(net.gen))
    pp.rundcpp(net,numba=False); Sbase=net.sn_mva
    slack=int(net.ext_grid.bus.iloc[0]); nb=len(net.bus)
    bidx={int(x):i for i,x in enumerate(net.bus.index)}; br=[]
    for li in net.line.index:
        if not net.line.at[li,"in_service"]: continue
        f=bidx[int(net.line.at[li,"from_bus"])]; t=bidx[int(net.line.at[li,"to_bus"])]
        x=net.line.at[li,"x_ohm_per_km"]*net.line.at[li,"length_km"]
        vb=net.bus.at[int(net.line.at[li,"from_bus"]),"vn_kv"]; zb=vb**2/Sbase
        br.append((f,t,x/zb,np.sqrt(3)*vb*net.line.at[li,"max_i_ka"],("line",int(li))))
    for ti in net.trafo.index:
        if not net.trafo.at[ti,"in_service"]: continue
        f=bidx[int(net.trafo.at[ti,"hv_bus"])]; t=bidx[int(net.trafo.at[ti,"lv_bus"])]
        xpu=net.trafo.at[ti,"vk_percent"]/100.0*(Sbase/max(net.trafo.at[ti,"sn_mva"],1e-6))
        rate=net.trafo.at[ti,"sn_mva"] if net.trafo.at[ti,"sn_mva"]>0 else np.median([b[3] for b in br])
        br.append((f,t,xpu,rate,("trafo",int(ti))))
    red=[i for i in range(nb) if i!=slack]; pos={bb:i for i,bb in enumerate(red)}
    nbr=len(br); F=np.zeros((nbr,nb-1)); Bred=np.zeros((nb-1,nb-1))
    for k,(f,t,x,r,l) in enumerate(br):
        bsus=1.0/x
        if f!=slack: F[k,pos[f]]+=bsus*Sbase
        if t!=slack: F[k,pos[t]]-=bsus*Sbase
        for a,c2,s in [(f,f,1),(t,t,1),(f,t,-1),(t,f,-1)]:
            if a!=slack and c2!=slack: Bred[pos[a],pos[c2]]+=s*bsus*Sbase
    H=np.vstack([F[k].copy() for k in range(nbr)]+[Bred[i].copy() for i in range(nb-1)])
    theta=np.deg2rad(net.res_bus.va_degree.values); tred=np.array([theta[i] for i in range(nb) if i!=slack])
    z=H@tred; sigma=0.01*np.maximum(np.abs(z),1.0); W=np.diag(1.0/sigma**2)
    return dict(F=F,H=H,theta=tred,br=br,ratings=np.array([b[3] for b in br]),
                Bred=Bred,pos=pos,slack=slack,z=z,sigma=sigma,W=W,net=net,nbr=nbr)

def lodf_col(L,gm):
    F=gm["F"]; br=gm["br"]; pos=gm["pos"]; slack=gm["slack"]; n=F.shape[1]
    Binv=np.linalg.pinv(gm["Bred"]); (fM,tM,xM,_,_)=br[L]; e=np.zeros(n)
    if fM!=slack: e[pos[fM]]+=1
    if tM!=slack: e[pos[tM]]-=1
    ptdf=F@(Binv@e); d=1.0-ptdf[L]
    return None if abs(d)<1e-6 else ptdf/d

def ac_unsafe_open(gm,lid):
    n=copy.deepcopy(gm["net"]);
    try:
        idx=[li for li in n.line.index if int(li)==lid]
        if not idx: return False
        n.line.at[lid,"in_service"]=False; pp.runpp(n,numba=False)
    except Exception: return True
    return bool(n.converged and n.res_line.loading_percent.max()>100)

def chi2_thr(gm,alpha=0.99):
    dof=gm["H"].shape[0]-gm["H"].shape[1]; return chi2.ppf(alpha,dof)

def goals_induce(gm,L):
    F=gm["F"]; ratings=gm["ratings"]; base=F@gm["theta"]
    sgn=float(np.sign(base[L] if abs(base[L])>1 else 1.0))
    return [(sgn*F[L],">=",1.00*ratings[L]-sgn*base[L]),(sgn*F[L],"<=",1.15*ratings[L]-sgn*base[L])]

def goals_approve(gm,L,margin=MARGIN):
    F=gm["F"]; ratings=gm["ratings"]; base=F@gm["theta"]; lo=lodf_col(L,gm)
    if lo is None: return None
    g=[]
    for j in range(gm["nbr"]):
        if j==L: continue
        rv=F[j]+lo[j]*F[L]; cst=base[j]+lo[j]*base[L]
        g.append((rv,"<=",margin*ratings[j]-cst)); g.append((-rv,"<=",margin*ratings[j]+cst))
    return g

def _bound_diag(gm,tau):
    """Per-meter magnitude cap for a compromised meter: tau*sigma_i if tau given, else MBIG."""
    m_=gm["H"].shape[0]
    return (tau*gm["sigma"]) if tau is not None else (MBIG*np.ones(m_))

def solve_min_meters(gm,goals,cap,protected=None,time_limit=10,tau=None,bnd=None):
    H=gm["H"]; m_,n=H.shape; N=n+2*m_; cons=[]
    if bnd is None: bnd=_bound_diag(gm,tau)
    A1=np.zeros((m_,N)); A1[:,n:n+m_]=np.eye(m_); A1[:,0:n]=-H; cons.append(LinearConstraint(csr_matrix(A1),0,0))
    A2=np.zeros((m_,N)); A2[:,n:n+m_]=np.eye(m_); A2[:,n+m_:]=-np.diag(bnd); cons.append(LinearConstraint(csr_matrix(A2),-np.inf,0))
    A3=np.zeros((m_,N)); A3[:,n:n+m_]=-np.eye(m_); A3[:,n+m_:]=-np.diag(bnd); cons.append(LinearConstraint(csr_matrix(A3),-np.inf,0))
    for (row,sense,rhs) in goals:
        g=np.zeros(N); g[0:n]=row
        cons.append(LinearConstraint(csr_matrix(g.reshape(1,-1)),-np.inf,rhs) if sense=="<=" else LinearConstraint(csr_matrix(g.reshape(1,-1)),rhs,np.inf))
    gb=np.zeros(N); gb[n+m_:]=1; cons.append(LinearConstraint(csr_matrix(gb.reshape(1,-1)),0,cap))
    lb=np.concatenate([-10*np.ones(n),-np.maximum(bnd,1e-9),np.zeros(m_)])
    ub=np.concatenate([10*np.ones(n),np.maximum(bnd,1e-9),np.ones(m_)])
    if protected:
        for i in protected: ub[n+m_+i]=0.0
    c_obj=np.zeros(N); c_obj[n+m_:]=1; integ=np.zeros(N); integ[n+m_:]=1
    res=milp(c=c_obj,constraints=cons,integrality=integ,bounds=Bounds(lb,ub),options={"time_limit":time_limit})
    if not res.success: return None,None
    ind=res.x[n+m_:]; used=[i for i in range(m_) if ind[i]>0.5]
    return len(used),used

def solve_capture_c(gm,goals,cap,protected=None,time_limit=10,tau=None,bnd=None):
    """Same MILP as solve_min_meters but returns the state-shift vector c (x[:n]), or None."""
    H=gm["H"]; m_,n=H.shape; N=n+2*m_; cons=[]
    if bnd is None: bnd=_bound_diag(gm,tau)
    A1=np.zeros((m_,N)); A1[:,n:n+m_]=np.eye(m_); A1[:,0:n]=-H; cons.append(LinearConstraint(csr_matrix(A1),0,0))
    A2=np.zeros((m_,N)); A2[:,n:n+m_]=np.eye(m_); A2[:,n+m_:]=-np.diag(bnd); cons.append(LinearConstraint(csr_matrix(A2),-np.inf,0))
    A3=np.zeros((m_,N)); A3[:,n:n+m_]=-np.eye(m_); A3[:,n+m_:]=-np.diag(bnd); cons.append(LinearConstraint(csr_matrix(A3),-np.inf,0))
    for (row,sense,rhs) in goals:
        g=np.zeros(N); g[0:n]=row
        cons.append(LinearConstraint(csr_matrix(g.reshape(1,-1)),-np.inf,rhs) if sense=="<=" else LinearConstraint(csr_matrix(g.reshape(1,-1)),rhs,np.inf))
    gb=np.zeros(N); gb[n+m_:]=1; cons.append(LinearConstraint(csr_matrix(gb.reshape(1,-1)),0,cap))
    lb=np.concatenate([-10*np.ones(n),-np.maximum(bnd,1e-9),np.zeros(m_)])
    ub=np.concatenate([10*np.ones(n),np.maximum(bnd,1e-9),np.ones(m_)])
    if protected:
        for i in protected: ub[n+m_+i]=0.0
    c_obj=np.zeros(N); c_obj[n+m_:]=1; integ=np.zeros(N); integ[n+m_:]=1
    res=milp(c=c_obj,constraints=cons,integrality=integ,bounds=Bounds(lb,ub),options={"time_limit":time_limit})
    return res.x[:n] if res.success else None

def solve_status(gm,goals,cap,protected=None,time_limit=10,tau=None,bnd=None):
    """Returns (count_or_None, status) with status in {optimal,infeasible,timeout,other}."""
    H=gm["H"]; m_,n=H.shape; N=n+2*m_; cons=[]
    if bnd is None: bnd=_bound_diag(gm,tau)
    A1=np.zeros((m_,N)); A1[:,n:n+m_]=np.eye(m_); A1[:,0:n]=-H; cons.append(LinearConstraint(csr_matrix(A1),0,0))
    A2=np.zeros((m_,N)); A2[:,n:n+m_]=np.eye(m_); A2[:,n+m_:]=-np.diag(bnd); cons.append(LinearConstraint(csr_matrix(A2),-np.inf,0))
    A3=np.zeros((m_,N)); A3[:,n:n+m_]=-np.eye(m_); A3[:,n+m_:]=-np.diag(bnd); cons.append(LinearConstraint(csr_matrix(A3),-np.inf,0))
    for (row,sense,rhs) in goals:
        g=np.zeros(N); g[0:n]=row
        cons.append(LinearConstraint(csr_matrix(g.reshape(1,-1)),-np.inf,rhs) if sense=="<=" else LinearConstraint(csr_matrix(g.reshape(1,-1)),rhs,np.inf))
    gb=np.zeros(N); gb[n+m_:]=1; cons.append(LinearConstraint(csr_matrix(gb.reshape(1,-1)),0,cap))
    lb=np.concatenate([-10*np.ones(n),-np.maximum(bnd,1e-9),np.zeros(m_)])
    ub=np.concatenate([10*np.ones(n),np.maximum(bnd,1e-9),np.ones(m_)])
    if protected:
        for i in protected: ub[n+m_+i]=0.0
    c_obj=np.zeros(N); c_obj[n+m_:]=1; integ=np.zeros(N); integ[n+m_:]=1
    res=milp(c=c_obj,constraints=cons,integrality=integ,bounds=Bounds(lb,ub),options={"time_limit":time_limit})
    smap={0:"optimal",1:"timeout",2:"infeasible",3:"unbounded",4:"other"}
    st=smap.get(res.status,"other")
    if res.success:
        ind=res.x[n+m_:]; return int(round(sum(1 for i in range(m_) if ind[i]>0.5))), "optimal"
    return None, st

def valid_targets(gm, cand_lines=None, cap=None):
    br=gm["br"]; F=gm["F"]; theta=gm["theta"]; ratings=gm["ratings"]; m_=gm["H"].shape[0]
    if cap is None: cap=m_
    out=[]
    for Lh,bb in enumerate(br):
        if bb[4][0]!="line": continue
        lid=bb[4][1]
        if cand_lines is not None and lid not in cand_lines: continue
        if not ac_unsafe_open(gm,lid): continue
        ga=goals_approve(gm,Lh)
        if ga is None: continue
        k,_=solve_min_meters(gm,ga,cap=cap)
        if k is None: continue
        out.append(Lh)
    return out
