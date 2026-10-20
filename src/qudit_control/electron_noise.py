"""Prepared-electron telegraph channels with a fixed laboratory waveform.

The classical jumps preserve the adiabatic nuclear label. The stationary excited
fraction sets the two rates; the independently specified initial fraction sets
preparation. Neither rate is identified with a measured STM lifetime.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from .extended import ExtendedProblem
from .model import fidelity, spin
from .requirements import no_jump_probability


def prepared_telegraph(pr,x,tau_c,p_exc,ntraj=4096,seed=20260919,dt=.005,p_initial=0.):
    if tau_c<=0 or not 0<=p_exc<=1 or ntraj<2 or not 0<=p_initial<=1:
        raise ValueError('Invalid telegraph parameters')
    # Both conditional Hamiltonians see exactly the original seven/14 carriers.
    problem=ExtendedProblem(B=pr.B,p=pr.p,T=pr.T,amax=pr.amax,ncoef=pr.ncoef,dt=dt,
               kind=pr.kind,frame='tracked',active=(0,1),carrier_branches=pr.carrier_branches,
               mechanism=pr.mechanism)
    _,prefix,*_=problem.propagators(x,True)
    ts=np.arange(problem.N+1)*problem.dt
    spl=[CubicSpline(ts,prefix[s],axis=0) for s in (0,1)]
    d=pr.d;T=pr.T;E=pr.E;eye=np.eye(d,dtype=complex)
    target_lab=np.exp(-2j*np.pi*E[0]*T)[:,None]*pr.G
    rates=np.array([p_exc,1-p_exc])/tau_c
    initial=np.array([1-p_initial,p_initial]);survive=np.exp(-rates*T)
    masses=initial*(-np.expm1(-rates*T));pjump=float(masses.sum())
    def cumul(s,t):
        u=spl[s](t)
        for _ in range(2):u=.5*u@(3*eye-u.conj().swapaxes(-1,-2)@u)
        return np.exp(-2j*np.pi*t[:,None]*E[s])[:,:,None]*u
    static=np.array([cumul(s,np.array([T]))[0] for s in (0,1)])
    fstatic=fidelity(static,target_lab)
    nojump_contribution=float(np.sum(initial*survive*fstatic))
    if pjump<1e-15:
        return dict(fidelity=nojump_contribution,sem=0.,nojump_probability=1.,conditional_jump_fidelity=None,
                    tau_c_us=tau_c,p_exc=p_exc,p_initial=p_initial,ntraj=ntraj,seed=seed,dt_us=problem.dt)
    rng=np.random.default_rng(seed)
    state=(rng.random(ntraj)<masses[1]/pjump).astype(int)
    r=rates[state];u=rng.random(ntraj)
    first=-np.log1p(-u*(-np.expm1(-r*T)))/r
    V=np.tile(eye,(ntraj,1,1))
    for s in (0,1):
        ids=np.flatnonzero(state==s)
        if len(ids):V[ids]=cumul(s,first[ids])
    t=first;state=1-state;alive=np.ones(ntraj,bool);nj=np.ones(ntraj,int)
    while np.any(alive):
        # A trajectory can advance twice in one iteration; jump times stay exact.
        for s in (0,1):
            ids=np.flatnonzero(alive&(state==s))
            if not len(ids):continue
            w=rng.exponential(1/rates[s],len(ids)) if rates[s]>0 else np.full(len(ids),np.inf)
            end=np.minimum(t[ids]+w,T)
            V[ids]=(cumul(s,end)@cumul(s,t[ids]).conj().swapaxes(-1,-2))@V[ids]
            jumps=end<T;t[ids]=end;alive[ids]=jumps;state[ids[jumps]]=1-s;nj[ids[jumps]]+=1
    vals=fidelity(V,target_lab)
    return dict(fidelity=nojump_contribution+pjump*float(vals.mean()),
                sem=pjump*float(vals.std(ddof=1))/np.sqrt(ntraj),
                nojump_probability=float(np.sum(initial*survive)),
                conditional_jump_fidelity=float(vals.mean()),conditional_mean_jumps=float(nj.mean()),
                unitarity_error=float(np.max(np.linalg.norm(V.conj().swapaxes(-1,-2)@V-eye,axis=(-2,-1)))),
                tau_c_us=tau_c,p_exc=p_exc,p_initial=p_initial,ntraj=ntraj,seed=seed,dt_us=problem.dt)


def prepared_dephasing(pr,x,Tnoise,kind='magnetic',dt=.00625):
    """CPTP Strang split; all d^2 matrix units, L=sqrt(2/Tnoise) A."""
    from .noise import dephasing_operator
    problem=ExtendedProblem(B=pr.B,p=pr.p,T=pr.T,amax=pr.amax,ncoef=pr.ncoef,dt=dt,
             kind=pr.kind,active=(0,),carrier_branches=pr.carrier_branches,mechanism=pr.mechanism)
    H=problem.hamiltonians(x)[0];w,v=np.linalg.eigh(H)
    half=(v*np.exp(-1j*np.pi*problem.dt*w)[:,None,:])@v.conj().swapaxes(-1,-2)
    d=pr.d;inputs=np.eye(d*d,dtype=complex).reshape(d*d,d,d);out=inputs.copy()
    A=dephasing_operator(pr.p.I,kind)
    damping=np.exp(-problem.dt*(A[:,None]-A[None,:])**2/Tnoise)
    for u in half:
        out=u@out@u.conj().T;out*=damping;out=u@out@u.conj().T
    ideal=pr.G@inputs@pr.G.conj().T
    overlap=float(np.einsum('aij,aij->',ideal.conj(),out).real)
    return dict(fidelity=(d+overlap)/(d*(d+1)),Tnoise_us=Tnoise,kind=kind,dt_us=problem.dt)
