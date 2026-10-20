"""Explicit stochastic-electron and nuclear-dephasing channels.

Electron switching is a classical continuous-time Markov process in an
adiabatically labelled nuclear basis. Jumps preserve the nuclear label. This is
NOT a microscopic substrate/tip scattering model, and is not identified with a
measured Ti switching rate. Its reduced nuclear dynamics retain telegraph memory.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from .model import fidelity,spin


def telegraph_gate(problem,x,tau_c,p_exc,ntraj=8192,seed=20260918,dt=.005,stratified=False):
    """Random-unitary channel from exact exponentially timed electron jumps.

    tau_c = 1/(k_up+k_down). When stratified=True the no-jump contribution
    is integrated exactly and ntraj trajectories are sampled conditional on
    at least one jump. This avoids missing rare jumps at high tau_c or low p.
    Interpolated cumulative unitaries are polar-corrected by two Newton steps;
    time-step convergence is still required and separately recorded.
    """
    if tau_c<=0 or not 0<=p_exc<=1 or ntraj<2: raise ValueError('Invalid telegraph parameters')
    from .control import ControlProblem
    pr=ControlProblem(B=problem.B,p=problem.p,T=problem.T,amax=problem.amax,ncoef=problem.ncoef,dt=dt,kind=problem.kind,common_frame=problem.common_frame)
    _,prefix,*_=pr.propagators(x,True)
    ts=np.arange(pr.N+1)*pr.dt
    spl=[CubicSpline(ts,prefix[s],axis=0) for s in range(2)]
    d=pr.d;T=pr.T;E=pr.E;eye=np.eye(d,dtype=complex)
    target_lab=np.exp(-2j*np.pi*np.mean(E,axis=0)*T)[:,None]*pr.G
    prob=np.array([1-p_exc,p_exc]);rates=np.array([p_exc,1-p_exc])/tau_c
    survive=np.exp(-rates*T);jumpmass=prob*(-np.expm1(-rates*T))
    nojump=float(np.sum(prob*survive));pjump=float(np.sum(jumpmass))
    rng=np.random.default_rng(seed)
    def cumul(s,times):
        ui=spl[s](times)
        # Near-identity Newton polar correction; remove interpolation nonunitarity,
        # not physical leakage (none is included in this projected model).
        for _ in range(2):ui=.5*ui@(3*eye-ui.conj().swapaxes(-1,-2)@ui)
        return np.exp(-2j*np.pi*times[:,None]*E[s])[...,None]*ui
    staticU=np.array([cumul(s,np.array([T]))[0] for s in (0,1)])
    staticF=fidelity(staticU,target_lab)
    if stratified and pjump<1e-15:
        return {'fidelity':float(np.sum(prob*staticF)),'sem':0.,'unitarity_error':0.,
                'nojump_probability':nojump,'sampled_nojump':None,'mean_jumps':0.,'expected_jumps':0.,
                'conditional_jump_fidelity':None,'tau_c_us':tau_c,'p_exc':p_exc,'ntraj':ntraj,
                'seed':seed,'dt_us':pr.dt,'stratified':True}
    state=(rng.random(ntraj)<(jumpmass[1]/pjump if stratified else p_exc)).astype(int)
    t=np.zeros(ntraj);V=np.tile(eye,(ntraj,1,1));alive=np.ones(ntraj,bool);jumps=np.zeros(ntraj,int)
    if stratified:
        # First dwell drawn from an exponential truncated to [0,T].
        u=rng.random(ntraj)
        first=-np.log1p(-u*(-np.expm1(-rates[state]*T)))/rates[state]
        for s in (0,1):
            ids=np.flatnonzero(state==s)
            if len(ids):V[ids]=cumul(s,first[ids])
        t=first;state=1-state;jumps[:]=1
    while np.any(alive):
        for s in (0,1):
            ids=np.flatnonzero(alive&(state==s))
            if not len(ids):continue
            wait=rng.exponential(1/rates[s],len(ids)) if rates[s]>0 else np.full(len(ids),np.inf)
            end=np.minimum(t[ids]+wait,T)
            u0=cumul(s,t[ids]);u1=cumul(s,end)
            V[ids]=(u1@u0.conj().swapaxes(-1,-2))@V[ids]
            switched=end<T;t[ids]=end;alive[ids]=switched
            state[ids[switched]]=1-s;jumps[ids[switched]]+=1
    vals=fidelity(V,target_lab)
    if stratified:
        mean=float(np.sum(prob*survive*staticF)+pjump*np.mean(vals))
        sem=float(pjump*np.std(vals,ddof=1)/np.sqrt(ntraj));mj=float(pjump*np.mean(jumps));sn=None
    else:
        mean=float(np.mean(vals));sem=float(np.std(vals,ddof=1)/np.sqrt(ntraj));mj=float(np.mean(jumps));sn=float(np.mean(jumps==0))
    return {'fidelity':mean,'sem':sem,
            'unitarity_error':float(np.max(np.linalg.norm(V.conj().swapaxes(-1,-2)@V-eye,axis=(-2,-1)))),
            'nojump_probability':nojump,'sampled_nojump':sn,'mean_jumps':mj,
            'expected_jumps':float(2*p_exc*(1-p_exc)*T/tau_c),
            'conditional_jump_fidelity':float(np.mean(vals)) if stratified else None,
            'tau_c_us':tau_c,'p_exc':p_exc,'ntraj':ntraj,'seed':seed,'dt_us':pr.dt,'stratified':stratified}


def dephasing_operator(I,kind):
    _,_,iz=spin(I)
    m=np.diag(iz)
    if kind=='magnetic': return m
    if kind=='quadrupolar': return (m*m-np.mean(m*m))/(2*I-1)
    raise ValueError('Unknown noise operator')


def dephasing_channel(problem,x,Tnoise,kind='magnetic',dt=.01):
    """CPTP Strang splitting for L=sqrt(2/Tnoise) A, propagated on all d^2 inputs.
    Tnoise is defined by a unit difference of A eigenvalues; for the normalized
    quadrupolar operator it is the outermost adjacent coherence, not the central.
    """
    from .control import ControlProblem
    pr=ControlProblem(B=problem.B,p=problem.p,T=problem.T,amax=problem.amax,ncoef=problem.ncoef,dt=dt,kind=problem.kind,common_frame=problem.common_frame)
    H=pr.hamiltonians(x);w,v=np.linalg.eigh(H)
    half=(v*np.exp(-1j*np.pi*pr.dt*w)[:,:,None,:])@v.conj().swapaxes(-1,-2)
    d=pr.d
    out=np.tile(np.eye(d*d,dtype=complex).reshape(d*d,d,d),(2,1,1,1))
    A=dephasing_operator(pr.p.I,kind)
    damp=np.exp(-pr.dt*(A[:,None]-A[None,:])**2/Tnoise)
    for k in range(pr.N):
        u=half[:,k,None,:,:]
        out=u@out@u.conj().swapaxes(-1,-2)
        out*=damp[None,None,:,:]
        out=u@out@u.conj().swapaxes(-1,-2)
    inputs=np.eye(d*d,dtype=complex).reshape(d*d,d,d)
    ideal=pr.W[:,None,:,:]@inputs[None,:,:,:]@pr.W.conj().swapaxes(-1,-2)[:,None,:,:]
    overlap=np.einsum('caij,caij->c',ideal.conj(),out).real
    fs=(d+overlap)/(d*(d+1))
    return {'fidelities':fs.tolist(),'worst':float(np.min(fs)),'Tnoise_us':Tnoise,'kind':kind,'dt_us':pr.dt}


def white_noise_slope(I,kind='magnetic'):
    A=dephasing_operator(I,kind);d=len(A)
    return float(2*(d*np.sum(A*A)-np.sum(A)**2)/(d*(d+1)))


def idle_telegraph_exact(E,T,tau_c,p_exc,G=None):
    """Exact free channel in the common nuclear reference via 2x2 matrix exponentials."""
    d=E.shape[1];common=np.mean(E,axis=0);prob=np.array([1-p_exc,p_exc])
    up=p_exc/tau_c;down=(1-p_exc)/tau_c
    K=np.array([[-up,down],[up,-down]],complex)
    total=0j
    gdiag=np.ones(d,complex) if G is None else np.diag(G)
    for m in range(d):
        for n in range(d):
            w=2*np.pi*(E[:,m]-E[:,n]-(common[m]-common[n]))
            total+=gdiag[m].conj()*gdiag[n]*np.sum(expm((K-1j*np.diag(w))*T)@prob)
    return float((d+total.real)/(d*(d+1)))
