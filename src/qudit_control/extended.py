"""Mechanism-resolved nuclear-qudit controls. MHz, microseconds and tesla.

Electron and common magnetic operators are expressed in nuclear-equivalent field
units, not calibrated junction voltage. Conditional (tracked) and common nuclear
frames correspond to distinct operational tasks.
"""
from dataclasses import replace
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from .model import Parameters, branches, hamiltonian, target, common_frame_targets, fidelity, MU_B, spin
from .control import ControlProblem, spline_basis
from .validation import envelope_function


def nominal_parameters():
    return Parameters(g_t=1.67)


def full_operator(p, mechanism='nuclear'):
    _,S,N=hamiltonian(1.,p)
    if mechanism=='nuclear': return N[0]
    if mechanism=='electron': return MU_B*p.g_t/p.gamma*S[0]
    if mechanism=='magnetic': return N[0]+MU_B*p.g_t/p.gamma*S[0]
    raise ValueError('mechanism must be nuclear, electron, or magnetic')


def projected(B,p,mechanism='nuclear'):
    E,_,V,purity=branches(B,p)
    O=full_operator(p,mechanism)
    D=V.conj().swapaxes(-1,-2)@O@V
    D=D-np.trace(D,axis1=1,axis2=2)[:,None,None]/E.shape[1]*np.eye(E.shape[1])
    return E,D,V,purity


def calibration_ensemble(B,p,db=.0001,dk=.002,da=.002,gain=.03):
    """Nominal and all 16 corners of four residual calibration errors."""
    from itertools import product
    out=[(B,p,1.,'nominal')]
    for sb,sk,sa,sg in product((-1,1),repeat=4):
        out.append((B+sb*db,replace(p,kappa=p.kappa+sk*dk,A_z=p.A_z+sa*da),
                    1+sg*gain,f'B{sb:+d}_k{sk:+d}_A{sa:+d}_g{sg:+d}'))
    return out


class ExtendedProblem(ControlProblem):
    def __init__(self,B=1.6,p=None,T=24.,amax=.05,ncoef=16,dt=.025,
                 kind='rotation',ensemble=None,frame='tracked',active=(0,1),
                 mechanism='nuclear',beta=1000.,carrier_branches=None,carrier_offset=None):
        if frame not in ('tracked','common') or set(active)-{0,1} or not active:
            raise ValueError('Invalid frame or active branches')
        if min(T,dt,amax)<=0 or ncoef<6: raise ValueError('Invalid controls')
        self.B=B;self.p=p or nominal_parameters();self.T=T;self.amax=amax
        self.ncoef=ncoef;self.kind=kind;self.common_frame=frame=='common';self.frame=frame
        self.active=tuple(active);self.carrier_branches=tuple(active if carrier_branches is None else carrier_branches);self.mechanism=mechanism;self.beta=beta
        self.N=int(np.ceil(T/dt));self.dt=T/self.N
        self.t=(np.arange(self.N)+.5)*self.dt
        self.basis=spline_basis(self.t,T,ncoef);self.K=self.basis.shape[1]
        self.E,self.D,self.V,self.purity=projected(B,self.p,mechanism)
        E,D=self.E,self.D;self.d=E.shape[1]
        self.freq=np.abs(np.diff(E[list(self.carrier_branches)],axis=1)).ravel()
        # Optional per-carrier frequency offsets (MHz), e.g. to track drive-induced
        # (second-order) shifts of the nuclear transitions. Stored with the pulse config.
        self.carrier_offset=None if carrier_offset is None else np.asarray(carrier_offset,float)
        if self.carrier_offset is not None:
            if self.carrier_offset.shape!=self.freq.shape:raise ValueError('carrier_offset must have one entry per carrier')
            self.freq=self.freq+self.carrier_offset
        self.J=len(self.freq);self.ij=np.triu_indices(self.d,1);self.L=len(self.ij[0])
        self.G=target(self.p.I,kind)
        Ws=common_frame_targets(E,T,self.G) if self.common_frame else np.tile(self.G,(2,1,1))
        # Shifted carriers define a shifted conditional frame.  The level shift
        # delta_m implied by the carrier offsets (cumulative along the ladder,
        # overall constant irrelevant) multiplies the tracked target by the
        # diagonal phase exp(-2 pi i delta_m T); this is set by the chosen
        # carriers, not fitted to any simulated output.
        self.frame_shift=np.zeros_like(E)
        if self.carrier_offset is not None and not self.common_frame:
            Ws=Ws.copy();nl=self.d-1
            for k,s in enumerate(self.carrier_branches):
                off=self.carrier_offset[k*nl:(k+1)*nl];dE=np.diff(E[s])
                level=np.r_[0.,np.cumsum(-np.sign(-dE)*off)]   # delta_{m+1}=delta_m - sign(E_m-E_{m+1}) off
                self.frame_shift[s]=level
                Ws[s]=np.exp(-2j*np.pi*level*T)[:,None]*self.G
        self.Wbranch=np.asarray(Ws)
        # Ensemble scenarios are (B, p, gain, label[, t0]).  A start time t0 (us) means
        # that the envelope begins at laboratory time t0 while every carrier runs
        # continuously from t=0: the resonant couplings are unchanged, the off-resonant
        # tone/transition couplings acquire the phases exp(2 pi i (delta-f) t0).  Training
        # over start times spanning the full range of carrier phases a sequence can encounter
        # (0-100 ms in the paper) gives pulses that are robust at held-out start times (composable gates).
        self.ensemble=ensemble or [(B,self.p,1.,'nominal')]
        phases=[];signs=[];drifts=[];targets=[];self.labels=[];self.start_times=[]
        for item in self.ensemble:
            bb,pp,gain,label=item[:4];t0=float(item[4]) if len(item)>4 else 0.
            EE,DD,_,_=projected(bb,pp,mechanism)
            for s in self.active:
                delta=E[s,self.ij[0]]-E[s,self.ij[1]];sg=np.sign(delta)
                det=delta[:,None]-sg[:,None]*self.freq[None,:]
                P=np.exp(2j*np.pi*(self.t+t0)[:,None,None]*det[None,:,:])*np.sinc(self.dt*det)[None,:,:]
                P*=DD[s,self.ij[0],self.ij[1]][None,:,None]*gain
                phases.append(P);signs.append(sg);drifts.append(EE[s]-E[s]);targets.append(self.target_at(s,t0))
                self.labels.append(label+f'_s{s}');self.start_times.append(t0)
        self.P=np.asarray(phases);self.sign=np.asarray(signs)
        self.drift=np.asarray(drifts);self.W=np.asarray(targets);self.C=len(targets)
        self.history=[];self.calls=0

    def target_at(self,s,t0=0.):
        """Conditional target of branch s for a pulse starting at laboratory time t0.

        In the nominal interaction picture a gate G defined in the carrier frame
        (levels E+delta) reads diag(e^{-2 pi i delta (T+t0)}) G diag(e^{+2 pi i delta t0});
        for unshifted carriers (delta=0) it is G itself, independent of t0.
        """
        W=self.Wbranch[s]
        if t0==0. or not np.any(self.frame_shift[s]):return W
        d=self.frame_shift[s]
        return np.exp(-2j*np.pi*d*t0)[:,None]*W*np.exp(2j*np.pi*d*t0)[None,:]

    def initial(self,seed=0):
        ix,_,_=spin(self.p.I)
        area=float(np.sum(self.basis)*self.dt)
        c=np.zeros((self.K,self.J,2)); j=0
        for s in self.carrier_branches:
            for m in range(self.d-1):
                desired=-1j*.25/area*ix[m,m+1]
                a=desired/self.D[s,m,m+1]
                if self.E[s,m]<self.E[s,m+1]:a=a.conjugate()
                if self.kind=='selective' and m!=0:a=0j
                if self.kind=='selective' and m==0:a/=2*ix[m,m+1]
                c[:,j,0]=a.real/self.amax;c[:,j,1]=a.imag/self.amax;j+=1
        if seed:
            c+=np.random.default_rng(seed).normal(0,.10,c.shape)
        return np.clip(c,-1/np.sqrt(2),1/np.sqrt(2)).ravel()

    def objective(self,x,beta=None,regularization=1e-7):
        return super().objective(x,self.beta if beta is None else beta,regularization)

    def optimize(self,x0=None,maxiter=1500,seed=0,checkpoint=None):
        x0=self.initial(seed) if x0 is None else x0
        start=time.time();iterations=[0]
        def cb(x):
            iterations[0]+=1
            if checkpoint is not None and iterations[0]%100==0:
                checkpoint(x,self.history,iterations[0],time.time()-start)
        opt=minimize(self.objective,x0,jac=True,method='L-BFGS-B',callback=cb,
                     bounds=[(-1/np.sqrt(2),1/np.sqrt(2))]*len(x0),
                     options={'maxiter':maxiter,'ftol':2e-13,'gtol':2e-8,'maxls':30,'maxcor':20})
        fs=fidelity(self.propagators(opt.x),self.W)
        return opt,{'elapsed_s':time.time()-start,'iterations':opt.nit,'evaluations':opt.nfev,
                    'success':bool(opt.success),'message':str(opt.message),'worst_train':float(fs.min()),
                    'mean_train':float(fs.mean()),'train_fidelities':fs.tolist(),'history':self.history}


def branch_targets(pr,t0=0.):
    """Conditional targets of both branches for a pulse starting at t0 (carrier frame included)."""
    if hasattr(pr,'target_at'):return np.asarray([pr.target_at(s,t0) for s in (0,1)])
    return common_frame_targets(pr.E,pr.T,pr.G) if pr.common_frame else np.tile(pr.G,(2,1,1))


def integrate_extended(pr,x=None,env=None,counterrotating=True,scenario=None,
                       rtol=2e-9,atol=2e-11,max_step=.003,breakpoints=None):
    """Adaptive projected propagation, independent of optimization stepping."""
    if env is None:env=envelope_function(pr,x)
    E,D=pr.E.copy(),pr.D.copy();drift=np.zeros_like(E);gain=1.;t0=0.
    if scenario is not None:
        bb,pp,gain=scenario[:3];t0=float(scenario[3]) if len(scenario)>3 else 0.
        EE,D,_,_=projected(bb,pp,pr.mechanism);drift=EE-E
    active=list(pr.active);d=pr.d;delta=E[:,:,None]-E[:,None,:]
    dlt=delta[active];dip=D[active];off=drift[active]
    det=dlt[:,:,:,None]-np.sign(dlt)[:,:,:,None]*pr.freq
    idx=np.arange(d)
    def rhs(t,y):
        # The envelope is a function of the time since the pulse started (t); the
        # carriers and the interaction-picture phases run with laboratory time t+t0.
        U=y.reshape(len(active),d,d);a=env(t);tl=t+t0
        if counterrotating:
            u=2*np.real(np.sum(a*np.exp(-2j*np.pi*pr.freq*tl)))
            H=gain*u*dip*np.exp(2j*np.pi*dlt*tl)
        else:
            c=np.where(dlt[:,:,:,None]>=0,a,np.conj(a))
            H=gain*dip*np.sum(c*np.exp(2j*np.pi*det*tl),axis=-1);H[:,idx,idx]=0
        H[:,idx,idx]+=off
        return (-2j*np.pi*H@U).ravel()
    y=np.tile(np.eye(d,dtype=complex),(len(active),1,1)).ravel();nfev=0
    ts=[0.]+list(breakpoints or [])+[pr.T]
    for a,b in zip(ts[:-1],ts[1:]):
        sol=solve_ivp(rhs,(a,b),y,method='DOP853',rtol=rtol,atol=atol,max_step=max_step)
        if not sol.success:raise RuntimeError(sol.message)
        y=sol.y[:,-1];nfev+=sol.nfev
    U=y.reshape(len(active),d,d)
    W=branch_targets(pr,t0)[active]
    f=fidelity(U,W)
    return {'fidelities':f.tolist(),'worst':float(f.min()),'unitarity_error':float(np.max(np.linalg.norm(U.conj().swapaxes(-1,-2)@U-np.eye(d),axis=(-2,-1)))),
            'nfev':nfev,'counterrotating':counterrotating,'rtol':rtol,'atol':atol,'max_step_us':max_step,'start_time_us':t0},U


def plain_envelope(pr):
    """Unshaped, matrix-element-calibrated multitone collective rotation."""
    ix,_,_=spin(pr.p.I);a=[]
    for s in pr.carrier_branches:
        for m in range(pr.d-1):
            z=-1j*.25/pr.T*ix[m,m+1]/pr.D[s,m,m+1]
            if pr.E[s,m]<pr.E[s,m+1]:z=z.conjugate()
            a.append(z)
    a=np.asarray(a)
    if np.max(np.abs(a))>pr.amax*(1+1e-12):raise ValueError('Plain pulse exceeds per-tone bound')
    return lambda t:a.copy()


def full_lab_midpoint(pr,x=None,env=None,dt=.0002,chunk=1000,t0=0.,return_blocks=False,return_full=False):
    """Full 2d-level propagation; exact exponential of each midpoint Hamiltonian.

    Electronic oscillations are retained in the matrix exponential. Convergence
    of the piecewise-constant waveform is checked by successive step halving.
    The reported branch fidelity counts leakage as failure (no renormalization).
    With return_full=True, also return the complete 2d-by-2d propagator in the
    static-energy interaction picture, in the ordered ground/excited branch
    eigenbasis. This retains interbranch paths when composing pulse sequences.
    """
    if return_blocks and return_full:
        raise ValueError("Choose return_blocks or return_full, not both")
    if env is None:env=envelope_function(pr,x)
    H,_,_=hamiltonian(pr.B,pr.p);O=full_operator(pr.p,pr.mechanism)
    # A global energy/drive trace does not change physical gate fidelities.
    H=H-np.trace(H)/len(H)*np.eye(len(H));O=O-np.trace(O)/len(O)*np.eye(len(O))
    real=np.max(np.abs(H.imag))+np.max(np.abs(O.imag))<1e-13
    if real:H=H.real;O=O.real
    N=int(np.ceil(pr.T/dt));dt=pr.T/N
    U=np.eye(len(H),dtype=complex);start=time.time()
    for lo in range(0,N,chunk):
        t=(np.arange(lo,min(N,lo+chunk))+.5)*dt
        amps=np.array([env(float(tt)) for tt in t])
        w=2*np.real(np.sum(amps*np.exp(-2j*np.pi*(t+t0)[:,None]*pr.freq),axis=1))
        hh=H[None,:,:]+w[:,None,None]*O[None,:,:]
        lam,V=np.linalg.eigh(hh)
        uk=(V*np.exp(-2j*np.pi*dt*lam)[:,None,:])@V.conj().swapaxes(-1,-2)
        for u in uk:U=u@U
    W=branch_targets(pr,t0)
    fs=[];leak=[];diagonal_fs=[];diagonal_phases=[];blocks=[]
    for s in pr.active:
        v=pr.V[s];block=v.conj().T@U@v
        # interaction picture with respect to the static Hamiltonian between laboratory times t0 and t0+T
        block=np.exp(2j*np.pi*pr.E[s]*(pr.T+t0))[:,None]*block*np.exp(-2j*np.pi*pr.E[s]*t0)[None,:]
        norm=float(np.trace(block.conj().T@block).real)
        fs.append(float((norm+abs(np.trace(W[s].conj().T@block))**2)/(pr.d*(pr.d+1))))
        leak.append(float(1-norm/pr.d))
        diagonal=np.diag(block@W[s].conj().T)
        diagonal_fs.append(float((norm+np.sum(np.abs(diagonal))**2)/(pr.d*(pr.d+1))))
        diagonal_phases.append(np.angle(diagonal).tolist());blocks.append(block)
    info={'best_diagonal_output_fidelities':diagonal_fs,'best_diagonal_output_phases_rad':diagonal_phases,'fidelities':fs,'worst':min(fs),'leakage':leak,'dt_us':dt,'steps':N,
            'elapsed_s':time.time()-start,'unitarity_error':float(np.linalg.norm(U.conj().T@U-np.eye(len(H)))),'start_time_us':t0}
    if return_full:
        # Keep the actual branch mean energies: discarding their relative phase
        # would alter paths that leave a branch and return in a later pulse.
        Vjoint=np.concatenate(pr.V,axis=1)
        energies=np.real(np.diag(Vjoint.conj().T@H@Vjoint))
        joint=Vjoint.conj().T@U@Vjoint
        joint=np.exp(2j*np.pi*energies*(t0+pr.T))[:,None]*joint*np.exp(-2j*np.pi*energies*t0)[None,:]
        return info,joint
    return (info,np.array(blocks)) if return_blocks else info
