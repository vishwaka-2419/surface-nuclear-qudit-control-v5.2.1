"""Bounded multitone spline control with exact exponential Frechet gradients.

A real laboratory waveform is 2 Re sum_j a_j(t) exp(-i 2pi f_j t).
All off-resonant tone/transition pairs are retained in the sign-correct RWA.
A sinc-integrated first Magnus step is used and independently refined.
"""
from dataclasses import replace
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import logsumexp
from .model import Parameters,branches,target,common_frame_targets,fidelity


def spline_basis(t,T,ncoef):
    k=3
    knots=np.r_[np.zeros(k+1),np.linspace(0,T,ncoef-k+1)[1:-1],np.full(k+1,T)]
    full=BSpline.design_matrix(np.clip(t,0,T),knots,k).toarray()
    # Two zero end coefficients enforce a=da/dt=0 at both ends.
    return full[:,2:-2]


def training_ensemble(B,p,db=.001,dk=.02,gain=.03):
    items=[(B,p,1.,'nominal')]
    for sb in (-1,1):
        for sk in (-1,1):
            for sg in (-1,1):
                items.append((B+sb*db,replace(p,kappa=p.kappa+sk*dk),1+sg*gain,
                              f'B{sb:+d}_k{sk:+d}_g{sg:+d}'))
    return items


class ControlProblem:
    def __init__(self,B=1.03,p=None,T=12.,amax=.05,ncoef=12,dt=.025,
                 kind='rotation',ensemble=None,common_frame=True):
        if T<=0 or dt<=0 or amax<=0 or ncoef<6:
            raise ValueError("Require positive duration, mesh, amplitude and at least six spline coefficients")
        self.B=B; self.p=p or Parameters(); self.T=T; self.amax=amax
        self.ncoef=ncoef; self.kind=kind; self.common_frame=common_frame
        self.N=int(np.ceil(T/dt)); self.dt=T/self.N
        self.t=(np.arange(self.N)+.5)*self.dt
        self.basis=spline_basis(self.t,T,ncoef); self.K=self.basis.shape[1]
        E,D,V,purity=branches(B,self.p); self.E=E; self.D=D; self.purity=purity
        self.freq=np.abs(np.diff(E,axis=1)).ravel()
        self.J=len(self.freq); self.d=E.shape[1]
        self.ij=np.triu_indices(self.d,1); self.L=len(self.ij[0])
        self.G=target(self.p.I,kind)
        targets=common_frame_targets(E,T,self.G) if common_frame else np.tile(self.G,(2,1,1))
        ensemble=ensemble or [(B,self.p,1.,'nominal')]
        self.ensemble=ensemble; self.labels=[]; phases=[]; drifts=[]; Ws=[]; signs=[]
        for bb,pp,gain,label in ensemble:
            EE,DD,_,_=branches(bb,pp)
            for s in range(2):
                delta=E[s,self.ij[0]]-E[s,self.ij[1]]
                sg=np.sign(delta)
                det=delta[:,None]-sg[:,None]*self.freq[None,:]
                phase=np.exp(2j*np.pi*self.t[:,None,None]*det[None,:,:])*np.sinc(self.dt*det)[None,:,:]
                phase*=DD[s,self.ij[0],self.ij[1]][None,:,None]*gain
                phases.append(phase); signs.append(sg)
                drifts.append(EE[s]-E[s]); Ws.append(targets[s]); self.labels.append(label+f'_s{s}')
        self.P=np.array(phases); self.sign=np.array(signs)
        self.drift=np.array(drifts); self.W=np.array(Ws); self.C=len(Ws)
        self.history=[]; self.calls=0

    def initial(self,seed=0):
        # Ideal corrected resonant phases, with a small deterministic random kick.
        signs=np.sign(self.E[:,self.ij[0][:self.d-1]] if False else self.E[:,:-1]-self.E[:,1:])
        coeff=np.zeros((self.K,self.J,2))
        # Integrate the shared smooth envelope to match theta=pi/2.
        area=float(np.sum(self.basis)*self.dt)
        a=min(.25/area,self.amax*.60) if self.kind=='rotation' else self.amax*.18
        coeff[:,:,1]=-signs.ravel()[None,:]*a/self.amax
        rng=np.random.default_rng(seed)
        coeff+=rng.normal(0,.025,coeff.shape)
        return np.clip(coeff,-1/np.sqrt(2),1/np.sqrt(2)).ravel()

    def amplitudes(self,x):
        c=np.asarray(x).reshape(self.K,self.J,2)
        a=self.amax*np.einsum('tk,kjq->tjq',self.basis,c)
        return a[...,0]+1j*a[...,1]

    def hamiltonians(self,x):
        a=self.amplitudes(x)
        h=np.einsum('ctlj,tj->ctl',self.P,a.real)+1j*np.einsum('ctlj,cl,tj->ctl',self.P,self.sign,a.imag)
        H=np.zeros((self.C,self.N,self.d,self.d),complex)
        H[:,:,self.ij[0],self.ij[1]]=h
        H[:,:,self.ij[1],self.ij[0]]=h.conj()
        idx=np.arange(self.d); H[:,:,idx,idx]=self.drift[:,None,:]
        return H

    def propagators(self,x,return_steps=False):
        H=self.hamiltonians(x); lam,V=np.linalg.eigh(H)
        e=np.exp(-2j*np.pi*self.dt*lam)
        Uk=(V*e[:,:,None,:])@V.conj().swapaxes(-1,-2)
        U=np.tile(np.eye(self.d,dtype=complex),(self.C,1,1))
        prefix=np.empty((self.C,self.N+1,self.d,self.d),complex); prefix[:,0]=U
        for k in range(self.N):
            U=Uk[:,k]@U; prefix[:,k+1]=U
        if return_steps: return U,prefix,Uk,lam,V,e
        return U

    def objective(self,x,beta=60.,regularization=1e-6):
        U,Fwd,Uk,lam,V,e=self.propagators(x,True)
        z=np.einsum('cij,cij->c',self.W.conj(),U)
        fs=(self.d+np.abs(z)**2)/(self.d*(self.d+1)); err=1-fs
        w=np.exp(beta*err-logsumexp(beta*err))
        loss=(logsumexp(beta*err)-np.log(self.C))/beta
        adj=-w[:,None,None]*z[:,None,None]*self.W/(self.d*(self.d+1))
        gu=np.empty_like(Uk)
        for k in range(self.N-1,-1,-1):
            gu[:,k]=adj@Fwd[:,k].conj().swapaxes(-1,-2)
            adj=Uk[:,k].conj().swapaxes(-1,-2)@adj
        A=V.conj().swapaxes(-1,-2)@gu@V
        # Stable divided difference of exp(-i tau lambda).
        tau=2*np.pi*self.dt
        dl=lam[:,:,:,None]-lam[:,:,None,:]
        F=-1j*tau*np.exp(-.5j*tau*(lam[:,:,:,None]+lam[:,:,None,:]))*np.sinc(tau*dl/(2*np.pi))
        GH=V@(A*F.conj())@V.conj().swapaxes(-1,-2)
        C=GH[:,:,self.ij[0],self.ij[1]]+GH[:,:,self.ij[1],self.ij[0]].conj()
        gr=2*np.real(np.einsum('ctl,ctlj->tj',C.conj(),self.P))
        gi=-2*np.imag(np.einsum('ctl,ctlj,cl->tj',C.conj(),self.P,self.sign))
        g=self.amax*np.einsum('tk,tjq->kjq',self.basis,np.stack((gr,gi),axis=-1))
        # Small coefficient smoothness penalty, not a claimed hardware bandwidth.
        c=np.asarray(x).reshape(self.K,self.J,2); diff=np.diff(c,axis=0)
        loss+=regularization*np.mean(diff**2)
        gp=np.zeros_like(c); gp[:-1]-=2*diff/diff.size; gp[1:]+=2*diff/diff.size
        g+=regularization*gp
        self.calls+=1
        if self.calls%10==0:
            self.history.append({'evaluation':self.calls,'loss':float(loss),'worst':float(np.min(fs)),'mean':float(np.mean(fs))})
        return float(loss),g.ravel()

    def optimize(self,x0=None,maxiter=160,seed=0):
        x0=self.initial(seed) if x0 is None else x0
        start=time.time()
        opt=minimize(self.objective,x0,jac=True,method='L-BFGS-B',bounds=[(-1/np.sqrt(2),1/np.sqrt(2))]*len(x0),
                     options={'maxiter':maxiter,'ftol':1e-12,'gtol':2e-7,'maxls':25,'maxcor':15})
        fs=fidelity(self.propagators(opt.x),self.W)
        return opt,{'elapsed_s':time.time()-start,'iterations':opt.nit,'evaluations':opt.nfev,
                    'success':bool(opt.success),'message':str(opt.message),
                    'worst_train':float(np.min(fs)),'mean_train':float(np.mean(fs)),
                    'train_fidelities':fs.tolist(),'history':self.history}
