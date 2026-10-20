"""Carrier control including second-order virtual electronic transitions.

All signed carrier-pair frequencies are retained, including sum-frequency
quadratic transitions. This is a perturbative design model; validation still
uses the unreduced electron-nuclear Hamiltonian.
"""
import numpy as np
from scipy.special import logsumexp
from .extended import ExtendedProblem
from .virtual_electron import second_order_blocks

class StarkProblem(ExtendedProblem):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if len(self.ensemble)!=1:raise ValueError('This implementation currently designs nominal controls only')
        self.qij=np.triu_indices(self.d,0)
        signed=np.r_[self.freq,-self.freq]
        pairs=signed[:,None]+signed[None,:]
        blocks=second_order_blocks(self);Q=[]
        for s in self.active:
            delta=self.E[s,self.qij[0]]-self.E[s,self.qij[1]]
            det=delta[:,None,None]-pairs[None,:,:]
            q=np.exp(2j*np.pi*self.t[:,None,None,None]*det[None,:,:,:])*np.sinc(self.dt*det)[None,:,:,:]
            q*=blocks[s,self.qij[0],self.qij[1]][None,:,None,None]
            Q.append(q)
        self.Q=np.asarray(Q)

    def hamiltonians(self,x):
        H=super().hamiltonians(x);a=self.amplitudes(x);b=np.concatenate((a,a.conj()),axis=1)
        h=np.einsum('ctqkl,tk,tl->ctq',self.Q,b,b,optimize=True)
        H[:,:,self.qij[0],self.qij[1]]+=h
        mask=self.qij[0]!=self.qij[1]
        H[:,:,self.qij[1][mask],self.qij[0][mask]]+=h[:,:,mask].conj()
        idx=np.arange(self.d);H[:,:,idx,idx]=H[:,:,idx,idx].real
        return H

    def objective(self,x,beta=None,regularization=1e-7):
        beta=self.beta if beta is None else beta
        U,Fwd,Uk,lam,V,e=self.propagators(x,True)
        z=np.einsum('cij,cij->c',self.W.conj(),U)
        fs=(self.d+abs(z)**2)/(self.d*(self.d+1));err=1-fs
        w=np.exp(beta*err-logsumexp(beta*err))
        loss=(logsumexp(beta*err)-np.log(self.C))/beta
        adj=-w[:,None,None]*z[:,None,None]*self.W/(self.d*(self.d+1))
        gu=np.empty_like(Uk)
        for k in range(self.N-1,-1,-1):
            gu[:,k]=adj@Fwd[:,k].conj().swapaxes(-1,-2)
            adj=Uk[:,k].conj().swapaxes(-1,-2)@adj
        A=V.conj().swapaxes(-1,-2)@gu@V;tau=2*np.pi*self.dt
        dl=lam[:,:,:,None]-lam[:,:,None,:]
        F=-1j*tau*np.exp(-.5j*tau*(lam[:,:,:,None]+lam[:,:,None,:]))*np.sinc(tau*dl/(2*np.pi))
        GH=V@(A*F.conj())@V.conj().swapaxes(-1,-2)
        C=GH[:,:,self.ij[0],self.ij[1]]+GH[:,:,self.ij[1],self.ij[0]].conj()
        gr=2*np.real(np.einsum('ctl,ctlj->tj',C.conj(),self.P))
        gi=-2*np.imag(np.einsum('ctl,ctlj,cl->tj',C.conj(),self.P,self.sign))
        CQ=GH[:,:,self.qij[0],self.qij[1]].copy();mask=self.qij[0]!=self.qij[1]
        CQ[:,:,mask]+=GH[:,:,self.qij[1][mask],self.qij[0][mask]].conj()
        a=self.amplitudes(x);b=np.concatenate((a,a.conj()),axis=1)
        AQ=2*np.einsum('ctq,ctqkl,tl->tk',CQ.conj(),self.Q,b,optimize=True)
        gr+=2*np.real(AQ[:,:self.J]+AQ[:,self.J:])
        gi+=-2*np.imag(AQ[:,:self.J]-AQ[:,self.J:])
        g=self.amax*np.einsum('tk,tjq->kjq',self.basis,np.stack((gr,gi),axis=-1))
        c=np.asarray(x).reshape(self.K,self.J,2);diff=np.diff(c,axis=0)
        loss+=regularization*np.mean(diff**2)
        gp=np.zeros_like(c);gp[:-1]-=2*diff/diff.size;gp[1:]+=2*diff/diff.size
        g+=regularization*gp
        self.calls+=1
        if self.calls%10==0:self.history.append(dict(evaluation=self.calls,loss=float(loss),worst=float(fs.min()),mean=float(fs.mean())))
        return float(loss),g.ravel()
