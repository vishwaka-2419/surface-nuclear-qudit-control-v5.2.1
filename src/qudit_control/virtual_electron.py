"""Second-order elimination diagnostic for off-resonant electronic driving.

This diagnostic is not used to certify optimality or to replace full-system
propagation. It isolates virtual electronic shifts missed by linear projection.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .model import hamiltonian, common_frame_targets, fidelity
from .extended import full_operator
from .validation import envelope_function


def second_order_blocks(pr):
    H,_,_=hamiltonian(pr.B,pr.p);O=full_operator(pr.p,pr.mechanism)
    E=np.array([np.diag(v.conj().T@H@v).real for v in pr.V])
    K=[]
    for s in (0,1):
        q=1-s;cross=pr.V[s].conj().T@O@pr.V[q]
        den=E[s,:,None]-E[q,None,:]
        k=.5*((cross/den)@cross.conj().T+cross@(cross/den).conj().T)
        k-=np.trace(k)/pr.d*np.eye(pr.d)
        K.append(k)
    return np.array(K)


def integrate_second_order(pr,x,rtol=2e-9,max_step=.002,gain=1.,t0=0.,env=None):
    K=second_order_blocks(pr);env=envelope_function(pr,x) if env is None else env;ids=list(pr.active);d=pr.d
    delta=pr.E[ids,:,None]-pr.E[ids,None,:];D=pr.D[ids];K=K[ids]
    def rhs(t,y):
        tl=t+t0
        u=gain*2*np.real(np.sum(env(t)*np.exp(-2j*np.pi*pr.freq*tl)))
        H=(u*D+u*u*K)*np.exp(2j*np.pi*delta*tl)
        return (-2j*np.pi*H@y.reshape(len(ids),d,d)).ravel()
    y=np.tile(np.eye(d,dtype=complex),(len(ids),1,1)).ravel()
    sol=solve_ivp(rhs,(0,pr.T),y,method='DOP853',rtol=rtol,atol=rtol/100,max_step=max_step)
    if not sol.success:raise RuntimeError(sol.message)
    U=sol.y[:,-1].reshape(len(ids),d,d)
    from .extended import branch_targets
    W=branch_targets(pr,t0)[ids]
    f=fidelity(U,W)
    ts=np.linspace(0,pr.T,int(pr.T/.001)+1);a=env(ts)
    u=2*np.real(np.sum(a*np.exp(-2j*np.pi*ts[:,None]*pr.freq),axis=1))
    rms2=float(np.trapezoid(u*u,ts)/pr.T)
    shifts=np.real(np.diagonal(K,axis1=1,axis2=2))*rms2
    return dict(fidelities=f.tolist(),worst=float(f.min()),rtol=rtol,max_step_us=max_step,nfev=sol.nfev,gain=gain,start_time_us=t0,
                mean_u_squared_MHz2=rms2,mean_diagonal_shift_MHz=shifts.tolist(),
                maximum_adjacent_mean_shift_MHz=float(np.max(np.abs(np.diff(shifts,axis=1)))))
