"""Diagnostic: how does the second-order model's residual scale with drive amplitude?

For a tracked pulse, the envelope is multiplied by a global factor lambda and the
propagators of the second-order model and of the full 16-level Hamiltonian are
compared directly (average fidelity between the two unitaries, worst branch).
A residual from third-order terms scales as lambda^6, from fourth-order as lambda^8.
Writes results/residual_order_check.json.  Not used in any figure or claim beyond the
SM remark on the origin of the sub-permille residual.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import full_operator,branch_targets
from qudit_control.model import hamiltonian,fidelity
from qudit_control.validation import envelope_function
from qudit_control.virtual_electron import second_order_blocks

def second_order_U(pr,env,lam):
    K=second_order_blocks(pr);ids=list(pr.active);d=pr.d
    delta=pr.E[ids,:,None]-pr.E[ids,None,:];D=pr.D[ids];K=K[ids]
    def rhs(t,y):
        u=lam*2*np.real(np.sum(env(t)*np.exp(-2j*np.pi*pr.freq*t)))
        H=(u*D+u*u*K)*np.exp(2j*np.pi*delta*t)
        return (-2j*np.pi*H@y.reshape(len(ids),d,d)).ravel()
    y=np.tile(np.eye(d,dtype=complex),(len(ids),1,1)).ravel()
    sol=solve_ivp(rhs,(0,pr.T),y,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.002)
    return sol.y[:,-1].reshape(len(ids),d,d)

def full_U(pr,env,lam,dt=.00005,chunk=1000):
    H,_,_=hamiltonian(pr.B,pr.p);O=full_operator(pr.p,pr.mechanism)
    H=H-np.trace(H)/len(H)*np.eye(len(H));O=O-np.trace(O)/len(O)*np.eye(len(O));H=H.real;O=O.real
    N=int(np.ceil(pr.T/dt));dt=pr.T/N;U=np.eye(len(H),dtype=complex)
    for lo in range(0,N,chunk):
        t=(np.arange(lo,min(N,lo+chunk))+.5)*dt;amps=np.array([env(float(tt)) for tt in t])
        w=lam*2*np.real(np.sum(amps*np.exp(-2j*np.pi*t[:,None]*pr.freq),axis=1))
        lam_,V=np.linalg.eigh(H[None]+w[:,None,None]*O[None]);uk=(V*np.exp(-2j*np.pi*dt*lam_)[:,None,:])@V.conj().swapaxes(-1,-2)
        for u in uk:U=u@U
    out=[]
    for s in pr.active:
        v=pr.V[s];block=v.conj().T@U@v;block=np.exp(2j*np.pi*pr.E[s]*pr.T)[:,None]*block;out.append(block)
    return np.array(out)

if __name__=='__main__':
    name=sys.argv[1] if len(sys.argv)>1 else 'tracked_magnetic_B1.6_dual'
    pr,x,_=load_pulse(ROOT,name);env=envelope_function(pr,x);W=branch_targets(pr)[list(pr.active)]
    rec=dict(pulse=name,points=[])
    for lam in (0.5,0.75,1.0):
        U2=second_order_U(pr,env,lam);Uf=full_U(pr,env,lam)
        # normalize the full return block to compare unitaries (leakage is negligible)
        f_between=[float(fidelity(Uf[i],U2[i])) for i in range(len(U2))]
        rec['points'].append(dict(lam=lam,model_vs_full_infidelity=[1-f for f in f_between],
                                  second_order_gate_fidelity=fidelity(U2,W).tolist(),full_gate_fidelity=fidelity(Uf,W).tolist()))
        print(name,'lambda',lam,'model-vs-full infidelity',[f'{1-f:.2e}' for f in f_between],flush=True)
    (ROOT/'results'/f'residual_order_{name}.json').write_text(json.dumps(rec,indent=2))
