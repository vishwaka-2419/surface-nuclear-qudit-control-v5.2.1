"""Calibration tolerance of the shift-tracked electron-mediated pulses.

The tracked pulses are frozen (carriers, offsets, envelopes) and evaluated over the
residual-error domain of the main text with the second-order model (L3): common
gain +-3%, field +-0.1 mT, kappa and A_z +-2 kHz, and all sixteen simultaneous
corners.  The worst corner of the seven-tone pulse is re-propagated in the full
16-level model (L4).  Writes results/tracked_robustness_<name>.json.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,argparse
from itertools import product
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import ExtendedProblem,projected,full_operator,branch_targets
from qudit_control.model import hamiltonian,fidelity
from qudit_control.validation import envelope_function
from qudit_control.virtual_electron import second_order_blocks


def second_order_scenario(pr,x,bb,pp,gain,rtol=2e-9,max_step=.002):
    """L3 propagation of the frozen pulse with a perturbed Hamiltonian (bb, pp) and gain.

    Carriers, frame and target are those of the nominal calibration (pr); the perturbed
    energies enter as a diagonal drift, the perturbed matrix elements and K as the couplings.
    """
    env=envelope_function(pr,x);ids=list(pr.active);d=pr.d
    EE,DD,_,_=projected(bb,pp,pr.mechanism)
    tmp=ExtendedProblem(B=bb,p=pp,T=pr.T,amax=pr.amax,ncoef=pr.ncoef,dt=.05,active=pr.active,mechanism=pr.mechanism,carrier_offset=pr.carrier_offset)
    KK=second_order_blocks(tmp)[ids];DD=DD[ids];drift=(EE-pr.E)[ids]
    delta=pr.E[ids,:,None]-pr.E[ids,None,:];idx=np.arange(d)
    def rhs(t,y):
        u=gain*2*np.real(np.sum(env(t)*np.exp(-2j*np.pi*pr.freq*t)))
        H=(u*DD+u*u*KK)*np.exp(2j*np.pi*delta*t);H[:,idx,idx]+=drift
        return (-2j*np.pi*H@y.reshape(len(ids),d,d)).ravel()
    y=np.tile(np.eye(d,dtype=complex),(len(ids),1,1)).ravel()
    sol=solve_ivp(rhs,(0,pr.T),y,method='DOP853',rtol=rtol,atol=rtol/100,max_step=max_step)
    if not sol.success:raise RuntimeError(sol.message)
    U=sol.y[:,-1].reshape(len(ids),d,d);f=fidelity(U,branch_targets(pr)[ids])
    return f.tolist(),float(f.min())


def full_scenario(pr,x,bb,pp,gain,dt=.00005,chunk=1000):
    """L4 propagation of the frozen pulse with a perturbed static Hamiltonian and gain."""
    env=envelope_function(pr,x);H,_,_=hamiltonian(bb,pp);O=full_operator(pp,pr.mechanism)
    H=H-np.trace(H)/len(H)*np.eye(len(H));O=O-np.trace(O)/len(O)*np.eye(len(O));H=H.real;O=O.real
    N=int(np.ceil(pr.T/dt));dt=pr.T/N;U=np.eye(len(H),dtype=complex)
    for lo in range(0,N,chunk):
        t=(np.arange(lo,min(N,lo+chunk))+.5)*dt;amps=np.array([env(float(tt)) for tt in t])
        w=gain*2*np.real(np.sum(amps*np.exp(-2j*np.pi*t[:,None]*pr.freq),axis=1))
        lam,V=np.linalg.eigh(H[None]+w[:,None,None]*O[None]);uk=(V*np.exp(-2j*np.pi*dt*lam)[:,None,:])@V.conj().swapaxes(-1,-2)
        for u in uk:U=u@U
    W=branch_targets(pr);fs=[]
    for s in pr.active:
        v=pr.V[s];block=np.exp(2j*np.pi*pr.E[s]*pr.T)[:,None]*(v.conj().T@U@v)
        norm=float(np.trace(block.conj().T@block).real);fs.append(float((norm+abs(np.trace(W[s].conj().T@block))**2)/(pr.d*(pr.d+1))))
    return fs,min(fs)


def run(name,full_worst=True):
    pr,x,_=load_pulse(ROOT,name);B=pr.B;p=pr.p
    rec=dict(name=name,gain=[],axes=[],corners=[])
    for g in (0.97,1.0,1.03):
        fs,f=second_order_scenario(pr,x,B,p,g);rec['gain'].append(dict(gain=g,fidelities=fs,worst=f));print('GAIN',name,g,f'{f:.5f}',flush=True)
    for lab,(bb,pp) in {'B-':(B-1e-4,p),'B+':(B+1e-4,p),'k-':(B,replace(p,kappa=p.kappa-.002)),'k+':(B,replace(p,kappa=p.kappa+.002)),'A-':(B,replace(p,A_z=p.A_z-.002)),'A+':(B,replace(p,A_z=p.A_z+.002))}.items():
        fs,f=second_order_scenario(pr,x,bb,pp,1.);rec['axes'].append(dict(axis=lab,fidelities=fs,worst=f));print('AXIS',name,lab,f'{f:.5f}',flush=True)
    for sb,sk,sa,sg in product((-1,1),repeat=4):
        bb=B+sb*1e-4;pp=replace(p,kappa=p.kappa+sk*.002,A_z=p.A_z+sa*.002);g=1+sg*.03
        fs,f=second_order_scenario(pr,x,bb,pp,g);rec['corners'].append(dict(label=f'B{sb:+d}_k{sk:+d}_A{sa:+d}_g{sg:+d}',fidelities=fs,worst=f))
    worst=min(rec['corners'],key=lambda z:z['worst']);rec['worst_corner']=worst;print('CORNERS',name,'min',f'{worst["worst"]:.5f}',worst['label'],flush=True)
    if full_worst:
        sb,sk,sa,sg=[int(z[1:]) for z in worst['label'].split('_')]
        fs,f=full_scenario(pr,x,B+sb*1e-4,replace(p,kappa=p.kappa+sk*.002,A_z=p.A_z+sa*.002),1+sg*.03)
        rec['worst_corner_full']=dict(fidelities=fs,worst=f,dt_us=.00005);print('FULL_CORNER',name,f'{f:.5f}',flush=True)
    (ROOT/'results'/f'tracked_robustness_{name}.json').write_text(json.dumps(rec,indent=2))
    return rec


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pulses',default='tracked_electron_B1.6_single,tracked_electron_B1.6_dual');ap.add_argument('--no-full',action='store_true');args=ap.parse_args()
    for n in args.pulses.split(','):run(n,full_worst=(not args.no_full) and n.endswith('single'))
