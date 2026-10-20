"""Drive-induced-shift tracking for electron-coupled nuclear drives.

The second-order (virtual electronic transition) matrix K_s is dominated by a
term linear in the nuclear quantum number m, i.e. a uniform shift of each
seven-line comb proportional to the mean squared drive <u^2>. This script
(i) predicts that shift from the plain multitone amplitude, (ii) moves every
carrier by the predicted amount, (iii) optimizes the envelopes with the full
second-order design model, (iv) updates the shift self-consistently, and
(v) validates with independent second-order propagation and the unreduced
16-level Hamiltonian.  No parameter other than the carrier frequencies and
the envelope coefficients is changed relative to the nominal designs.

Usage: python scripts/stark_tracking_study.py [--cases electron_single,...]
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,gc,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.stark_control import StarkProblem
from qudit_control.extended import ExtendedProblem,full_lab_midpoint,plain_envelope,integrate_extended
from qudit_control.virtual_electron import second_order_blocks,integrate_second_order
from qudit_control.validation import resource_metrics,envelope_function


def mean_u2(pr,x=None,env=None):
    """Time average of u(t)^2 for the real waveform on a 1-ns grid (MHz^2)."""
    if env is None:env=envelope_function(pr,x)
    t=np.linspace(0,pr.T,int(pr.T/.001)+1);a=env(t)
    u=2*np.real(np.sum(a*np.exp(-2j*np.pi*t[:,None]*pr.freq),axis=1))
    return float(np.trapezoid(u*u,t)/pr.T)


def predicted_offsets(pr,u2):
    """Uniform-in-time estimate of the drive-induced shift of every carrier (MHz).

    Level m of branch s shifts by (K_s)_mm <u^2>; the physical transition frequency
    |E_m-E_{m+1}| therefore shifts by sign(E_m-E_{m+1}) [(K_s)_mm-(K_s)_{m+1,m+1}] <u^2>.
    """
    K=second_order_blocks(pr);off=[]
    for s in pr.carrier_branches:
        dK=np.real(np.diag(K[s]));dE=np.diff(pr.E[s])
        off.extend(np.sign(-dE)*(-np.diff(dK))*u2)   # sign(E_m-E_{m+1}) * (K_mm-K_{m+1,m+1})
    return np.asarray(off)


def run_case(mechanism,active,budget=(400,300),rounds=2,validate_steps=(.0001,.00005,.000025),warm=False):
    source=f'nominal_{mechanism}_B1.6_{active}';pr0,x0,meta=load_pulse(ROOT,source)
    name=f'tracked_{mechanism}_B1.6_{active}';log=[]
    x=None
    prev=ROOT/'results'/f'{name}.json'
    if warm and prev.exists():
        # Continue the self-consistency iteration from the archived tracked pulse.
        old=json.loads(prev.read_text());log=old['log']
        off=np.asarray(log[-1]['updated_offsets_kHz'])/1e3;x=np.load(ROOT/'results/pulses'/f'{name}.npz')['x']
        print('WARM',name,'offsets kHz',np.round(1e3*off,2),flush=True)
    else:
        # Round 0 prediction from the plain (unshaped) multitone amplitude.
        u2=mean_u2(pr0,env=plain_envelope(pr0));off=predicted_offsets(pr0,u2)
        log.append(dict(stage='plain_prediction',mean_u2_MHz2=u2,offsets_kHz=(1e3*off).tolist()))
        print('PREDICT',name,'u2',u2,'offsets kHz',np.round(1e3*off,2),flush=True)
    r0=sum(1 for e in log if e['stage'].startswith('round'))
    for r in range(r0,r0+rounds):
        stages=[]
        for dt,iters in zip((.025,.0125),budget):
            p=StarkProblem(B=pr0.B,p=pr0.p,T=pr0.T,amax=pr0.amax,ncoef=pr0.ncoef,dt=dt,
                           active=pr0.active,mechanism=mechanism,carrier_offset=off)
            x_start=p.initial(seed=0) if x is None else x
            op,info=p.optimize(x0=x_start,maxiter=iters);x=op.x;info['dt_us']=dt;info['round']=r;stages.append(info)
            print('STAGE',name,'round',r,'dt',dt,'worst_train',info['worst_train'],'iters',info['iterations'],round(info['elapsed_s'],1),'s',flush=True)
            del p;gc.collect()
        pr=ExtendedProblem(B=pr0.B,p=pr0.p,T=pr0.T,amax=pr0.amax,ncoef=pr0.ncoef,dt=.0125,
                           active=pr0.active,mechanism=mechanism,carrier_offset=off)
        u2=mean_u2(pr,x);new_off=predicted_offsets(pr,u2)
        log.append(dict(stage=f'round{r}',mean_u2_MHz2=u2,offsets_kHz=(1e3*off).tolist(),updated_offsets_kHz=(1e3*new_off).tolist(),stages=stages))
        print('UPDATE',name,'round',r,'u2',u2,'new offsets kHz',np.round(1e3*new_off,2),flush=True)
        off=new_off
    # Final problem with the last offsets actually used for the optimized x.
    off_used=np.asarray(log[-1]['offsets_kHz'])/1e3
    pr=ExtendedProblem(B=pr0.B,p=pr0.p,T=pr0.T,amax=pr0.amax,ncoef=pr0.ncoef,dt=.0125,
                       active=pr0.active,mechanism=mechanism,carrier_offset=off_used)
    config=dict(meta['config']);config.update(dt=.0125,maxiter=sum(budget)*rounds,carrier_offset=off_used.tolist())
    record=dict(config=config,status='completed',design_model='second_order_electron_tracked',log=log,
                warm_start='plain multitone at shifted carriers',iterations=sum(s['iterations'] for e in log if 'stages' in e for s in e['stages']))
    np.savez_compressed(ROOT/'results/pulses'/f'{name}.npz',x=x)
    (ROOT/'results'/f'{name}.json').write_text(json.dumps(record,indent=2))
    physical=integrate_second_order(pr,x)
    print('SECOND_ORDER',name,physical['worst'],'adjacent shift kHz',1e3*physical['maximum_adjacent_mean_shift_MHz'],flush=True)
    linear=integrate_extended(pr,x,max_step=.004)[0]   # same pulse, linear projected model only
    vals=[]
    for dt in validate_steps:
        rr=full_lab_midpoint(pr,x,dt=dt);vals.append(rr);print('FULL',name,dt,rr['worst'],flush=True)
    v=dict(name=name,second_order_physical=physical,linear_projected=linear,full_electron=vals,
           resources=resource_metrics(pr,x),carrier_offset_kHz=(1e3*off_used).tolist())
    (ROOT/'results'/f'validation_{name}.json').write_text(json.dumps(v,indent=2))
    return v


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default='electron_single,electron_dual,magnetic_single,magnetic_dual')
    ap.add_argument('--budget',default='400,300');ap.add_argument('--rounds',type=int,default=2);ap.add_argument('--warm',action='store_true');args=ap.parse_args()
    budget=tuple(int(b) for b in args.budget.split(','))
    for case in args.cases.split(','):
        mech,act=case.split('_');run_case(mech,act,budget=budget,rounds=args.rounds,warm=args.warm)
