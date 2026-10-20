"""Start-time-robust (composable) gate design and validation.

Trains the same bounded multitone pulses as the nominal study, but on an ensemble
of pulse start times t0 with continuously running carriers, so that the pulse
cannot rely on a particular phase of its off-resonant couplings.

Two training ranges are used.  A narrow window (``--tmax 10``, results named
``window_<case>``) samples start times from 0-10 us; pulses trained this way
generalize inside the window but not beyond it, because couplings whose detunings
differ by a few kilohertz (the two combs have slightly different spacings) keep
an almost fixed relative phase over 10 us and drift apart over hundreds of
microseconds (SI Supplementary Note 9).  The wide range (default ``--tmax 1e5``, results named
``composable_<case>``) samples start times from 0-100 ms, spanning many cycles
of the identified kilohertz-scale relative detunings. The phases remain linked
by their common start time; independent or exhaustive phase coverage is not assumed.

Validation of each pulse (never using the training start times): adaptive
real-waveform propagation (L2) at held-out start times drawn from the same range
and at the two start times singled out by an external audit; a dense scan over
the range on the design mesh (L1) with the worst scanned point re-propagated at
L2 and in the full 16-level model (L4); the full model at t0 = 0 and at the
worst held-out point (L4); and two- and three-pulse sequences with short and
long idles (L2; the zero-idle two-pulse test also at L4).

Usage: python scripts/composable_study.py [--cases N_B1.6_dual,...] [--ntrain 16]
       [--tmax 1e5] [--maxiter 600] [--ntest 48] [--nscan 400] [--prefix composable] [--warm NAME]
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.extended import ExtendedProblem,nominal_parameters,integrate_extended,full_lab_midpoint
from qudit_control.validation import resource_metrics
from qudit_control.model import fidelity
sys.path.insert(0,str(ROOT/'scripts'));from start_time_study import sequence
OUT=ROOT/'results';(OUT/'pulses').mkdir(exist_ok=True)
TRAIN_SEED=20261020;TEST_SEED=20261021;SCAN_SEED=20261024
AUDIT_POINTS=(43.417,769.615835)   # start times at which an external audit evaluated the window-trained 1.6 T pulse


def train_starts(n,tmax):
    rng=np.random.default_rng(TRAIN_SEED);return np.r_[0.,np.sort(rng.uniform(0,tmax,n-1))]

def test_starts(n,tmax):
    rng=np.random.default_rng(TEST_SEED);return np.sort(rng.uniform(0,tmax,n))

def scan_starts(n,tmax):
    rng=np.random.default_rng(SCAN_SEED);return np.sort(rng.uniform(0,tmax,n))


def design_mesh_scan(B,p,T,amax,ncoef,dt,active,x,t0s,chunk=20):
    """Worst-branch fidelity on the optimizer mesh (L1) at many start times."""
    out=[]
    for k in range(0,len(t0s),chunk):
        part=t0s[k:k+chunk]
        pr=ExtendedProblem(B=B,p=p,T=T,amax=amax,ncoef=ncoef,dt=dt,active=active,
                           ensemble=[(B,p,1.,f's{i}',float(t0)) for i,t0 in enumerate(part)])
        f=fidelity(pr.propagators(x),pr.W).reshape(len(part),len(active))
        out+=[dict(t0_us=float(t0),worst=float(ff.min()),fidelities=ff.tolist()) for t0,ff in zip(part,f)]
    return out


def run(case,ntrain,maxiter,tmax,ntest,nscan,prefix,T=12.,ncoef=14,dt=.0125,amax=.05,warm_name=None):
    kind,B,act=case.split('_');B=float(B[1:]);active=(0,1) if act=='dual' else (0,)
    p=nominal_parameters();name=f'{prefix}_{case}';starts=train_starts(ntrain,tmax)
    ensemble=[(B,p,1.,f't{k}',float(t0)) for k,t0 in enumerate(starts)]
    pr=ExtendedProblem(B=B,p=p,T=T,amax=amax,ncoef=ncoef,dt=dt,active=active,ensemble=ensemble)
    cfg=dict(B=B,T=T,amax=amax,ncoef=ncoef,dt=dt,active=list(active),parameters=asdict(p),seed=0,maxiter=maxiter,
             train_start_times_us=starts.tolist(),train_range_us=tmax,train_seed=TRAIN_SEED)
    # warm start: the window-trained pulse of the same case if it exists, else the nominal pulse
    nominal={'N_B1.03_dual':'nominal_N_B1.03_dual','N_B1.35_dual':'nominal_N_B1.35_dual','N_B1.6_dual':'nominal_N_B1.6_dual',
             'N_B1.714_dual':'nominal_N_B1.714_dual','N_B1.6_single':'nominal_nuclear_B1.6_single'}.get(case)
    warm=None
    for cand in ([warm_name] if warm_name else [])+([f'window_{case}'] if prefix!='window' else [])+([nominal] if nominal else []):
        if cand and (OUT/'pulses'/f'{cand}.npz').exists():warm=cand;break
    x0=np.load(OUT/'pulses'/f'{warm}.npz')['x'] if warm else None
    t=time.time();opt,info=pr.optimize(x0=x0,maxiter=maxiter);x=opt.x
    info.update(config=cfg,status='completed',labels=pr.labels,warm_start=warm)
    np.savez_compressed(OUT/'pulses'/f'{name}.npz',x=x);(OUT/f'{name}.json').write_text(json.dumps(info,indent=2))
    print('TRAINED',name,'worst_train',f'{info["worst_train"]:.6f}','iters',info['iterations'],round(time.time()-t),'s',flush=True)
    # ---- validation (nominal single-scenario problem; start times enter through the scenario)
    prv=ExtendedProblem(B=B,p=p,T=T,amax=amax,ncoef=ncoef,dt=dt,active=active)
    def l2(t0):
        r,_=integrate_extended(prv,x,scenario=(B,p,1.,float(t0)),max_step=.004);return dict(t0_us=float(t0),worst=r['worst'],fidelities=r['fidelities'])
    held=[l2(t0) for t0 in test_starts(ntest,tmax)]
    audit=[l2(t0) for t0 in AUDIT_POINTS]
    worst=min(held,key=lambda z:z['worst'])
    scan=design_mesh_scan(B,p,T,amax,ncoef,dt,active,x,scan_starts(nscan,tmax))
    scan_worst=min(scan,key=lambda z:z['worst'])
    scan_worst_l2=l2(scan_worst['t0_us'])
    full=[full_lab_midpoint(prv,x,dt=.00005,t0=0.),full_lab_midpoint(prv,x,dt=.00005,t0=worst['t0_us']),
          full_lab_midpoint(prv,x,dt=.00005,t0=scan_worst['t0_us'])]
    seqs=[]
    for idle in (0.,1.37,5.0,250.37,1000.37):
        s=sequence(prv,x,[0.,T+idle],full_dt=.00005 if idle==0. else None);s['idle_us']=idle;seqs.append(s)
    # The third-pulse offsets are cumulative; consecutive gaps are 0.73,2.38 and 87.3,1926.4 us.
    s3=sequence(prv,x,[0.,T+0.73,2*T+3.11]);s3['idle_us']='0.73,2.38';seqs.append(s3)
    s3w=sequence(prv,x,[0.,T+87.3,2*T+2013.7]);s3w['idle_us']='87.3,1926.4';seqs.append(s3w)
    v=dict(name=name,train_range_us=tmax,train_start_times_us=starts.tolist(),
           held_out=held,held_out_min=worst['worst'],held_out_median=float(np.median([z['worst'] for z in held])),
           audit_points=audit,design_mesh_scan=dict(n=nscan,seed=SCAN_SEED,min=scan_worst['worst'],t0_min_us=scan_worst['t0_us'],
                                                   median=float(np.median([z['worst'] for z in scan])),points=scan,
                                                   worst_point_L2=scan_worst_l2),
           full_electron=full,sequences=seqs,resources=resource_metrics(prv,x))
    (OUT/f'validation_{name}.json').write_text(json.dumps(v,indent=2))
    print('VALIDATED',name,'held-out min',f'{worst["worst"]:.5f}','median',f'{v["held_out_median"]:.5f}',
          'scan min',f'{scan_worst["worst"]:.5f}','@',f'{scan_worst["t0_us"]:.1f}','full@heldworst',f'{full[1]["worst"]:.5f}',
          'full@scanworst',f'{full[2]["worst"]:.5f}','audit',[f'{a["worst"]:.5f}' for a in audit],
          'seq',[f'{s["worst"]:.4f}' for s in seqs],flush=True)
    return v


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default='N_B1.6_dual,N_B1.6_single,N_B1.35_dual,N_B1.03_dual,N_B1.714_dual')
    ap.add_argument('--ntrain',type=int,default=16);ap.add_argument('--maxiter',type=int,default=600);ap.add_argument('--T',type=float,default=12.)
    ap.add_argument('--tmax',type=float,default=1e5);ap.add_argument('--ntest',type=int,default=48);ap.add_argument('--nscan',type=int,default=400)
    ap.add_argument('--prefix',default='composable');ap.add_argument('--warm',default=None,help='name of an archived pulse to start from (default: window_<case>, else the nominal pulse)')
    args=ap.parse_args()
    for case in args.cases.split(','):run(case,args.ntrain,args.maxiter,args.tmax,args.ntest,args.nscan,args.prefix,T=args.T,warm_name=args.warm)
