"""Start-time (carrier-phase) sensitivity and pulse-sequence tests.

A pulse replayed at laboratory time t0 with continuously running carriers has the
same resonant couplings but rephased off-resonant couplings, exp(2 pi i (delta-f) t0).
This script (i) scans the fidelity of archived pulses versus t0 with the adaptive
real-waveform propagation, (ii) checks selected points with the full 16-level
propagation, and (iii) propagates two-pulse sequences with idle periods against
the target G^2 (interaction picture: idles contribute the identity).

Usage: python scripts/start_time_study.py [--pulses a,b,...] [--n 24] [--tmax 10]
Writes results/start_time_<pulse>.json and results/start_time_summary.csv.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,argparse,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import integrate_extended,full_lab_midpoint,branch_targets
from qudit_control.model import fidelity
from qudit_control.virtual_electron import integrate_second_order


def scan(name,t0s,full_points=(),full_dt=.00005):
    pr,x,meta=load_pulse(ROOT,name);base=(pr.B,pr.p,1.)
    rec=dict(pulse=name,T_us=pr.T,scan=[],full=[],sequences=[])
    second=pr.mechanism!='nuclear'
    for t0 in t0s:
        if second:
            r=integrate_second_order(pr,x,t0=float(t0));f=r['worst'];fs=r['fidelities']
        else:
            r,_=integrate_extended(pr,x,scenario=base+(float(t0),),max_step=.004);f=r['worst'];fs=r['fidelities']
        rec['scan'].append(dict(t0_us=float(t0),worst=float(f),fidelities=fs))
        print('SCAN',name,f'{t0:.3f}',f'{f:.5f}',flush=True)
    for t0 in full_points:
        r=full_lab_midpoint(pr,x,dt=full_dt,t0=float(t0));rec['full'].append(dict(t0_us=float(t0),worst=r['worst'],fidelities=r['fidelities'],dt_us=full_dt))
        print('FULL',name,f'{t0:.3f}',f'{r["worst"]:.5f}',flush=True)
    return pr,x,rec


def sequence(pr,x,starts,target_power=None,full_dt=None):
    """Propagate the same pulse at the listed start times against its target product.

    The optional full-model check composes complete 2d-dimensional propagators
    in the static-energy interaction picture. Idles are identities in this
    picture; interbranch leakage and return paths are retained.
    """
    second=pr.mechanism!='nuclear';ids=list(pr.active);d=pr.d
    Utot=np.tile(np.eye(d,dtype=complex),(len(ids),1,1));Wtot=np.tile(np.eye(d,dtype=complex),(len(ids),1,1))
    for t0 in starts:
        if second:
            # second-order model propagator is not returned by integrate_second_order; use the full model instead
            raise NotImplementedError('sequence test for electron-coupled pulses uses the full model (full_dt)')
        r,U=integrate_extended(pr,x,scenario=(pr.B,pr.p,1.,float(t0)),max_step=.004)
        Utot=U@Utot;Wtot=branch_targets(pr,float(t0))[ids]@Wtot
    f=fidelity(Utot,Wtot)
    out=dict(starts_us=[float(t) for t in starts],fidelities=f.tolist(),worst=float(f.min()))
    # population transfer |m=-I> -> |m=+I> for two pi/2 rotations (collective rotation of 2 x pi/2 = pi about y)
    if pr.kind=='rotation' and len(starts)==2:
        psi=np.zeros(d,complex);psi[-1]=1.   # m=-7/2 is the last basis state
        pops=[float(abs(U[0,:]@psi)**2) for U in Utot]   # probability of m=+7/2 (first basis state)
        ideal=float(abs((Wtot[0][0,:]@psi))**2)
        out.update(transfer_probability=pops,ideal_transfer_probability=ideal)
    if full_dt:
        Uf=np.eye(2*d,dtype=complex)
        for t0 in starts:
            _,joint=full_lab_midpoint(pr,x,dt=full_dt,t0=float(t0),return_full=True)
            Uf=joint@Uf
        ff=[];leak=[]
        for k,s in enumerate(ids):
            block=Uf[s*d:(s+1)*d,s*d:(s+1)*d]
            norm=float(np.trace(block.conj().T@block).real)
            ff.append(float((norm+abs(np.trace(Wtot[k].conj().T@block))**2)/(d*(d+1))))
            leak.append(float(1-norm/d))
        out.update(full_fidelities=ff,full_worst=min(ff),full_leakage=leak,full_dt_us=full_dt,
                   full_composition="complete static-energy interaction-picture propagators")
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pulses',default='nominal_N_B1.35_dual,nominal_N_B1.6_dual,nominal_nuclear_B1.6_single,nominal_N_B1.03_dual,nominal_N_B1.714_dual,tracked_electron_B1.6_single,robust_N_B1.6_dual_refined')
    ap.add_argument('--n',type=int,default=24);ap.add_argument('--tmax',type=float,default=10.);ap.add_argument('--full',type=int,default=2);args=ap.parse_args()
    rng=np.random.default_rng(20261019);rows=[]
    for name in args.pulses.split(','):
        t0s=np.r_[0.,np.sort(rng.uniform(0,args.tmax,args.n-1))]
        pr,x,rec=scan(name,t0s,full_points=[])
        # full-model check at the worst scanned start time and at the auditor's example point
        worst=min(rec['scan'],key=lambda z:z['worst'])['t0_us']
        fulls=[worst]+([2.71] if pr.mechanism=='nuclear' else [])
        for t0 in fulls[:args.full]:
            r=full_lab_midpoint(pr,x,dt=.00005,t0=float(t0));rec['full'].append(dict(t0_us=float(t0),worst=r['worst'],fidelities=r['fidelities'],dt_us=.00005))
            print('FULL',name,f'{t0:.3f}',f'{r["worst"]:.5f}',flush=True)
        if pr.mechanism=='nuclear' and pr.kind=='rotation':
            for idle in (0.,1.37):
                seq=sequence(pr,x,[0.,pr.T+idle],full_dt=.00005 if idle==0. else None);seq['idle_us']=idle;rec['sequences'].append(seq)
                print('SEQ',name,'idle',idle,'worst',f'{seq["worst"]:.5f}',seq.get('full_worst'),'transfer',seq.get('transfer_probability'),flush=True)
        (ROOT/'results'/f'start_time_{name}.json').write_text(json.dumps(rec,indent=2))
        ws=[z['worst'] for z in rec['scan']]
        rows.append(dict(pulse=name,T_us=pr.T,n_start_times=len(ws),min_fidelity=min(ws),median_fidelity=float(np.median(ws)),max_fidelity=max(ws),
                         fidelity_t0_0=rec['scan'][0]['worst'],full_at_worst=rec['full'][0]['worst'] if rec['full'] else None,
                         two_pulse_zero_idle=rec['sequences'][0]['worst'] if rec['sequences'] else None,two_pulse_full=rec['sequences'][0].get('full_worst') if rec['sequences'] else None))
        pd.DataFrame(rows).to_csv(ROOT/'results/start_time_summary.csv',index=False)
    print('done')
