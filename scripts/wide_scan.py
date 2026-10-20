"""Wide-range start-time scan of archived pulses (nominal and window-trained).

For each pulse: design-mesh fidelity (L1) at ``--nscan`` start times drawn
uniformly from [0, tmax] (same seed as composable_study.py), adaptive
real-waveform propagation (L2) at the worst scanned point and at the two audit
start times, and the full 16-level model (L4) at the worst scanned point.

Usage: python scripts/wide_scan.py [--pulses a,b,...] [--tmax 1e5] [--nscan 400]
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from qudit_control.io import load_pulse
from qudit_control.extended import integrate_extended,full_lab_midpoint
from composable_study import design_mesh_scan,scan_starts,AUDIT_POINTS,SCAN_SEED
OUT=ROOT/'results'

def run(name,tmax,nscan):
    pr,x,meta=load_pulse(ROOT,name);cfg=meta['config']
    if pr.mechanism!='nuclear':raise NotImplementedError('wide scan is implemented for the direct nuclear drive')
    scan=design_mesh_scan(pr.B,pr.p,pr.T,pr.amax,pr.ncoef,pr.dt,pr.active,x,scan_starts(nscan,tmax))
    worst=min(scan,key=lambda z:z['worst'])
    def l2(t0):
        r,_=integrate_extended(pr,x,scenario=(pr.B,pr.p,1.,float(t0)),max_step=.004);return dict(t0_us=float(t0),worst=r['worst'],fidelities=r['fidelities'])
    rec=dict(pulse=name,T_us=pr.T,train_range_us=cfg.get('train_range_us',None),scan_range_us=tmax,n=nscan,seed=SCAN_SEED,
             scan_min=worst['worst'],t0_min_us=worst['t0_us'],scan_median=float(np.median([z['worst'] for z in scan])),points=scan,
             worst_point_L2=l2(worst['t0_us']),audit_points=[l2(t0) for t0 in AUDIT_POINTS])
    r=full_lab_midpoint(pr,x,dt=.00005,t0=worst['t0_us']);rec['worst_point_full']=dict(t0_us=worst['t0_us'],worst=r['worst'],fidelities=r['fidelities'],dt_us=.00005)
    (OUT/f'wide_scan_{name}.json').write_text(json.dumps(rec,indent=2))
    print('WIDE',name,'scan min',f'{worst["worst"]:.5f}','@',f'{worst["t0_us"]:.1f}','L2',f'{rec["worst_point_L2"]["worst"]:.5f}','L4',f'{r["worst"]:.5f}',
          'audit',[f'{a["worst"]:.5f}' for a in rec['audit_points']],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pulses',default='nominal_N_B1.6_dual,nominal_nuclear_B1.6_single,nominal_N_B1.35_dual,nominal_N_B1.03_dual,nominal_N_B1.714_dual,window_N_B1.6_dual,window_N_B1.6_single,window_N_B1.35_dual,window_N_B1.03_dual,window_N_B1.714_dual')
    ap.add_argument('--tmax',type=float,default=1e5);ap.add_argument('--nscan',type=int,default=400);args=ap.parse_args()
    for n in args.pulses.split(','):run(n,args.tmax,args.nscan)
