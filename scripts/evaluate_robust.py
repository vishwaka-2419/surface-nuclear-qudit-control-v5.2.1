"""Corner, held-out, and real-waveform checks, without changing any pulse."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import ExtendedProblem,calibration_ensemble,integrate_extended,plain_envelope,full_lab_midpoint
from qudit_control.model import fidelity
from qudit_control.validation import resource_metrics


def evaluate(pr,x,points,dt=.00625):
 out=[]
 for lo in range(0,len(points),4):
  ens=points[lo:lo+4]
  p=ExtendedProblem(B=pr.B,p=pr.p,T=pr.T,amax=pr.amax,ncoef=pr.ncoef,dt=dt,
        active=pr.active,mechanism=pr.mechanism,ensemble=ens,frame=pr.frame)
  f=fidelity(p.propagators(x),p.W).reshape(len(ens),len(pr.active))
  out.extend(f.min(axis=1).tolist())
 return np.array(out)


def run(name):
 pr,x,meta=load_pulse(ROOT,name)
 corners=calibration_ensemble(pr.B,pr.p)
 fs=evaluate(pr,x,corners)
 z=2*qmc.Sobol(4,scramble=True,seed=2026091904).random_base2(7)-1
 points=[(pr.B+u[0]*.0001,replace(pr.p,kappa=pr.p.kappa+u[1]*.002,A_z=pr.p.A_z+u[2]*.002),1+u[3]*.03,f'heldout{i}') for i,u in enumerate(z)]
 hf=evaluate(pr,x,points)
 pd.DataFrame(dict(delta_B_T=z[:,0]*.0001,delta_kappa_MHz=z[:,1]*.002,delta_A_MHz=z[:,2]*.002,gain_error=z[:,3]*.03,fidelity=hf)).to_csv(ROOT/'results'/f'heldout_{name}.csv',index=False)
 pd.DataFrame(dict(label=[r[3] for r in corners],fidelity=fs)).to_csv(ROOT/'results'/f'corners_{name}.csv',index=False)
 iw=int(fs.argmin());scenario=corners[iw][:3]
 nom,_=integrate_extended(pr,x,max_step=.004)
 corner,_=integrate_extended(pr,x,max_step=.004,scenario=scenario)
 full=full_lab_midpoint(pr,x,dt=.0001)
 full2=full_lab_midpoint(pr,x,dt=.00005)
 info=dict(name=name,nominal_physical=nom,selected_corner_physical=corner,selected_corner_label=corners[iw][3],corner_rwa_min=float(fs.min()),heldout_rwa_min=float(hf.min()),heldout_rwa_median=float(np.median(hf)),n_heldout=len(hf),heldout_seed=2026091904,full_electron=[full,full2],resources=resource_metrics(pr,x))
 (ROOT/'results'/f'robust_validation_{name}.json').write_text(json.dumps(info,indent=2))
 # One-axis test scans are distinct from the simultaneous corner test.
 scans=[]
 for axis,scale in [('gain',.06),('A_z',.006),('B',.0003),('kappa',.006)]:
  xx=np.linspace(-scale,scale,25);pts=[]
  for i,v in enumerate(xx):
   b=pr.B+v if axis=='B' else pr.B;p=replace(pr.p,**{axis: getattr(pr.p,axis)+v}) if axis in ('A_z','kappa') else pr.p
   g=1+v if axis=='gain' else 1.;pts.append((b,p,g,str(i)))
  vals=evaluate(pr,x,pts,dt=.0125)
  scans.extend(dict(axis=axis,offset=float(v),fidelity=float(f)) for v,f in zip(xx,vals))
 pd.DataFrame(scans).to_csv(ROOT/'results'/f'axis_test_{name}.csv',index=False)
 print('ROBUST_VALIDATED',name,fs.min(),hf.min(),corner['worst'],flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
 names=['nominal_N_B1.6_dual','nominal_nuclear_B1.6_single',
        'robust_N_B1.6_dual_seed1','robust_N_B1.6_dual_seed2','robust_N_B1.6_dual_seed3',
        'robust_N_B1.6_single_seed1','robust_N_B1.6_dual_corners','robust_N_B1.6_dual_refined']
 while names:
  progressed=False
  for name in names.copy():
   out=ROOT/'results'/f'robust_validation_{name}.json'
   if out.exists():names.remove(name);continue
   p=ROOT/'results'/f'{name}.json'
   if p.exists():
    try:done=json.loads(p.read_text()).get('status')=='completed'
    except json.JSONDecodeError:done=False
    if done:run(name);names.remove(name);progressed=True
  if names and not args.wait:break
  if names and not progressed:time.sleep(10)
