"""Full-corner and finer-mesh refinement of an independently started control.

The refinement ensemble contains the nominal point and all 16 corners. It is
separate from the axis-only training of the initial multistart calculation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import ExtendedProblem,calibration_ensemble
source='robust_N_B1.6_dual_seed1'
pr,x,meta=load_pulse(ROOT,source)
for name,dt,niter in [('robust_N_B1.6_dual_corners',.025,400),('robust_N_B1.6_dual_refined',.0125,250)]:
 ens=calibration_ensemble(pr.B,pr.p)
 p=ExtendedProblem(B=pr.B,p=pr.p,T=pr.T,amax=pr.amax,ncoef=pr.ncoef,dt=dt,
                   active=pr.active,mechanism=pr.mechanism,beta=2000.,ensemble=ens)
 cfg=dict(meta['config']);cfg.update(dt=p.dt,seed=1,maxiter=niter,robust=True,beta=2000.)
 def checkpoint(xx,h,n,t):
  np.savez_compressed(ROOT/'results/pulses'/f'{name}.npz',x=xx)
  (ROOT/'results'/f'{name}.json').write_text(json.dumps(dict(config=cfg,history=h,iterations=n,elapsed_s=t,status='running',training='full_corners',warm_start=source),indent=2))
  print(name,n,h[-1] if h else {},round(t,1),flush=True)
 op,info=p.optimize(x0=x,maxiter=niter,checkpoint=checkpoint);x=op.x
 info.update(config=cfg,status='completed',training='full_corners',warm_start=source,labels=p.labels)
 np.savez_compressed(ROOT/'results/pulses'/f'{name}.npz',x=x)
 (ROOT/'results'/f'{name}.json').write_text(json.dumps(info,indent=2))
 print('DONE',name,info['worst_train'],info['elapsed_s'],flush=True)
 source=name
