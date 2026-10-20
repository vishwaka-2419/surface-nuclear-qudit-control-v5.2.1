"""Nominal synthesis including virtual electron excitation and full validation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,gc
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.stark_control import StarkProblem
from qudit_control.extended import full_lab_midpoint
from qudit_control.virtual_electron import integrate_second_order
from qudit_control.validation import resource_metrics
for active in ('single','dual'):
 for mechanism in ('electron','magnetic'):
  source=f'nominal_{mechanism}_B1.6_{active}';pr,x,meta=load_pulse(ROOT,source)
  name=f'stark_{mechanism}_B1.6_{active}'
  stages=[]
  for dt,iters in ((.025,350),(.0125,250)):
   p=StarkProblem(B=pr.B,p=pr.p,T=pr.T,amax=pr.amax,ncoef=pr.ncoef,dt=dt,
                   active=pr.active,mechanism=mechanism)
   op,info=p.optimize(x0=x,maxiter=iters);x=op.x;info['dt_us']=dt;stages.append(info)
   print('STARK_STAGE',name,dt,info['worst_train'],info['elapsed_s'],flush=True)
   del p;gc.collect()
  config=dict(meta['config']);config.update(dt=.0125,maxiter=600)
  record=dict(config=config,status='completed',design_model='second_order_electron',stages=stages,
              warm_start=source,iterations=sum(s['iterations'] for s in stages))
  np.savez_compressed(ROOT/'results/pulses'/f'{name}.npz',x=x)
  (ROOT/'results'/f'{name}.json').write_text(json.dumps(record,indent=2))
  physical=integrate_second_order(pr,x)
  vals=[]
  for dt in (.0001,.00005,.000025):
   rr=full_lab_midpoint(pr,x,dt=dt);vals.append(rr);print('STARK_FULL',name,dt,rr['worst'],flush=True)
  v=dict(name=name,second_order_physical=physical,full_electron=vals,resources=resource_metrics(pr,x))
  (ROOT/'results'/f'validation_{name}.json').write_text(json.dumps(v,indent=2))
