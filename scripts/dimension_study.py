"""Synthetic dimension transfer at equal worst-case root-sum-square envelope bound."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.extended import ExtendedProblem,nominal_parameters,integrate_extended
from qudit_control.validation import resource_metrics
rows=[]
for d in (6,8,10):
 I=(d-1)/2;p=replace(nominal_parameters(),I=I,kappa=-3.3*2*I*(2*I-1)/3)
 J=2*(d-1);amp=.05*np.sqrt(14/J)
 pr=ExtendedProblem(B=1.6,p=p,T=12.,amax=amp,ncoef=14,dt=.0125)
 op,info=pr.optimize(maxiter=500,seed=7)
 actual,_=integrate_extended(pr,op.x,max_step=.003)
 name=f'synthetic_d{d}';cfg=dict(B=1.6,T=12.,amax=amp,ncoef=14,dt=.0125,parameters=asdict(p),active=[0,1],mechanism='nuclear',frame='tracked',seed=7,maxiter=500)
 info.update(config=cfg,status='completed',physical=actual,resources=resource_metrics(pr,op.x))
 (ROOT/'results'/f'{name}.json').write_text(json.dumps(info,indent=2));np.savez_compressed(ROOT/'results/pulses'/f'{name}.npz',x=op.x)
 rows.append(dict(d=d,I=I,J=J,kappa_MHz=p.kappa,amax_MHz=amp,rss_bound_MHz=np.sqrt(J)*amp,physical_fidelity=actual['worst'],training_fidelity=info['worst_train']))
 pd.DataFrame(rows).to_csv(ROOT/'results/dimension_transfer.csv',index=False)
 print(name,actual['worst'],flush=True)
