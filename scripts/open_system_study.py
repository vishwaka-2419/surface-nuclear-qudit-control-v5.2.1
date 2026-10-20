"""Prepared-electron noise study. Fixed seeds and rare-jump stratification."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.electron_noise import prepared_telegraph,prepared_dephasing
from qudit_control.requirements import thermal_population
pr,x,_=load_pulse(ROOT,'nominal_nuclear_B1.6_single')
pe=thermal_population(pr.B,pr.p,.4)
rows=[]
for p in (pe,1e-4,1e-5):
 for tau in (.1,1.,10.,100.,1000.,10000.):
  r=prepared_telegraph(pr,x,tau,p,ntraj=8192,dt=.005)
  rows.append(r);print('NOISE',p,tau,r['fidelity'],r['sem'],flush=True)
  pd.DataFrame(rows).to_csv(ROOT/'results/prepared_telegraph.csv',index=False)
conv=[]
for dt in (.01,.005,.0025):
 r=prepared_telegraph(pr,x,.1,1e-4,ntraj=8192,dt=dt);conv.append(r)
 (ROOT/'results/telegraph_convergence.json').write_text(json.dumps(conv,indent=2))
noise=[]
for kind in ('magnetic','quadrupolar'):
 for t in (100.,1000.,10000.,100000.):
  r=prepared_dephasing(pr,x,t,kind);noise.append(r);print('DEPHASE',kind,t,r['fidelity'],flush=True)
pd.DataFrame(noise).to_csv(ROOT/'results/prepared_dephasing.csv',index=False)
