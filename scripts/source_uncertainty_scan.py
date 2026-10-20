"""Alternative interpretations of source errors, not a selected confidence set."""
from pathlib import Path
import sys,json
from dataclasses import replace
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.extended import nominal_parameters
from qudit_control.model import metrics,MU_N
p=nominal_parameters();R=.247/.302;rel=np.hypot(.011/.247,.010/.302);mean=56.7*R
rows=[]
for sk47 in (.8,16.):
 sk49=np.hypot(R*sk47,mean*rel)
 for scenario,g,dg in [('bulk_coefficient',p.gamma/MU_N,0.),('surface_fit',.37,.04)]:
  lo=3*(mean-sk49)/(56*MU_N*(g+dg));hi=3*(mean+sk49)/(56*MU_N*(g-dg));center=3*mean/(56*MU_N*g)
  rows.append(dict(sigma_k47_MHz=sk47,source_gamma=scenario,g_center=g,g_half_range=dg,k49_MHz=-mean,sigma_k49_MHz=sk49,B_r1_T=center,B_lower_T=lo,B_upper_T=hi))
pd.DataFrame(rows).to_csv(ROOT/'results/source_uncertainty_intervals.csv',index=False)
rows=[]
for B in np.linspace(.8,1.8,101):
 for eta,tilt in [(0,0),(.2,0),(.5,0),(0,5),(0,15),(.2,15)]:
  pp=replace(p,eta=eta,efg_tilt=tilt)
  try:inter,intra,purity=metrics(B,pp);valid=True
  except ValueError:inter=intra=purity=float('nan');valid=False
  rows.append(dict(B_T=B,eta=eta,tilt_deg=tilt,inter_MHz=inter,intra_MHz=intra,purity=purity,valid_labels=valid))
pd.DataFrame(rows).to_csv(ROOT/'results/efg_spectral_stress.csv',index=False)
print(pd.read_csv(ROOT/'results/source_uncertainty_intervals.csv').to_string(index=False))
