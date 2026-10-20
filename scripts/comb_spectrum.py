"""Nuclear transition frequencies of both electron branches versus field.

Writes results/comb_spectrum.csv (full 16-level model, nominal anisotropic
parameters) and results/comb_spectrum_secular.csv (axial secular ENDOR
expression) for the overview figure.  Nothing here feeds the pulse designs.
"""
import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.model import branches
from qudit_control.extended import nominal_parameters
p=nominal_parameters();rows=[];sec=[]
I=p.I;m=np.arange(I,-I,-1.)   # upper level of each adjacent transition, m -> m-1
for B in np.linspace(0.6,1.9,261):
    E,_,_,_=branches(B,p)
    for s,lab in ((0,'ground'),(1,'excited')):
        f=np.abs(np.diff(E[s]))
        for k,fk in enumerate(f):rows.append(dict(B_T=B,branch=lab,link=k,f_MHz=fk))
    for s,lab in ((-.5,'ground'),(.5,'excited')):
        nu=p.gamma*B+p.A_z*s+3*p.kappa*(2*m-1)/(4*I*(2*I-1))
        for k,v in enumerate(nu):sec.append(dict(B_T=B,branch=lab,link=k,f_MHz=abs(v)))
pd.DataFrame(rows).to_csv(ROOT/'results/comb_spectrum.csv',index=False)
pd.DataFrame(sec).to_csv(ROOT/'results/comb_spectrum_secular.csv',index=False)
print('wrote comb spectra')
