"""Requirements, operator matrix elements, source sensitivity, and frame tests."""
import sys,json
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np,pandas as pd
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.extended import nominal_parameters,projected
from qudit_control.requirements import *
from qudit_control.model import metrics,finite_comb_distance,spin
OUT=ROOT/'results';p=nominal_parameters();ix,iy,iz=spin(3.5)
rows=[]
for B in (1.03,1.35,1.6,1.714):
    for T in (12.,48.,60.,250.):
        pe=maximum_excited_fraction(T,.1)
        rows.append(dict(B_T=B,T_us=T,tau_c_reference_us=.1,p_thermal_04K=thermal_population(B,p,.4),
                         p_exc_sufficient=pe,temperature_sufficient_mK=1000*polarization_temperature(B,p,pe),
                         motional_tau_fs=1e9*motional_narrowing_time(T,p.A_z,thermal_population(B,p,.4)),
                         mean_generator_kHz=1000/(4*T),pure_magnetic_Bpeak_mT=1000/(2*T*p.gamma)))
pd.DataFrame(rows).to_csv(OUT/'operating_requirements.csv',index=False)
rows=[]
for T in (12.,48.,60.,250.):
    for tau in np.logspace(-4,5,181):
        pe=maximum_excited_fraction(T,tau)
        rows.append(dict(T_us=T,tau_c_us=tau,p_exc_sufficient=pe))
pd.DataFrame(rows).to_csv(OUT/'polarization_requirement_map.csv',index=False)
rows=[]
for B in np.linspace(.8,1.8,101):
 for mech in ('nuclear','electron','magnetic'):
    E,D,V,purity=projected(B,p,mech)
    gains=np.real(np.diagonal(D,1,axis1=1,axis2=2)/np.diag(ix,1))
    for s in (0,1):
        rows.append(dict(B_T=B,mechanism=mech,branch=s,gain_min=gains[s].min(),gain_mean=gains[s].mean(),gain_max=gains[s].max()))
pd.DataFrame(rows).to_csv(OUT/'operator_gains.csv',index=False)
rows=[]
for B in np.linspace(.65,1.8,231):
    for name,pp in [('anisotropic',p),('isotropic',replace(p,g_t=p.g_z)),('surface_gamma',replace(p,gamma=7.622593285*.37))]:
        inter,intra,purity=metrics(B,pp)
        rows.append(dict(B_T=B,scenario=name,inter_MHz=inter,intra_MHz=intra,purity=purity))
pd.DataFrame(rows).to_csv(OUT/'field_spectrum_new.csv',index=False)
rows=[]
for eps in np.r_[-np.logspace(-4,-.7,51)[::-1],0,np.logspace(-4,-.7,51)]:
    rows.append(dict(gain_error=eps,plain_infidelity=amplitude_error(3.5,eps),BB1_infidelity=amplitude_error(3.5,eps,bb1=True)))
pd.DataFrame(rows).to_csv(OUT/'analytic_composite_baseline.csv',index=False)
# Two ideal phase-tracked gates; population readout in the m basis.
E,D,V,purity=projected(1.6,p);G=expm(-1j*np.pi/2*iy);rho=np.zeros((8,8),complex);rho[-1,-1]=1
rows=[]
for idle in np.linspace(0,.1,101):
    tracked=[];untracked=[]
    for s in (0,1):
        R=np.diag(np.exp(-2j*np.pi*E[s]*idle))
        Utracked=R@G@G
        Uuntracked=G@R@G
        tracked.append(np.real(np.diag(Utracked@rho@Utracked.conj().T)))
        untracked.append(np.real(np.diag(Uuntracked@rho@Uuntracked.conj().T)))
    rows.append(dict(idle_us=idle,tracked_population_distance=np.linalg.norm(tracked[0]-tracked[1],1),
                     untracked_population_distance=np.linalg.norm(untracked[0]-untracked[1],1)))
pd.DataFrame(rows).to_csv(OUT/'phase_tracking_populations.csv',index=False)
summary=dict(parameters=asdict(p),gate_dimension=8,unitary_minimum_fidelity=1/9,depolarizing_fidelity=1/8,
             complete_diagonal_dephasing_fidelity=2/9,magnetic_slope=magnetic_dephasing_slope(3.5),
             requirement_12us_at_1p6T=pd.read_csv(OUT/'operating_requirements.csv').query('B_T==1.6 and T_us==12').iloc[0].to_dict())
(OUT/'analytical_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
