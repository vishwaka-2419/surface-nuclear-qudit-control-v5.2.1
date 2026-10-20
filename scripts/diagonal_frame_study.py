"""A diagnostic optimized final phase update, not an unannounced fidelity gauge.

The phase correction is obtained from the simulated output and is an upper
benchmark for a calibratable diagonal compiler correction. It is reported
separately from fidelity to the originally fixed target.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import full_lab_midpoint
out=[]
for mech in ('electron','magnetic'):
 for active in ('single','dual'):
  name=f'nominal_{mech}_B1.6_{active}';pr,x,_=load_pulse(ROOT,name)
  r=full_lab_midpoint(pr,x,dt=.000025);r['name']=name;out.append(r)
  (ROOT/'results/diagonal_output_diagnostic.json').write_text(json.dumps(out,indent=2))
  print(name,r['worst'],r['best_diagonal_output_fidelities'],flush=True)
