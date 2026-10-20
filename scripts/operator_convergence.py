"""Refine full electron propagation for the alternative drive operators."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import full_lab_midpoint
for mech in ('electron','magnetic'):
 for active in ('single','dual'):
  name=f'nominal_{mech}_B1.6_{active}';pr,x,_=load_pulse(ROOT,name)
  p=ROOT/'results'/f'validation_{name}.json';d=json.loads(p.read_text())
  for dt in (2.5e-5,1.25e-5,6.25e-6):
   if any(abs(v.get('dt_us', -1)-dt)<1e-14 for v in d['full_electron']): continue
   r=full_lab_midpoint(pr,x,dt=dt);d['full_electron'].append(r);p.write_text(json.dumps(d,indent=2))
   print(name,dt,r['worst'],r['leakage'],flush=True)
