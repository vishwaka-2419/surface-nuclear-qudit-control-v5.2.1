import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.virtual_electron import integrate_second_order
rows=[]
for mechanism in ('nuclear','electron','magnetic'):
 for active in ('single','dual'):
  name=(f'nominal_N_B1.6_dual' if mechanism=='nuclear' and active=='dual' else f'nominal_{mechanism}_B1.6_{active}')
  pr,x,_=load_pulse(ROOT,name);r=integrate_second_order(pr,x);r['name']=name;rows.append(r)
  print(name,r['worst'],r['maximum_adjacent_mean_shift_MHz'],flush=True)
  (ROOT/'results/virtual_electron_shifts.json').write_text(json.dumps(rows,indent=2))
