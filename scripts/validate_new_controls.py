"""Independent projected and full-electron validations of archived controls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,argparse
from pathlib import Path
import numpy as np,pandas as pd
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.extended import *
from qudit_control.validation import resource_metrics
from qudit_control.model import fidelity
OUT=ROOT/'results'
def validate(name):
    pr,x,meta=load_pulse(ROOT,name)
    print('VALIDATE',name,flush=True)
    info,UU=integrate_extended(pr,x,max_step=.004)
    rwa,_=integrate_extended(pr,x,counterrotating=False,max_step=.004)
    full=[]
    for step in (.0002,.0001,.00005):
        full.append(full_lab_midpoint(pr,x,dt=step));print('FULL',name,step,full[-1]['worst'],flush=True)
    baseline=None
    if pr.kind=='rotation':
        baseline,_=integrate_extended(pr,env=plain_envelope(pr),max_step=.004)
    refined=[]
    for step in (.025,.0125,.00625):
        rp,xx,_=load_pulse(ROOT,name,dt=step)
        refined.append({'dt_us':rp.dt,'worst':float(fidelity(rp.propagators(xx),rp.W).min())})
    data=dict(name=name,physical_projected=info,adaptive_rwa=rwa,full_electron=full,
              plain_multitone=baseline,mesh_refinement=refined,resources=resource_metrics(pr,x))
    (OUT/f'validation_{name}.json').write_text(json.dumps(data,indent=2))
    print('VALIDATED',name,info['worst'],full[-1]['worst'],flush=True)
    return data
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    expected=10
    while True:
        names=[]
        for f in sorted(OUT.glob('*.json')):
            if not (f.stem.startswith('nominal_') or f.stem.startswith('selective_')):continue
            m=json.loads(f.read_text())
            if m.get('status')=='completed' and not (OUT/f'validation_{f.stem}.json').exists():names.append(f.stem)
        for n in names:validate(n)
        nval=len(list(OUT.glob('validation_nominal_*.json')))+len(list(OUT.glob('validation_selective_*.json')))
        if not args.wait or nval>=expected:break
        time.sleep(5)
