"""Reproduce finite-budget optimization. No hardware connection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,argparse
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.extended import ExtendedProblem,nominal_parameters
OUT=ROOT/'results';(OUT/'pulses').mkdir(exist_ok=True)
def axis_ensemble(B,p):
    items=[(B,p,1.,'nominal')]
    for s in (-1,1):
        items.extend([(B+s*.0001,p,1.,f'B{s}'),(B,replace(p,kappa=p.kappa+s*.002),1.,f'k{s}'),
                      (B,replace(p,A_z=p.A_z+s*.002),1.,f'A{s}'),(B,p,1+s*.03,f'gain{s}')])
    return items

def run(job):
    name=job.pop('name');iters=job.pop('maxiter');seed=job.pop('seed',0);rob=job.pop('robust',False)
    p=nominal_parameters();B=job.get('B',1.6)
    cfg=dict(job);cfg.update(parameters=asdict(p),seed=seed,maxiter=iters,robust=rob)
    pr=ExtendedProblem(p=p,ensemble=axis_ensemble(B,p) if rob else None,**job)
    def checkpoint(x,h,n,t):
        np.savez_compressed(OUT/'pulses'/f'{name}.npz',x=x)
        (OUT/f'{name}.json').write_text(json.dumps(dict(config=cfg,history=h,iterations=n,elapsed_s=t,status='running'),indent=2))
        print(name,n,h[-1] if h else {},round(t,1),flush=True)
    opt,info=pr.optimize(seed=seed,maxiter=iters,checkpoint=checkpoint)
    info.update(config=cfg,status='completed',labels=pr.labels)
    np.savez_compressed(OUT/'pulses'/f'{name}.npz',x=opt.x)
    (OUT/f'{name}.json').write_text(json.dumps(info,indent=2))
    print('DONE',name,info['worst_train'],info['elapsed_s'],flush=True)
    return name

def jobs(group):
    if group=='nominal':
        a=[dict(name=f'nominal_N_B{b:g}_dual',B=b,T=12.,amax=.05,ncoef=14,dt=.0125,active=(0,1),maxiter=600) for b in (1.03,1.35,1.6,1.714)]
        a += [dict(name=f'nominal_{m}_B1.6_single',B=1.6,T=12.,amax=.05,ncoef=14,dt=.0125,active=(0,),mechanism=m,maxiter=800) for m in ('nuclear','electron','magnetic')]
        a += [dict(name=f'nominal_{m}_B1.6_dual',B=1.6,T=12.,amax=.05,ncoef=14,dt=.0125,active=(0,1),mechanism=m,maxiter=800) for m in ('electron','magnetic')]
        a += [dict(name='selective_N_B1.6_dual',B=1.6,T=20.,kind='selective',amax=.05,ncoef=16,dt=.0125,active=(0,1),maxiter=800)]
        return a
    if group=='robust':
        a=[dict(name=f'robust_N_B1.6_dual_seed{s}',B=1.6,T=48.,amax=.05,ncoef=22,dt=.05,active=(0,1),maxiter=1200,seed=s,robust=True) for s in (1,2,3)]
        a += [dict(name='robust_N_B1.6_single_seed1',B=1.6,T=48.,amax=.05,ncoef=22,dt=.05,active=(0,),maxiter=1200,seed=1,robust=True)]
        return a
    raise ValueError(group)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['nominal','robust'],required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    work=[j for j in jobs(args.group) if not (OUT/(j['name']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:list(ex.map(run,work))
