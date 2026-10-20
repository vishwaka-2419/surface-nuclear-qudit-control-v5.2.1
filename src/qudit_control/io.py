"""Load a released pulse with its complete model metadata."""
import inspect
import json
from pathlib import Path
import numpy as np
from .model import Parameters
from .extended import ExtendedProblem

PROBLEM_KEYS=set(inspect.signature(ExtendedProblem.__init__).parameters)-{'self'}

def load_pulse(root,name,**overrides):
    """Return (problem, coefficients, metadata) for an archived pulse.

    The archived configuration may carry study metadata that is not a model
    argument (optimizer settings, training start times and seeds, robustness
    ensembles); those keys are kept in ``meta['config']`` but not passed to
    ``ExtendedProblem``.  The ensemble used for training is deliberately not
    reconstructed: the returned problem is the nominal single-scenario model in
    which the archived validations were carried out.
    """
    root=Path(root);meta=json.loads((root/'results'/f'{name}.json').read_text())
    cfg=dict(meta['config']);p=Parameters(**cfg.pop('parameters'))
    cfg={k:v for k,v in cfg.items() if k in PROBLEM_KEYS}
    cfg.update(overrides);cfg['p']=p
    pr=ExtendedProblem(**cfg)
    x=np.load(root/'results/pulses'/f'{name}.npz')['x']
    return pr,x,meta
