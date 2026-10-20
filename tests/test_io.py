"""The pulse loader must accept archived configurations that carry study metadata."""
import json
from dataclasses import asdict
import numpy as np
from qudit_control.extended import ExtendedProblem, nominal_parameters
from qudit_control.io import load_pulse, PROBLEM_KEYS


def test_load_pulse_ignores_study_metadata(tmp_path):
    p = nominal_parameters()
    pr = ExtendedProblem(B=1.6, p=p, T=12., amax=.05, ncoef=14, dt=.0125, active=(0, 1))
    x = pr.initial()
    (tmp_path / 'results' / 'pulses').mkdir(parents=True)
    cfg = dict(B=1.6, T=12., amax=.05, ncoef=14, dt=.0125, active=[0, 1], parameters=asdict(p), seed=0, maxiter=3,
               train_start_times_us=[0., 1.5, 7.2], train_range_us=1e5, train_seed=20261020, robust=None)
    (tmp_path / 'results' / 'demo.json').write_text(json.dumps(dict(config=cfg, status='completed')))
    np.savez_compressed(tmp_path / 'results' / 'pulses' / 'demo.npz', x=x)
    pr2, x2, meta = load_pulse(tmp_path, 'demo')
    assert np.allclose(x2, x) and pr2.T == 12. and pr2.J == pr.J and len(pr2.ensemble) == 1
    assert meta['config']['train_start_times_us'] == [0., 1.5, 7.2]
    assert 'train_start_times_us' not in PROBLEM_KEYS and 'carrier_offset' in PROBLEM_KEYS
