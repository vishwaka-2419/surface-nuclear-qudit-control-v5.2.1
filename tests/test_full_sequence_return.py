"""Full-space return preserves leakage-inclusive branch fidelities."""
import numpy as np
import pytest
from qudit_control.extended import ExtendedProblem, full_lab_midpoint, branch_targets


def test_full_interaction_propagator_matches_reported_branch_fidelities():
    pr = ExtendedProblem(T=.12, ncoef=8, dt=.01, active=(0, 1))
    x = pr.initial()
    t0 = .173
    info, joint = full_lab_midpoint(pr, x, dt=.0001, t0=t0, return_full=True)
    assert joint.shape == (2*pr.d, 2*pr.d)
    assert np.linalg.norm(joint.conj().T @ joint - np.eye(2*pr.d)) < 1e-9
    for k, s in enumerate(pr.active):
        block = joint[s*pr.d:(s+1)*pr.d, s*pr.d:(s+1)*pr.d]
        target = branch_targets(pr, t0)[s]
        value = (np.linalg.norm(block)**2 + abs(np.trace(target.conj().T @ block))**2)/(pr.d*(pr.d+1))
        assert abs(value - info['fidelities'][k]) < 1e-11


def test_full_propagator_return_options_are_exclusive():
    pr = ExtendedProblem(T=.12, ncoef=8, dt=.01)
    with pytest.raises(ValueError):
        full_lab_midpoint(pr, pr.initial(), return_blocks=True, return_full=True)
