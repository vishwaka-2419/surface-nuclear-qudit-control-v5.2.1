"""Carrier-defined frames and the second-order comb shift."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qudit_control.extended import ExtendedProblem,nominal_parameters,branch_targets,full_operator
from qudit_control.virtual_electron import second_order_blocks
from qudit_control.model import MU_B,hamiltonian
from qudit_control.requirements import electron_splitting


def _problem(offset=None,active=(0,1)):
    return ExtendedProblem(B=1.6,p=nominal_parameters(),T=12.,amax=.05,ncoef=14,dt=.05,active=active,mechanism='electron',carrier_offset=offset)


def test_carrier_offsets_reconstruct_level_shifts():
    """Offsets derived from an arbitrary level shift must reproduce that shift (up to a constant)."""
    pr=_problem();rng=np.random.default_rng(3)
    delta=rng.normal(0,.01,(2,pr.d))
    off=[]
    for s in (0,1):
        dE=np.diff(pr.E[s]);off.extend(np.sign(-dE)*(-np.diff(delta[s])))   # physical transition shift
    pr2=_problem(np.asarray(off))
    for s in (0,1):
        rec=pr2.frame_shift[s];assert np.allclose(rec-rec[0],delta[s]-delta[s,0],atol=1e-12)
        W=branch_targets(pr2)[s];expect=np.exp(-2j*np.pi*(delta[s]-delta[s,0])*pr.T)[:,None]*pr.G
        assert np.allclose(W,expect)
    assert np.allclose(pr2.freq,pr.freq+np.asarray(off))


def test_zero_offset_is_identity():
    pr=_problem(np.zeros(14));assert np.allclose(branch_targets(pr),np.tile(pr.G,(2,1,1)));assert np.allclose(pr.frame_shift,0)


def test_second_order_shift_matches_leading_order():
    """Adjacent diagonal differences of K_s equal A_z (beta g_t/gamma)^2/(4 nu_e^2) to within 10%."""
    pr=_problem();p=pr.p;K=second_order_blocks(pr)
    nu=electron_splitting(1.6,p);analytic=(MU_B*p.g_t/p.gamma)**2*p.A_z/(4*nu**2)
    for s in (0,1):
        d=np.abs(np.diff(np.real(np.diag(K[s]))))
        assert np.all(np.abs(d/analytic-1)<.1)
        # both combs shift towards lower frequency: physical transition shift negative
        shift=np.sign(-np.diff(pr.E[s]))*(-np.diff(np.real(np.diag(K[s]))))
        assert np.all(shift<0)
    # K is essentially diagonal for the electron-only operator
    for s in (0,1):
        od=K[s]-np.diag(np.diag(K[s]));assert np.max(np.abs(od))<1e-3*np.max(np.abs(np.diag(K[s])))


def test_offset_requires_one_entry_per_carrier():
    import pytest
    with pytest.raises(ValueError):_problem(np.zeros(7))
