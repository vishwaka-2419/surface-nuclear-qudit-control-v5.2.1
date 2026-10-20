"""Additional channel checks: ensemble consistency, trace overlaps and exact jumps."""
import numpy as np
from scipy.linalg import expm
from qudit_control.control import ControlProblem
from qudit_control.model import Parameters, fidelity, common_frame_targets
from qudit_control.noise import telegraph_gate, dephasing_channel, dephasing_operator
from qudit_control.validation import integrate_physical

def test_polarized_electron_is_static():
 pr=ControlProblem(T=.3,ncoef=8,dt=.005)
 x=pr.initial()
 val=telegraph_gate(pr,x,tau_c=.05,p_exc=0,ntraj=32,dt=.005)
 f=float(fidelity(pr.propagators(x),pr.W)[0])
 assert abs(val['fidelity']-f)<2e-10
 assert val['mean_jumps']==0
 assert val['nojump_probability']==1

def test_jump_statistics():
 pr=ControlProblem(T=.3,ncoef=8,dt=.005);x=np.zeros(pr.K*pr.J*2)
 val=telegraph_gate(pr,x,tau_c=.1,p_exc=.25,ntraj=2048,seed=113,dt=.005)
 assert abs(val['sampled_nojump']-val['nojump_probability'])<.035
 assert abs(val['mean_jumps']-val['expected_jumps'])<.10
 assert val['unitarity_error']<1e-10

def test_zero_drive_dephasing_against_analytic_superoperator():
 pr=ControlProblem(T=.3,ncoef=8,dt=.01);x=np.zeros(pr.K*pr.J*2)
 A=dephasing_operator(3.5,'quadrupolar');tn=20.
 damping=np.exp(-pr.T*(A[:,None]-A[None,:])**2/tn)
 out=dephasing_channel(pr,x,tn,'quadrupolar',dt=.01)
 exact=[]
 for W in pr.W:
  z=np.diag(W)
  overlap=np.sum(z.conj()[:,None]*z[None,:]*damping).real
  exact.append((pr.d+overlap)/(pr.d*(pr.d+1)))
 assert np.max(np.abs(np.array(out['fidelities'])-np.array(exact)))<1e-12

def test_independent_rwa_agrees_with_refined_propagator():
 pr=ControlProblem(T=.05,ncoef=8,dt=.00025)
 x=pr.initial();a=pr.propagators(x)
 b,info,_=integrate_physical(pr,x,include_counterrotating=False,max_step=.0001)
 assert np.max(np.abs(fidelity(a,pr.W)-fidelity(b,pr.W)))<1e-7

def test_stratified_jumps_against_exact_free_channel():
 from qudit_control.noise import idle_telegraph_exact
 pr=ControlProblem(T=.3,ncoef=8,dt=.005);x=np.zeros(pr.K*pr.J*2)
 val=telegraph_gate(pr,x,tau_c=.1,p_exc=.25,ntraj=4096,seed=9001,dt=.005,stratified=True)
 exact=idle_telegraph_exact(pr.E,pr.T,.1,.25,pr.G)
 assert abs(val['fidelity']-exact)<4*val['sem']+2e-4
 assert val['unitarity_error']<1e-10
