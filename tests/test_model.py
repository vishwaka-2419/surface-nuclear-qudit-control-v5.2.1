import numpy as np
import pytest
from scipy.linalg import expm
from dataclasses import replace
from qudit_control.model import *
from qudit_control.control import ControlProblem,spline_basis,training_ensemble
from qudit_control.noise import white_noise_slope,idle_telegraph_exact

@pytest.mark.parametrize('I',[.5,2.5,3.5,4.5])
def test_spin_algebra(I):
 x,y,z=spin(I)
 assert np.max(np.abs(x@y-y@x-1j*z))<1e-13
 assert np.max(np.abs(x@x+y@y+z@z-I*(I+1)*np.eye(len(x))))<1e-13

@pytest.mark.parametrize('I',[2.5,3.5,4.5])
def test_wrong_rotation_fidelity(I):
 _,y,_=spin(I);a=expm(-.5j*np.pi*y);b=expm(.5j*np.pi*y)
 assert abs(fidelity(a,b)-1/(2*I+2))<1e-13

def test_common_frame_counterexample():
 p=Parameters(A_t=0,B_tip=0); E,*_=branches(1,p);G=target(p.I)
 T=1/(2*p.A_z)
 wrong=np.tile(G,(2,1,1));W=common_frame_targets(E,T,G)
 assert np.max(np.abs(fidelity(wrong,W)-1/9))<1e-11

@pytest.mark.parametrize('B',[1.03,1.35,1.6])
def test_axial_spectrum(B):
 p=Parameters(A_t=0,B_tip=0); E,*_=branches(B,p)
 m=np.arange(p.I,-p.I-1,-1)
 q=p.kappa/(4*p.I*(2*p.I-1))*(3*m*m-p.I*(p.I+1))
 exact=np.array([(s*p.A_z+p.gamma*B)*m+q for s in (-.5,.5)])
 exact-=exact.mean(axis=1)[:,None]
 assert np.max(np.abs(E-exact))<1e-10

def test_hermitian_and_purity():
 p=Parameters(eta=.3,efg_tilt=10)
 H,*_=hamiltonian(1.03,p);assert np.linalg.norm(H-H.conj().T)<1e-12
 E,D,V,pur=branches(1.03,p)
 assert np.min(pur)>.8
 assert np.max(np.abs(D-D.conj().swapaxes(-1,-2)))<1e-12

def test_frechet_gradient():
 p=Parameters();pr=ControlProblem(B=1.03,T=2.,ncoef=8,dt=.05,ensemble=training_ensemble(1.03,p,db=.0001,dk=.002,gain=.01))
 x=pr.initial();val,g=pr.objective(x)
 for k in [0,13,40,74]:
  xp=x.copy();xm=x.copy();xp[k]+=1e-5;xm[k]-=1e-5
  fd=(pr.objective(xp)[0]-pr.objective(xm)[0])/2e-5
  assert np.isclose(fd,g[k],atol=2e-8,rtol=2e-5)

def test_bounds_and_endpoints():
 pr=ControlProblem(T=12,ncoef=14)
 x=np.random.default_rng(9).uniform(-1/np.sqrt(2),1/np.sqrt(2),(pr.K,pr.J,2)).ravel()
 from qudit_control.validation import envelope_function
 env=envelope_function(pr,x); t=np.linspace(0,12,1001)
 assert np.max(np.abs(env(t)))<=pr.amax*(1+1e-12)
 assert np.max(np.abs(env([0,12])))<1e-14
 assert np.max(np.abs(env.derivative()([0,12])))<1e-14

def test_hermitian_control_unitarity():
 pr=ControlProblem(T=2,ncoef=8,dt=.025)
 x=pr.initial();H=pr.hamiltonians(x);U=pr.propagators(x)
 assert np.max(np.abs(H-H.conj().swapaxes(-1,-2)))<1e-14
 assert np.max(np.abs(U@U.conj().swapaxes(-1,-2)-np.eye(pr.d)))<2e-12

def test_noise_slopes():
 assert abs(white_noise_slope(3.5)-28/3)<1e-12
 assert abs(white_noise_slope(2.5)-5)<1e-12
 assert abs(white_noise_slope(4.5)-15)<1e-12

@pytest.mark.parametrize('p',[0,.1,.5,1.])
def test_idle_zero_hyperfine(p):
 E=np.tile(np.arange(8),(2,1));assert abs(idle_telegraph_exact(E,12,.1,p)-1)<1e-13

def test_nominal_16_level_reference():
 inter,_,_=metrics(1.6,Parameters())
 assert np.isclose(inter,1.03255322656,atol=1e-7)

def test_finite_comb_does_not_obey_global_half_spacing_ceiling():
 p=Parameters(A_t=0,B_tip=0);B=6.
 actual,_,_=metrics(B,p);axial=finite_comb_distance(B,p)
 spacing=3*abs(p.kappa)/(2*p.I*(2*p.I-1))
 assert np.isclose(actual,axial,atol=1e-10)
 assert axial>spacing/2
 assert p.A_z/2>abs(p.gamma*B)+(p.I-.5)*spacing
