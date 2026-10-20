"""Independent algebra, physical conventions, gradients and channel tests."""
import numpy as np
import pytest
from scipy.linalg import expm
from qudit_control.model import spin,fidelity
from qudit_control.extended import nominal_parameters,projected,full_operator,ExtendedProblem,calibration_ensemble,integrate_extended,full_lab_midpoint
from qudit_control.requirements import *
from qudit_control.validation import envelope_function
from qudit_control.virtual_electron import second_order_blocks

@pytest.mark.parametrize('mechanism',['nuclear','electron','magnetic'])
def test_anisotropic_operator_projection(mechanism):
 p=nominal_parameters();E,D,V,purity=projected(1.6,p,mechanism)
 assert p.g_t!=p.g_z
 assert np.max(abs(D-D.conj().swapaxes(-1,-2)))<1e-10
 assert np.max(abs(np.trace(D,axis1=1,axis2=2)))<1e-10
 assert min(purity)>.99
 if mechanism=='electron':
  ix,_,_=spin(p.I);r=np.real(np.diagonal(D,1,axis1=1,axis2=2)/np.diag(ix,1))
  assert np.all(r[0]<-3) and np.all(r[1]>3)

@pytest.mark.parametrize('active',[(0,),(0,1)])
def test_extended_linear_gradient(active):
 pr=ExtendedProblem(T=.3,ncoef=8,dt=.025,active=active)
 x=pr.initial(3);v,g=pr.objective(x)
 for k in (0,5,len(x)-2):
  xp=x.copy();xm=x.copy();xp[k]+=2e-6;xm[k]-=2e-6
  fd=(pr.objective(xp)[0]-pr.objective(xm)[0])/4e-6
  assert np.isclose(g[k],fd,rtol=2e-4,atol=3e-8)

def test_seven_carriers_propagate_both_branches():
 pr=ExtendedProblem(T=.3,ncoef=8,dt=.025,active=(0,1),carrier_branches=(0,))
 assert pr.J==7 and pr.C==2
 H=pr.hamiltonians(pr.initial());assert H.shape==(2,12,8,8)
 assert np.max(abs(H-H.conj().swapaxes(-1,-2)))<1e-12

def test_finite_control_bounds():
 pr=ExtendedProblem(T=2.,ncoef=10,active=(0,))
 x=np.random.default_rng(34).uniform(-1/np.sqrt(2),1/np.sqrt(2),pr.K*pr.J*2)
 f=envelope_function(pr,x);t=np.linspace(0,pr.T,401)
 assert np.max(abs(f(t)))<=pr.amax+1e-13
 assert np.max(abs(f([0,pr.T])))<1e-13
 assert np.max(abs(f.derivative()([0,pr.T])))<1e-13

def test_corner_set_has_16_distinct_corners():
 ens=calibration_ensemble(1.6,nominal_parameters())
 assert len(ens)==17
 assert len(set((b,p.kappa,p.A_z,g) for b,p,g,l in ens))==17

@pytest.mark.parametrize('pe',[0.,1e-4,.17,.5,1.])
def test_preparation_separate_from_stationarity(pe):
 T=.73;tau=.13
 assert np.isclose(no_jump_probability(T,tau,pe,0),np.exp(-pe*T/tau))
 assert np.isclose(no_jump_probability(T,tau,pe,1),np.exp(-(1-pe)*T/tau))
 expected=(1-pe)*np.exp(-pe*T/tau)+pe*np.exp(-(1-pe)*T/tau)
 assert np.isclose(no_jump_probability(T,tau,pe,pe),expected)

@pytest.mark.parametrize('d',[6,8,10])
def test_sufficient_bound_saturates_allocated_budget(d):
 pe=maximum_excited_fraction(12.,.1,d=d)
 assert abs(guaranteed_fidelity(12.,.1,pe,.001,d=d)-.99)<1e-14
 assert guaranteed_fidelity(12.,.1,pe/2,.001,d=d)>.99

@pytest.mark.parametrize('T,tau',[(0,.1),(-1,.1),(12,0)])
def test_bad_requirement_input(T,tau):
 with pytest.raises(ValueError):maximum_excited_fraction(T,tau)

def test_thermal_requirement_and_time_units():
 p=nominal_parameters();pe=maximum_excited_fraction(12.,.1)
 t=polarization_temperature(1.6,p,pe)
 assert np.isclose(pe,8.490105510636162e-5,rtol=1e-12)
 assert .066<t<.068
 assert np.isclose(thermal_population(1.6,p,t),pe,rtol=1e-12)
 tau=motional_narrowing_time(12.,p.A_z,thermal_population(1.6,p,.4))
 assert .8<tau*1e9<.9  # microseconds to femtoseconds

def test_fidelity_floors_are_distinct():
 d=8;_,iy,_=spin(3.5)
 assert np.isclose(fidelity(expm(-1j*np.pi*iy),np.eye(d)),1/(d+1))
 # Choi/superoperator expressions for depolarization and complete dephasing.
 assert np.isclose((d+1)/(d*(d+1)),1/d)
 assert np.isclose((d+d)/(d*(d+1)),2/(d+1))

def test_phase_tracking_population_cancellation():
 p=nominal_parameters();E,*_=projected(1.6,p);G=expm(-.5j*np.pi*spin(p.I)[1])
 for t in [.0005,.00371,.0293,.087234]:
  pops=[]
  for s in (0,1):
   R=np.diag(np.exp(-2j*np.pi*E[s]*t));U=R@G@G;pops.append(abs(U[:,-1])**2)
  assert np.max(abs(pops[0]-pops[1]))<2e-14

def test_final_diagonal_fidelity_is_calculated_separately():
 G=expm(-.5j*np.pi*spin(3.5)[1]);D=np.diag(np.exp(1j*np.linspace(-1,1,8)));U=D@G
 d=8;diag=np.diag(U@G.conj().T)
 corrected=(d+np.sum(abs(diag))**2)/(d*(d+1))
 assert abs(corrected-1)<1e-13 and fidelity(U,G)<.9

def test_bb1_removes_leading_amplitude_error():
 a=amplitude_error(3.5,.03);b=amplitude_error(3.5,.03,bb1=True)
 assert a>.009 and b<1e-7


def test_full_short_propagation_is_unitary_and_leakage_not_normalized():
 pr=ExtendedProblem(T=.12,ncoef=8,active=(0,),dt=.01)
 x=pr.initial();r=full_lab_midpoint(pr,x,dt=.0001)
 assert r['unitarity_error']<1e-9
 assert r['worst']<=1+1e-10
 assert abs(r['leakage'][0])<1e-6
 assert r['best_diagonal_output_fidelities'][0]>=r['worst']-1e-12

@pytest.mark.parametrize('mechanism',['nuclear','electron','magnetic'])
def test_second_order_blocks_are_hermitian(mechanism):
 pr=ExtendedProblem(T=.2,ncoef=8,active=(0,),mechanism=mechanism)
 k=second_order_blocks(pr)
 assert np.max(abs(k-k.conj().swapaxes(-1,-2)))<1e-12
 assert np.max(abs(np.trace(k,axis1=1,axis2=2)))<5e-12  # scalar subtraction roundoff for large electronic prefactor


def test_second_order_control_gradient():
 from qudit_control.stark_control import StarkProblem
 pr=StarkProblem(T=.3,ncoef=8,dt=.025,amax=.01,active=(0,),mechanism='electron')
 x=pr.initial(4);v,g=pr.objective(x)
 for k in (0,6,len(x)-3):
  xp=x.copy();xm=x.copy();xp[k]+=2e-6;xm[k]-=2e-6
  fd=(pr.objective(xp)[0]-pr.objective(xm)[0])/4e-6
  assert np.isclose(g[k],fd,rtol=4e-4,atol=3e-8)
