"""Analytical operating requirements; explicit model assumptions in each function."""
import numpy as np
from scipy.linalg import expm
from .model import spin,fidelity,MU_B
K_PER_MHZ=4.799243073366221e-5

def electron_splitting(B,p):
    phi=np.deg2rad(p.tip_angle)
    return MU_B*np.hypot(p.g_z*(B+p.B_tip*np.cos(phi)),p.g_t*p.B_tip*np.sin(phi))

def thermal_population(B,p,temperature):
    if temperature<=0:raise ValueError('Temperature must be positive')
    return float(1/(1+np.exp(electron_splitting(B,p)*K_PER_MHZ/temperature)))

def no_jump_probability(T,tau_c,p_exc,p_initial=0.):
    if T<0 or tau_c<=0 or not 0<=p_exc<=1 or not 0<=p_initial<=1:raise ValueError('Invalid rates')
    return (1-p_initial)*np.exp(-p_exc*T/tau_c)+p_initial*np.exp(-(1-p_exc)*T/tau_c)

def guaranteed_fidelity(T,tau_c,p_exc,coherent_error=0.,d=8,p_initial=0.):
    P=no_jump_probability(T,tau_c,p_exc,p_initial)
    return P*(1-coherent_error)+(1-P)/(d+1)

def maximum_excited_fraction(T,tau_c,total_error=.01,coherent_error=.001,d=8):
    """Sufficient bound for an initially prepared ground electron, not a necessity."""
    if T<=0 or tau_c<=0 or d<2 or not 0<=coherent_error<total_error<d/(d+1):raise ValueError('Invalid budget')
    P=(1-total_error-1/(d+1))/(1-coherent_error-1/(d+1))
    return float(min(1.,-tau_c/T*np.log(P)))

def polarization_temperature(B,p,p_exc):
    if not 0<p_exc<.5:raise ValueError('Require a polarized thermal two-state population')
    return electron_splitting(B,p)*K_PER_MHZ/np.log((1-p_exc)/p_exc)

def magnetic_dephasing_slope(I):
    d=int(2*I+1)
    return 2*d*I*(I+1)/(3*(d+1))

def motional_narrowing_time(T,A_z,p_exc,error=.009,I=3.5):
    """Leading fast-fluctuation estimate in a mean-frequency tracked frame.
    This is not the fidelity of the frozen-branch waveform in the fast regime.
    """
    return error/(magnetic_dephasing_slope(I)*(2*np.pi*A_z)**2*p_exc*(1-p_exc)*T)

def amplitude_error(I,epsilon,theta=np.pi/2,bb1=False):
    ix,iy,iz=spin(I)
    R=lambda ang,phi:expm(-1j*(1+epsilon)*ang*(np.cos(phi)*ix+np.sin(phi)*iy))
    G=expm(-1j*theta*iy)
    if not bb1:U=R(theta,np.pi/2)
    else:
        phi=np.arccos(-theta/(4*np.pi));a=np.pi/2
        U=R(np.pi,a+phi)@R(2*np.pi,a+3*phi)@R(np.pi,a+phi)@R(theta,a)
    return float(1-fidelity(U,G))
