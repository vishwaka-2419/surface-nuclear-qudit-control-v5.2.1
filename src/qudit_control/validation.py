"""Independent physical-waveform integration, resource metrics, and testing."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import solve_ivp
from .model import branches,common_frame_targets,fidelity


def envelope_function(problem,x):
    n=problem.ncoef;k=3;T=problem.T
    knots=np.r_[np.zeros(k+1),np.linspace(0,T,n-k+1)[1:-1],np.full(k+1,T)]
    coef=np.zeros((n,problem.J),complex)
    c=np.asarray(x).reshape(problem.K,problem.J,2)
    coef[2:-2]=problem.amax*(c[...,0]+1j*c[...,1])
    return BSpline(knots,coef,k,extrapolate=False)


def integrate_physical(problem,x,include_counterrotating=True,rtol=2e-10,atol=2e-12,max_step=.002,dense=False,scenario=None):
    """Solve the projected laboratory drive in the exact nominal interaction frame.

    This implementation does NOT use the optimizer's sinc/Magnus stepping or
    its cached tone phases. Both rotating terms and diagonal drive terms are
    included when include_counterrotating=True. Electronic inter-manifold
    leakage is not part of this projected validation.
    """
    d=problem.d;env=envelope_function(problem,x);T=problem.T
    delta=problem.E[:,:,None]-problem.E[:,None,:]
    D=problem.D
    drift=np.zeros_like(problem.E)
    gain=1.0
    if scenario is not None:
        bb,pp,gain=scenario
        actualE,D,_,_=branches(bb,pp)
        drift=actualE-problem.E
    freq=problem.freq
    def rhs(t,y):
        U=y.reshape(2,d,d)
        a=env(t)
        if include_counterrotating:
            wave=2*np.real(np.sum(a*np.exp(-2j*np.pi*freq*t)))
            H=gain*wave*D*np.exp(2j*np.pi*delta*t)
        else:
            # Direct RWA written independently for every matrix element.
            det=delta[:,:,:,None]-np.sign(delta)[:,:,:,None]*freq
            c=np.where(delta[:,:,:,None]>=0,a,a.conj())
            H=gain*D*np.sum(c*np.exp(2j*np.pi*det*t),axis=-1)
            idx=np.arange(d);H[:,idx,idx]=0
        idx=np.arange(d);H[:,idx,idx]+=drift
        return (-2j*np.pi*H@U).ravel()
    y0=np.tile(np.eye(d,dtype=complex),(2,1,1)).ravel()
    sol=solve_ivp(rhs,(0,T),y0,method='DOP853',rtol=rtol,atol=atol,max_step=max_step,dense_output=dense)
    if not sol.success: raise RuntimeError(sol.message)
    U=sol.y[:,-1].reshape(2,d,d)
    W=common_frame_targets(problem.E,T,problem.G) if problem.common_frame else np.tile(problem.G,(2,1,1))
    info={'fidelities':fidelity(U,W).tolist(),'worst':float(np.min(fidelity(U,W))),
          'unitarity_error':float(np.max(np.linalg.norm(U.conj().swapaxes(-1,-2)@U-np.eye(d),axis=(-2,-1)))),
          'nfev':sol.nfev,'rtol':rtol,'atol':atol,'max_step_us':max_step,
          'counterrotating':include_counterrotating}
    return U,info,sol


def resource_metrics(problem,x):
    # 1 ns sampling resolves carriers up to ~100 MHz and is independent of the optimizer mesh.
    dt=.001; t=np.arange(0,problem.T+dt/2,dt)
    env=envelope_function(problem,x);a=env(t)
    da=env.derivative()(t)
    wave=2*np.real(np.sum(a*np.exp(-2j*np.pi*t[:,None]*problem.freq),axis=1))
    # Spectral bandwidth of the baseband envelopes, not an assertion of a sharp cutoff.
    nt=2**int(np.ceil(np.log2(len(t)*4)))
    spec=np.abs(np.fft.fft(a,n=nt,axis=0))**2
    f=np.fft.fftfreq(nt,dt)
    order=np.argsort(np.abs(f));pw=np.sum(spec,axis=1)[order]
    cdf=np.cumsum(pw)/np.sum(pw)
    b99=float(np.abs(f)[order][np.searchsorted(cdf,.99)])
    return {'peak_tone_MHz':float(np.max(np.abs(a))),
            'peak_rss_MHz':float(np.max(np.sqrt(np.sum(np.abs(a)**2,axis=1)))),
            'sum_bound_peak_MHz':float(2*np.max(np.sum(np.abs(a),axis=1))),
            'sampled_real_waveform_peak_MHz':float(np.max(np.abs(wave))),
            'real_waveform_rms_MHz':float(np.sqrt(np.mean(wave**2))),
            'fluence_MHz2_us':float(np.trapezoid(np.sum(np.abs(a)**2,axis=1),t)),
            'peak_envelope_slew_MHz_per_us':float(np.max(np.abs(da))),
            'envelope_99pct_power_halfband_MHz':b99,
            'min_carrier_MHz':float(np.min(problem.freq)),
            'max_carrier_MHz':float(np.max(problem.freq)),
            'equivalent_single_tone_B_peak_mT':float(2*np.max(np.abs(a))/problem.p.gamma*1000)}
