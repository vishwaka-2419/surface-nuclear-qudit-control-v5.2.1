"""Frequency-unit spin Hamiltonians. MHz times microseconds are cycles.

Nuclear Zeeman convention: H_N/h = gamma_coeff * B * I_z, where
 gamma_coeff = -g_N * mu_N/h > 0 for a negative nuclear magnetic g factor.
The electronic effective g tensor is positive in this convention.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

MU_B = 13996.24555  # MHz/T; retained for comparison with submitted code
MU_N = 7.622593285


def spin(j: float):
    if j < 0 or not np.isclose(2*j, round(2*j)):
        raise ValueError("Spin must be a nonnegative integer or half-integer")
    m = np.arange(j, -j-0.5, -1.)
    p = np.zeros((len(m), len(m)), complex)
    for c in range(1, len(m)):
        p[c-1,c] = np.sqrt(j*(j+1)-m[c]*(m[c]+1))
    return (p+p.T.conj())/2, (p-p.T.conj())/(2j), np.diag(m)


@dataclass(frozen=True)
class Parameters:
    I: float = 3.5
    A_z: float = 130.0
    A_t: float = 10.0
    kappa: float = -46.3738410596
    gamma: float = MU_N*0.3146857142857143
    g_z: float = 0.56
    g_t: float = 0.56
    B_tip: float = 0.0679
    tip_angle: float = 5.0
    eta: float = 0.0
    efg_tilt: float = 0.0


def hamiltonian(B: float, p: Parameters):
    if p.I<1 or p.g_z<=0:
        raise ValueError("Quadrupolar branch model requires I>=1 and positive effective electron g_z")
    ix,iy,iz = spin(p.I); sx,sy,sz = spin(.5)
    d=len(ix); i2=np.eye(2); idd=np.eye(d)
    S=[np.kron(a,idd) for a in (sx,sy,sz)]
    N=[np.kron(i2,a) for a in (ix,iy,iz)]
    phi=np.deg2rad(p.tip_angle)
    alpha=np.deg2rad(p.efg_tilt)
    xp=np.cos(alpha)*ix-np.sin(alpha)*iz
    zp=np.sin(alpha)*ix+np.cos(alpha)*iz
    q=p.kappa/(4*p.I*(2*p.I-1))*(3*zp@zp-p.I*(p.I+1)*idd+p.eta*(xp@xp-iy@iy))
    H=MU_B*(p.g_z*(B+p.B_tip*np.cos(phi))*S[2]+p.g_t*p.B_tip*np.sin(phi)*S[0])
    H=H+p.gamma*B*N[2]+p.A_z*S[2]@N[2]+p.A_t*(S[0]@N[0]+S[1]@N[1])+np.kron(i2,q)
    return H, S, N


def branches(B: float, p: Parameters):
    """Adiabatic labels via maximum-overlap assignment to |s,m>.

    Returned order is ground (s=-1/2), excited (s=+1/2), each with m descending.
    Reject a strongly mixed assignment rather than silently relabeling levels.
    """
    H,S,N=hamiltonian(B,p)
    e,v=np.linalg.eigh(H)
    rows,cols=linear_sum_assignment(-np.abs(v)**2)
    order=cols[np.argsort(rows)]; v=v[:,order]; e=e[order]
    phase=np.angle(np.diag(v)); v=v*np.exp(-1j*phase)[None,:]
    d=int(2*p.I+1)
    indices=[np.arange(d,2*d),np.arange(d)]
    es=[]; ds=[]; vs=[]; purity=[]
    for ind in indices:
        w=v[:,ind]
        es.append(e[ind]-np.mean(e[ind]))
        ds.append(w.conj().T@N[0]@w)
        vs.append(w)
        purity.append(float(np.min(np.abs(v[ind,ind])**2)))
    if min(purity)<0.9:
        raise ValueError("Product-state branch purity below 0.9; use a full joint electron-nuclear model")
    return np.array(es),np.array(ds),np.array(vs),np.array(purity)


def metrics(B: float, p: Parameters):
    E,D,V,purity=branches(B,p)
    f=np.abs(np.diff(E,axis=1))
    inter=np.min(np.abs(f[0,:,None]-f[1,None,:]))
    intra=min(np.min(np.diff(np.sort(x))) for x in f)
    return float(inter),float(intra),float(np.min(purity))


def target(I: float, kind: str='rotation'):
    ix,iy,iz=spin(I); d=len(ix)
    if kind=='rotation':
        return expm(-.5j*np.pi*iy)
    if kind=='selective':
        h=np.zeros((d,d),complex); h[0,1]=-1j/2; h[1,0]=1j/2
        return expm(-.5j*np.pi*h)
    if kind=='quadratic':
        q=iz@iz-np.trace(iz@iz)/d*np.eye(d)
        return expm(-1j*np.pi/4*q)
    raise ValueError(f'Unknown target {kind}')


def common_frame_targets(E: np.ndarray, T: float, G: np.ndarray):
    """Same logical gate in ONE common frame, not separate electron frames.

    Physical desired nuclear unitary = exp[-i 2pi E_common T] G.
    Its targets in nominal conditional interaction frames include the different
    conditional drift phases. Overall electron-branch phases are immaterial for
    the explicitly classical electron state assumed here.
    """
    ecommon=np.mean(E,axis=0)
    return np.exp(2j*np.pi*(E-ecommon)*T)[:,:,None]*G[None,:,:]


def fidelity(U: np.ndarray, V: np.ndarray):
    d=U.shape[-1]
    z=np.einsum('...ij,...ij->...',V.conj(),U)
    return (d+np.abs(z)**2)/(d*(d+1))


def finite_comb_distance(B: float, p: Parameters) -> float:
    """Axial strong-coupling finite-comb result, not a global speed limit."""
    spacing=3*abs(p.kappa)/(2*p.I*(2*p.I-1))
    ell=np.arange(int(-2*p.I+1),int(2*p.I))
    return float(np.min(np.abs(ell*spacing-2*p.gamma*B)))
