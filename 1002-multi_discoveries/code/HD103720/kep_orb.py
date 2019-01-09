import numpy as np
from rv import solve_kep_eqn

def kep_orb(t, P, tau, k, w, e0, offset):

    M_anom  = 2*np.pi/P * (t.flatten() - tau)
    e_anom  = solve_kep_eqn(M_anom, e0)
    f       = 2*np.arctan( np.sqrt((1+e0)/(1-e0))*np.tan(e_anom*.5) )
    rv      = k * (np.cos(f + w) + e0*np.cos(w))
    return rv + offset