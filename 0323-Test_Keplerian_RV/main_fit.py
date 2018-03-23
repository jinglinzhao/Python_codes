import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit,fsolve
from rv import *
from rv_fit import * 



# t = time of measurement
t = np.arange(2000)

# n = angular frequency of planet
T = 80.
n = 1/T

# tau = time of pericenter passage
tau = 10

# k = amplitude of radial velocity (depends on planet mass and eccentricity)
k = -20

# w = related to the argument of pericenter by a shift of pi.
w = 3.1415

# e = eccentricity of orbit
e = 0.86

'''
The radial velocity at time t is given by
vr = k*(cos(f + w)+e*cos(w)),
where f is related to the number of periods since pericenter passage, n*(t-tau)
'''
y = fitting_function(t,n,tau,k,w,e)


# Plot the data using the star.plot function
plt.close('all')
plt.plot(t, y, '-')
plt.ylabel('RV' r"$[m/s]$")
plt.xlabel('t')
plt.title('Simulated RV')
plt.show()