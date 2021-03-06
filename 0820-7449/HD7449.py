import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model

#==============================================================================
# Import data 
#==============================================================================


x 		= np.loadtxt('MJD.dat')
idx     = x < 57100
x 		= x[idx]
y 		= np.loadtxt('RV_HARPS.dat')
y 		= (y[idx] - np.mean(y[idx])) * 1000
yerr 	= np.loadtxt('RV_noise.dat') #m/s
yerr 	= yerr[idx]

jitter_raw  = np.loadtxt('jitter_raw.txt')
jitter_raw	= jitter_raw[idx]
# jitter_smooth = np.loadtxt('jitter_smooth.txt')

# import time
# import os
# import shutil
# time0   = time.time()
# os.makedirs(str(time0))
# shutil.copy('HD85390-2_planet+jitter.py', str(time0)+'/HD85390-2_planet+jitter.py')  
# os.chdir(str(time0))

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel("RV [m/s]")
plt.xlabel("Time [d]")
plt.savefig('HD85390-1-RV.png')
# plt.show()


#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/5000
max_f   = 1
spp     = 10

frequency0, power0 = LombScargle(x, y, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(x, jitter_raw, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.axhline(y=0, color='k')
# ax.axvline(x=394, color='k')
# ax.axvline(x=843, color='k')
# ax.axvline(x=3442, color='k')
plt.plot(1/frequency0, power0, '-', label='HARPS', linewidth=2.0)
plt.plot(1/frequency1, power1, '--', label='Jitter')
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('HD7449-0-Periodogram.png')