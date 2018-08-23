import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# Import data 
#==============================================================================

jitter_raw  = np.loadtxt('jitter_raw.txt')
x 		= np.loadtxt('MJD.dat')
y 		= np.loadtxt('RV_HARPS.dat')
y 		= (y - np.mean(y)) * 1000
yerr 	= np.loadtxt('RV_noise.dat') #m/s

#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/500
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
plt.plot(1/frequency0, power0, '-', label='HARPS', linewidth=2.0)
plt.plot(1/frequency1, power1, '--', label='Jitter_H')
# ax.axvline(x=39.6, color='k')
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('Gl581-0-Periodogram.png')
plt.show()