import numpy as np
import matplotlib.pyplot as plt
# from celerite.modeling import Model

#==============================================================================
# Import data 
#==============================================================================

jitter_raw_H  = np.loadtxt('jitter_raw_H.txt')
jitter_raw_L  = np.loadtxt('jitter_raw_L.txt')
jitter_raw_HL  = np.loadtxt('jitter_raw_HL.txt')

x 		= np.loadtxt('MJD.dat')
idx     = x < 57300
x 		= x[idx]
y 		= np.loadtxt('RV_HARPS.dat')
y 		= (y[idx] - np.mean(y[idx])) * 1000
yerr 	= np.loadtxt('RV_noise.dat') #m/s
yerr 	= yerr[idx]



#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/1000
max_f   = 1
spp     = 10

frequency0, power0 = LombScargle(x, y, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(x, jitter_raw_H, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

frequency2, power2 = LombScargle(x, jitter_raw_L, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

frequency3, power3 = LombScargle(x, jitter_raw_HL, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.axhline(y=0, color='k')
plt.plot(1/frequency0, power0, '-', label='HARPS', linewidth=2.0)
plt.plot(1/frequency1, power1, '--', label='Jitter_H')
plt.plot(1/frequency2, power2, '--', label='Jitter_L')
plt.plot(1/frequency3, power3, '--', label='Jitter_HL')
ax.axvline(x=39.6, color='k')
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('Gl176-0-Periodogram.png')
plt.show()