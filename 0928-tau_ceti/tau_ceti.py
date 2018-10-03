import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing

#==============================================================================
# Import data 
#==============================================================================

x 		= np.loadtxt('MJD.dat')
y 		= np.loadtxt('RV_HARPS.dat')
y 		= (y - np.mean(y)) * 1000
yerr 	= np.loadtxt('RV_noise.dat') #m/s


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
plt.errorbar(x, y-YY, yerr=yerr, fmt=".r", capsize=0)
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
# plt.savefig('HD10700-1-RV.png')
plt.show()

# 54099

# 54422: 4 day period?



# XX  = np.loadtxt('XX.txt')
YY  = np.loadtxt('YY.txt')
ZZ  = np.loadtxt('ZZ.txt')

sl      = 0.5         # smoothing length
XY      = gaussian_smoothing(x, y-YY, x, sl)
XX 	 	= gaussian_smoothing(x, y, x, sl)
plt.figure()
# plt.errorbar(x, XX, yerr=yerr, fmt=".k", capsize=0)
# plt.errorbar(x, XY, yerr=yerr, fmt=".r", capsize=0)
plt.plot(x, XX, '.k')
plt.plot(x, XY, '.r')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
# plt.savefig('HD10700-1-RV.png')
plt.show()

# idx = (x > 53500) & (x < 53800)
idx = (x > 54250) & (x < 54500)

x = x[idx]
XX = XX[idx]
XY = XY[idx]
yerr = yerr[idx]

#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/100
max_f   = 1
spp     = 10

frequency0, power0 = LombScargle(x, XX, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(x, XY, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

plt.figure()
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axhline(y=0, color='k')
# ax.axvline(x=394, color='k')
# ax.axvline(x=843, color='k')
# ax.axvline(x=3442, color='k')
plt.plot(1/frequency0, power0, 'b-', label='HARPS', linewidth=2.0)
plt.plot(1/frequency1, power1, 'r--', label='Jitter')
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
# plt.savefig('HD10700-0-Periodogram.png')
plt.show()