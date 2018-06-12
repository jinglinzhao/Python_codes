#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Extract the CCF from the .fits file and save them for use

#############################################

import sys
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from math import sqrt

############# DEFINE FUNCTIONS ##############
def gaussian(x, a, b, c, d):
    val = a * np.exp(-(x - b)**2 / c**2) + d
    return val    

# Root mean square
def rms(num):
    return sqrt(sum(n*n for n in num)/len(num))
#############################################

FILE        = glob.glob('/Volumes/DataSSD/CoRot-7/CCF/*fits')
N           = len(FILE)
N_start     = 0
N_end       = N
n_file      = N_end - N_start
MJD         = np.zeros(n_file)
RV_g        = np.zeros(n_file)

for n in range(N):
    
    nidx   = n-N_start
    
    # progress bar
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*int((n+1-N_start)*50./(N_end-N_start)), int((n+1-N_start)*100./(N_end-N_start))))
    sys.stdout.flush()    

    # Verify the qualified spectra
    bad_spe     = 0                                                             # reset

    hdulist     = fits.open(FILE[n])
    contrast    = hdulist[0].header['HIERARCH ESO DRS CCF CONTRAST']
    if (contrast < 35):
        print('\n', n, '-- low contrast')
        bad_spe = 1
    
    noise = hdulist[0].header['HIERARCH ESO DRS CCF NOISE'] * 1000              # m/s
    if (noise > 10):                                                            # ******** TO BE EXPLORED ********
        print('\n', n, '-- large photon noise')
        bad_spe = 1

    bc          = hdulist[0].header['HIERARCH ESO DRS CCF RVC']                 # barycentric RV        
    v0          = hdulist[0].header['CRVAL1'] - bc                              # velocity on the left (N_starting point)
    if (v0 < -30) | (v0 > -10):                                                 # Use only the qualified data (e.g. get rid of lines off centre)
        print('\n', n, '-- line off centre')
        bad_spe = 1

    if bad_spe:
        continue

    CCF         = hdulist[0].data                                               # ccf 2-d array
    ccf         = CCF[- 1, :]                                                   # ccf 1-d array (whole range)
    delta_v     = hdulist[0].header['CDELT1']                                   # velocity grid size 
    v           = v0 + np.arange(CCF.shape[1]) * delta_v                        # velocity array (whole range)
    vidx        = (v<=10) & (v>-10)                                                # km/s    
    popt, pcov  = curve_fit( gaussian, v[vidx], 1 - ccf[vidx] / ccf[vidx].max() ) 
    ccf_new 	= ((1 - ccf[vidx] / ccf[vidx].max()) - popt[3]) / popt[0]

    RV_g[nidx]  = popt[1] * 1000                                                # m/s    
	MJD[nidx]   = hdulist[0].header['MJD-OBS']
    
	plt.plot(v[vidx], ccf_new)
	plt.title('CCF - CoRot-7')
	plt.xlabel('km/s')
	plt.ylabel('Normalized flux')

	np.savetxt(str(MJD[nidx]), np.vstack((v[vidx], ccf_new)))

plt.show()

    
    