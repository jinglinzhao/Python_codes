#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:25:27 2017

@author: jzhao
"""

'''
Implement GP into RV data
'''


#==============================================================================
# Import RV data 
#==============================================================================

from functions import read_rdb
import numpy as np
import matplotlib.pyplot as plt

DIR         = '/Volumes/DataSSD/SOAP_2/outputs/HERMIT_2spot_0720/'
file_name   = DIR + 'phase_rv.dat'
data_ccf    = read_rdb(file_name)
rv_ccf      = np.array(data_ccf['RV_tot']) * 1000   # in unit of m/s
rv_ccf      = rv_ccf - rv_ccf[0]                    # relative to the first rv

if 0: # test plot
    t = np.arange(len(rv_ccf))
    plt.plot(t, rv_ccf, '.')


#==============================================================================
# Inject a planet
#==============================================================================

from celerite.modeling import Model

class Model(Model):
    parameter_names = ("amp", "P", "phase", "offset")

    def get_value(self, t):
        return self.amp * np.sin(2*np.pi*t/self.P + self.phase) + self.offset
    
    
truth = dict(amp=0.005, P=25*0.31, phase=1, offset=0) 


#==============================================================================
# Import line profile
#==============================================================================

import os 
import glob
from astropy.io import fits
from scipy.interpolate import CubicSpline

os.chdir(DIR + 'fits')
FILE    = glob.glob('*fits')
N       = 3 * len(FILE)
v       = np.linspace(-20,20,401)
CCF     = np.zeros([401, N]) 
v_new   = np.linspace(-10,10,201)
CCF_new = np.zeros([201, N]) 

for n in range(N):
    i           = n % 25
    hdulist     = fits.open(FILE[i])
    CCF[:,n]    = hdulist[0].data
    v_planet    = Model(**truth).get_value(n)
    cs          = CubicSpline(v+v_planet, CCF[:,n])  
    CCF_new[:,n]= cs(v_new)
    
#plt.plot(np.arange(N), CCF_new[70, :])
plt.plot(v_new, CCF[100:301,n], v_new, CCF_new[:,n], '--', v_new, CCF[100:301,n]-CCF_new[:,n], '-.')
plt.plot(v_new[-50], CCF[301-50,n], 'ro')
plt.plot(v_new[-60], CCF[301-60,n], 'bo')
plt.plot(v_new[-70], CCF[301-70,n], 'mo')
plt.plot(v_new[-70], CCF[301-70,n]-CCF_new[-70,n], 'mo')
plt.plot(v_new[-60], CCF[301-60,n]-CCF_new[-60,n], 'bo')
plt.plot(v_new[-50], CCF[301-50,n]-CCF_new[-50,n], 'ro')
plt.ylabel("Normalized flux")
plt.xlabel("Wavelength [km/s]")
plt.legend(['static line profile', 'shifted line profile', 'variation'])

    
#==============================================================================
# Generate white noise
#==============================================================================

t       = np.arange(N)
y       = CCF_new[50, :]
'''
snr     = abs(np.random.randn() * 9e6 + 1e6)**0.5                               # random number between 1e6 and 1e7 in the bracket
delta_y = np.random.normal(0, y**0.5/ snr)
yerr    = y**0.5/ snr
'''
snr     = 3000
delta_y = np.random.normal(0, y**0.5/ snr)
yerr    = y**0.5/ snr

'''
for n in range(N):
    snr             = abs(np.random.randn() * 9e6 + 1e6)**0.5                   # random number between 1e6 and 1e7 in the bracket
    CCF_err[:,n]    = CCF_new[:,n]**0.5/ snr
    delta_CCF[:,n]  = np.random.normal(0, CCF_new[:,n]**0.5/ snr)
    

if 1:
    plt.errorbar(t, CCF_new[60,:] + delta_CCF[60,:], yerr=CCF_err[60,:], fmt=".k", capsize=0)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.title("simulated data");
'''

if 1:
    plt.errorbar(t, y+delta_y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(t, y+delta_y, color="b", alpha=0.5)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$ [days]")
    plt.title("Position 1");
    plt.show()



#==============================================================================
# Modelling
#==============================================================================

import celerite
celerite.__version__
from celerite import terms

class CustomTerm(terms.Term):
    parameter_names = ("amp", "P", "phase", "offset")
    
bounds = dict(amp = (0, 0.002), P = (3, 10), phase = (0, 2*np.pi), offset=(min(y), max(y)))
kernel1 = CustomTerm(amp = 0.001, P = 8, phase = 1, offset = np.median(y), bounds=bounds)
kernel2  = terms.SHOTerm(np.log(1.1), np.log(1.1), np.log(25))
kernel  = kernel1 * kernel2
#gp      = celerite.GP(kernel, mean=Model(amp=np.var(y)/2, P=7, phase=0))
gp  = celerite.GP(kernel, mean=Model(amp=(np.var(y))**0.5, P=7.75, phase=1,  offset = np.median(y)), fit_mean = True)     
#gp  = celerite.GP(kernel, mean=Model(**truth ), fit_mean = True)                            
#gp  = celerite.GP(kernel, mean=Model(**truth ), fit_mean = True, log_white_noise=np.log(1), fit_white_noise=True) 
gp.compute(t, yerr)   

def lnprob2(p):
    gp.set_parameter_vector(p)                                                  # Set the parameter values to the given vector
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()                    # Compute the logarithm of the marginalized likelihood of a set of observations under the Gaussian process model. 

#==============================================================================
# run MCMC on this model
#==============================================================================

import emcee
initial = gp.get_parameter_vector()                                             # Get an array of the parameter values in the correct order. len(initial) = 5. 
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

print("Running first burn-in...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 1000)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 1000);


'''
#==============================================================================
# plot the posterior samples on top of the data
#==============================================================================

# Plot the data.
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

# The positions where the prediction should be computed.
x = np.linspace(0, 74, 1001)

# Plot 24 posterior samples.
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=24)]:
    gp.set_parameter_vector(s)
    mu = gp.sample_conditional(y, x)
    plt.plot(x, mu, color="#4682b4", alpha=0.3)

plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.title("fit with GP noise model");
plt.show()
'''

#==============================================================================
# Plot the model
#==============================================================================

x_pred = np.linspace(0, 74, 1001)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color="b", alpha=0.2)
plt.plot(x_pred, pred, "b", lw=1.5, alpha=0.5)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("t")
plt.ylabel("y");
plt.title("fit with GP noise model 2");
plt.show()


#==============================================================================
# Corner plots
#==============================================================================
import corner
tri_cols = ["amp", "P", "phase"]
tri_labels = [r"$amplitude$", r"$Period$", r"$Phase$"]
tri_truths = [truth[k] for k in tri_cols]
#tri_range = [(-2, -0.01), (-3, -0.5), (-1, 1)]
names = gp.get_parameter_names()
# inds = np.array([names.index("mean:"+k) for k in tri_cols])
inds = [0,1,2]
corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels);
plt.show()




