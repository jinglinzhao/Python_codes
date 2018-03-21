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
# Import data 
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
# Tile the data and generate white noise
#==============================================================================

y      = np.tile(rv_ccf, 3)
N      = len(y)
t      = np.arange(N)
yerr   = 1 + np.random.rand(N)
y      = y + yerr

if 0:
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.title("simulated data");


#==============================================================================
# Inject a planet
#==============================================================================

from celerite.modeling import Model

class Model(Model):
    parameter_names = ("amp", "P", "phase")

    def get_value(self, t):
        return self.amp * np.sin(2*np.pi*t/self.P + self.phase)
    
    
truth = dict(amp=5, P=25*0.31, phase=0.1) 
y_planet = Model(**truth).get_value(t)
y        = y + y_planet

if 1:
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.title("simulated data");
    plt.show()


#==============================================================================
# Modelling
#==============================================================================

import celerite
celerite.__version__
from celerite import terms

kernel  = terms.SHOTerm(np.log(2), np.log(2), np.log(25))
#gp      = celerite.GP(kernel, mean=Model(amp=np.var(y)/2, P=7, phase=0))
gp  = celerite.GP(kernel, mean=Model(**truth ), fit_mean = True)                                 
#gp  = celerite.GP(kernel, mean=Model(**truth ), log_white_noise=np.log(1), fit_white_noise=True) 
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

plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="b", alpha=0.2)
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
#samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
#fig = corner.corner(samples)

tri_cols = ["amp", "P", "phase"]
tri_labels = [r"$amplitude$", r"$Period$", r"$Phase$"]
tri_truths = [truth[k] for k in tri_cols]
#tri_range = [(-2, -0.01), (-3, -0.5), (-1, 1)]
names = gp.get_parameter_names() 
inds = np.array([names.index("mean:"+k) for k in tri_cols])
#inds  e= [0,1,2]
corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels);
plt.show()




