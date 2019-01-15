#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==============================================================================
# Import data 
#==============================================================================
import numpy as np

star    = 'HD103720'
DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
MJD     = np.loadtxt(DIR + '/MJD.dat')
RV_HARPS= np.loadtxt(DIR + '/RV_HARPS.dat') * 1000
RV_noise= np.loadtxt(DIR + '/RV_noise.dat')

DIR2    = '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
GG      = np.loadtxt(DIR2 + 'GG.txt')
YY      = np.loadtxt(DIR2 + 'YY.txt')
jitter  = GG - YY

# convert to x, y, yerr
x       = MJD
y       = RV_HARPS - np.mean(RV_HARPS)
yerr    = RV_noise


#==============================================================================
# Model
#==============================================================================

import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model
import math


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'offset', 'm')

    def get_value(self, t):
        M_anom  = 2*np.pi/self.P * (t - self.tau)
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = self.k * (np.cos(f + self.w) + self.e0*np.cos(self.w))

        return rv + self.offset + self.m * jitter

# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
P       = 4.5557
tau     = 3.
e       = 0.086
offset  = 0.
k       = 89
w       = 262 / 360 * 2 * np.pi
m       = 5

guess   = dict(P=P, tau=tau, k=k, w=w, e0=e, offset=offset, m=m)


#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P, tau, k, w, e0, offset, m = theta
    if (4.4 < P < 4.7) and (80 < k < 100) and (0 < w < 2*np.pi) and (0. < e0 < 0.15) and (1 < m < 8):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P, tau, k, w, e0, offset, m = theta
    fit_curve   = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset=offset, m=m)
    y_fit       = fit_curve.get_value(np.array(x))
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2. ))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    


import emcee
ndim = len(guess)
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)
initial = [P, tau, k, w, e, offset, m] 

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 2000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 2000)

print("Running fourth burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 2000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples     = sampler.chain[:, 9000:, :].reshape((-1, ndim))
real_samples    = copy.copy(log_samples)


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", "offset", r"$m$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('../../output/HD103720/103720pj_MCMC_-Trace.png')
plt.show()


import corner
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('../../output/HD103720/103720pj_MCMC-Corner.png')
plt.show()


#==============================================================================
# Output
#==============================================================================

a0, a1, a2, a3, a4, a5, a6 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((len(guess),3))
aa[0,:] = [a0[i] for i in range(3)]
aa[1,:] = [a1[i] for i in range(3)]
aa[2,:] = [a2[i] for i in range(3)]
aa[3,:] = [a3[i] for i in range(3)]
aa[4,:] = [a4[i] for i in range(3)]
aa[5,:] = [a5[i] for i in range(3)]
aa[6,:] = [a6[i] for i in range(3)]
np.savetxt('../../103720pj_MCMC_result.txt', aa, fmt='%.6f')


P, tau, k, w, e0, offset, m = aa[:,0]
fit_curve = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset=offset, m=m)
y_fit   = fit_curve.get_value(np.array(x))


plt.figure()
plt.errorbar(x, y_fit, yerr=yerr, fmt=".", capsize=0, label='MCMC fit')
plt.errorbar(x,   y,   yerr=yerr,   fmt=".", capsize=0, label='HARPS')
plt.ylabel("RV [m/s]")
plt.xlabel("MJD")
plt.legend(loc="upper center")
plt.savefig('../../output/HD103720/103720pj_MCMC_fit.png')
plt.show()

residual    = np.array(y) - fit_curve.get_value(np.array(x))
chi2        = sum(residual**2 / np.array(yerr)**2) / (len(x)-len(guess))
rms         = np.sqrt(np.mean(residual**2))
wrms        = np.sqrt(sum((residual/yerr)**2)/sum(1/yerr**2))


# plot residuals 
plt.figure()
plt.errorbar(x, residual, yerr=yerr, fmt=".", capsize=0, label='HARPS', alpha=0.5)
plt.ylabel("Residual [m/s]")
plt.xlabel("MJD [day]")
plt.legend(loc="upper center")
plt.savefig('../../output/HD103720/103720pj_residual.png')
plt.show()

if 1:

    #==============================================================================
    # Periodogram
    #==============================================================================

    from astropy.stats import LombScargle

    min_f   = 0.01
    max_f   = 5
    spp     = 1000  # spp=1000 will take a while; for quick results set spp = 10

    frequency0, power0 = LombScargle(x, y, yerr).autopower(minimum_frequency=min_f,
                                                       maximum_frequency=max_f,
                                                       samples_per_peak=spp)

    frequency1, power1 = LombScargle(x, jitter, yerr).autopower(minimum_frequency=min_f,
                                                       maximum_frequency=max_f,
                                                       samples_per_peak=spp)


    ax = plt.subplot(111)
    ax.axvline(x=17, color='k', ls='-.')
    plt.plot(1/frequency0, power0, label='HARPS RV', ls='-', alpha=0.5)
    plt.plot(1/frequency1, power1, label='jitter', ls='--', alpha=0.5)
    plt.xlim(0, 30)
    plt.xlabel("Period")
    plt.ylabel("Power")
    plt.legend()
    plt.savefig('../../output/HD103720/103720-Periodogram.png')
    plt.show()












