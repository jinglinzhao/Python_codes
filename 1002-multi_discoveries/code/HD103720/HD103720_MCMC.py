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
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'offset')

    def get_value(self, t):
        M_anom  = 2*np.pi/self.P * (t - self.tau)
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = self.k * (np.cos(f + self.w) + self.e0*np.cos(self.w))

        return rv + self.offset

# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
P       = 4.5557
tau     = 3.24
e       = 0.073
offset  = 0.
k       = 89.5
w       = 262 / 360 * 2 * np.pi

guess   = dict(P=P, tau=tau, k=k, w=w, e0=e, offset=offset)


#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P, tau, k, w, e0, offset = theta
    if (4.4 < P < 4.7) and (80 < k < 100) and (0 < w < 2*np.pi) and (0. < e0 < 0.15):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P, tau, k, w, e0, offset = theta
    fit_curve   = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset=offset)
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
initial = [P, tau, k, w, e, offset] 

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 1000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 1000)

print("Running fourth burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 1000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 2000);    


time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))




#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples         = sampler.chain[:, 5000:, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", 
            "offset"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('../../output/HD103720/103720_MCMC_-Trace.png')
plt.show()


import corner
labels=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", "offset"]
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('../../output/HD103720/103720_MCMC-Corner.png')
plt.show()


#==============================================================================
# Output
#==============================================================================

a0, a1, a2, a3, a4, a5 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((len(guess),3))
aa[0,:] = [a0[i] for i in range(3)]
aa[1,:] = [a1[i] for i in range(3)]
aa[2,:] = [a2[i] for i in range(3)]
aa[3,:] = [a3[i] for i in range(3)]
aa[4,:] = [a4[i] for i in range(3)]
aa[5,:] = [a5[i] for i in range(3)]
np.savetxt('../../output/HD103720/103720_MCMC_result.txt', aa, fmt='%.6f')


P, tau, k, w, e0, offset = aa[:,0]
fit_curve = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset=offset)
t_fit   = np.linspace(min(x)-20, max(x), num=10001, endpoint=True)
y_fit   = fit_curve.get_value(np.array(t_fit))


plt.figure()
plt.plot(t_fit, y_fit, label='MCMC fit')
plt.errorbar(x,   y,    yerr=yerr,   fmt=".", capsize=0, label='HARPS')
plt.ylabel("RV [m/s]")
plt.xlabel("MJD")
plt.legend(loc="upper center")
plt.savefig('../../output/HD103720/103720_MCMC_fit.png')
plt.show()

companion   = fit_curve.get_value(np.array(x))
residual    = np.array(y) - companion
chi2        = sum(residual**2 / np.array(yerr)**2) / (len(x)-len(guess))
rms         = np.sqrt(np.mean(residual**2))
wrms        = np.sqrt(sum((residual/yerr)**2)/sum(1/yerr**2))



# plot residuals 
plt.figure()
plt.errorbar(x, residual, yerr=yerr, fmt=".", capsize=0, label='HARPS', alpha=0.5)
plt.ylabel("Residual [m/s]")
plt.xlabel("MJD [day]")
plt.legend(loc="upper center")
plt.savefig('../../output/HD103720/103720_residual.png')
plt.show()

np.savetxt('/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/BI.txt', companion)


if 0:
    inds = np.random.randint(len(log_samples), size=100)
    plt.figure()
    for ind in inds:
        sample = log_samples[ind]
        fit_curve = Model(P=sample[0], tau=sample[1], k=sample[2], w=sample[3], e0=sample[4], off_aat=sample[5],
                     off_chiron=sample[6], off_feros=sample[7], off_mj1=sample[8], off_mj3=sample[9])
        y_fit   = fit_curve.get_value(np.array(t_fit))
        plt.plot(t_fit, y_fit, "g", alpha=0.1)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel("RV [m/s]")
    plt.xlabel("Shifted JD [d]")
    plt.savefig('76920_MCMC_5sets-4-MCMC_100_realizations.png')
    # plt.close()

