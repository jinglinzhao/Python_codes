#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Based on Demo_76920_celerite.py
Introduce individual offsets 
'''

import matplotlib.pyplot as plt


#==============================================================================
# Import data 
#==============================================================================
import numpy as np
from functions import gaussian_smoothing

BJD         = np.loadtxt('MJD.dat')
BJD         = BJD - min(BJD)
BJD         = BJD[:-1]
RV_HARPS    = np.loadtxt('RV_HARPS.dat')
RV_HARPS    = (RV_HARPS - np.mean(RV_HARPS))*1000
RV_HARPS    = RV_HARPS[:-1]
Jitter      = np.loadtxt('RV_jitter.txt')
Jitter      = Jitter[:-1]
RV_noise    = np.loadtxt('RV_noise.dat')
RV_noise    = RV_noise[:-1]
weight      = 1 / RV_noise**2
t_resample = np.linspace(min(BJD), max(BJD), 10000)
Jitter2     = gaussian_smoothing(BJD, Jitter, t_resample, 1.5, weight)


plt.errorbar(BJD, RV_HARPS, yerr=RV_noise, fmt=".", capsize=0)
# plt.plot(BJD, Jitter, '.')
# plt.plot(t_resample, Jitter2, '-')
plt.show()



#==============================================================================
# Periodogram
#==============================================================================
from astropy.stats import LombScargle

min_f   = 0.01
max_f   = 5
spp     = 10

frequency0, power0 = LombScargle(BJD, RV_HARPS, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency1, power1 = LombScargle(BJD, Jitter, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

ax = plt.subplot(111)
ax.set_xscale('log')
# ax.axhline(y=0, color='k')
ax.axvline(x=14.275, color='k')
plt.plot(1/frequency0, power0, '-', label='RV_HARPS')
plt.plot(1/frequency1, power1, '-.', label='Jitter')
# plt.xlim([0, 25])
plt.legend()
plt.show()


#==============================================================================
# Model
#==============================================================================

from rv import solve_kep_eqn
import celerite
from celerite.modeling import Model


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'offset', 'alpha')

    def get_value(self, t):
        M_anom  = 2*np.pi/np.exp(self.P) * (t.flatten() - np.exp(self.tau))
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = np.exp(self.k)*(np.cos(f + self.w) + self.e0*np.cos(self.w))

        return rv + 100*self.offset + self.alpha * Jitter


#==============================================================================
# log likelihood
#==============================================================================

def lnprior(theta):
    P, tau, k, w, e0, offset, alpha = theta
    if (0 < P < 5) and (0 < k < 5) and (-2*np.pi < w < 2*np.pi) and (0. < e0 < 0.8):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P, tau, k, w, e0, offset, alpha = theta
    model = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset=offset, alpha=alpha)
    return -0.5*(np.sum( ((y-model.get_value(np.array(x)))/yerr)**2. ))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    

#==============================================================================
# run MCMC on this model
#==============================================================================

import emcee
ndim = 7
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(BJD, RV_HARPS, RV_noise), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = [3, 1, 3, 0, 0.2, 0, 1] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 1000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
# sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 1000)
# sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 2000);

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples         = sampler.chain[:, 1000:, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,0:3] = np.exp(real_samples[:,0:3])
real_samples[:,-1:] = real_samples[:,-1:]*100

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$\log\ P$", r"$\log\ T_{0}$", r"$K$", r"$\omega$", r"$e$", 
            r"$\frac{\delta}{100}$", r"$\alpha$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
# plt.savefig('76920_MCMC_6sets-2-Trace.png')
plt.show()


import corner
labels=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", r"$\delta$", r"$\alpha$"]
fig = corner.corner(real_samples[:,:], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.show()



samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
a0, a1, a2, a3, a4, a5, a6, a7, a8 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

aa = np.zeros((6,3))
aa[0,:] = [a3[i] for i in range(3)]
aa[1,:] = [a4[i] for i in range(3)]
aa[2,:] = [a5[i] for i in range(3)]
aa[3,:] = [a6[i] for i in range(3)]
aa[4,:] = [a7[i] for i in range(3)]
aa[5,:] = [a8[i] for i in range(3)]

np.savetxt('fit_parameter.txt', aa, fmt='%.6f')



if 1:
    plt.errorbar(RV_AAT[:,0],   RV_AAT[:,1],    yerr=RV_AAT[:,2],   fmt=".", capsize=0, label='AAT')
    plt.errorbar(RV_CHIRON[:,0],RV_CHIRON[:,1], yerr=RV_CHIRON[:,2],fmt=".", capsize=0, label='CHIRON')
    plt.errorbar(RV_FEROS[:,0], RV_FEROS[:,1],  yerr=RV_FEROS[:,2], fmt=".", capsize=0, label='FEROS')
    plt.errorbar(RV_MJ1[:,0],   RV_MJ1[:,1],    yerr=RV_MJ1[:,2],   fmt=".", capsize=0, label='MJ1')
    plt.errorbar(RV_MJ3[:,0],   RV_MJ3[:,1],    yerr=RV_MJ3[:,2],   fmt=".", capsize=0, label='MJ3')
    plt.ylabel(r"$RV [m/s]$")
    plt.xlabel(r"$JD$")
    plt.title("Adjusted RV time series")
    plt.legend()
    plt.show()


