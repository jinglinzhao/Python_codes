#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==============================================================================
# Import data 
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt

star        = 'HD216770'
DIR         = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
MJD1        = np.loadtxt(DIR + '/K5/MJD.dat')
RV_HARPS1   = np.loadtxt(DIR + '/K5/RV_HARPS.dat') * 1000
RV_noise1   = np.loadtxt(DIR + '/K5/RV_noise.dat')

MJD2        = np.loadtxt(DIR + '/G2/MJD.dat')
RV_HARPS2   = np.loadtxt(DIR + '/G2/RV_HARPS.dat') * 1000
RV_noise2   = np.loadtxt(DIR + '/G2/RV_noise.dat')

if 0:
    plt.figure()
    plt.errorbar(MJD1, RV_HARPS1, yerr=RV_noise1, fmt=".r", capsize=0, alpha=0.5, label='HARPS K5')
    plt.errorbar(MJD2, RV_HARPS2, yerr=RV_noise2, fmt=".b", capsize=0, alpha=0.5, label='HARPS G2')
    plt.ylabel("RV [m/s]")
    plt.xlabel("MJD [days]")
    plt.legend()
    plt.savefig('../../output/'+star+'/'+star+'-0-RV.png')
    plt.show()

# convert to x, y, yerr
x       = np.hstack((MJD1, MJD2))
y       = np.hstack((RV_HARPS1, RV_HARPS2))
y       = y - np.mean(y)
yerr    = np.hstack((RV_noise1, RV_noise2))


#==============================================================================
# Periodogram
#==============================================================================
if 0:

    from astropy.stats import LombScargle

    min_f   = 0.005
    max_f   = 5
    spp     = 20  # spp=1000 will take a while; for quick results set spp = 10

    frequency0, power0 = LombScargle(MJD1, RV_HARPS1, RV_noise1).autopower(minimum_frequency=min_f,
                                                       maximum_frequency=max_f,
                                                       samples_per_peak=spp)

    frequency1, power1 = LombScargle(MJD2, RV_HARPS2, RV_noise2).autopower(minimum_frequency=min_f,
                                                       maximum_frequency=max_f,
                                                       samples_per_peak=spp)


    ax = plt.subplot(111)
    plt.plot(1/frequency0, power0, label='K5', ls='-', alpha=0.5)
    plt.plot(1/frequency1, power1, label='G2', ls='--', alpha=0.5)
    plt.xlim(0, 1/min_f)
    plt.xlabel("Period")
    plt.ylabel("Power")
    plt.legend()
    plt.savefig('../../output/'+star+'/'+star+'-Periodogram.png')
    plt.show()

#==============================================================================
# Model
#==============================================================================

import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model
import math


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'offset1', 'offset2')

    def get_value(self, t):
        M_anom  = 2*np.pi/self.P * (t - self.tau)
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = self.k * (np.cos(f + self.w) + self.e0*np.cos(self.w))

        offset = np.zeros(len(t))
        for i in range(len(t)):
            if t[i]<57161:
                offset[i] = self.offset1
            else:
                offset[i] = self.offset2

        return rv + offset

# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
P       = 118.
tau     = 0.
e       = 0.37
offset1  = 0.
offset2  = 0.
k       = 31
w       = 0

guess   = dict(P=P, tau=tau, k=k, w=w, e0=e, offset1=offset1, offset2=offset2)


#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P, tau, k, w, e0, offset1, offset2 = theta
    if (0 < P) and (0 < k) and (-np.pi < w < np.pi) and (0. < e0 < 0.5):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P, tau, k, w, e0, offset1, offset2 = theta
    fit_curve   = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset1=offset1, offset2=offset2)
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
initial = [P, tau, k, w, e, offset1, offset2]

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 2000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 2000)

# print("Running fourth burn-in...")
# p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
# p0, _, _ = sampler.run_mcmc(p0, 2000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 1000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))




#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples         = sampler.chain[:, 6000:, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", 
            "offset1", "offset2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('../../output/'+star+'/'+star+'_MCMC_-Trace.png')
plt.show()


import corner
labels=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", "offset1", "offset2"]
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('../../output/'+star+'/'+star+'_MCMC-Corner.png')
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
np.savetxt('../../output/'+star+'/'+star+'_MCMC_result.txt', aa, fmt='%.6f')


P, tau, k, w, e0, offset1, offset2 = aa[:,0]
fit_curve = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset1=offset1, offset2=offset2)
y_fit   = fit_curve.get_value(np.array(x))


plt.figure()
plt.errorbar(x,   y, yerr=yerr,   fmt=".", capsize=0, label='HARPS')
plt.errorbar(x, y_fit, yerr=yerr,   fmt=".", capsize=0,  label='MCMC fit')
plt.ylabel("RV [m/s]")
plt.xlabel("MJD")
plt.legend(loc="upper center")
plt.savefig('../../output/'+star+'/'+star+'_MCMC_fit.png')
plt.show()

companion   = fit_curve.get_value(np.array(x))
residual    = np.array(y) - companion
chi2        = sum(residual**2 / np.array(yerr)**2) / (len(x)-len(guess))
rms         = np.sqrt(np.mean(residual**2))
wrms        = np.sqrt(sum((residual/yerr)**2)/sum(1/yerr**2))


BI = fit_curve.get_value(x[x<57161])
np.savetxt('/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/'+star+'/BI.txt', BI-np.mean(BI), fmt='%.6f')


# plot residuals 
plt.figure()
plt.errorbar(x, residual, yerr=yerr, fmt=".", capsize=0, label='HARPS', alpha=0.5)
plt.ylabel("Residual [m/s]")
plt.xlabel("MJD [day]")
plt.legend(loc="upper center")
plt.savefig('../../output/'+star+'/'+star+'_residual.png')
plt.show()


# Phase folded plot # 

plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(12,8))
frame1  = fig.add_axes((.15, .35, .8, .55))

phase       = x/P - [int(x[i]/P) for i in range(len(x))]
phase_x     = np.hstack((phase, phase+1))
phase_y     = np.zeros(len(x))
for i in range(len(x)):
    if x[i] < 57161:
        phase_y[i] = y[i] - offset1
    else:
        phase_y[i] = y[i] - offset2
phase_y     = np.hstack((phase_y, phase_y))
phase_err   = np.hstack((yerr, yerr))
idx = (phase_y == phase_y[-1]) | (phase_y == phase_y[-2])
plt.errorbar(phase_x[~idx], phase_y[~idx], yerr=phase_err[~idx], fmt="ko", capsize=0, label='MJD<57161', alpha=0.5)
plt.errorbar(phase_x[idx], phase_y[idx], yerr=phase_err[idx], fmt="ks", capsize=0, label='MJD>57161', alpha=0.5)
plt.ylabel("RV [m/s]")
# plt.title('HD 36051')


# Plot model #
class PlotModel(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0',)

    def get_value(self, t):
        M_anom  = 2*np.pi/self.P * (t - self.tau)
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = self.k * (np.cos(f + self.w) + self.e0*np.cos(self.w))

        return rv

plot_x      = np.linspace(0, 2, num=1001)
curve       = PlotModel(P=P, tau=tau, k=k, w=w, e0=e0)
plot_y      = curve.get_value(np.array(plot_x*P))
plt.plot(plot_x, plot_y, alpha=0.5)
plt.legend()
frame1.set_xticklabels([])

res = phase_y - curve.get_value(np.array(phase_x * P))
np.sqrt(np.mean(res**2))

frame2  = fig.add_axes((.15, .15, .8, .2))   
frame2.axhline(color="gray", ls='--')
plt.errorbar(phase_x[~idx], res[~idx], yerr=phase_err[~idx], fmt="ko", capsize=0, label='MJD<57161', alpha=0.5)
plt.errorbar(phase_x[idx], res[idx], yerr=phase_err[idx], fmt="ks", capsize=0, label='MJD>57161', alpha=0.5)
plt.xlabel("Phase")
plt.ylabel("Residual [m/s]")
plt.ylim(-19.5,19.5)
plt.savefig('../../output/'+star+'/'+star+'_fit.png')
plt.show()
plt.close('all')
# np.savetxt('/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/BI.txt', companion)




