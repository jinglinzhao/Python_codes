#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Based on 76920_MCMC_sets.py
'''


#==============================================================================
# Import data 
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt


BJD         = np.loadtxt('MJD_2012.txt')
RV_HARPS    = np.loadtxt('RV_HARPS_2012.txt')
RV_HARPS_err= np.loadtxt('RV_noise_2012.txt')
GP_y_2012   = np.loadtxt('GP_y_2012.txt')
GP_err_2012 = np.loadtxt('GP_err_2012.txt')
Jitter      = np.loadtxt('Jitter_model_2012.txt')
Jitter_err  = np.loadtxt('RV_noise_2012.txt')
Jitter_smooth = np.loadtxt('jitter_smooth_2012.txt')


plt.figure()
plt.errorbar(BJD, RV_HARPS, yerr=RV_HARPS_err, fmt=".", capsize=0, label='HARPS')
plt.errorbar(BJD, GP_y_2012, yerr=GP_err_2012, fmt=".", capsize=0, label='Scaled jitter')
plt.errorbar(BJD, Jitter, yerr=Jitter_err, fmt=".", capsize=0, label='RAW jitter')
plt.legend()
plt.show()


x       = BJD
y       = RV_HARPS
yerr    = Jitter_err

#==============================================================================
# Model
#==============================================================================

from rv import solve_kep_eqn
from celerite.modeling import Model


class Model(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'm', 'offset')

    def get_value(self, t):

        # Planet 1
        M_anom1 = 2*np.pi/np.exp(self.P1) * (t - np.exp(self.tau1))
        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1     = np.exp(self.k1)*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))
        
        # Planet 2
        M_anom2 = 2*np.pi/np.exp(self.P2) * (t - np.exp(self.tau2))
        e_anom2 = solve_kep_eqn(M_anom2, self.e2)
        f2      = 2*np.arctan( np.sqrt((1+self.e2)/(1-self.e2))*np.tan(e_anom2*.5) )
        rv2     = np.exp(self.k2)*(np.cos(f2 + self.w2) + self.e2*np.cos(self.w2))

        # correct jitter by a factor 'm'
        rv_j    = np.zeros(len(t))
        for i in range(len(t)):
            if t[i] in BJD:
                rv_j = self.m * Jitter_smooth[i]

        return rv1 + rv2 + rv_j + self.offset


# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
truth   = dict(log_P1=np.log(0.853), log_tau1=np.log(1), log_k1=np.log(3.42), w1=0., e1=0.12,
                log_P2=np.log(3.70), log_tau2=np.log(1), log_k2=np.log(6.01), w2=0., e2=0.12,
                m=2., offset=0.)

#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, m, offset = theta
    if (-10 < m < 10) and (0. < e1 < 1.) and (0. < e2 < 1.) and (-np.pi < w1 < np.pi) and (-np.pi < w2 < np.pi):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, m, offset = theta
    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
                        P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, m=m, offset=offset)
    y_fit       = fit_curve.get_value(np.array(x))
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2. ))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    


import emcee
ndim = len(truth)
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
pos = [[val for key, val in truth.items()] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
pos, prob, state  = sampler.run_mcmc(pos, 3000)

print("Running second burn-in...")
pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
pos, prob, state  = sampler.run_mcmc(pos, 5000)

print("Running production...")
sampler.run_mcmc(pos, 5000);


time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


import copy
log_samples         = sampler.chain[:, :, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,0:3] = np.exp(real_samples[:,0:3])


import corner
labels=[r"$P_{1}$", r"$\tau_{1}$", r"$k_{1}$", r"$\omega_{1}$", r"$e_{1}$", 
        r"$P_{2}$", r"$\tau_{2}$", r"$k_{2}$", r"$\omega_{2}$", r"$e_{2}$", r"$m$", "offset"]
fig = corner.corner(log_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
# plt.show()
plt.savefig('CoRot-7_MCMC-1-Corner.png')


#==============================================================================
# Trace
#==============================================================================

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$\log\ P_{1}$", r"$\log\ \tau_{1}$", r"$\log\ k_{1}$", r"$\omega_{1}$", r"$e_{1}$", 
            r"$\log\ P_{2}$", r"$\log\ \tau_{2}$", r"$\log\ k_{2}$", r"$\omega_{2}$", r"$e_{2}$", 
            r"$m$", "offset"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
# plt.savefig('CoRot-7_MCMC-2-Trace.png')
plt.show()


a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((len(truth),3))
aa[0,:] = [a0[i] for i in range(3)]
aa[1,:] = [a1[i] for i in range(3)]
aa[2,:] = [a2[i] for i in range(3)]
aa[3,:] = [a3[i] for i in range(3)]
aa[4,:] = [a4[i] for i in range(3)]
aa[5,:] = [a5[i] for i in range(3)]
aa[6,:] = [a6[i] for i in range(3)]
aa[7,:] = [a7[i] for i in range(3)]
aa[8,:] = [a8[i] for i in range(3)]
aa[9,:] = [a9[i] for i in range(3)]
aa[10,:] = [a10[i] for i in range(3)]
aa[11,:] = [a11[i] for i in range(3)]
np.savetxt('76920_MCMC_5sets_result.txt', aa, fmt='%.6f')

residual = fit_curve.get_value(np.array(x)) - np.array(y)
chi2 = sum(residual**2 / np.array(yerr)**2)
rms = np.sqrt(np.mean(residual**2))


if 0: # I need to add the offset into the data before plotting
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, m, offset = aa[:,0]
    fit_curve = Model(P1=np.log(P1), tau1=np.log(tau1), k1=np.log(k1), w1=w1, e1=e1, 
                        P2=np.log(P2), tau2=np.log(tau2), k2=np.log(k2), w2=w2, e2=e2, 
                        m=m, offset=offset)
    y_fit   = fit_curve.get_value(x)
    plt.figure()
    plt.plot(x, y_fit, 'o', label='MCMC fit')
    plt.errorbar(BJD, RV_HARPS, yerr=RV_HARPS_err, fmt=".", capsize=0, label='HARPS')
    plt.title("RV time series")
    plt.ylabel("RV [m/s]")
    plt.xlabel("Shifted JD [d]")
    plt.legend()
    plt.savefig('CoRot-7_MCMC-3-fit.png')
    # plt.show()




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


plt.show()







