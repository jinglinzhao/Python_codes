#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==============================================================================
# Import data 
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt

star 	= 'HD103720'

DIR1    = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
MJD     = np.loadtxt(DIR1 + '/MJD.dat')
MJD0 	= min(MJD) 	# MJD zero point
RV_noise= np.loadtxt(DIR1 + '/RV_noise.dat')

DIR2 	= '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
GG 		= np.loadtxt(DIR2 + 'GG.txt')
YY 		= np.loadtxt(DIR2 + 'YY.txt')

# convert to t, y, yerr
t       = MJD - MJD0
y       = GG - YY
yerr    = RV_noise

if 0: # plot the jitter time series
    plt.errorbar(t, y, yerr=yerr, fmt="o", capsize=0, alpha=0.5)
    plt.ylabel('RV [m/s]')
    plt.xlabel('time [day]')
    plt.show()


#==============================================================================
# GP 
#==============================================================================

from george import kernels

k1 = 10 * kernels.ExpSquaredKernel(metric=10**2)
# k2 = 1**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=1, log_period=1.9)
k2 = 20 * kernels.ExpSquaredKernel(200**2) * kernels.ExpSine2Kernel(gamma=2, log_period=np.log(17),
                            bounds=dict(gamma=(-3,30), log_period=(np.log(17-5),np.log(17+5))))
k3 = 20 * kernels.RationalQuadraticKernel(log_alpha=np.log(100), metric=120**2)
k4 = 10 * kernels.ExpSquaredKernel(1000**2)
# kernel = k1 + k2 + k3 + k4
kernel =  k3 + k2 + k4
import george
#gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)

gp.compute(t, yerr)


#==============================================================================
# Optimization 
#==============================================================================
if 0: 

	import scipy.optimize as op

	# Define the objective function (negative log-likelihood in this case).
	def nll(p):
	    gp.set_parameter_vector(p)
	    ll = gp.log_likelihood(y, quiet=True)
	    return -ll if np.isfinite(ll) else 1e25

	# And the gradient of the objective function.
	def grad_nll(p):
	    gp.set_parameter_vector(p)
	    return -gp.grad_log_likelihood(y, quiet=True)

	# You need to compute the GP once before starting the optimization.
	gp.compute(t, yerr)

	# Print the initial ln-likelihood.
	print(gp.log_likelihood(y))

	# Run the optimization routine.
	p0 = gp.get_parameter_vector()
	results = op.minimize(nll, p0, jac=grad_nll, method="Newton-CG")

	# Update the kernel and print the final log-likelihood.
	gp.set_parameter_vector(results.x)
	print(gp.log_likelihood(y))

	# print the rotation period 
	print(np.exp(results.x[6]))    


#==============================================================================
# MCMC
#==============================================================================

initial = gp.get_parameter_vector()

# Define the objective function (negative log-likelihood in this case).
def lnprob(p):
    if np.any((initial-0.1*abs(initial) > p) + (p > initial+0.1*abs(initial))):
        return -np.inf
    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)    


import emcee

initial = gp.get_parameter_vector()
names = gp.get_parameter_names()
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-3 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))



#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, 8000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)

fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=names
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('HD103720-2-Trace.png')
plt.show()

import corner
fig = corner.corner(real_samples, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('HD103720-3-Corner.png')
plt.show()


#==============================================================================
# Output
#==============================================================================
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = map(lambda v: 
    (v[1], v[0], v[2], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
aa = np.zeros((len(gp),5))
aa[0,:] = [a0[i] for i in range(5)]
aa[1,:] = [a1[i] for i in range(5)]
aa[2,:] = [a2[i] for i in range(5)]
aa[3,:] = [a3[i] for i in range(5)]
aa[4,:] = [a4[i] for i in range(5)]
aa[5,:] = [a5[i] for i in range(5)]
aa[6,:] = [a6[i] for i in range(5)]
aa[7,:] = [a7[i] for i in range(5)]
aa[8,:] = [a8[i] for i in range(5)]
aa[9,:] = [a9[i] for i in range(5)]
np.savetxt('HD103720.txt', aa, fmt='%.6f')


#==============================================================================
# Plots
#==============================================================================

gp.set_parameter_vector(aa[:,0])

# Make the maximum likelihood prediction
x = np.linspace(min(t), max(t), 20000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)
gp_predict = np.transpose(np.vstack((x,mu,std)))

# x: oversampled time (column 1)
# mu: Gaussian processes prediction of the most likely value (column 2)
# std: standard deivation of walkers in all runs in MCMC (column 3)
color = "#ff7f0e"
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, mu, color=color)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.savefig('HD103720-prediction-4-MCMC.png') 
plt.show()


# Estimated jitter
mu, var 	= gp.predict(y, t, return_var=True)
std 		= np.sqrt(var)
gp_predict 	= np.transpose(np.vstack((t+MJD0,mu,std)))
np.savetxt('gp_predict.txt', gp_predict, fmt='%.8f')
if 1:
    plt.errorbar(t, y, yerr=yerr, fmt="o", capsize=0, alpha=0.5, label='jitter')
    plt.errorbar(t, mu, yerr=std, fmt="o", capsize=0, alpha=0.5, label='jitter_gp')
    plt.ylabel('RV [m/s]')
    plt.xlabel('time [day]')
    plt.legend()
    plt.show()


