import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing
from rv import solve_kep_eqn
import os

#==============================================================================
# Import data 
#==============================================================================
star    = 'HD128621'
print('*'*len(star))
print(star)
print('*'*len(star))

plt.rcParams.update({'font.size': 14})

DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
t 		= np.loadtxt(DIR + '/MJD.dat') + 0.5
XX      = np.loadtxt(DIR + '/RV_HARPS.dat')
XX      = XX * 1000
yerr 	= np.loadtxt(DIR + '/RV_noise.dat') #m/s
FWHM    = np.loadtxt(DIR + '/FWHM.dat')

YY  = np.loadtxt('../data/'+star+'/YY.txt')
ZZ  = np.loadtxt('../data/'+star+'/ZZ.txt')

XY 	= XX - YY
ZX 	= ZZ - XX

os.chdir('../output/'+star)

# Filter the unwanted data #
if 1: # Valid for data HD128621_2_2010-03-22..2010-06-12[PART]
    # idx  = ~((XY>3) | ((t>55340) & (XX>0)))
    # the following does the job equally good 
    idx = (FWHM<6.3) &  (FWHM>6.24)
if 0: # valid for part 3 [part] (i.e. 2011-02-18..2011-05-15)
    idx = ~ ((XY>1.5) | (XX>30) | ((t>55658) & (t<55659) & (XX>5)) | ((t>55679) & (t<55680) & (XX>-10)) | ((t>55692) & (t<55693) & (XY<0)) | (FWHM<6.22))
if 0: # valid for part 1 [part]
    idx = (FWHM>6.23) & (t<54975)


plt.figure()
plt.errorbar(t[idx], XY[idx], yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label=r'$\Delta RV_L$')
plt.errorbar(t[~idx], XY[~idx], yerr=yerr[~idx], fmt=".r", capsize=0, alpha=0.2, label='outlier')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV1.png')
# plt.show()

plt.figure()
plt.errorbar(t[idx], ZX[idx], yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label=r'$\Delta RV_H$')
plt.errorbar(t[~idx], ZX[~idx], yerr=yerr[~idx], fmt=".r", capsize=0, alpha=0.2, label='outlier')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV2.png')
# plt.show()    


#==============================================================================
# GP 
#==============================================================================

t   = t[idx]
# y   = XY[idx]
y   = ZX[idx]
y   = y - np.mean(y)
yerr = yerr[idx] * 2**0.5

from george import kernels

# k1 = 1**2 * kernels.ExpSquaredKernel(metric=10**2)
# k2 = 1**2 * kernels.ExpSquaredKernel(80**2) * kernels.ExpSine2Kernel(gamma=8, log_period=np.log(36.2))
# boundary doesn't seem to take effect
k2 = 1**2 * kernels.ExpSquaredKernel(80**2) * kernels.ExpSine2Kernel(gamma=11, log_period=np.log(36.2),
                            bounds=dict(gamma=(-3,30), log_period=(np.log(36.2-5),np.log(36.2+6))))
# k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
# k4 = 1**2 * kernels.ExpSquaredKernel(40**2)
# kernel = k1 + k2 + k3 + k4
kernel = k2

import george
# gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
# gp.freeze_parameter('kernel:k2:log_period')
# gp.freeze_parameter('kernel:k2:gamma')

#==============================================================================
# Optimization 
#==============================================================================
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
print(np.exp(results.x[4]))



#==============================================================================
# MCMC
#==============================================================================

# Define the objective function (negative log-likelihood in this case).
def lnprob(p):
    # if np.any((results.x-0.5*abs(results.x) > p) + (p > results.x+0.5*abs(results.x))):
    #     return -np.inf
    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)    


import emcee

initial = results.x
# initial = gp.get_parameter_vector()
names = gp.get_parameter_names()
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))