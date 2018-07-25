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

# all_rvs 	= np.genfromtxt('all_rvs_1outlier_removed.dat', dtype = None)
all_rvs     = np.genfromtxt('all_rvs_N102_outlier_and_5MJ_removed.dat', dtype=None)

DATA_AAT 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'AAT']
DATA_CHIRON = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'CHIRON']
DATA_FEROS 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FEROS']
DATA_MJ1 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ1']
DATA_MJ3 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ3']
DATA_FIDEOS = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FIDEOS']


RV_AAT 		= np.zeros( (len(DATA_AAT), 3) )
RV_CHIRON 	= np.zeros( (len(DATA_CHIRON), 3) )
RV_FEROS 	= np.zeros( (len(DATA_FEROS), 3) )
RV_MJ1 		= np.zeros( (len(DATA_MJ1), 3) )
RV_MJ3 		= np.zeros( (len(DATA_MJ3), 3) )
RV_FIDEOS   = np.zeros( (len(DATA_FIDEOS), 3) )


for k in range(len(DATA_AAT)):
	RV_AAT[k, :] 	= [ DATA_AAT[k][i] for i in range(3) ]

for k in range(len(DATA_CHIRON)):
	RV_CHIRON[k, :]	= [ DATA_CHIRON[k][i] for i in range(3) ]

for k in range(len(DATA_FEROS)):
	RV_FEROS[k, :]	= [ DATA_FEROS[k][i] for i in range(3) ]

for k in range(len(DATA_MJ1)):
	RV_MJ1[k, :]	= [ DATA_MJ1[k][i] for i in range(3) ]

for k in range(len(DATA_MJ3)):
	RV_MJ3[k, :]	= [ DATA_MJ3[k][i] for i in range(3) ]

for k in range(len(DATA_FIDEOS)):
    RV_FIDEOS[k, :] = [ DATA_FIDEOS[k][i] for i in range(3) ]    


# Concatenate the five data sets # 
RV_ALL  = np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3, RV_FIDEOS))
RV_SORT = sorted(RV_ALL, key=lambda x: x[0])
t       = np.array([RV_SORT[i][0] for i in range(len(RV_SORT))])
y       = np.array([RV_SORT[i][1] for i in range(len(RV_SORT))])
yerr    = np.array([RV_SORT[i][2] for i in range(len(RV_SORT))])
yerr    = (yerr**2 + 20**2)**0.5 

#==============================================================================
# Gaussian Processes
#==============================================================================

import celerite
celerite.__version__
from celerite.modeling import Model
from celerite import terms
from rv import solve_kep_eqn


class Model(Model):
    parameter_names = ('log_P', 'log_tau', 'log_k', 'w', 'e0', 'off_aat', 'off_chiron', 'off_feros', 'off_mj1', 'off_mj3', 'off_fideos')

    def get_value(self, t):
        M_anom  = 2*np.pi/np.exp(self.log_P) * (t.flatten() - np.exp(self.log_tau))
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = np.exp(self.log_k)*(np.cos(f + self.w) + self.e0*np.cos(self.w))

        offset = np.zeros(len(t))
        for i in range(len(t)):
            if t[i] in RV_AAT[:,0]:
                offset[i] = self.off_aat
            elif t[i] in RV_CHIRON[:,0]:
                offset[i] = self.off_chiron
            elif t[i] in RV_FEROS[:,0]:
                offset[i] = self.off_feros           
            elif t[i] in RV_MJ1[:,0]:
                offset[i] = self.off_mj1
            elif t[i] in RV_MJ3[:,0]:
                offset[i] = self.off_mj3             
            elif t[i] in RV_FIDEOS[:,0]:
                offset[i] = self.off_fideos

        return rv + 100*offset


# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
OFFSET_AAT      = 3.0/100
OFFSET_CHIRON   = -70./100
OFFSET_FEROS    = -8.4/100
OFFSET_MJ1      = -12.8/100
OFFSET_MJ3      = -54.4/100
OFFSET_FIDEOS   = -83.2/100

truth   = dict(log_P=np.log(415.9), log_tau=np.log(4812), log_k=np.log(186.8), w=-0.06, e0=0.856,
                off_aat=OFFSET_AAT, off_chiron=OFFSET_CHIRON, off_feros=OFFSET_FEROS, 
                off_mj1=OFFSET_MJ1, off_mj3=OFFSET_MJ3, off_fideos=OFFSET_FIDEOS)


kernel  = terms.SHOTerm(log_S0=np.log(2), log_Q=np.log(2), log_omega0=np.log(5))
kernel.freeze_parameter("log_Q")

# mean: An object (following the modeling protocol) that specifies the mean function of the GP.
gp  = celerite.GP(kernel, mean=Model(**truth), fit_mean = True)

# compute(x, yerr=0.0, **kwargs). Pre-compute the covariance matrix and factorize it for a set of times and uncertainties.
gp.compute(t, yerr)                                                             



#==============================================================================
# log likelihood
#==============================================================================

def lnprob2(p):
    
    # Trivial uniform prior.
    _, _, P, tau, k, w, e0, off_aat, off_chiron, off_feros, off_mj1, off_mj3, off_fideos = p
    if (5.8 < P < 6.1) and (4.6 < k < 5.7) and (-np.pi < w < np.pi) and (0.7 < e0 < 0.99):
        return 0.0
    return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()



#==============================================================================
# run MCMC on this model
#==============================================================================

import emcee

# Get an array of the parameter values in the correct order. len(initial) = 5. 
initial = gp.get_parameter_vector()                                             
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 1000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
# sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 1000)
# sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 2000);

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# plot the posterior samples on top of the data
#==============================================================================
# Plot the data.

plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

# The positions where the prediction should be computed.
x = np.linspace(min(RV_ALL[:,0]), max(RV_ALL[:,0]), num=1000, endpoint=True)

# Plot 24 posterior samples.
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=24)]:
    gp.set_parameter_vector(s)
    mu = gp.sample_conditional(y, x)
    plt.plot(x, mu, color="#4682b4", alpha=0.3)

plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
#plt.xlim(-5, 5)
plt.title("fit with GP noise model");
plt.show()


x = np.linspace(min(RV_ALL[:,0]), max(RV_ALL[:,0]), num=10000, endpoint=True)
pred_mean, pred_var = gp.predict(y, x, return_var=True)
pred_std = np.sqrt(pred_var)

color = "#ff7f0e"
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, pred_mean, color=color)
plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,
                 edgecolor="none")
plt.show()


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples         = sampler.chain[:, :, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,3:6] = np.exp(real_samples[:,3:6])
real_samples[:,-6:] = real_samples[:,-6:]*100

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=['1', '2', '3', r"$\log\ P$", r"$\log\ T_{0}$", r"$\log\ K$", r"$\omega$", r"$e$", 
            r"$\frac{\delta_{AAT}}{100}$", r"$\frac{\delta_{CHIRON}}{100}$", 
            r"$\frac{\delta_{FEROS}}{100}$", r"$\frac{\delta_{MJ1}}{100}$", 
            r"$\frac{\delta_{MJ3}}{100}$", r"$\frac{\delta_{FIDEOS}}{100}$"]
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
labels=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", r"$\delta_{AAT}$", r"$\delta_{CHIRON}$", 
            r"$\delta_{FEROS}$", r"$\delta_{MJ1}$", r"$\delta_{MJ3}$", r"$\delta_{FIDEOS}$"]
fig = corner.corner(real_samples[:,3:], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
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


