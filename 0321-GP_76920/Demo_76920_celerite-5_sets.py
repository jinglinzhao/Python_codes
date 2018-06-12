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

all_rvs 	= np.genfromtxt('all_rvs_1outlier_removed.dat', dtype = None)

DATA_AAT 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'AAT']
DATA_CHIRON = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'CHIRON']
DATA_FEROS 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FEROS']
DATA_MJ1 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ1']
DATA_MJ3 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ3']


#==============================================================================
# apply the offset
#==============================================================================

OFFSET_CHIRON   = -73.098470
OFFSET_FEROS	= -6.5227172
OFFSET_MJ1 		= -14.925970
OFFSET_MJ3 		= -56.943472

RV_AAT 		= np.zeros( (len(DATA_AAT), 3) )
RV_CHIRON 	= np.zeros( (len(DATA_CHIRON), 3) )
RV_FEROS 	= np.zeros( (len(DATA_FEROS), 3) )
RV_MJ1 		= np.zeros( (len(DATA_MJ1), 3) )
RV_MJ3 		= np.zeros( (len(DATA_MJ3), 3) )


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


# Concatenate the five data sets # 
RV_ALL  = np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3))
RV_SORT = sorted(RV_ALL, key=lambda x: x[0])
t       = [RV_SORT[i][0] for i in range(len(RV_SORT))]
y       = [RV_SORT[i][1] for i in range(len(RV_SORT))]
yerr    = [RV_SORT[i][2] for i in range(len(RV_SORT))]


#==============================================================================
# Gaussian Processes
#==============================================================================

import celerite
celerite.__version__
from celerite.modeling import Model
from celerite import terms
from rv import solve_kep_eqn


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'off_aat', 'off_chiron', 'off_feros', 'off_mj1', 'off_mj3')

    def get_value(self, t):
        M_anom  = 2*np.pi/np.exp(self.P) * (t.flatten() - np.exp(self.tau))
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = np.exp(self.k)*(np.cos(f + self.w) + self.e0*np.cos(self.w))

        offset = np.zeros(len(t))
        for i in range(len(t)):
            if t[i] in RV_AAT[:,0]:
                offset[i] = self.off_aat
            if t[i] in RV_CHIRON[:,0]:
                offset[i] = self.off_chiron
            if t[i] in RV_FEROS[:,0]:
                offset[i] = self.off_feros           
            if t[i] in RV_MJ1[:,0]:
                offset[i] = self.off_mj1
            if t[i] in RV_MJ3[:,0]:
                offset[i] = self.off_mj3             

        return rv + 100*offset


# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
truth   = dict(log_P=np.log(415.9), log_tau=np.log(4812), log_k=np.log(186.8), w=-0.06, e0=0.856,
                off_aat=0, off_chiron=0, off_feros=0, off_mj1=0, off_mj3=0)

kernel  = terms.SHOTerm(np.log(2), np.log(2), np.log(5))

# mean: An object (following the modeling protocol) that specifies the mean function of the GP.
gp  = celerite.GP(kernel, mean=Model(**truth), fit_mean = True)

# compute(x, yerr=0.0, **kwargs). Pre-compute the covariance matrix and factorize it for a set of times and uncertainties.
gp.compute(t, yerr)                                                             



#==============================================================================
# log likelihood
#==============================================================================

def lnprob2(p):
    
    # Trivial uniform prior.
    if ((p[3]>450) or (p[3]<350) or (p[5]<100) or (p[5]>300) or (p[6]<-np.pi) or (p[6]>np.pi) or (p[7] < 0.7) or (p[7] > 0.99) or (p[8] < -30) or (p[8] > 30)
        or (p[9] < -30) or (p[9] > 30) or (p[10] < -30) or (p[10] > 30) or (p[11] < -30) or (p[11] > 30) or (p[12] < -30) or (p[12] > 30)):
        print(p)
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
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

print("Running first burn-in...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 500)
# p0, lp, _ = sampler.run_mcmc(p0, 2000)


print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 500)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 1000);



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
# Corner plots
#==============================================================================
import corner

tri_cols = ['P', 'tau', 'k', 'w', 'e0', 'off_aat', 'off_chiron', 'off_feros', 'off_mj1', 'off_mj3']
tri_labels = ['P', 'tau', 'k', 'w', 'e0', 'OFFSET_AAT', 'OFFSET_CHIRON', 'OFFSET_FEROS', 'OFFSET_MJ1', 'OFFSET_MJ3']
tri_truths = [truth[k] for k in tri_cols]
names = gp.get_parameter_names()
inds = np.array([names.index("mean:"+k) for k in tri_cols])
corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels)
plt.savefig('corner.png')
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


