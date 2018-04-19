#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Based on Demo_george.py and implemented the data of HD76920.
Change to celerite
'''

#==============================================================================
# Simulated Dataset
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model

'''
class Model(Model):
    parameter_names = ('n', 'tau', 'k', 'w', 'e0', 'offset')

    def get_value(self, t):
         e_anom = solve_kep_eqn(self.n*(t.flatten()-self.tau), self.e0)
         f 		= 2*np.arctan2(np.sqrt(1+self.e0)*np.sin(e_anom*.5),np.sqrt(1-self.e0)*np.cos(e_anom*.5))
         return self.k*(np.cos(f + self.w) + self.e0*np.cos(self.w)) + self.offset
'''


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'offset')

    def get_value(self, t):
        M_anom  = 2*np.pi/self.P * (t.flatten() - self.tau)
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        # print(self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        return self.k*(np.cos(f + self.w) + self.e0*np.cos(self.w)) + self.offset

# bounds = dict(P=(350.,450.), tau=(4000.,5000.), k=(100.,300.), w=(-2*np.pi, 2*np.pi), e0=(0.8, 0.95), offset=(-100.,100.))


# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
#truth = dict(amp=-2.0, location=0.1, log_sigma2=np.log(0.4))             
# might consider using log scale
truth 	= dict(P=415.4, tau=4867, k=186.8, w=-0.06, e0=0.856, offset=0)        



#==============================================================================
# Import data 
#==============================================================================

# all_rvs 	= np.genfromtxt('all_rvs.dat', dtype = None)
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
	RV_CHIRON[k, 1] = RV_CHIRON[k, 1] - OFFSET_CHIRON

for k in range(len(DATA_FEROS)):
	RV_FEROS[k, :]	= [ DATA_FEROS[k][i] for i in range(3) ]
	RV_FEROS[k, 1] 	= RV_FEROS[k, 1] - OFFSET_FEROS

for k in range(len(DATA_MJ1)):
	RV_MJ1[k, :]	= [ DATA_MJ1[k][i] for i in range(3) ]
	RV_MJ1[k, 1] 	= RV_MJ1[k, 1] - OFFSET_MJ1

for k in range(len(DATA_MJ3)):
	RV_MJ3[k, :]	= [ DATA_MJ3[k][i] for i in range(3) ]
	RV_MJ3[k, 1] 	= RV_MJ3[k, 1] - OFFSET_MJ3


if 1:
    plt.errorbar(RV_AAT[:,0], 	RV_AAT[:,1], 	yerr=RV_AAT[:,2], 	fmt=".", capsize=0, label='AAT')
    plt.errorbar(RV_CHIRON[:,0],RV_CHIRON[:,1], yerr=RV_CHIRON[:,2],fmt=".", capsize=0, label='CHIRON')
    plt.errorbar(RV_FEROS[:,0], RV_FEROS[:,1], 	yerr=RV_FEROS[:,2], fmt=".", capsize=0, label='FEROS')
    plt.errorbar(RV_MJ1[:,0], 	RV_MJ1[:,1], 	yerr=RV_MJ1[:,2], 	fmt=".", capsize=0, label='MJ1')
    plt.errorbar(RV_MJ3[:,0], 	RV_MJ3[:,1], 	yerr=RV_MJ3[:,2], 	fmt=".", capsize=0, label='MJ3')
    plt.ylabel(r"$RV [m/s]$")
    plt.xlabel(r"$JD$")
    plt.title("Adjusted RV time series")
    plt.legend()
    # plt.show()


# Concatenate the five data sets # 
RV_ALL  = np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3))
RV_SORT = sorted(RV_ALL, key=lambda x: x[0])
t       = [RV_SORT[i][0] for i in range(len(RV_SORT))]
y       = [RV_SORT[i][1] for i in range(len(RV_SORT))]
yerr    = [RV_SORT[i][2] for i in range(len(RV_SORT))]
#yerr    = [(yerr[i]**2 + 7**2)**0.5 for i in range(len(RV_SORT))]

#==============================================================================
# Modelling correlated noise
#==============================================================================

import celerite
celerite.__version__
from celerite import terms

# bounds = dict(P=(350,400), k=(100,300), w=(-2*np.pi, 2*np.pi), e0=(0.8, 0.95), offset=(-100,100))
# bounds = dict(log_S0 = (0,2), log_Q=(0,2), log_omega0=(0,2), P=(350,400), k=(100,300), w=(-2*np.pi, 2*np.pi), e0=(0.8, 0.95), offset=(-100,100))
# bounds[3] = (350,400)
# bounds[4] = (0, 10000)
# bounds[5] = (100,300)
# bounds[6] = (-2*np.pi, 2*np.pi)
# bounds[7] = (0.8, 0.95)
# bounds[8] = (-100,100)

kernel  = terms.SHOTerm(np.log(2), np.log(2), np.log(5))
# kernel  = terms.SHOTerm(np.log(2), np.log(2), np.log(5), 415.4, 4867, 186.8, 0, 0.856, 0, bounds=bounds)
# mean: An object (following the modeling protocol) that specifies the mean function of the GP.
gp  = celerite.GP(kernel, mean=Model(**truth), fit_mean = True)

# compute(x, yerr=0.0, **kwargs). Pre-compute the covariance matrix and factorize it for a set of times and uncertainties.
gp.compute(t, yerr)                                                             





####################################################################

if 0:

    from scipy.optimize import minimize

    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    initial_params = gp.get_parameter_vector()
    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))

####################################################################


#==============================================================================
# Test  1
#==============================================================================

if 0: 

    kwargs = dict(**truth)
    kwargs["bounds"] = dict(location=(-2, 2))
    mean_model = Model(**kwargs)
    gp = george.GP(np.var(y) * kernels.Matern32Kernel(10.0), mean=mean_model)
    gp.compute(t, yerr)
    #####

    def lnprob2(p):
        
        print(p)
        # Set the parameter values to the given vector
        gp.set_parameter_vector(p)                                                  

        # Compute the logarithm of the marginalized likelihood of a set of observations under the Gaussian process model. 
        return gp.log_likelihood(y, quiet=True) + gp.log_prior()                    



#==============================================================================
# Test 2
#==============================================================================

def lnprob2(p):
    
    # Trivial uniform prior.
    if ((0 > p[7]) or (p[7] > 0.99) or (p[3]>450) or (p[3]<350) or (p[5]<100) or (p[5]>300) or (p[6]<-np.pi) or (p[6]>np.pi)):
        print(p)
        return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()

##################


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
p0, lp, _ = sampler.run_mcmc(p0, 1000)
# p0, lp, _ = sampler.run_mcmc(p0, 2000)


print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 1000)
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

tri_cols = ['P', 'tau', 'k', 'w', 'e0', 'offset']
tri_labels = ['P', 'tau', 'k', 'w', 'e0', 'offset']
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






