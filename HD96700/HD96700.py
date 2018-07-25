
import matplotlib.pyplot as plt


#==============================================================================
# Import data 
#==============================================================================
import numpy as np
from functions import gaussian_smoothing

BJD         = np.loadtxt('MJD.dat')
BJD         = BJD - min(BJD)
# BJD         = BJD[:-1]
RV_HARPS    = np.loadtxt('RV_HARPS.dat')
RV_HARPS    = (RV_HARPS - np.mean(RV_HARPS))*1000
# RV_HARPS    = RV_HARPS[:-1]
Jitter      = np.loadtxt('RV_jitter.txt')
# Jitter      = Jitter[:-1]
RV_noise    = np.loadtxt('RV_noise.dat')
# RV_noise    = RV_noise[:-1]
weight      = 1 / RV_noise**2
t_resample = np.linspace(min(BJD), max(BJD), 10000)
Jitter2     = gaussian_smoothing(BJD, Jitter, BJD, 1.5, weight)


# plt.errorbar(BJD, RV_HARPS, yerr=RV_noise, fmt=".", capsize=0)
plt.plot(BJD, Jitter, '.')
plt.plot(BJD, Jitter2, '.')
plt.show()



#==============================================================================
# Periodogram
#==============================================================================
from astropy.stats import LombScargle

min_f   = 0.001
max_f   = 5
spp     = 10

frequency0, power0 = LombScargle(BJD, RV_HARPS, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency1, power1 = LombScargle(BJD, Jitter2, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

ax = plt.subplot(111)
ax.set_xscale('log')
# ax.axhline(y=0, color='k')
ax.axvline(x=8.1256, color='k')
ax.axvline(x=103.49, color='k')
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
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'offset', 'alpha')

    def get_value(self, t):
        M_anom1  = 2*np.pi/np.exp(self.P1) * (t.flatten() - np.exp(self.tau1))
        e_anom1  = solve_kep_eqn(M_anom1, self.e1)
        f1       = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1      = np.exp(self.k1)*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))

        M_anom2  = 2*np.pi/np.exp(self.P2) * (t.flatten() - np.exp(self.tau2))
        e_anom2  = solve_kep_eqn(M_anom2, self.e2)
        f2       = 2*np.arctan( np.sqrt((1+self.e2)/(1-self.e2))*np.tan(e_anom2*.5) )
        rv2      = np.exp(self.k2)*(np.cos(f2 + self.w2) + self.e2*np.cos(self.w2))

        return rv1 + rv2 + 100*self.offset + self.alpha * Jitter2 * 0


#==============================================================================
# log likelihood
#==============================================================================

def lnprior(theta):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset, alpha = theta
    if (0 < P1 < 3) and (0 < k1 < 5) and (-2*np.pi < w1 < 2*np.pi) and (0. < e1 < 0.4) and \
    	(3 < P2 < 6) and (1 < k2 < 5) and (-2*np.pi < w2 < 2*np.pi) and (0.2 < e2 < 0.7):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset, alpha = theta
    model = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
    			  P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, 
    			  offset=offset, alpha=alpha)
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
ndim = 12
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(BJD, RV_HARPS, RV_noise), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = [2, 1, 3, 0, 0.2, 4, 1, 3, 0, 0.4, 0, 1] + 1e-4 * np.random.randn(nwalkers, ndim)
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
log_samples         = sampler.chain[:, :, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,0:3] = np.exp(real_samples[:,0:3])
real_samples[:,5:8] = np.exp(real_samples[:,5:8])
real_samples[:,-1:] = real_samples[:,-1:]*100

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$\log\ P1$", r"$\log\ T_{1}$", r"$\log\ K1$", r"$\omega1$", r"$e1$", 
			r"$\log\ P2$", r"$\log\ T_{2}$", r"$\log\ K2$", r"$\omega2$", r"$e2$", 
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
labels=[r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", 
		r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", 
		r"$\delta$", r"$\alpha$"]
fig = corner.corner(real_samples[:,:], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('96700_MCMC-corner.png')
plt.show()










