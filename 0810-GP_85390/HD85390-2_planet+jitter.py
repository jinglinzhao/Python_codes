import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model

#==============================================================================
# Import data 
#==============================================================================

all_rvs     = np.loadtxt('HD85390_quad.vels')
jitter_raw  = np.loadtxt('jitter_raw.txt')
jitter_smooth = np.loadtxt('jitter_smooth.txt')

x 		= all_rvs[:,0]
idx     = x < 57300
x 		= x[idx]
y 		= all_rvs[idx,1]
yerr 	= all_rvs[idx,2]


import time
import os
import shutil
time0   = time.time()
os.makedirs(str(time0))
shutil.copy('HD85390-2_planet+jitter.py', str(time0)+'/HD85390-2_planet+jitter.py')  
os.chdir(str(time0))

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel("RV [m/s]")
plt.xlabel("Shifted JD [d]")
plt.savefig('HD85390-1-RV.png')
# plt.show()

truth_P1 = 400
truth_P2 = 850
truth_P3 = 3300


#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/5000
max_f   = 1
spp     = 10

frequency0, power0 = LombScargle(x, y, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(x, jitter_raw, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

ax = plt.subplot(111)
ax.set_xscale('log')
ax.axhline(y=0, color='k')
ax.axvline(x=394, color='k')
ax.axvline(x=843, color='k')
ax.axvline(x=3442, color='k')
plt.plot(1/frequency0, power0, '-', label='HARPS', linewidth=2.0)
plt.plot(1/frequency1, power1, '--', label='Jitter')
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
# plt.xlim([0, /25])
plt.legend()
plt.savefig('HD85390-0-Periodogram.png')
plt.show()




#==============================================================================
# Model
#==============================================================================
class Model(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'offset', 'alpha')

    def get_value(self, t):

        # Planet 1
        M_anom1 = 2*np.pi/np.exp(self.P1*10) * (t - np.exp(self.tau1))
        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1     = np.exp(self.k1)*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))
        
        # Planet 2
        M_anom2 = 2*np.pi/np.exp(self.P2*10) * (t - np.exp(self.tau2))
        e_anom2 = solve_kep_eqn(M_anom2, self.e2)
        f2      = 2*np.arctan( np.sqrt((1+self.e2)/(1-self.e2))*np.tan(e_anom2*.5) )
        rv2     = np.exp(self.k2)*(np.cos(f2 + self.w2) + self.e2*np.cos(self.w2))

        return rv1 + rv2 + self.offset + self.alpha * jitter1

class Model2(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'offset')

    def get_value(self, t):

        # Planet 1
        M_anom1 = 2*np.pi/np.exp(self.P1*10) * (t - np.exp(self.tau1))
        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1     = np.exp(self.k1)*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))
        
        # Planet 2
        M_anom2 = 2*np.pi/np.exp(self.P2*10) * (t - np.exp(self.tau2))
        e_anom2 = solve_kep_eqn(M_anom2, self.e2)
        f2      = 2*np.arctan( np.sqrt((1+self.e2)/(1-self.e2))*np.tan(e_anom2*.5) )
        rv2     = np.exp(self.k2)*(np.cos(f2 + self.w2) + self.e2*np.cos(self.w2))

        return rv1 + rv2 + self.offset
#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset, alpha = theta
    # if (0.5 < P1 < 0.7) and (0 < tau1) and (-2 < k1 < 3) and (-np.pi < w1 < np.pi) and (0 < e1 < 0.5) and \
    #    (0.6 < P2 < 0.8) and (0 < tau2) and (-2 < k2 < 3) and (-np.pi < w2 < np.pi) and (0 < e2 < 0.5) and \
    #    (0.7 < P3 < 0.9) and (0 < tau3) and (-2 < k3 < 3) and (-np.pi < w3 < np.pi) and (0 < e3 < 0.5):
    if (0. < P1) and (0 < tau1 < 10) and (-2 < k1 < 3) and (-2*np.pi < w1 < 2*np.pi) and (0 < e1 < 0.8) and \
       (0. < P2) and (0 < tau2 < 10) and (-2 < k2 < 3) and (-2*np.pi < w2 < 2*np.pi) and (0 < e2 < 0.8):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset, alpha = theta
    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
                        P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, offset=offset, alpha=alpha)
    y_fit       = fit_curve.get_value(x)
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    


import emcee
ndim = 12
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
pos = [[0.6, 1., np.log(np.std(y)), 0, 0.1,\
        0.7, 1., np.log(np.std(y)), 0, 0.1, 0., 1] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
pos, prob, state  = sampler.run_mcmc(pos, 5000)

print("Running second burn-in...")
pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
pos, prob, state  = sampler.run_mcmc(pos, 5000)

print("Running third burn-in...")
pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
pos, prob, state  = sampler.run_mcmc(pos, 5000)

print("Running production...")
sampler.run_mcmc(pos, 10000);

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples         = sampler.chain[:, 11000:, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,0]   = real_samples[:,0]*10
real_samples[:,5]   = real_samples[:,5]*10
real_samples[:,0:3] = np.exp(real_samples[:,0:3])
real_samples[:,5:8] = np.exp(real_samples[:,5:8])


fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=[r"$\log\ P1$", r"$\log\ T_{1}$", r"$\log\ K1$", r"$\omega1$", r"$e1$", 
            r"$\log\ P2$", r"$\log\ T_{2}$", r"$\log\ K2$", r"$\omega2$", r"$e2$", 
            "offset", r"$\alpha$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('HD85390-2-Trace.png')
# plt.show()


import corner
labels=[r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", "offset", r"$\alpha$"]
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('HD85390-3-Corner.png')
# plt.show()


#==============================================================================
# Output
#==============================================================================

a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((12,3))
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
aa[10,:]= [a10[i] for i in range(3)]
aa[11,:]= [a11[i] for i in range(3)]
np.savetxt('HD85390_fit.txt', aa, fmt='%.6f')



P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset, alpha = aa[:,0]
fit_curve   = Model(P1=np.log(P1)/10, tau1=np.log(tau1), k1=np.log(k1), w1=w1, e1=e1, 
                    P2=np.log(P2)/10, tau2=np.log(tau2), k2=np.log(k2), w2=w2, e2=e2, offset=offset, alpha=alpha)
y_fit       = fit_curve.get_value(x)

fit_curve2  = Model2(P1=np.log(P1)/10, tau1=np.log(tau1), k1=np.log(k1), w1=w1, e1=e1, 
                     P2=np.log(P2)/10, tau2=np.log(tau2), k2=np.log(k2), w2=w2, e2=e2, offset=offset)
t_fit       = np.linspace(min(x), max(x), num=10001, endpoint=True)
y_fit2      = fit_curve2.get_value(t_fit)

residual    = y_fit - y
chi2        = sum(residual**2 / yerr**2)
rms         = np.sqrt(np.mean(residual**2))

fig = plt.figure(figsize=(10, 7))
frame1 = fig.add_axes((.15,.3,.8,.6))
frame1.axhline(y=0, color='k', ls='--', alpha=.3)
plt.plot(t_fit, y_fit2, 'b', alpha=.5)
plt.plot(x, y_fit, 'bo', alpha=.5)
plt.plot(x, alpha*jitter1, 'ro', alpha=.5)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel("Radial velocity [m/s]")

frame2  = fig.add_axes((.15,.1,.8,.2))   
frame2.axhline(y=0, color='k', ls='--', alpha=.3)
plt.errorbar(x, residual, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD - 2450000")
plt.ylabel('Residual [m/s]')
plt.savefig('HD85390-4-MCMC_fit.png')

plt.close("all")

os.chdir('..')


















