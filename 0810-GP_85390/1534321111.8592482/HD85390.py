import numpy as np
import matplotlib.pyplot as plt


#==============================================================================
# Import data 
#==============================================================================

all_rvs = np.loadtxt('HD85390_quad.vels')
x 		= all_rvs[:,0]
x 		= x - min(x)
y 		= all_rvs[:,1]
yerr 	= all_rvs[:,2]


import time
import os
import shutil
time0   = time.time()
os.makedirs(str(time0))
shutil.copy('HD85390.py', str(time0)+'/HD85390.py')  
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
# Model
#==============================================================================

from rv import solve_kep_eqn
from celerite.modeling import Model

class Model(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'P3', 'tau3', 'k3', 'w3', 'e3', 'offset')

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

        # Planet 3
        M_anom3 = 2*np.pi/np.exp(self.P3*10) * (t - np.exp(self.tau3))
        e_anom3 = solve_kep_eqn(M_anom3, self.e3)
        f3      = 2*np.arctan( np.sqrt((1+self.e3)/(1-self.e3))*np.tan(e_anom3*.5) )
        rv3     = np.exp(self.k3)*(np.cos(f3 + self.w3) + self.e3*np.cos(self.w3))

        return rv1 + rv2 + rv3 + self.offset


#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, P3, tau3, k3, w3, e3, offset = theta
    if (0.5 < P1 < 0.7) and (0 < tau1) and (-2 < k1 < 3) and (-np.pi < w1 < np.pi) and (0 < e1 < 0.3) and \
       (0.6 < P2 < 0.8) and (0 < tau2) and (-2 < k2 < 3) and (-np.pi < w2 < np.pi) and (0 < e2 < 0.3) and \
       (0.7 < P3 < 0.9) and (0 < tau3) and (-2 < k3 < 3) and (-np.pi < w3 < np.pi) and (0 < e3 < 0.3):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, P3, tau3, k3, w3, e3, offset = theta
    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
                        P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, 
                        P3=P3, tau3=tau3, k3=k3, w3=w3, e3=e3, offset=offset)
    y_fit       = fit_curve.get_value(x)
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    


import emcee
ndim = 16
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
pos = [[0.6, 1., np.log(np.std(y)), 0, 0.1,\
        0.7, 1., np.log(np.std(y)), 0, 0.1,\
        0.8, 1., np.log(np.std(y)), 0, 0.1, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
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
real_samples[:,10]  = real_samples[:,10]*10
real_samples[:,0:3] = np.exp(real_samples[:,0:3])
real_samples[:,5:8] = np.exp(real_samples[:,5:8])
real_samples[:,10:13] = np.exp(real_samples[:,10:13])


fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=[r"$\log\ P1$", r"$\log\ T_{1}$", r"$\log\ K1$", r"$\omega1$", r"$e1$", 
            r"$\log\ P2$", r"$\log\ T_{2}$", r"$\log\ K2$", r"$\omega2$", r"$e2$", 
            r"$\log\ P3$", r"$\log\ T_{3}$", r"$\log\ K3$", r"$\omega3$", r"$e3$",
            r"$\frac{offset}{100}$"]
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
labels=[r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", r"$P3$", r"$T_{3}$", r"$K3$", r"$\omega3$", r"$e3$", "offset"]
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('HD85390-3-Corner.png')
# plt.show()


#==============================================================================
# Output
#==============================================================================

a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((16,3))
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
aa[12,:]= [a12[i] for i in range(3)]
aa[13,:]= [a13[i] for i in range(3)]
aa[14,:]= [a14[i] for i in range(3)]
aa[15,:]= [a15[i] for i in range(3)]
np.savetxt('HD85390_fit.txt', aa, fmt='%.6f')



P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, P3, tau3, k3, w3, e3, offset = aa[:,0]
fit_curve   = Model(P1=np.log(P1)/10, tau1=np.log(tau1), k1=np.log(k1), w1=w1, e1=e1, 
                    P2=np.log(P2)/10, tau2=np.log(tau2), k2=np.log(k2), w2=w2, e2=e2, 
                    P3=np.log(P3)/10, tau3=np.log(tau3), k3=np.log(k3), w3=w3, e3=e3, offset=offset)
t_fit       = np.linspace(min(x), max(x), num=10001, endpoint=True)
y_fit       = fit_curve.get_value(np.array(t_fit))

residual    = fit_curve.get_value(x) - y
chi2        = sum(residual**2 / yerr**2)
rms         = np.sqrt(np.mean(residual**2))


fig = plt.figure(figsize=(10, 7))
frame1 = fig.add_axes((.15,.3,.8,.6))
frame1.axhline(y=0, color='k', ls='--', alpha=.3)
plt.plot(t_fit, y_fit, alpha=.5)
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


















