# based on GP_85390_george_SE_2_planets.py

import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn

#==============================================================================
# Import data 
#==============================================================================
star    = 'HD7449'

if star == 'HD117618':
    AAT     = np.loadtxt('../data/HD117618_AAT.vels')
    HARPS1  = np.loadtxt('../data/HD117618_HARPSbinjit.vels')
    HARPS2  = np.loadtxt('../data/HD117618_HARPSbinjit2.vels')
    t       = np.hstack((AAT[:,0], HARPS1[:,0], HARPS2[:,0]))
    y       = np.hstack((AAT[:,1], HARPS1[:,1], HARPS2[:,1]))
    yerr    = np.hstack((AAT[:,2], HARPS1[:,2], HARPS2[:,2]))

if star == 'HD7449':
    HARPS1  = np.loadtxt('../data/HD7449_HARPSbinjit.vels')
    HARPS2  = np.loadtxt('../data/HD7449_HARPSbinjit2.vels')
    t       = np.hstack((HARPS1[:,0], HARPS2[:,0]))
    y       = np.hstack((HARPS1[:,1], HARPS2[:,1]))
    yerr    = np.hstack((HARPS1[:,2], HARPS2[:,2]))

import time
import os
import shutil
time0   = time.time()
dir_name = star + '_' + str(time0)
os.makedirs('../output/'+dir_name)
shutil.copy('GP_multi_stars.py', '../output/'+dir_name+'/GP_multi_stars.py')  
os.chdir('../output/'+dir_name)

plt.figure()
if star == 'HD117618':
    plt.errorbar(AAT[:,0], AAT[:,1], yerr=AAT[:,2], fmt=".k", capsize=0)
    plt.errorbar(HARPS1[:,0], HARPS1[:,1], yerr=HARPS1[:,2], fmt=".b", capsize=0)
    plt.errorbar(HARPS2[:,0], HARPS2[:,1], yerr=HARPS2[:,2], fmt=".r", capsize=0)
if star == 'HD7449':
    plt.errorbar(HARPS1[:,0], HARPS1[:,1], yerr=HARPS1[:,2], fmt=".b", capsize=0)
    plt.errorbar(HARPS2[:,0], HARPS2[:,1], yerr=HARPS2[:,2], fmt=".r", capsize=0)    
plt.ylabel("RV [m/s]")
plt.xlabel("JD−2,400,000")
plt.savefig(star+'-0-RV.png')
# plt.show()

#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/15000
max_f   = 1
spp     = 10

plt.figure()
ax = plt.subplot(111)
ax.set_xscale('log')
ax.axhline(y=0, color='k')

if star == 'HD117618':
    ax.axvline(x=25.8, color='k')
    ax.axvline(x=318, color='k')
    frequency0, power0 = LombScargle(AAT[:,0], AAT[:,1], AAT[:,2]).autopower(minimum_frequency=min_f,
                                                                            maximum_frequency=max_f,
                                                                            samples_per_peak=spp)
if star == 'HD7449':
    ax.axvline(x=1275, color='k')
    ax.axvline(x=4046, color='k')
    frequency0, power0 = LombScargle(AAT[:,0], AAT[:,1], AAT[:,2]).autopower(minimum_frequency=min_f,
                                                                            maximum_frequency=max_f,
                                                                            samples_per_peak=spp)
plt.plot(1/frequency0, power0, '-', linewidth=2.0)
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.savefig(star+'-1-Periodogram.png')


#==============================================================================
# Model
#==============================================================================
import george
from george.modeling import Model

class Model(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'd_aat', 'd_harps1', 'd_harps2')

    def get_value(self, t):

        # Planet 1
        M_anom1 = 2*np.pi/(100*self.P1) * (t - 1000*self.tau1)
        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))
        
        # Planet 2
        M_anom2 = 2*np.pi/(100*self.P2) * (t - 1000*self.tau2)
        e_anom2 = solve_kep_eqn(M_anom2, self.e2)
        f2      = 2*np.arctan( np.sqrt((1+self.e2)/(1-self.e2))*np.tan(e_anom2*.5) )
        rv2     = 100*self.k2*(np.cos(f2 + self.w2) + self.e2*np.cos(self.w2))

        offset = np.zeros(len(t))
        for i in range(len(t)):
            if t[i] in AAT[:,0]:
                offset[i] = self.d_aat
            elif t[i] in HARPS1[:,0]:
                offset[i] = self.d_harps1
            elif t[i] in HARPS2[:,0]:
                offset[i] = self.d_harps2           

        return rv1 + rv2 + offset

#==============================================================================
# GP
#==============================================================================
from george import kernels

k1      = kernels.ExpSine2Kernel(gamma = 1, log_period = np.log(100), 
                                bounds=dict(gamma=(0,100), log_period=(0,10)))
k2      = np.std(y) * kernels.ExpSquaredKernel(100)
kernel  = k1 * k2
truth   = dict(P1=0.5, tau1=0.1, k1=np.std(y)/100, w1=0., e1=0.4, 
             P2=3., tau2=0.1, k2=np.std(y)/100, w2=0., e2=0.4, 
             d_aat=0., d_harps1=0., d_harps2=0.)
kwargs  = dict(**truth)
kwargs["bounds"] = dict(P1=(0, 1000), k1=(0,0.3), w1=(-2*np.pi,2*np.pi), e1=(0,0.9), 
                        P2=(0, 1000), k2=(0,0.3), w2=(-2*np.pi,2*np.pi), e2=(0,0.9))
mean_model = Model(**kwargs)
gp = george.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(t, yerr)

def lnprob2(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()

#==============================================================================
# MCMC
#==============================================================================
import emcee

initial = gp.get_parameter_vector()
names = gp.get_parameter_names()
ndim, nwalkers = len(initial), 36
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 3000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 3000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, 3000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)
real_samples[:,1]   = 10*real_samples[:,1]
real_samples[:,6]   = 10*real_samples[:,6]
real_samples[:,0:3] = 100*real_samples[:,0:3]
real_samples[:,5:8] = 100*real_samples[:,5:8]
idx = real_samples[:,3] > 0
real_samples[idx,3] = real_samples[idx, 3] - 2*np.pi
idx = real_samples[:,8] < 3
real_samples[idx,8] = real_samples[idx, 8] + 2*np.pi


fig, axes   = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels      = np.hstack(([r"$\frac{P_{1}}{100}$", r"$\frac{T_{1}}{1000}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", 
              r"$\frac{P_{2}}{100}$", r"$\frac{T_{2}}{1000}$", r"$\frac{K_{2}}{100}$", r"$\omega2$", r"$e2$", 
              "d_aat", "d_harps1", "d_harps2"], names[-4:]))
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig(star+'-2-Trace.png')
# plt.show()


import corner
labels= np.hstack(([r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", 
		r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", 
        "d_aat", "d_harps1", "d_harps2"], names[-4:]))
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig(star+'-3-Corner.png')
# plt.show()


#==============================================================================
# Output
#==============================================================================
vv = np.zeros(11)
# aaa= np.zeros(12)
# np.hstack((vv, aaa)) = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))

a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11 = map(lambda v: 
	(v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
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

solution = np.zeros(len(gp))
solution[12] = v1[0]
solution[13] = v2[0]
solution[14] = v3[0]
solution[15] = v4[0]
solution[16] = v5[0]
solution[17] = v6[0]
solution[18] = v7[0]
solution[19] = v8[0]
solution[20] = v9[0]
solution[21] = v10[0]
solution[22] = v11[0]
solution[0:12] = aa[:,0]

P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset1, offset2 = aa[:,0]
fig = plt.figure(figsize=(10, 7))
frame1 = fig.add_axes((.15,.3,.8,.6))
frame1.axhline(y=0, color='k', ls='--', alpha=.3)
t_sample    = np.linspace(min(x), max(x), num=10001, endpoint=True)
# Planet 1 #
Planet1     = Model(P1=P1/100, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, 
                    P2=P2/100, tau2=tau2/1000, k2=0, w2=w2, e2=e2, offset1=offset1, offset2=0)
y1          = Planet1.get_value(t_sample)
plt.plot(t_sample, y1, 'b-.', alpha=.3, label='Planet1')
# Planet 2 #
Planet2     = Model(P1=P1/100, tau1=tau1/1000, k1=0, w1=w1, e1=e1, 
                    P2=P2/100, tau2=tau2/1000, k2=k2/100, w2=w2, e2=e2, offset1=0, offset2=offset2)
y2          = Planet2.get_value(t_sample)
plt.plot(t_sample, y2, 'b--', alpha=.3, label='Planet2')
# Planet1 + Planet2 #
y12         = y1 + y2
plt.plot(t_sample, y12, 'b-', alpha=.5, label='Planet1+Planet2')
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label='HARPS RV')
plt.legend()
plt.ylabel("Radial velocity [m/s]")

fit_curve   = Model(P1=P1/100, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, 
                    P2=P2/100, tau2=tau2/1000, k2=k2/100, w2=w2, e2=e2, offset1=offset1, offset2=offset2)
y_fit       = fit_curve.get_value(x)

residual    = y_fit - y
chi2        = sum(residual**2 / yerr**2)
rms         = np.sqrt(np.mean(residual**2))
wrms        = np.sqrt(sum((residual/yerr)**2) / sum(1/yerr**2))

frame2  = fig.add_axes((.15,.1,.8,.2))   
frame2.axhline(y=0, color='k', ls='--', alpha=.3)
plt.errorbar(x, residual, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD - 2400000")
plt.ylabel('Residual [m/s]')
plt.savefig('HD85390-4-MCMC_fit.png')
plt.close("all")


# Make the maximum likelihood prediction
gp.set_parameter_vector(solution)
t = np.linspace(min(x), max(x), 10000)
mu, var = gp.predict(y, t, return_var=True)
std = np.sqrt(var)


# Plot the data
color = "#ff7f0e"
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(t, mu, color=color)
plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
# plt.xlim(-5, 5)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("HD85390 - maximum likelihood prediction");
plt.savefig('HD85390-5-prediction.png')
plt.close('all')

# os.chdir('..')


















