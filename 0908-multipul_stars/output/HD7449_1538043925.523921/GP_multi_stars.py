# based on GP_85390_george_SE_2_planets.py

import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn

#==============================================================================
# Import data 
#==============================================================================
star    = 'HD7449'
print('*'*len(star))
print(star)
print('*'*len(star))

if star == 'HD117618':
    AAT     = np.loadtxt('../data/HD117618_AAT.vels')
    HARPS1  = np.loadtxt('../data/HD117618_HARPSbinjit.vels')
    HARPS2  = np.loadtxt('../data/HD117618_HARPSbinjit2.vels')
    x       = np.hstack((AAT[:,0], HARPS1[:,0], HARPS2[:,0]))
    y       = np.hstack((AAT[:,1], HARPS1[:,1], HARPS2[:,1]))
    yerr    = np.hstack((AAT[:,2], HARPS1[:,2], HARPS2[:,2]))

if star == 'HD7449':    
#http://www.openexoplanetcatalogue.com/planet/HD%207449%20b/
# The host star shows signs of short-term activity due to magnetic features on the stellar surface. 
# Rotational period ~ 14 d
    HARPS1  = np.loadtxt('../data/HD7449_HARPSbinjit.vels')
    HARPS2  = np.loadtxt('../data/HD7449_HARPSbinjit2.vels')
    x       = np.hstack((HARPS1[:,0], HARPS2[:,0]))
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
    ax.axvline(x=1250, color='k')
    ax.axvline(x=10675, color='k')
    frequency0, power0 = LombScargle(HARPS1[:,0], HARPS1[:,1], HARPS1[:,2]).autopower(minimum_frequency=min_f,
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

if star == 'HD117618':

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

if star == 'HD7449':

    class Model(Model):
        parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'd_harps1', 'd_harps2')

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

            offset      = np.zeros(len(t))
            idx         = t < 57161
            offset[idx] = self.d_harps1
            offset[~idx]= self.d_harps2     

            return rv1 + rv2 + offset

#==============================================================================
# GP
#==============================================================================
from george import kernels

if star == 'HD117618':
    k1      = kernels.ExpSine2Kernel(gamma = 1, log_period = np.log(100), 
                                    bounds=dict(gamma=(-1,100), log_period=(0,10)))
    k2      = np.std(y) * kernels.ExpSquaredKernel(1.)
    k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
    kernel  = k1 * k2 + k3
    truth   = dict(P1=0.25, tau1=0.1, k1=np.std(y)/100, w1=0., e1=0.4, 
                   P2=3.1, tau2=0.1, k2=np.std(y)/100, w2=0., e2=0.4, 
                   d_aat=0., d_harps1=0., d_harps2=0.)
    kwargs  = dict(**truth)
    kwargs["bounds"] = dict(P1=(0.2, 0.3), k1=(0,0.3), w1=(-2*np.pi,2*np.pi), e1=(0,0.9), 
                            P2=(2.5, 3.5), k2=(0,0.3), w2=(-2*np.pi,2*np.pi), e2=(0,0.9))

if star == 'HD7449':
    k1      = kernels.ExpSine2Kernel(gamma = 1, log_period = np.log(14), 
                                    bounds=dict(gamma=(-3,1), log_period=(1,3)))
    k2      = kernels.ConstantKernel(log_constant=np.log(1.0), bounds=dict(log_constant=(-3,3))) * kernels.ExpSquaredKernel(1.)
    kernel  = k1 * k2    
    truth   = dict(P1=12.50, tau1=0.1, k1=np.std(y)/100, w1=0., e1=0.8, 
                   P2=160., tau2=0.1, k2=np.std(y)/100, w2=0., e2=0.2, 
                   d_harps1=0., d_harps2=0.)
    kwargs  = dict(**truth)
    kwargs["bounds"] = dict(P1=(12.0, 13.0), k1=(0,1.), w1=(-2*np.pi,2*np.pi), e1=(0.7,0.95), 
                            P2=(35, 200), k2=(0,2.), w2=(-2*np.pi,2*np.pi), e2=(0.,0.5))

mean_model = Model(**kwargs)
gp = george.GP(kernel, mean=mean_model, fit_mean=True, white_noise=np.log(0.5**2), fit_white_noise=True)
# gp.freeze_parameter('kernel:k2:k1:log_constant')
gp.compute(x, yerr)
lnp1 = gp.log_likelihood(y)

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
p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 4000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 4000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-2 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 4000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 5000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, -5000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)
real_samples[:,1]   = 10*real_samples[:,1]
real_samples[:,6]   = 10*real_samples[:,6]
real_samples[:,0:3] = 100*real_samples[:,0:3]
real_samples[:,5:8] = 100*real_samples[:,5:8]
idx = real_samples[:,3] > 0
real_samples[idx,3] = real_samples[idx, 3] - 2*np.pi
idx = real_samples[:,8] < 0
real_samples[idx,8] = real_samples[idx, 8] + 2*np.pi


fig, axes   = plt.subplots(ndim, figsize=(20, 14), sharex=True)
if star == 'HD117618':
    labels      = np.hstack(([r"$\frac{P_{1}}{100}$", r"$\frac{T_{1}}{1000}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", 
                              r"$\frac{P_{2}}{100}$", r"$\frac{T_{2}}{1000}$", r"$\frac{K_{2}}{100}$", r"$\omega2$", r"$e2$", 
                              "d_aat", "d_harps1", "d_harps2"], names[-4:]))
if star == 'HD7449':
    labels      = np.hstack(([r"$\frac{P_{1}}{100}$", r"$\frac{T_{1}}{1000}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", 
                              r"$\frac{P_{2}}{100}$", r"$\frac{T_{2}}{1000}$", r"$\frac{K_{2}}{100}$", r"$\omega2$", r"$e2$", 
                              "d_harps1", "d_harps2"], names[-5:]))
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig(star+'-2-Trace.png')
# plt.show()

import corner
if star == 'HD117618':
    labels= np.hstack(([r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", 
                        r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", 
                        "d_aat", "d_harps1", "d_harps2"], names[-5:]))
if star == 'HD7449':
    labels= np.hstack(([r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", 
                        r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", 
                        "d_harps1", "d_harps2"], names[-5:]))
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig(star+'-3-Corner.png')
# plt.show()

#==============================================================================
# Output
#==============================================================================
aa = np.zeros((len(truth),3))
solution = np.zeros(len(gp))

if star == 'HD117618':
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, v1, v2, v3, v4= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                                                            zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
    aa[12,:]= [a12[i] for i in range(3)]

if star == 'HD7449':
    # a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, v1, v2, v3, v4 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
    #                                                         zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, v1, v2, v3, v4, v5 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                                                            zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
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
np.savetxt(star+'_fit.txt', aa, fmt='%.6f')

solution[0:len(truth)]  = aa[:,0]
solution[len(truth)]    = v1[0]
solution[len(truth)+1]  = v2[0]
solution[len(truth)+2]  = v3[0]
solution[len(truth)+3]  = v4[0]
solution[len(truth)+4]  = v5[0]

if 0:

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

if star == 'HD117618':
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, d_aat, d_harps1, d_harps2 = aa[:,0]
    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
                        P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, 
                        d_aat=d_aat, d_harps1=d_harps1, d_harps2=d_harps2)
if star == 'HD7449':
    P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, d_harps1, d_harps2 = aa[:,0]
    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
                        P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, 
                        d_harps1=d_harps1, d_harps2=d_harps2)

fig         = plt.figure(figsize=(20, 14))
frame1      = fig.add_axes((.15,.3,.8,.6))
frame1.axhline(y=0, color='k', ls='--', alpha=.3)
frame1.axvline(x=57161, color='k', ls='--', alpha=.5)
t           = np.linspace(min(x), max(x), 10000, endpoint=True)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label='HARPS RV')
plt.plot(t, fit_curve.get_value(t), 'b-', alpha=.5, label='Planet fit')

y_fit       = fit_curve.get_value(x)
residual    = y_fit - y
chi2        = sum(residual**2 / yerr**2)
rms         = np.sqrt(np.mean(residual**2))
wrms        = np.sqrt(sum((residual/yerr)**2) / sum(1/yerr**2))

frame2  = fig.add_axes((.15,.1,.8,.2))   
frame2.axhline(y=0, color='k', ls='--', alpha=.3)
frame2.axvline(x=57161, color='k', ls='--', alpha=.5)
plt.errorbar(x, residual, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("JD−2,400,000")
plt.ylabel('Residual [m/s]')
plt.savefig(star+'-4-MCMC_fit.png')
plt.close("all")

# Make the maximum likelihood prediction
gp.set_parameter_vector(solution)
lnp2 = gp.log_likelihood(y)
mu, var = gp.predict(y, t, return_var=True)
std = np.sqrt(var)

# Plot the data
plt.figure(figsize=(20, 14))
color = "#ff7f0e"
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(t, mu, color=color)
plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title(star+ " - maximum likelihood prediction");
plt.savefig(star+'-5-prediction.png')
plt.close('all')

print(star+' finished')
print(lnp1)
print(lnp2)
# os.chdir('..')


















