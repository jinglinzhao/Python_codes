# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 
# change the way the planet is fitted 
#######################################
# After 03/09
# based on 0423-MCMC/MCMC_FT2.py

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import gran_gen
from functions import gaussian_smoothing


#==============================================================================
# Setup
#==============================================================================

real_a      = 2.0
real_k      = 0.7
real_phi    = 1.0

N = 200
amplitude1  = np.zeros(N)
period1     = np.zeros(N)
amplitude2  = np.zeros(N)
period2     = np.zeros(N)

yes1_a1     = 0     # FT correction: amplitude within 1 sigma
yes1_k1     = 0     # FT correction: period within 1 sigma
yes1_a2     = 0     # FT correction: amplitude within 2 sigma
yes1_k2     = 0     # FT correction: period within 2 sigma
yes1_both1  = 0     # FT correction: both within 1 sigma
yes1_both2  = 0     # FT correction: both within 2 sigma

yes2_a1     = 0     # No correction: amplitude within 1 sigma
yes2_k1     = 0     # No correction: period within 1 sigma
yes2_a2     = 0     # No correction: amplitude within 2 sigma
yes2_k2     = 0     # No correction: period within 2 sigma
yes2_both1  = 0     # No correction: both within 1 sigma
yes2_both2  = 0     # No correction: both within 2 sigma


mode    = 5; 
n_group = 6
n_obs   = 40

burn_in_1_step  = 2000
burn_in_2_step  = 1000
production_step = 2000

# Read Data 
t       = np.arange(200)
XX  = np.loadtxt('XX.txt')
YY  = np.loadtxt('YY.txt') 
ZZ  = np.loadtxt('ZZ.txt') 
RV_jitter   = np.loadtxt('RV_jitter.txt')
RV_jitter   = np.hstack((RV_jitter,RV_jitter))


#==============================================================================
# Begin looping
#==============================================================================

import time
time0   = time.time()
os.makedirs(str(time0))
os.chdir(str(time0))


# from multiprocessing import Pool

for n in range(N):

    print('----' + '-' * len(str(n)))
    print('n = %i' %n)
    print('----' + '-' * len(str(n)))

    new_dir = str(n) + '/'
    os.makedirs(new_dir)

    ################
    # Present data #
    ################

    # re-sample
    # x       = gran_gen(n_group, n_obs)
    print('Observation samples:')
    x = np.sort(random.sample(range(200), 40))
    # x = t    
    print(x)    
    RV_X   = np.array([XX[i] for i in x])
    RV_Y   = np.array([YY[i] for i in x])
    RV_Z   = np.array([ZZ[i] for i in x])
    # RV_diff = RV_X - RV_Y
    # RV_diff = gaussian_smoothing(x, RV_diff, x, 2)

    if 1: # sub-sampling 
        fig = plt.figure()
        plt.plot(t, XX, 'o', label='full samples')
        plt.plot(x, RV_X, 'r.', label='sub-sampling')
        plt.title('Sampling = %i' % n_obs)
        plt.xlabel(r"$t$")
        plt.ylabel('Measured RV [m/s]')
        plt.legend()
        plt.savefig(new_dir + '0-RV_sampling.png')

    if 1:   # Time series 
        fig = plt.figure()
        frame1 = fig.add_axes((.1,.3,.8,.6))
        plt.plot(x, RV_Y, 'rs', label='FT')
        plt.plot(x, RV_X, 'bo', label='Gaussian')
        plt.title(r'$N_{sample} = %i$' % n_obs)
        plt.ylabel('RV (m/s)')
        plt.legend()

        # frame2  = fig.add_axes((.1,.1,.8,.2))   
        # plt.plot(x, RV_diff, 'k.', label='jitter model')
        # plt.xlabel(r"$t$")
        # plt.ylabel(r'$\Delta$ RV (m/s)')
        plt.savefig(new_dir + '1-Time_series.png')

    if 1:   # Comparison 
        fig = plt.figure()
        xx  = [min(np.hstack((RV_X, RV_Y)))-0.5, max(np.hstack((RV_X, RV_Y)))+0.5]
        plt.plot(xx, xx, '--', label=r'y = x')
        plt.plot(RV_X, RV_Y, '*')
        plt.title('Comparison')
        plt.ylabel(r"$RV_{FT} [m/s]$")
        plt.xlabel(r"$RV_{IN} [m/s]$")
        plt.legend()
        plt.savefig(new_dir + '2-Comparison.png')


    #==============================================================================
    # MCMC without jitter correction
    #==============================================================================

    print('# MCMC without jitter correction #')

    # each data is equally weighted 
    yerr    = 1 + np.zeros(RV_X.shape)

    def lnprior2(theta2):
        a2, k2, phi2, b2 = theta2
        if (-5 < a2 <5) and (-5 < k2 <5) and (-2*np.pi < phi2 < 2*np.pi) and (-10. < b2 < 10):
            return 0.0
        return -np.inf

    # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
    def lnlike2(theta2, x, y, yerr):
        a2, k2, phi2, b2 = theta2
        model = np.exp(a2) * np.sin(x/100. * np.exp(k2) * 2. * np.pi + phi2) + b2
        return -0.5*(np.sum( ((y-model)/yerr)**2. ))

    def lnprob2(theta2, x, y, yerr):
        lp2 = lnprior2(theta2)
        if not np.isfinite(lp2):
            return -np.inf
        return lp2 + lnlike2(theta2, x, y, yerr)    


    import emcee
    ndim    = 4
    nwalkers = 32
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(x, RV_X, yerr)) # Note that running with multiple threads takes three times the time

    print("Running first burn-in...")
    pos2     = [[np.log(np.var(RV_X)), np.log(1), 1., 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, _  = sampler2.run_mcmc(pos2, burn_in_1_step)

    print("Running second burn-in...")
    pos2 = [pos2[np.argmax(prob2)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, _  = sampler2.run_mcmc(pos2, burn_in_2_step)

    print("Running production...")
    sampler2.run_mcmc(pos2, production_step)

    #==============================================================================
    # Trace and corner plots 
    #==============================================================================

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels_log=[r"$\log\ A$", r"$\log\ \nu$", r"$\omega$", r"b"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot( np.rot90(sampler2.chain[:, :, i], 3), "k", alpha=0.3)
        ax.set_xlim(0, sampler2.chain.shape[1])
        ax.set_ylabel(labels_log[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number");
    plt.savefig(new_dir + '3-Trace2.png')


    import copy
    log_samples         = sampler2.chain[:, 3000:, :].reshape((-1, ndim))
    real_samples        = copy.copy(log_samples)
    real_samples[:,0:2] = np.exp(real_samples[:,0:2])

    import corner
    fig = corner.corner(real_samples, labels=[r"$A$", r"$\nu$", r"$\omega$", r"$b$"], truths=[real_a, real_k, real_phi, 100],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(new_dir + '3-MCMC2.png')


    #==============================================================================
    # Statistics
    #==============================================================================
    a2, k2, phi2, b2 = map(lambda v: np.array(v), zip(*np.percentile(real_samples, [50, 16, 84, 2.5, 97.5], axis=0)))

    if (a2[1] < real_a < a2[2]):
        yes2_a1 += 1
    if (k2[1] < real_k < k2[2]):
        yes2_k1 += 1
    if (a2[3] < real_a < a2[4]):
        yes2_a2 += 1
    if (k2[3] < real_k < k2[4]):
        yes2_k2 += 1
    if (a2[1] < real_a < a2[2]) and (k2[1] < real_k < k2[2]):
        yes2_both1 += 1
        print('Bingo - No correction - 1 sigma')
    if (a2[3] < real_a < a2[4]) and (k2[3] < real_k < k2[4]):
        yes2_both2 += 1
        print('Bingo - No correction - 2 sigma')

    print(np.vstack((a2, k2, phi2, b2)))


    fig = plt.figure()
    RV2_pos = a2[0] * np.sin(x/100. * k2[0] * 2. * np.pi + phi2[0]) + b2[0]
    RV2_os  = a2[0] * np.sin(t/100. * k2[0] * 2. * np.pi + phi2[0]) + b2[0]
    frame1  = fig.add_axes((.1,.3,.8,.6))
    frame1.axhline(color="gray", ls='--')
    plt.plot(x, RV_X, 'o', label='Data')
    plt.xlim(0,200) 
    plt.plot(x, RV2_pos, 'g^')
    plt.plot(t, RV2_os, 'g', label='Planet model')
    plt.title('No correction')
    plt.ylabel("RV (m/s)")
    plt.legend()
    frame1.set_xticklabels([])

    frame2  = fig.add_axes((.1,.1,.8,.2))   
    frame2.axhline(color="gray", ls='--')
    rms     = np.sqrt(np.var(RV2_pos - RV_X))
    plt.plot(x, RV2_pos - RV_X , 'k.', label=r'rms$=%.2f m/s$' %rms)
    plt.xlim(0,200) 
    plt.xlabel(r"$t$")
    plt.ylabel("Residual (m/s)")
    plt.legend()
    plt.savefig(new_dir + '5-Fit2.png')
    plt.close('all')


    #==============================================================================
    # MCMC with jitter correction (5 parameters)
    #==============================================================================

    import george
    from george.modeling import Model

    class Model(Model):
        parameter_names = ('a', 'k', 'phi', 'kz', 'offset_Z')

        def get_value(self, t):

            return self.kz*RV_X - (self.kz-1)*self.a*np.sin(t/100. * self.k * 2. * np.pi + self.phi) + self.offset_Z


    #==============================================================================
    # GP
    #==============================================================================
    from george import kernels

    k1      = kernels.ExpSine2Kernel(gamma = 1, log_period = np.log(100), 
                                    bounds=dict(gamma=(0,100), log_period=(0,7)))
    k2      = np.std(y) * kernels.ExpSquaredKernel(100)
    kernel  = k1 * k2

    y = RV_Z
    truth = dict(a=2., k=0.7, phi=1., kz=2., offset_Z=0.)
    kwargs = dict(**truth)
    kwargs["bounds"] = dict(a=(0,5), k=(0.1,3), phi=(-2*np.pi,2*np.pi), kz=(1,3), offset_Z=(-5,5))
    mean_model = Model(**kwargs)
    gp = george.GP(kernel, mean=mean_model, fit_mean=True)
    gp.compute(x, yerr)
    names = gp.get_parameter_names()

    def lnprob2(p):
        gp.set_parameter_vector(p)
        return gp.log_likelihood(y, quiet=True) + gp.log_prior()           

    if 0:
        def lnprior(theta):
            a, k, phi, kz, offset_Z = theta
            if (0 < a < 5) and (0 < k < 5) and (-2*np.pi < phi < 2*np.pi) and (1<kz<5):
                return 0.0
            return -np.inf

        # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
        def lnlike(theta, x, y, yerr):
            a, k, phi, kz, offset_Z = theta
            model = kz*RV_X - (kz-1)*a * np.sin(x/100. * k * 2. * np.pi + phi) + offset_Z 
            return -0.5*(np.sum( ((y-model)/yerr)**2. ))

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)    


    import emcee

    initial = gp.get_parameter_vector()
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=14)
    # import emcee
    # ndim        = 5
    # nwalkers    = 32
    # sampler     = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_Z, yerr))

    print("Running first burn-in...")
    pos         = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    pos, prob, _  = sampler.run_mcmc(pos, 3000)

    print("Running second burn-in...")
    pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
    pos, prob, _ = sampler.run_mcmc(pos, 2000)    

    print("Running production...")
    pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
    pos, prob, _ = sampler.run_mcmc(pos, 3000)    
    
    #==============================================================================
    # Trace and corner plots 
    #==============================================================================

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    # labels_log=np.hstack(([r"$A$", r"$\nu$", r"$\omega$", "kz", "offset_Z"], np.array(names[-4:])))
    labels=[r"$A$", r"$\nu$", r"$\omega$", "kz", "offset_Z", '1', '2', '3','4']
    for i in range(ndim):
        ax = axes[i]
        ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
        ax.set_xlim(0, sampler.chain.shape[1])
        ax.set_ylabel(labels_log[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number");
    plt.savefig(new_dir + '3-Trace1.png')


    import copy
    log_samples         = sampler.chain[:, 3000:, :].reshape((-1, ndim))
    real_samples        = copy.copy(log_samples)
    # real_samples[:,1] = np.exp(real_samples[:,1])

    import corner
    fig = corner.corner(real_samples, labels=labels, truths=[real_a, real_k, real_phi, 100, 100, 0, 0, 0,0],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(new_dir + '3-MCMC1.png')


    #==============================================================================
    # Statistics
    #==============================================================================    
    a, k, phi, kz, offset_Z, _,_,_,_= map(lambda v: np.array(v), zip(*np.percentile(real_samples, [50, 16, 84, 2.5, 97.5], axis=0)))

    if (a[1] < real_a < a[2]):
        yes1_a1 += 1
    if (k[1] < real_k < k[2]):
        yes1_k1 += 1
    if (a[3] < real_a < a[4]):
        yes1_a2 += 1
    if (k[3] < real_k < k[4]):
        yes1_k2 += 1
    if (a[1] < real_a < a[2]) and (k[1] < real_k < k[2]):
        yes1_both1 += 1
        print('Bingo - FT correction - 1 sigma')
    if (a[3] < real_a < a[4]) and (k[3] < real_k < k[4]):
        yes1_both2 += 1
        print('Bingo - FT correction - 2 sigma')

    print(np.vstack((a, k, phi, kz, offset_Z)))


    # #==============================================================================
    # # FT correction Plots
    # #==============================================================================    

    fig = plt.figure()
    RV2_pos = a[0] * np.sin(x/100. * k[0] * 2. * np.pi + phi[0])
    RV2_os  = a[0] * np.sin(t/100. * k[0] * 2. * np.pi + phi[0])
    frame1  = fig.add_axes((.1,.3,.8,.6))
    frame1.axhline(color="gray", ls='--')
    plt.plot(x, RV_X, 'o', label='Data')
    plt.xlim(0,200) 
    plt.plot(x, RV2_pos, 'g^')
    plt.plot(t, RV2_os, 'g', label='Planet model')
    plt.title('No correction')
    plt.ylabel("RV (m/s)")
    plt.legend()
    frame1.set_xticklabels([])

    frame2  = fig.add_axes((.1,.1,.8,.2))   
    frame2.axhline(color="gray", ls='--')
    rms     = np.sqrt(np.var(RV2_pos - RV_X))
    plt.plot(x, RV2_pos - RV_X , 'k.', label=r'rms$=%.2f m/s$' %rms)
    plt.xlim(0,200) 
    plt.xlabel(r"$t$")
    plt.ylabel("Residual (m/s)")
    plt.legend()
    plt.savefig(new_dir + '5-Fit1.png')
    plt.close('all')

    # Statistics #
    amplitude1[n]   = a[0]
    period1[n]      = k[0]

    amplitude2[n]   = a2[0]
    period2[n]      = k2[0]


#==============================================================================
# Hostogram
#==============================================================================    

# bins  = np.linspace(bin_min, bin_max, 15)
bins = 15

ax = plt.subplot(111)
ax.axvline(x=real_a, color='k', ls='-.')
# bins  = np.linspace(2, 7.5, 20)
plt.hist([amplitude1, amplitude2], bins, color=['r','b'], alpha=0.8)
# x_his = 1.0
# y_his = 14
# y_space = 1
# plt.text(x_his, y_his, 'FT correction', weight='bold', color='b')
# plt.text(x_his+0.1, y_his-y_space, r'1$\sigma$: %.2f' %(yes1_a1/N), color='b')
# plt.text(x_his+0.1, y_his-y_space*2, r'2$\sigma$: %.2f' %(yes1_a2/N), color='b')
# plt.text(x_his, y_his-y_space*3, 'No correction', weight='bold', color='r')
# plt.text(x_his+0.1, y_his-y_space*4, r'1$\sigma$: %.2f' %(yes2_a1/N), color='r')
# plt.text(x_his+0.1, y_his-y_space*5, r'2$\sigma$: %.2f' %(yes2_a2/N), color='r')
plt.xlabel('Amplitude (m/s)')
plt.ylabel('Count')
plt.savefig('Histogram_1.png')
plt.show()

ax = plt.subplot(111)
ax.axvline(x=real_k, color='k', ls='-.')
# bins  = np.linspace(3.65, 4.0, 20)
plt.hist([period1, period2], bins, color=['r','b'], alpha=0.8)
# x_his = 3.9
# y_his = 25
# y_space = y_his / 20
# plt.text(x_his, y_his, 'FT correction', weight='bold', color='b')
# plt.text(x_his+0.01, y_his-y_space, r'1$\sigma$: %.2f' %(yes1_k1/N), color='b')
# plt.text(x_his+0.01, y_his-y_space*2, r'2$\sigma$: %.2f' %(yes1_k2/N), color='b')
# plt.text(x_his, y_his-y_space*3, 'No correction', weight='bold', color='r')
# plt.text(x_his+0.01, y_his-y_space*4, r'1$\sigma$: %.2f' %(yes2_k1/N), color='r')
# plt.text(x_his+0.01, y_his-y_space*5, r'2$\sigma$: %.2f' %(yes2_k2/N), color='r')
plt.xlabel(r'$\nu_{orbital}$ / $\nu_{rot}$')
plt.ylabel('Count')
plt.savefig('Histogram_2.png')
plt.show()

print('FT correction:')
print(yes1_both1/N, yes1_both2/N)
print('No correction')
print(yes2_both1/N, yes2_both2/N)
Performance = np.vstack((np.hstack((yes1_both1/N, yes1_both2/N)), np.hstack((yes2_both1/N, yes2_both2/N))))
np.savetxt('Performance.txt', Performance, fmt='%.2f')


#==============================================================================
# Ende
#==============================================================================    
os.chdir('..')
time1 = time.time()
print('\nRuntime = %.2f seconds' %(time1 - time0))


if 0:
    plt.plot(np.array(t), RV_Y - 0.7602*np.array(RV_X), '.')
    plt.show()



