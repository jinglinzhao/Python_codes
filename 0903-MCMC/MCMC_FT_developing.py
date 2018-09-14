# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 

# Based on MCMC_FT.py

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

N = 100
amplitude1      = np.zeros(N)
period1         = np.zeros(N)
amplitude1_XY   = np.zeros(N)
period1_XY      = np.zeros(N)
amplitude1_ZX   = np.zeros(N)
period1_ZX      = np.zeros(N)
amplitude1_XYZ  = np.zeros(N)
period1_XYZ     = np.zeros(N)
amplitude2      = np.zeros(N)
period2         = np.zeros(N)

yes1_a1     = 0     # FT correction: amplitude within 1 sigma
yes1_k1     = 0     # FT correction: period within 1 sigma
yes1_a2     = 0     # FT correction: amplitude within 2 sigma
yes1_k2     = 0     # FT correction: period within 2 sigma
yes1_both1_XY   = 0     # FT correction: both within 1 sigma
yes1_both2_XY   = 0     # FT correction: both within 2 sigma
yes1_both1_ZX   = 0     # FT correction: both within 1 sigma
yes1_both2_ZX   = 0     # FT correction: both within 2 sigma
yes1_both1_XYZ  = 0     # FT correction: both within 1 sigma
yes1_both2_XYZ  = 0     # FT correction: both within 2 sigma

yes2_a1     = 0     # No correction: amplitude within 1 sigma
yes2_k1     = 0     # No correction: period within 1 sigma
yes2_a2     = 0     # No correction: amplitude within 2 sigma
yes2_k2     = 0     # No correction: period within 2 sigma
yes2_both1  = 0     # No correction: both within 1 sigma
yes2_both2  = 0     # No correction: both within 2 sigma


mode    = 5; 
n_group = 12
n_obs   = 60

burn_in_1_step  = 2000
burn_in_2_step  = 1000
production_step = 2000

# Read Data 
GG  = np.loadtxt('GG.txt')
YY  = np.loadtxt('YY.txt')
ZZ  = np.loadtxt('ZZ.txt')
t   = np.arange(len(GG))
RV_jitter   = np.loadtxt('RV_jitter.txt')
RV_jitter   = RV_jitter - np.mean(RV_jitter)
RV_jitter   = np.hstack((RV_jitter,RV_jitter, RV_jitter, RV_jitter))


#==============================================================================
# Visualization
#==============================================================================

if 0: 
    # plt.plot(t, RV_jitter, '--', label='inout jitter')
    plt.plot(t, GG, 'k^', label=r'$RV_{Gaussian}$')
    plt.plot(t, YY, 'bo', label=r'$RV_{FT,L}$')
    plt.plot(t, ZZ, 'rs', label=r'$RV_{FT,H}$')
    plt.xlabel(r"$t$")
    plt.ylabel('RV [m/s]')
    plt.legend()
    plt.show()

# #==============================================================================
# # GP smoothing
# #==============================================================================
# import pymc3 as pm
# from theano import shared
# from pymc3.distributions.timeseries import GaussianRandomWalk
# from scipy import optimize
# import scipy.stats as stats

#==============================================================================
# Test smoothing
#==============================================================================
if 0:
    sl      = 8         # smoothing length
    YYs     = gaussian_smoothing(t, YY, t, sl)
    ZZs     = gaussian_smoothing(t, ZZ, t, sl)
    plt.plot(t, YYs, 'bo', label=r'$RV_{FT,L}$')
    plt.plot(t, ZZs, 'rs', label=r'$RV_{FT,H}$')
    plt.plot(t, GG, '-')
    # plt.show()
    # plt.plot(RV_jitter, GG-YY, 's')
    # plt.plot(RV_jitter, GG-YYs, 'o')
    # # plt.plot(RV_jitter, ZZ-GG, 's')
    # plt.plot(RV_jitter, ZZs-GG, 'o')
    plt.show()

#==============================================================================
# Begin looping
#==============================================================================

import time
time0   = time.time()
os.makedirs(str(time0))
os.chdir(str(time0))

for n in range(N):

    print('----' + '-' * len(str(n)))
    print('n = %i' %n)
    print('----' + '-' * len(str(n)))

    new_dir = str(n) + '/'
    os.makedirs(new_dir)

    #====================================
    # Present data
    #====================================

    # re-sample
    x       = gran_gen(n_group, n_obs)
    # x = np.sort(random.sample(range(200), n_obs))
    print('Observation samples:')
    print(x)    
    X   = np.array([GG[i] for i in x])
    Y   = np.array([YY[i] for i in x])
    Z   = np.array([ZZ[i] for i in x])
    J   = np.array([RV_jitter[i] for i in x])

    # smoothing 
    sl      = 2.5         # smoothing length
    XY      = gaussian_smoothing(x, X-Y, x, sl)
    ZX      = gaussian_smoothing(x, Z-X, x, sl)    

    Y_s     = gaussian_smoothing(x, Y, x, sl)
    Z_s     = gaussian_smoothing(x, Z, x, sl)
        # XY_s    = gaussian_smoothing(x, XY, x, sl)
        # ZX_s    = gaussian_smoothing(x, ZX, x, sl)
    if 0:
        XY  = X - Y_s
        ZX  = Z_s - X

    if 0:
        # plt.plot(x, Y, 'bo', label=r'$RV_{FT,L}$')
        # plt.plot(x, Ys, '-')
        plt.plot(x, Z, 'rs', label=r'$RV_{FT,H}$') 
        plt.plot(x, Zs, '-')    
        plt.show()

    if 1: # sub-sampling 
        fig = plt.figure()
        plt.plot(t, GG, 'o', label='full samples')
        plt.plot(x, X, 'r.', label='sub-sampling')
        plt.title('Sampling = %i' % n_obs)
        plt.xlabel(r"$t$")
        plt.ylabel('Measured RV [m/s]')
        plt.legend()
        plt.savefig(new_dir + '0-RV_sampling.png')

    if 1:   # Time series 
        fig = plt.figure()
        frame1 = fig.add_axes((.1,.3,.8,.6))
        plt.plot(x, Y_s, 'rs', label='FT_Y')
        plt.plot(x, X, 'bo', label='Gaussian')
        plt.title(r'$N_{sample} = %i$' % n_obs)
        plt.ylabel('RV (m/s)')
        plt.xlim(0,max(t))        
        plt.legend()

        frame2  = fig.add_axes((.1,.1,.8,.2))   
        plt.plot(x, XY, 'k.', label='jitter model')
        plt.xlabel(r"$t$")
        plt.ylabel(r'$\Delta$ RV (m/s)')
        plt.xlim(0,max(t))        
        plt.savefig(new_dir + '1_Y-Time_series.png')

    if 1:   # Time series 
        fig = plt.figure()
        frame1 = fig.add_axes((.1,.3,.8,.6))
        plt.plot(x, Z_s, 'rs', label='FT_Z')
        plt.plot(x, X, 'bo', label='Gaussian')
        plt.title(r'$N_{sample} = %i$' % n_obs)
        plt.ylabel('RV (m/s)')
        plt.xlim(0,max(t))         
        plt.legend()

        frame2  = fig.add_axes((.1,.1,.8,.2))   
        plt.plot(x, ZX, 'k.', label='jitter model')
        plt.xlabel(r"$t$")
        plt.ylabel(r'$\Delta$ RV (m/s)')
        plt.xlim(0,max(t))        
        plt.savefig(new_dir + '1_Z-Time_series.png')


    if 1:   # Comparison 
        fig = plt.figure()
        plt.plot(J, XY*4, '*', label='4*FT_Y')
        plt.plot(J, ZX, 'o', label='FT_Z')
        plt.title('Linearity')
        plt.ylabel(r"$RV [m/s]$")
        plt.xlabel("Jitter [m/s]")
        plt.legend()
        plt.savefig(new_dir + '2-Linearity.png')
        plt.close('all')

    #==============================================================================
    # MCMC without jitter correction
    #==============================================================================

    print('# MCMC without jitter correction #')

    # each data is equally weighted 
    yerr    = 0.5 + np.zeros(X.shape) # for S/N = 2000

    def lnprior2(theta2):
        a2, k2, phi2, b2 = theta2
        if (-5 < a2 <5) and (-5 < k2 <5) and (-2*np.pi < phi2 < 2*np.pi) and (-5. < b2 < 5):
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
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(x, X, yerr)) # Note that running with multiple threads takes three times the time

    print("Running first burn-in...")
    pos2     = [[np.log(np.var(X)), np.log(1), 1., 0.] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, burn_in_1_step)

    print("Running second burn-in...")
    pos2 = [pos2[np.argmax(prob2)] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, burn_in_2_step)

    print("Running production...")
    pos2 = [pos2[np.argmax(prob2)] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
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
    plt.errorbar(x/100, X, yerr=yerr, fmt="ok", capsize=0, label='Simulated RV', ecolor='blue', mfc='blue', mec='blue', alpha=0.5)
    plt.xlim(0, 4)
    plt.plot(x/100, RV2_pos, 'gs')
    plt.plot(t/100, RV2_os, 'g--', label='Planet model')
    plt.title('No correction')
    plt.ylabel("RV [m/s]")
    plt.legend()
    frame1.set_xticklabels([])

    frame2  = fig.add_axes((.1,.1,.8,.2))   
    frame2.axhline(color="gray", ls='--')
    rms     = np.sqrt(np.var(RV2_pos - X))
    plt.errorbar(x/100, X-RV2_pos, yerr=yerr, fmt=".k", capsize=0, alpha=0.5, label=r'rms$=%.2f$ m/s' %rms)
    plt.xlim(0, 4) 
    plt.xlabel(r"$P_{rot}$")
    plt.ylabel("Residual [m/s]")
    plt.legend()
    plt.savefig(new_dir + '5-Fit2.png')
    plt.close('all')


 
    #==============================================================================
    # MCMC with jitter correction
    #==============================================================================

    XYZ = 3.7784*XY+0.5*ZX
    for proto_jitter in [XY, ZX, XYZ]:

        if (proto_jitter == XY).all():
            print('# MCMC with XY jitter correction #')
            suffix = '_XY'
        if (proto_jitter == ZX).all():
            print('# MCMC with ZX jitter correction #')
            suffix = '_ZX'
        if (proto_jitter == XYZ).all():
            print('# MCMC with XYZ jitter correction #')
            suffix = '_XYZ'

        def lnprior(theta):
            a, k, phi, m, b = theta
            if (-5 < a < 5) and (-5 < k < 5) and (-2*np.pi < phi < 2*np.pi) and (-3 < m < 3) and (-5. < b < 5.):
                return 0.0
            return -np.inf

        # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
        def lnlike(theta, x, y, yerr):
            a, k, phi, m, b = theta
            model = np.exp(a) * np.sin(x/100. * np.exp(k) * 2*np.pi + phi) + (proto_jitter + b) * np.exp(m)
            return -0.5*(np.sum( ((y-model)/yerr)**2. ))

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)    

        import emcee
        ndim        = 5
        nwalkers    = 32
        sampler     = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, X, yerr))

        print("Running first burn-in...")
        pos         = [[np.log(np.var(X)), np.log(1), 1., 0., 0.] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_1_step)

        print("Running second burn-in...")
        pos = [pos[np.argmax(prob)] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_2_step)

        print("Running production...")
        pos = [pos[np.argmax(prob)] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        sampler.run_mcmc(pos, production_step)


        #==============================================================================
        # Trace and corner plots 
        #==============================================================================

        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels_log=[r"$\log\ A$", r"$\log\ \nu$", r"$\omega$", r"$\log\ m$", r"$b$"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
            ax.set_xlim(0, sampler.chain.shape[1])
            ax.set_ylabel(labels_log[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number");
        plt.savefig(new_dir + '3-Trace1' + suffix + '.png')


        import copy
        log_samples         = sampler.chain[:, 3000:, :].reshape((-1, ndim))
        real_samples        = copy.copy(log_samples)
        real_samples[:,0:2] = np.exp(real_samples[:,0:2])
        real_samples[:,3] = np.exp(real_samples[:,3])

        import corner
        fig = corner.corner(real_samples, labels=[r"$A$", r"$\nu$", r"$\omega$", r"$m$", r"$b$"], truths=[real_a, real_k, real_phi, 100, 100],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig(new_dir + '3-MCMC1' + suffix + '.png')

        #==============================================================================
        # Statistics
        #==============================================================================    
        a, k, phi, m, b = map(lambda v: np.array(v), zip(*np.percentile(real_samples, [50, 16, 84, 2.5, 97.5], axis=0)))

        if (a[1] < real_a < a[2]):
            yes1_a1 += 1
        if (k[1] < real_k < k[2]):
            yes1_k1 += 1
        if (a[3] < real_a < a[4]):
            yes1_a2 += 1
        if (k[3] < real_k < k[4]):
            yes1_k2 += 1
        if (a[1] < real_a < a[2]) and (k[1] < real_k < k[2]):
            if (proto_jitter == XY).all():
                yes1_both1_XY += 1
            if (proto_jitter == ZX).all():
                yes1_both1_ZX += 1
            if (proto_jitter == XYZ).all():
                yes1_both1_XYZ += 1                                
            print('Bingo - FT correction - 1 sigma')
        if (a[3] < real_a < a[4]) and (k[3] < real_k < k[4]):
            if (proto_jitter == XY).all():
                yes1_both2_XY += 1
            if (proto_jitter == ZX).all():
                yes1_both2_ZX += 1
            if (proto_jitter == XYZ).all():
                yes1_both2_XYZ += 1                   
            print('Bingo - FT correction - 2 sigma')

        print(np.vstack((a, k, phi, m, b)))


        #==============================================================================
        # FT correction Plots
        #==============================================================================    

        # fig = plt.figure()
        # RV_diff0    = RV_IN0 - RV_FT0
        # Jitter_pos0 = (RV_diff0 + b[0]) * m[0]
        # # Jitter_pos0 = np.hstack((Jitter_pos0,Jitter_pos0))
        # Jitter_pos  = (RV_diff + b[0]) * m[0]
        # Jitter_in   = RV_jitter - RV_jitter[0]
        # Jitter_in   = np.array([Jitter_in[i] for i in x%100])
        # plt.plot(x, Jitter_in - np.mean(Jitter_in), '*', label='Jitter_in')
        # plt.plot(x, Jitter_pos - np.mean(Jitter_pos), 'o', label='Jitter_pos')
        # plt.legend()
        # plt.savefig(new_dir + '4-Jitter_correction.png')

        fig = plt.figure()
        RV_pos  = a[0] * np.sin(x/100. * k[0] * 2. * np.pi + phi[0])
        RV_os   = a[0] * np.sin(t/100. * k[0] * 2. * np.pi + phi[0])
        Jitter_pos = (proto_jitter + b[0]) * m[0]
        frame1  = fig.add_axes((.1,.3,.8,.6))
        frame1.axhline(color="gray", ls='--')
        plt.errorbar(x/100, X, yerr=yerr, fmt="ok", capsize=0, label='Simulated RV', ecolor='blue', mfc='blue', mec='blue', alpha=0.5)
        plt.plot(x/100, RV_pos, 'gs')
        plt.plot(t/100, RV_os, 'g--', label='Planet model')
        plt.plot(x/100, Jitter_pos, '.', label='Jitter correction', color='darkorange')
        plt.xlim(0, 4)
        plt.title('Jitter correction')
        plt.ylabel("RV [m/s]")
        plt.legend()
        frame1.set_xticklabels([])

        frame2  = fig.add_axes((.1,.1,.8,.2))   
        frame2.axhline(color="gray", ls='--')
        res     = X - (Jitter_pos+RV_pos)
        rms     = 10
        if (np.sqrt(np.var(res)) < rms):
            rms = np.sqrt(np.var(res))
            amplitude1  = a[0]
            period1     = k[0]
        plt.errorbar(x/100, res, yerr=yerr, fmt=".k", capsize=0, alpha=0.5, label=r'rms$=%.2f$ m/s' %rms)
        plt.xlim(0,4)
        plt.xlabel(r"$P_{rot}$")
        plt.ylabel("Residual [m/s]")
        plt.legend()
        plt.savefig(new_dir + '5-Fit1' + suffix + '.png')
        plt.close('all')


        # Statistics #
        if (proto_jitter == XY).all():
            amplitude1_XY[n]   = a[0]
            period1_XY[n]      = k[0]
        if (proto_jitter == ZX).all():
            amplitude1_ZX[n]   = a[0]
            period1_ZX[n]      = k[0]
        if (proto_jitter == XYZ).all():
            amplitude1_XYZ[n]   = a[0]
            period1_XYZ[n]      = k[0]

    amplitude2[n]   = a2[0]
    period2[n]      = k2[0]


#==============================================================================
# Hostogram
#==============================================================================    

from scipy.stats import norm
if 0:
    for amp in [amplitude1_XY, amplitude1_ZX, amplitude1_XYZ, amplitude2]:
        mu, std = norm.fit(amp)
        std = std + 1
        x_plot = np.linspace(mu-3*std, mu+3*std, 100)
        p_plot = norm.pdf(x_plot, mu, std)
        plotting = plt.plot(x_plot, p_plot, linewidth=2)
    plt.legend(plotting[:], ['XY', 'ZX', 'XYZ', 'No correction'])
    plt.savefig('amp_dist.png')

mu, std = norm.fit(amplitude1_XY)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='XY')

mu, std = norm.fit(amplitude1_ZX)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='ZX')

mu, std = norm.fit(amplitude1_XYZ)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='XYZ')

mu, std = norm.fit(amplitude2)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='No correction')

plt.legend()

# plt.savefig('amp_dist.png')


# bins  = np.linspace(bin_min, bin_max, 15)
bins = 15
ax = plt.subplot(111)
ax.axvline(x=real_a, color='k', ls='-.')
plt.hist([amplitude1_XY, amplitude1_ZX, amplitude1_XYZ, amplitude2], bins, density=True, alpha=0.7)
plt.xlabel('Amplitude [m/s]')
plt.ylabel('Number density')
plt.savefig('Histogram_1.png')
plt.close('all')
# plt.show()


#####################

mu, std = norm.fit(period1_XY)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='XY')

mu, std = norm.fit(period1_ZX)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='ZX')

mu, std = norm.fit(period1_XYZ)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='XYZ')

mu, std = norm.fit(period2)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='No correction')

# plt.savefig('amp_dist.png')


# bins  = np.linspace(bin_min, bin_max, 15)
bins = 15
ax = plt.subplot(111)
ax.axvline(x=real_k, color='k', ls='-.')
plt.hist([period1_XY, period1_ZX, period1_XYZ, period2], bins, density=True, alpha=0.7)
plt.xlabel('nu')
plt.ylabel('Number density')
plt.savefig('Histogram_2.png')
plt.close('all')


######################
if 0:
    nu_mu1, nu_std1 = norm.fit(period1)
    x_plot = np.linspace(nu_mu1-3*nu_std1, nu_mu1+3*nu_std1, 100)
    p_plot = norm.pdf(x_plot, nu_mu1, nu_std1)
    plt.plot(x_plot, p_plot, 'r',  linewidth=2)

    nu_mu2, nu_std2 = norm.fit(period2)
    x_plot = np.linspace(nu_mu2-3*nu_std2, nu_mu2+3*nu_std2, 100)
    p_plot = norm.pdf(x_plot, nu_mu2, nu_std2)
    plt.plot(x_plot, p_plot, 'b',  linewidth=2)

    ax = plt.subplot(111)
    ax.axvline(x=real_k, color='k', ls='-.')
    # bins  = np.linspace(3.65, 4.0, 20)
    plt.hist([period1, period2], bins=30, density=True, color=['r','b'], alpha=0.7)
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
    # plt.show()

# print('FT correction:')
# print(yes1_both1/N, yes1_both2/N)
# print('No correction')
# print(yes2_both1/N, yes2_both2/N)
# Performance = np.vstack((np.hstack((yes1_both1/N, yes1_both2/N)), np.hstack((yes2_both1/N, yes2_both2/N))))
# np.savetxt('Performance.txt', Performance, fmt='%.2f')

mu, std = norm.fit(amplitude1)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='Jitter correction')

mu, std = norm.fit(amplitude2)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='No correction')

plt.legend()

bins = 15
ax = plt.subplot(111)
ax.axvline(x=real_a, color='k', ls='-.')
plt.hist([amplitude1, amplitude2], bins, density=True, alpha=0.7)
plt.xlabel('Amplitude [m/s]')
plt.ylabel('Number density')
plt.savefig('Histogram_new1.png')
plt.close('all')


mu, std = norm.fit(period1)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='Jitter correction')

mu, std = norm.fit(period2)
x_plot = np.linspace(mu-3*std, mu+3*std, 100)
p_plot = norm.pdf(x_plot, mu, std)
plotting = plt.plot(x_plot, p_plot, linewidth=2, label='No correction')

bins = 15
ax = plt.subplot(111)
ax.axvline(x=real_k, color='k', ls='-.')
plt.hist([period1, period2], bins, density=True, alpha=0.7)
plt.xlabel('nu')
plt.ylabel('Number density')
plt.savefig('Histogram_new2.png')
plt.close('all')

#==============================================================================
# Ende
#==============================================================================    
# os.chdir('..')
time1 = time.time()
print('\nRuntime = %.2f seconds' %(time1 - time0))


