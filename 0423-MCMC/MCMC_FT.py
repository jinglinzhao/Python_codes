# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import gran_gen


#==============================================================================
# Setup
#==============================================================================

real_a      = 4.0
real_k      = 1.8
real_phi    = 1.0

N = 100
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
production_step = 3000

# Read Data 
t       = np.arange(200)
RV_IN0  = np.loadtxt('RV_IN.txt')
RV_FT0  = np.loadtxt('RV_FT.txt') 
RV_jitter   = np.loadtxt('RV_jitter.dat')
RV_jitter   = np.hstack((RV_jitter,RV_jitter))


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

    new_dir = str(n)
    os.makedirs(new_dir)
    os.chdir(new_dir)


    ################
    # Present data #
    ################

    # re-sample
    x       = gran_gen(n_group, n_obs)
    print('Observation samples:')
    print(x)
    x = np.sort(random.sample(range(200), 40))
    RV_IN   = np.array([RV_IN0[i] for i in x])
    RV_FT   = np.array([RV_FT0[i] for i in x])
    RV_diff = RV_IN - RV_FT


    if 1: # sub-sampling 
        fig = plt.figure()
        plt.plot(t, RV_IN0, 'o', label='full samples')
        plt.plot(x, RV_IN, 'r.', label='sub-sampling')
        plt.title('Sampling = %i)' % n_obs)
        plt.xlabel(r"$t$")
        plt.ylabel('Measured RV [m/s]')
        plt.legend()
        plt.savefig('0-RV_sampling.png')


    if 1:   # Time series 
        fig = plt.figure()
        frame1 = fig.add_axes((.1,.3,.8,.6))
        plt.plot(x, RV_IN, 'b^', label=r'RV$_{IN}$')
        plt.plot(x, RV_FT, 'r.', label=r'RV$_{FT}$')
        plt.title(r'Time Series $(N_{observation} = %i)$' % n_obs)
        plt.ylabel(r"$RV [m/s]$")
        plt.legend()

        frame2  = fig.add_axes((.1,.1,.8,.2))   
        plt.plot(x, RV_diff, 'k.', label='jitter model')
        plt.xlabel(r"$t$")
        plt.ylabel('Scaled jitter [m/s]')
        plt.savefig('1-Time_series.png')


    if 1:   # Comparison 
        fig = plt.figure()
        xx  = [min(np.hstack((RV_IN, RV_FT)))-0.5, max(np.hstack((RV_IN, RV_FT)))+0.5]
        plt.plot(xx, xx, '--', label=r'y = x')
        plt.plot(RV_IN, RV_FT, '*')
        plt.title('Comparison')
        plt.ylabel(r"$RV_{FT} [m/s]$")
        plt.xlabel(r"$RV_{IN} [m/s]$")
        plt.legend()
        plt.savefig('2-Comparison.png')


    #==============================================================================
    # MCMC without jitter correction
    #==============================================================================

    print('# MCMC without jitter correction #')

    # each data is equally weighted 
    yerr    = 1 + np.zeros(RV_IN.shape)

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
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(x, RV_IN, yerr)) # Note that running with multiple threads takes three times the time

    print("Running first burn-in...")
    pos2     = [[np.log(np.var(RV_IN)), np.log(5), 1., 0.] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, burn_in_1_step)

    print("Running second burn-in...")
    pos2 = [pos2[np.argmax(prob2)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, burn_in_2_step)

    print("Running production...")
    sampler2.run_mcmc(pos2, production_step)

    #==============================================================================
    # Trace and corner plots 
    #==============================================================================

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels_log=[r"$\log\ P$", r"$\log\ K$", r"$\omega$", r"$\delta$"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot( np.rot90(sampler2.chain[:, :, i], 3), "k", alpha=0.3)
        ax.set_xlim(0, sampler2.chain.shape[1])
        ax.set_ylabel(labels_log[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.savefig('3-Trace2.png')


    import copy
    log_samples         = sampler2.chain[:, 4000:, :].reshape((-1, ndim))
    real_samples        = copy.copy(log_samples)
    real_samples[:,0:2] = np.exp(real_samples[:,0:2])

    import corner
    fig = corner.corner(real_samples, labels=["$a$", "$k$", "$phi$", "$b$"], truths=[real_a, real_k, real_phi, 0],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('3-MCMC2.png')


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
    plt.plot(x, RV_IN, '^', label='data')
    plt.xlim(0,200) 
    plt.plot(x, RV2_pos, 'go', label='prediction')
    plt.plot(t, RV2_os, 'g', label='model')
    plt.title('Fit without FT correction')
    plt.ylabel(r"$RV [m/s]$")
    plt.legend()
    frame1.set_xticklabels([])

    frame2  = fig.add_axes((.1,.1,.8,.2))   
    frame2.axhline(color="gray", ls='--')
    rms     = np.sqrt(np.var(RV2_pos - RV_IN))
    plt.plot(x, RV2_pos - RV_IN , 'r.', label=r'rms$=%.2f m/s$' %rms)
    plt.xlim(0,200) 
    plt.xlabel(r"$t$")
    plt.ylabel(r"$residual\ [m/s]$")
    plt.legend()
    plt.savefig('5-Fit2.png')
    plt.close('all')


    #==============================================================================
    # MCMC with jitter correction (4 parameters)
    #==============================================================================

    if (mode == 4):
        print('# MCMC with jitter correction (4 parameters) #')

        def lnprior(theta):
            a, k, phi, b = theta
            if (-5 < a < 5) and (-5 < k < 5) and (-2*np.pi < phi < 2*np.pi) and (-10. < b < 10.):
                return 0.0
            return -np.inf

        # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
        def lnlike(theta, x, y, yerr):
            a, k, phi, b = theta
            m = 0.85
            model = np.exp(a) * np.sin(x/100. * np.exp(k) * 2. * np.pi + phi) + (RV_diff + b)/(1-m)    
            return -0.5*(np.sum( ((y-model)/yerr)**2. ))

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)    


        import emcee
        ndim    = 4
        nwalkers = 32
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_IN, yerr))

        print("Running first burn-in...")
        pos     = [[np.log(np.var(RV_IN)), np.log(5), 1., 0.] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_1_step)

        print("Running second burn-in...")
        pos = [pos[np.argmax(prob)] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_2_step)

        print("Running production...")
        sampler.run_mcmc(pos, production_step)


        #==============================================================================
        # Trace and corner plots 
        #==============================================================================

        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels_log=[r"$\log\ P$", r"$\log\ K$", r"$\omega$", r"$\delta$"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
            ax.set_xlim(0, sampler.chain.shape[1])
            ax.set_ylabel(labels_log[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        plt.savefig('3-Trace1.png')


        import copy
        log_samples         = sampler.chain[:, 4000:, :].reshape((-1, ndim))
        real_samples        = copy.copy(log_samples)
        real_samples[:,0:2] = np.exp(real_samples[:,0:2])

        import corner
        fig = corner.corner(real_samples, labels=["$a$", "$k$", "$phi$", "$b$"], truths=[real_a, real_k, real_phi, 0],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('3-MCMC1.png')


        #==============================================================================
        # Statistics
        #==============================================================================        
        a, k, phi, b = map(lambda v: np.array(v), zip(*np.percentile(real_samples, [50, 16, 84, 2.5, 97.5], axis=0)))

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

        print(np.vstack((a, k, phi, b)))

        m = [0.85]


    #==============================================================================
    # MCMC with jitter correction (5 parameters)
    #==============================================================================

    if (mode == 5):
        print('# MCMC with jitter correction (5 parameters) #')


        def lnprior(theta):
            a, k, phi, m, b = theta
            if (-5 < a < 5) and (-5 < k < 5) and (-2*np.pi < phi < 2*np.pi) and (-5 < m < 3) and (-10. < b < 10.):
                return 0.0
            return -np.inf

        # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
        def lnlike(theta, x, y, yerr):
            a, k, phi, m, b = theta
            model = np.exp(a) * np.sin(x/100. * np.exp(k) * 2. * np.pi + phi) + (RV_diff + b) * np.exp(m)
            # model = a * np.sin(x/100. * 7. * 2. * np.pi + phi) * (1-m) + b  + m * np.asarray(RV_IN)
            # model =  -a * np.sin(x/100. * k * 2. * np.pi + phi) * (1-m)/m + b  + np.asarray(RV_FT)/m
            return -0.5*(np.sum( ((y-model)/yerr)**2. ))

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)    


        import emcee
        ndim        = 5
        nwalkers    = 32
        sampler     = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_IN, yerr))

        print("Running first burn-in...")
        pos         = [[np.log(np.var(RV_IN)), np.log(5), 1., 0.75, 2.] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_1_step)

        print("Running second burn-in...")
        pos = [pos[np.argmax(prob)] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_2_step)

        print("Running production...")
        sampler.run_mcmc(pos, production_step)


        #==============================================================================
        # Trace and corner plots 
        #==============================================================================

        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels_log=[r"$\log\ P$", r"$\log\ K$", r"$\omega$", r"$m$", r"$\delta$"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
            ax.set_xlim(0, sampler.chain.shape[1])
            ax.set_ylabel(labels_log[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        plt.savefig('3-Trace1.png')


        import copy
        log_samples         = sampler.chain[:, 4000:, :].reshape((-1, ndim))
        real_samples        = copy.copy(log_samples)
        real_samples[:,0:2] = np.exp(real_samples[:,0:2])
        real_samples[:,3] = np.exp(real_samples[:,3])

        import corner
        fig = corner.corner(real_samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"], truths=[real_a, real_k, real_phi, 0.72, 0],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('3-MCMC1.png')


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
            yes1_both1 += 1
            print('Bingo - FT correction - 1 sigma')
        if (a[3] < real_a < a[4]) and (k[3] < real_k < k[4]):
            yes1_both2 += 1
            print('Bingo - FT correction - 2 sigma')

        print(np.vstack((a, k, phi, m, b)))


    #==============================================================================
    # FT correction Plots
    #==============================================================================    

    fig = plt.figure()
    RV_diff0    = RV_IN0 - RV_FT0
    Jitter_pos0 = RV_diff0 * m[0]
    # Jitter_pos0 = np.hstack((Jitter_pos0,Jitter_pos0))
    Jitter_pos  = RV_diff * m[0]
    Jitter_in   = RV_jitter - RV_jitter[0]
    Jitter_in   = np.array([Jitter_in[i] for i in x%100])
    plt.plot(x, Jitter_in - np.mean(Jitter_in), '*', label='Jitter_in')
    plt.plot(t, Jitter_pos0 - np.mean(Jitter_pos0), '-', label='Jitter_pos')
    plt.legend()
    plt.savefig('4-Jitter_correction.png')


    fig = plt.figure()
    RV_pos  = a[0] * np.sin(x/100. * k[0] * 2. * np.pi + phi[0]) + b[0] * m[0]
    RV_os   = a[0] * np.sin(t/100. * k[0] * 2. * np.pi + phi[0]) + b[0] * m[0]
    frame1  = fig.add_axes((.1,.3,.8,.6))
    frame1.axhline(color="gray", ls='--')
    plt.plot(x, RV_IN, '^', label='data')
    plt.plot(x, RV_pos, 'go')
    plt.plot(t, RV_os, 'g--', label='planet model')
    plt.plot(x, Jitter_pos0[x], 'k.', label='jitter correction')
    plt.xlim(0,200) 
    plt.title('Fit with FT correction')
    plt.ylabel(r"$RV [m/s]$")
    plt.legend()
    frame1.set_xticklabels([])

    frame2  = fig.add_axes((.1,.1,.8,.2))   
    frame2.axhline(color="gray", ls='--')
    rms     = np.sqrt(np.var(RV_IN - (RV_pos+Jitter_pos0[x])))
    plt.plot(x, RV_IN - (RV_pos+Jitter_pos0[x]), 'r.', label=r'rms$=%.2f m/s$' %rms)
    plt.xlim(0,200) 
    plt.xlabel(r"$t$")
    plt.ylabel(r"$residual [m/s]$")
    plt.legend()
    plt.savefig('5-Fit1.png')
    plt.close('all')


    # Statistics #
    amplitude1[n]   = a[0]
    period1[n]      = k[0]

    amplitude2[n]   = a2[0]
    period2[n]      = k2[0]

    # Finish # 
    os.chdir('..')


#==============================================================================
# Hostogram
#==============================================================================    

bin_min = min(min(amplitude1), min(amplitude2))
bin_max = max(max(amplitude1), max(amplitude2))
# bins  = np.linspace(bin_min, bin_max, 15)
bins = 15

histogram1 = plt.figure()
plt.hist([amplitude1, amplitude2], bins, color=['b','r'], alpha=0.8)
# label = ['FT correction 1$\sigma$: %.2f' %percentage1, 'No correction 1$\sigma$ %.2f' %percentage2]
x_his = 1.5
y_his = 35
plt.text(x_his, y_his, 'FT correction', weight='bold', color='b')
plt.text(x_his+0.1, y_his-2.5, r'1$\sigma$: %.2f' %(yes1_a1/N), color='b')
plt.text(x_his+0.1, y_his-5, r'2$\sigma$: %.2f' %(yes1_a2/N), color='b')
plt.text(x_his, y_his-7.5, 'No correction', weight='bold', color='r')
plt.text(x_his+0.1, y_his-10, r'1$\sigma$: %.2f' %(yes2_a1/N), color='r')
plt.text(x_his+0.1, y_his-12.5, r'2$\sigma$: %.2f' %(yes2_a2/N), color='r')
plt.xlabel('Amplitude')
plt.ylabel('Count')
# plt.legend()
plt.show()
plt.savefig('Histogram_1.png')

histogram2 = plt.figure()
plt.hist([period1, period2], bins, color=['b','r'], alpha=0.8)
x_his = 5
y_his = 60
plt.text(x_his, y_his, 'FT correction', weight='bold', color='b')
plt.text(x_his+0.1, y_his-5, r'1$\sigma$: %.2f' %(yes1_k1/N), color='b')
plt.text(x_his+0.1, y_his-10, r'2$\sigma$: %.2f' %(yes1_k2/N), color='b')
plt.text(x_his, y_his-15, 'No correction', weight='bold', color='r')
plt.text(x_his+0.1, y_his-20, r'1$\sigma$: %.2f' %(yes2_k1/N), color='r')
plt.text(x_his+0.1, y_his-25, r'2$\sigma$: %.2f' %(yes2_k2/N), color='r')
plt.xlabel('Amplitude')
plt.ylabel('Count')
plt.show()
plt.savefig('Histogram_2.png')

#==============================================================================
# Ende
#==============================================================================    
os.chdir('..')
time1 = time.time()
print('\nRuntime = %.2f seconds' %(time1 - time0))


if 0:
    plt.plot(np.array(t), RV_FT - 0.7602*np.array(RV_IN), '.')
    plt.show()



