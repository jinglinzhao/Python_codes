# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from functions import gran_gen


#########
# Setup #
#########

real_a      = 4.0
real_k      = 1.8
real_phi    = 1.0

N = 50
amplitude1  = np.zeros(N)
period1     = np.zeros(N)
yes1        = 0
yes1_2sigma = 0
amplitude2  = np.zeros(N)
period2     = np.zeros(N)
yes2        = 0
yes2_2sigma = 0

mode    = 4; 
# mode    = 5;
n_group = 6
n_obs   = 40

burn_in_1_step  = 3000
burn_in_2_step  = 3000
production_step = 3000
remove_step     = 1000

# Read Data 
t       = np.arange(200)
RV_IN0  = np.loadtxt('RV_IN.txt')
RV_FT0  = np.loadtxt('RV_FT.txt') 
RV_jitter   = np.loadtxt('RV_jitter.dat')
RV_jitter   = np.hstack((RV_jitter,RV_jitter))


#########
# Begin #
#########

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
    # x = np.sort(random.sample(range(200), 100))
    RV_IN   = np.array([RV_IN0[i] for i in x])
    RV_FT   = np.array([RV_FT0[i] for i in x])
    RV_diff = RV_IN - RV_FT


    if 1: # sub-sampling 
        fig = plt.figure()
        plt.plot(t, RV_IN0, 'o', label='full samples')
        plt.plot(x, RV_IN, 'r.', label='sub-sampling')
        plt.title('Sampling = %i)$' % n_obs)
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



    ##################################
    # MCMC without jitter correction #
    ##################################

    print('# MCMC without jitter correction #')

    # each data is equally weighted 
    yerr    = 1 + np.zeros(RV_IN.shape)

    def lnprior2(theta2):
        a2, k2, phi2, b2 = theta2
        if (0. < a2 < 10) and (0 < k2 < 10) and (-10 < phi2 < 10) and (-10. < b2 < 10):
            return 0.0
        return -np.inf

    # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
    def lnlike2(theta2, x, y, yerr):
        a2, k2, phi2, b2 = theta2
        model = a2 * np.sin(x/100. * k2 * 2. * np.pi + phi2) + b2
        return -0.5*(np.sum( ((y-model)/yerr)**2. ))

    def lnprob2(theta2, x, y):
        lp2 = lnprior2(theta2)
        if not np.isfinite(lp2):
            return -np.inf
        return lp2 + lnlike2(theta2, x, y, yerr)    



    import emcee
    ndim    = 4
    nwalkers = 32
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(x, RV_IN))
    pos2     = [[(max(RV_IN)-min(RV_IN))/2, 5., 1., 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


    print("Running first burn-in...")
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, burn_in_1_step)
    sampler2.reset()


    if 1:
        print("Running second burn-in...")
        pos2 = [pos2[np.argmax(prob2)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
        pos2, prob2, state2  = sampler2.run_mcmc(pos2, burn_in_2_step)
        # samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
        sampler2.reset()


    print("Running production...")
    sampler2.run_mcmc(pos2, production_step);
    samples2 = sampler2.chain[:, remove_step:, :].reshape((-1, ndim))


    import corner
    fig = corner.corner(samples2, labels=["$a$", "$k$", "$phi$", "$b$"], truths=[real_a, real_k, real_phi, 0],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('3-MCMC2.png')


    # Statistics #
    a2, k2, phi2, b2 = map(lambda v: np.array(v), zip(*np.percentile(samples2, [50, 16, 84, 2.5, 97.5], axis=0)))

    if (a2[1] < real_a < a2[2]) and (k2[1] < real_k < k2[2]):
        yes2 += 1
        print('Bingo - No correction - 1 sigma')

    if (a2[3] < real_a < a2[4]) and (k2[3] < real_k < k2[4]):
        yes2_2sigma += 1
        print('Bingo - No correction - 2 sigma')

    print(np.vstack((a2, k2, phi2, b2)))


    fig = plt.figure()
    RV2_pos = a2[0] * np.sin(x/100. * k2[0] * 2. * np.pi + phi2[0]) + b2[0]
    RV2_os  = a2[0] * np.sin(t/100. * k2[0] * 2. * np.pi + phi2[0]) + b2[0]
    frame1  = fig.add_axes((.1,.3,.8,.6))
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



    ##############################################
    # MCMC with jitter correction (4 parameters) #
    ##############################################

    # Define the posterior PDF
    # Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
    # We take the logarithm since emcee needs it.

    # As prior, we assume an 'uniform' prior (i.e. constant prob. density)

    if (mode == 4):
        print('# MCMC with jitter correction (4 parameters) #')

        def lnprior(theta):
            a, k, phi, b = theta
            if (0. < a < 10) and (0 < k < 10) and (-10 < phi < 10) and (-3. < b < 3):
                return 0.0
            return -np.inf

        # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
        def lnlike(theta, x, y, yerr):
            a, k, phi, b = theta
            m = 0.77
            model = a * np.sin(x/100. * k * 2. * np.pi + phi) + (RV_diff + b)/(1-m)    
            return -0.5*(np.sum( ((y-model)/yerr)**2. ))

        def lnprob(theta, x, y):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)    


        import emcee
        ndim    = 4
        nwalkers = 32
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_IN))
        pos     = [[(max(RV_IN)-min(RV_IN))/2, 5, 1., 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


        print("Running first burn-in...")
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_1_step)
        sampler.reset()
        # fig1 = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"])

        if 1:
            print("Running second burn-in...")
            pos = [pos[np.argmax(prob)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
            pos, prob, state  = sampler.run_mcmc(pos, burn_in_2_step)
            sampler.reset()


        print("Running production...")
        sampler.run_mcmc(pos, production_step);
        samples = sampler.chain[:, remove_step:, :].reshape((-1, ndim))


        fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$b$"], truths=[real_a, real_k, real_phi, 0],
                quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('3-MCMC1.png')



        # Statistics #
        a, k, phi, b = map(lambda v: np.array(v), zip(*np.percentile(samples, [50, 16, 84, 2.5, 97.5], axis=0)))

        if (a[1] < real_a < a[2]) and (k[1] < real_k < k[2]):
            yes1 += 1
            print('Bingo - FT correction - 1 sigma')

        if (a[3] < real_a < a[4]) and (k[3] < real_k < k[4]):
            yes1_2sigma += 1
            print('Bingo - FT correction - 2 sigma')

        print(np.vstack((a, k, phi, b)))

        m = [0.77]



    ############################################## 
    # MCMC with jitter correction (5 parameters) #
    ############################################## 

    if (mode == 5):
        print('# MCMC with jitter correction (5 parameters) #')

        def lnprior(theta):
            a, k, phi, m, b = theta
            if (0. < a < 10) and (0 < k < 10) and (-2*np.pi < phi < 2*np.pi) and (0.65 < m < 0.85) and (-3. < b < 3):
                return 0.0
            return -np.inf

        # As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
        def lnlike(theta, x, y, yerr):
            a, k, phi, m, b = theta
            model = a * np.sin(x/100. * k * 2. * np.pi + phi) + (RV_diff + b)/(1-m)
            # model = a * np.sin(x/100. * 7. * 2. * np.pi + phi) * (1-m) + b  + m * np.asarray(RV_IN)
            # model =  -a * np.sin(x/100. * k * 2. * np.pi + phi) * (1-m)/m + b  + np.asarray(RV_FT)/m
            return -0.5*(np.sum( ((y-model)/yerr)**2. ))

        def lnprob(theta, x, y):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)    


        import emcee
        ndim    = 5
        nwalkers = 32
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_IN))
        pos     = [[(max(RV_IN)-min(RV_IN))/2, 5, 1., 0.75, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


        print("Running first burn-in...")
        pos, prob, state  = sampler.run_mcmc(pos, burn_in_1_step)
        sampler.reset()
        # fig1 = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"])

        if 1:
            print("Running second burn-in...")
            pos = [pos[np.argmax(prob)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
            pos, prob, state  = sampler.run_mcmc(pos, burn_in_2_step)
            sampler.reset()


        print("Running production...")
        sampler.run_mcmc(pos, production_step);
        samples = sampler.chain[:, remove_step:, :].reshape((-1, ndim))


        fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"], truths=[real_a, real_k, real_phi, 0.72, 0],
                quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('3-MCMC1.png')


        # Statistics #
        a, k, phi, m, b = map(lambda v: np.array(v), zip(*np.percentile(samples, [50, 16, 84, 2.5, 97.5], axis=0)))

        if (a[1] < real_a < a[2]) and (k[1] < real_k < k[2]):
            yes1 += 1
            print('Bingo - FT correction - 1 sigma')

        if (a[3] < real_a < a[4]) and (k[3] < real_k < k[4]):
            yes1_2sigma += 1
            print('Bingo - FT correction - 2 sigma')

        print(np.vstack((a, k, phi, b)))


    ###############
    # Final Plots # 
    ###############

    fig = plt.figure()
    RV_diff0    = RV_IN0 - RV_FT0
    Jitter_pos0 = (RV_diff0 + b[0]) / (1-m[0]) 
    # Jitter_pos0 = np.hstack((Jitter_pos0,Jitter_pos0))
    Jitter_pos  = (RV_diff + b[0]) / (1-m[0]) 
    Jitter_in   = RV_jitter - RV_jitter[0]
    Jitter_in   = np.array([Jitter_in[i] for i in x%100])
    plt.plot(x, Jitter_in - np.mean(Jitter_in), '*', label='Jitter_in')
    plt.plot(t, Jitter_pos0 - np.mean(Jitter_pos0), '-', label='Jitter_pos')
    plt.legend()
    plt.savefig('4-Jitter_correction.png')


    fig = plt.figure()
    RV_pos  = a[0] * np.sin(x/100. * k[0] * 2. * np.pi + phi[0]) + b[0]
    RV_os   = a[0] * np.sin(t/100. * k[0] * 2. * np.pi + phi[0]) + b[0]
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
    plt.savefig('5-Fit.png')
    plt.close('all')


    # Statistics #
    amplitude1[n]   = a[0]
    period1[n]      = k[0]


    amplitude2[n]   = a2[0]
    period2[n]      = k2[0]



    # Finish # 
    os.chdir('..')


##############
# Statistics # 
##############

percentage1 = yes1 / N
percentage2 = yes2 / N

bin_min = min(min(amplitude1), min(amplitude2))
bin_max = max(max(amplitude1), max(amplitude2))
# bins  = np.linspace(bin_min, bin_max, 15)
bins = 15

histogram1 = plt.figure()
plt.hist([amplitude1, amplitude2], bins, alpha=0.8, label = ['FT correction %.2f' %percentage1, 'No correction %.2f' %percentage2]  )
# plt.hist(amplitude1, bins, alpha=0.8, label = 'FT correction')
# plt.hist(amplitude2, bins, alpha=0.8, label = 'No correction')
plt.xlabel('Amplitude')
plt.ylabel('Count')
plt.legend()
plt.savefig('Histogram_1.png')

histogram2 = plt.figure()
plt.hist([period1, period2], bins, alpha=0.8, label = ['FT correction','No correction'])
# plt.hist(period1, num_bins, alpha=0.5, label = 'FT correction')
# plt.hist(period2, num_bins, alpha=0.5, label = 'No correction')
plt.xlabel('Period')
plt.ylabel('Count')
plt.legend()
plt.savefig('Histogram_2.png')




# Finish #
os.chdir('..')
time1 = time.time()
print('\nRuntime = %.2f seconds' %(time1 - time0))


if 0:
    plt.plot(np.array(t), RV_FT - 0.7602*np.array(RV_IN), '.')
    plt.show()



