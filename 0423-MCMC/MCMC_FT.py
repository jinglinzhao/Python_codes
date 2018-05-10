# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 


import numpy as np
import matplotlib.pyplot as plt
import random


################
# Present data #
################

# Read Data 
t       = np.arange(100)
RV_IN0 = np.loadtxt('RV_IN.txt')
RV_FT0  = np.loadtxt('RV_FT.txt') 
RV_jitter   = np.loadtxt('RV_jitter.dat')

# re-sample
N_night = 5
x_start = np.sort(random.sample(range(95), N_night))

x       = np.hstack([i + np.sort(random.sample(range(6), random.randint(1, 5))) for i in x_start])
x       = np.unique(x)
n_obs   = np.size(x)
print(x)
# x       = np.sort(random.sample(range(100), 30))
RV_IN   = np.array([RV_IN0[i] for i in x])
RV_FT   = np.array([RV_FT0[i] for i in x])
RV_diff = RV_IN - RV_FT


if 1:   # Time series 
    fig = plt.figure()
    frame1  = fig.add_axes((.1,.3,.8,.6))
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
    # plt.show()




##################################
# MCMC without jitter correction #
##################################

import time
time0   = time.time()
yerr    = 1 + np.zeros(RV_IN.shape)
# yerr = RV_diff + 0.05

def lnprior2(theta2):
    a2, k2, phi2, b2 = theta2
    if (0.1 < a2 < 10) and (0 < k2 < 10) and (0.001 < phi2 < 2*np.pi) and (-10. < b2 < 10):
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
pos2     = [[0.8, 5., 1., 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


print("Running first burn-in...")
pos2, prob2, state2  = sampler2.run_mcmc(pos2, 3000)
sampler2.reset()


if 1:
    print("Running second burn-in...")
    pos2 = [pos2[np.argmax(prob2)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, 3000)
    # samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    sampler2.reset()


print("Running production...")
sampler2.run_mcmc(pos2, 2000);
samples2 = sampler2.chain[:, 500:, :].reshape((-1, ndim))



fig = corner.corner(samples2, labels=["$a$", "$k$", "$phi$", "$b$"],
        truths=[4, 7, 1.0, 0.0137])
plt.savefig('3-MCMC2.png')
# plt.show()


a2, k2, phi2, b2 = map(lambda v: np.array((v[1], v[2]-v[1], v[1]-v[0])), zip(*np.percentile(samples2, [16, 50, 84], axis=0)))
print(np.vstack((a2, k2, phi2, b2)))


fig = plt.figure()
RV2_pos = a2[0] * np.sin(x/100. * k2[0] * 2. * np.pi + phi2[0]) + b2[0]
RV2_os  = a2[0] * np.sin(t/100. * k2[0] * 2. * np.pi + phi2[0]) + b2[0]
frame1  = fig.add_axes((.1,.3,.8,.6))
plt.plot(x, RV_IN, 'b^', label='data')
plt.xlim(0,100) 
plt.plot(x, RV2_pos, 'ro', label='prediction')
plt.plot(t, RV2_os, 'g', label='model')
plt.title('Fit without FT correction')
plt.ylabel(r"$RV [m/s]$")
plt.legend()
frame1.set_xticklabels([])

frame2  = fig.add_axes((.1,.1,.8,.2))   
frame2.axhline(color="gray", ls='--')
rms     = np.sqrt(np.var(RV2_pos - RV_IN))
plt.plot(x, RV2_pos - RV_IN , 'r.', label=r'rms$=%.2f m/s$' %rms)
plt.xlim(0,100) 
plt.xlabel(r"$t$")
plt.ylabel(r"$residual\ [m/s]$")
plt.legend()
plt.savefig('5-Fit2.png')
plt.close('all')



###############################
# MCMC with jitter correction #
###############################

# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)


def lnprior(theta):
    a, k, phi, m, b = theta
    # a, phi, m, b = theta
    # if (0 < a < 5.0) and (0.0 < phi < 2*np.pi) and (0 < m < 1) and (-10.0 < b < 10.0):
    if (0.1 < a < 10) and (0 < k < 10) and (0.001 < phi < np.pi) and (0.5 < m < 1) and (-3. < b < 3):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    a, k, phi, m, b = theta
    # a, phi, m, b = theta
    model = a * np.sin(x/100. * k * 2. * np.pi + phi) + (RV_diff + b)/(1-m)
    # model = a * np.sin(x/100. * 7. * 2. * np.pi + phi) * (1-m) + b  + m * np.asarray(RV_IN)
    # model =  -a * np.sin(x/100. * k * 2. * np.pi + phi) * (1-m)/m + b  + np.asarray(RV_FT)/m
    # print(model)
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
pos     = [[a2[0], k2[0], 1., 0.8, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


print("Running first burn-in...")
pos, prob, state  = sampler.run_mcmc(pos, 3000)
sampler.reset()
# fig1 = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"])

if 1:
    print("Running second burn-in...")
    pos = [pos[np.argmax(prob)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos, prob, state  = sampler.run_mcmc(pos, 3000)
    sampler.reset()


print("Running production...")
sampler.run_mcmc(pos, 2000);
samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))


import corner
fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"],
        truths=[4, 7, 1.0, 0.74, 0.0137])
plt.savefig('3-MCMC1.png')
# plt.show()


a, k, phi, m, b = map(lambda v: np.array((v[1], v[2]-v[1], v[1]-v[0])), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print(np.vstack((a, k, phi, m, b)))


fig = plt.figure()
RV_diff0    = RV_IN0 - RV_FT0
Jitter_pos0 = (RV_diff0 + b[0]) / (1-m[0]) 
Jitter_pos  = (RV_diff + b[0]) / (1-m[0]) 
Jitter_in   = RV_jitter - RV_jitter[0]
Jitter_in   = np.array([Jitter_in[i] for i in x])
plt.plot(x, Jitter_in - np.mean(Jitter_in), '*', label='Jitter_in')
plt.plot(t, Jitter_pos0 - np.mean(Jitter_pos0), '-', label='Jitter_pos')
plt.legend()
plt.savefig('4-Jitter_correction.png')
# plt.show()


fig = plt.figure()
RV_pos  = a[0] * np.sin(x/100. * k[0] * 2. * np.pi + phi[0]) + Jitter_pos 
RV_os   = a[0] * np.sin(t/100. * k[0] * 2. * np.pi + phi[0]) + Jitter_pos0
frame1  = fig.add_axes((.1,.3,.8,.6))
plt.plot(x, RV_IN, '^', label='data')
plt.plot(x, RV_pos, 'ro', label='prediction')
plt.plot(t, RV_os, 'g--', label='model')
plt.xlim(0,100) 
plt.title('Fit with FT correction')
plt.ylabel(r"$RV [m/s]$")
plt.legend()
frame1.set_xticklabels([])

frame2  = fig.add_axes((.1,.1,.8,.2))   
frame2.axhline(color="gray", ls='--')
rms     = np.sqrt(np.var(RV_IN - RV_pos))
plt.plot(x, RV_IN - RV_pos, 'r.', label=r'rms$=%.2f m/s$' %rms)
plt.xlim(0,100) 
plt.xlabel(r"$t$")
plt.ylabel(r"$residual [m/s]$")
plt.legend()
plt.savefig('5-Fit.png')
# plt.show()


time1=time.time()
print(time1-time0)


if 0:
    plt.plot(np.array(t), RV_FT - 0.7602*np.array(RV_IN), '.')
    plt.show()









