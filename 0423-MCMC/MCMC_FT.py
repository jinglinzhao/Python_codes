# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 


import numpy as np
import matplotlib.pyplot as plt
import random



# Read Data 

t       = np.arange(100)
RV_tot  = np.loadtxt('RV_tot.txt')
RV_FT   = np.loadtxt('RV_FT.txt') 

# re-sample
x       = np.sort(random.sample(range(100), 100))
RV_tot  = [RV_tot[i] for i in x]
RV_FT   = [RV_FT[i] for i in x]

if 1:   # Linearity 
    fig = plt.figure()
    plt.plot(RV_tot, RV_FT, '.')
    plt.ylabel(r"$RV_{FT} [m/s]$")
    plt.xlabel(r"$RV_{tot} [m/s]$")

if 1:   # Time series 
    fig = plt.figure()
    plt.plot(x, RV_tot, '-.', label='RV_{tot}')
    plt.plot(x, RV_FT, '--', label='RV_{FT}')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$RV m/s]$")
    plt.legend()
    plt.show()








# consider the errors 
RV_tot  = np.array(RV_tot)
RV_FT   = np.array(RV_FT)
RV_diff = RV_tot - RV_FT
yerr    = 1 + np.sin( (RV_diff - min(RV_diff)) / (max(RV_diff) - min(RV_diff)) * np.pi) * 0
yerr= yerr/10
plt.errorbar(np.array(t), RV_tot - RV_FT, yerr=yerr, fmt='.')
plt.xlabel(r"$t$")
plt.ylabel(r"$RV_{diff} -> jitter (scaled) [m/s]$")
plt.show()




# MCMC

# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    a, k, phi, m, b = theta
    # a, phi, m, b = theta
    # if (0 < a < 5.0) and (0.0 < phi < 2*np.pi) and (0 < m < 1) and (-10.0 < b < 10.0):
    if (0.1 < a < 2) and (6 < k < 8) and (0.001 < phi < np.pi) and (0.7 < m < 0.9) and (-1. < b < 1.0):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    a, k, phi, m, b = theta
    # a, phi, m, b = theta
    model = a * np.sin(x/100. * k * 2. * np.pi + phi) + (RV_diff + b)/(1-m)
    # model = a * np.sin(x/100. * 7. * 2. * np.pi + phi) * (1-m) + b  + m * np.asarray(RV_tot)
    # model =  -a * np.sin(x/100. * k * 2. * np.pi + phi) * (1-m)/m + b  + np.asarray(RV_FT)/m
    # print(model)
    return -0.5*(np.sum( ((y-model)/yerr)**2. ))

def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    



import emcee
ndim = 5
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_tot), threads=14)

# pos = [[4., np.pi, 0.8, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pos = [[0.8, 7., np.pi, 0.8, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


import time
time0 = time.time()
# burnin phase
pos, prob, state  = sampler.run_mcmc(pos, 2000)

samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
import corner
fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"],
        truths=[1, 7, 1.0, 0.7602, 0.0137])
# fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"],
#         range=[[0,5], [0,10], [0,2*np.pi], [0.5,1], [-10,10]],
#         truths=[1., 7, 1.0, 0.7602, 0.0283])
# fig = corner.corner(samples, labels=["$a$", "$phi$", "$m$", "$b$"])
plt.show()

sampler.reset()
# perform MCMC
pos, prob, state  = sampler.run_mcmc(pos, 20000)
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

time1=time.time()
print(time1-time0)


#let's plot the results
import corner
# fig = corner.corner(samples, labels=["$a$", "$phi$", "$m$", "$b$"])
fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"],
        truths=[1, 7, 1.0, 0.7602, 0.0137])
plt.show()


# a0, a1, a2, a3= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
a0, a1, a2, a3, a4 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print(a0)
print(a1)
print(a2)
print(a3)
print(a4)


plt.plot(np.array(t), RV_FT - 0.7602*np.array(RV_tot), '.')
plt.show()









