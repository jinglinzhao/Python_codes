# Debug histroy: 
# increase the burn-in steps so that the corner plots are well sampled. 


import numpy as np
import matplotlib.pyplot as plt
import random



# Read Data 

t       = np.arange(100)
RV_tot  = np.loadtxt('RV_tot.txt')
RV_FT   = np.loadtxt('RV_FT.txt') 
RV_jitter   = np.loadtxt('RV_jitter.dat')

# re-sample
x       = np.sort(random.sample(range(100), 15))
RV_tot  = [RV_tot[i] for i in x]
RV_FT   = [RV_FT[i] for i in x]


if 1:   # Time series 
    fig = plt.figure()
    plt.plot(x, RV_tot, '*', label='RV_{tot}')
    plt.plot(x, RV_FT, '.', label='RV_{FT}')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$RV m/s]$")
    plt.legend()
    plt.savefig('Time_series.png')

if 0:   # Linearity 
    fig = plt.figure()
    plt.plot(RV_tot, RV_FT, '.')
    plt.ylabel(r"$RV_{FT} [m/s]$")
    plt.xlabel(r"$RV_{tot} [m/s]$")
    plt.savefig('Linearity.png')
    # plt.show()


# consider the errors 
fig = plt.figure()
RV_tot  = np.array(RV_tot)
RV_FT   = np.array(RV_FT)
RV_diff = RV_tot - RV_FT
yerr    = 1 + np.sin( (RV_diff - min(RV_diff)) / (max(RV_diff) - min(RV_diff)) * np.pi) * 0
plt.errorbar(np.array(x), RV_tot - RV_FT, yerr=yerr/10, fmt='.')
plt.xlabel(r"$t$")
plt.ylabel(r"$RV_{diff} -> jitter (scaled) [m/s]$")
plt.savefig('Scaled_jitter.png')
# plt.show()




# MCMC

# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    a, k, phi, m, b = theta
    # a, phi, m, b = theta
    # if (0 < a < 5.0) and (0.0 < phi < 2*np.pi) and (0 < m < 1) and (-10.0 < b < 10.0):
    if (0.1 < a < 10) and (0 < k < 10) and (0.001 < phi < np.pi) and (0.1 < m < 1) and (-3. < b < 3):
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
ndim    = 5
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, RV_tot), threads=30)
pos     = [[0.8, 5., 1., 0.8, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


import time
time0 = time.time()




print("Running first burn-in...")
pos, prob, state  = sampler.run_mcmc(pos, 3000)
sampler.reset()
# fig1 = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"])

if 1:
    print("Running second burn-in...")
    pos = [pos[np.argmax(prob)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos, prob, state  = sampler.run_mcmc(pos, 3000)
    # samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    sampler.reset()


print("Running production...")
sampler.run_mcmc(pos, 2000);
samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))




import corner
fig = corner.corner(samples, labels=["$a$", "$k$", "$phi$", "$m$", "$b$"],
        truths=[0.5, 7, 1.0, 0.7602, 0.0137])
plt.savefig('MCMC1.png')
# plt.show()



# a0, a1, a2, a3= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
a, k, phi, m, b = map(lambda v: np.array((v[1], v[2]-v[1], v[1]-v[0])), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print(a)
print(k)
print(phi)
print(m)
print(b)


fig = plt.figure()
Jitter_pos  = (RV_diff + b[0]) / (1-m[0]) 
Jitter_in   = RV_jitter - RV_jitter[0]
Jitter_in   = np.array([Jitter_in[i] for i in x])
plt.plot(x, Jitter_in - np.mean(Jitter_in), '*', label='Jitter_in')
plt.plot(x, Jitter_pos - np.mean(Jitter_pos), '.', label='Jitter_pos')
plt.legend()
plt.savefig('Jitter_correction.png')
# plt.show()

fig = plt.figure()
Planet_pos  = a[0] * np.sin(x/100. * k[0] * 2. * np.pi + phi[0]) 
plt.plot(x, RV_tot, '*', label='data')
plt.plot(x, Planet_pos, '.', label='fit')
plt.legend()
plt.savefig('Fit.png')
# plt.show()



##################################
# MCMC without jitter correction #
##################################

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
sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(x, RV_tot), threads=30)
pos2     = [[0.8, 5., 1., 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 


print("Running first burn-in...")
pos2, prob2, state2  = sampler2.run_mcmc(pos2, 3000)
sampler2.reset()


if 1:
    print("Running second burn-in...")
    pos2 = [pos2[np.argmax(prob2)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
    pos2, prob2, state2  = sampler2.run_mcmc(pos2, 3000)
    # samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    sampler.reset()


print("Running production...")
sampler2.run_mcmc(pos2, 2000);
samples2 = sampler2.chain[:, 500:, :].reshape((-1, ndim))





fig = corner.corner(samples2, labels=["$a$", "$k$", "$phi$", "$b$"],
        truths=[1, 7, 1.0, 0.0137])
plt.savefig('MCMC2.png')
# plt.show()
plt.close('all')



a2, k2, phi2, b2 = map(lambda v: np.array((v[1], v[2]-v[1], v[1]-v[0])), zip(*np.percentile(samples2, [16, 50, 84], axis=0)))
print(a2)
print(k2)
print(phi2)
print(b2)



time1=time.time()
print(time1-time0)


if 0:
    plt.plot(np.array(t), RV_FT - 0.7602*np.array(RV_tot), '.')
    plt.show()









