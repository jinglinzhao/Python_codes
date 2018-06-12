# http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html

import numpy as np
import matplotlib.pyplot as plt


# Generate synthetic data from a model.
# For simplicity, let us assume a LINEAR model y = m*x + b
# where we want to fit m and b
m_true = -0.9594
b_true = 4.294
N = 50

x = np.sort(10*np.random.rand(N))
yerr = 0.2 + 0.5*np.random.rand(N)
y = m_true*x + b_true
y += yerr * np.random.randn(N)

fig = plt.figure()
# fig.set_size_inches(12, 8)
plt.errorbar(x, y, yerr=yerr, fmt='.k')
# plt.show()



# Now, let's setup some parameters that define the MCMC
ndim = 2
nwalkers = 500

# Initialize the chain
# Choice 1: chain uniformly distributed in the range of the parameters
pos_min = np.array([-5., 0.])
pos_max = np.array([5., 10.])
psize = pos_max - pos_min
# pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]
pos = [[m_true*1.2,b_true*0.8] + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]

# Visualize the initialization
import corner
fig = corner.corner(pos, labels=["$m$", "$b$"], truths=[m_true, b_true])
plt.show()
# fig.savefig("triangle.png")



# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    m, b = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf

'''
def lnprior(theta):
    m, b = theta
    if (-5.0 > m) or (m > 0.5) or (0.0 > b) or (b > 10.0):
    	return -np.inf
'''

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    return -0.5*(np.sum( ((y-model)/yerr)**2. ))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)



# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))



import time
time0 = time.time()
# perform MCMC
pos, prob, state  = sampler.run_mcmc(pos, 300)
time1=time.time()
print(time1-time0)

samples = sampler.flatchain
samples.shape



#let's plot the results
fig = corner.corner(samples, labels=["$m$", "$b$"], range=[[-1.1, -0.8], [3.5, 5.]],
                      truths=[m_true, b_true])
# fig.set_size_inches(10,10)
plt.show()















