import numpy as np

##########################
# Read and reformat data #
##########################

BJD 	= np.loadtxt('ksiboo_BJD.dat')
BJD 	= BJD.reshape((BJD.shape[0], 1))
data 	= np.loadtxt('ksiboo.dat')
ALL 	= np.concatenate((BJD, data), axis=1)

# array size #
#print(BJD.shape)
#print(data.shape)
#print(ALL.shape)
# BJD.shape 	= (16, 1)
# data.shape 	= (16, 6)
# ALL.shape 	= (16, 7)


##################
# Visualize data # 
##################
import matplotlib.pyplot as plt

if 0:
    plt.errorbar(ALL[:,0], 	ALL[:,1], 	yerr=ALL[:,2], 	fmt="o", capsize=0, label='Q')
    plt.errorbar(ALL[:,0], 	ALL[:,3], 	yerr=ALL[:,4], 	fmt="^", capsize=0, label='U')
    plt.errorbar(ALL[:,0], 	ALL[:,5], 	yerr=ALL[:,6], 	fmt="s", capsize=0, label='P')
    plt.ylabel('Polarization [ppm]')
    plt.xlabel(r"$JD$")
    plt.legend()
    plt.show()

# convert between two scripts # 
t 	 = ALL[:,0] - min(ALL[:,0])
y 	 = ALL[:,3]
yerr = ALL[:,4]



################
# Optimization # 
################
from george import kernels

k1 = 1**2 * kernels.ExpSquaredKernel(metric=10**2)
k2 = 1**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=1, log_period=1.8)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)
kernel = k1 + k2 + k3 + k4
#kernel = k1  +  k2

import george
#gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
gp.compute(t, yerr)
print(gp.log_likelihood(y))
print(gp.grad_log_likelihood(y))


import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t, yerr)

# Print the initial ln-likelihood.
print(gp.log_likelihood(y))

# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="Newton-CG")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(y))
print(np.exp(results.x[6]))
# 6.530548
# 5.7345
# 6.51176

# Make the maximum likelihood prediction
x = np.linspace(min(t), max(t), 1000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)

# Plot the data
# plt.figure()
color = "#ff7f0e"
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, mu, color=color)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("Column3 - maximum likelihood prediction");
plt.savefig('ksiboo-prediction-2.png')
plt.show()

##############################
# Sampling & Marginalization # 
##############################

def lnprob(p):
    # Trivial uniform prior.
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)


########
# MCMC #
########

import emcee

gp.compute(t, yerr)

# Set up the sampler.
nwalkers, ndim = 36, len(gp)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Initialize the walkers.
p0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 500)

print("Running production chain")
sampler.run_mcmc(p0, 500);


##########
# Plot 1 #
##########
'''
x = np.linspace(min(t), max(t), 250)
for i in range(50):
    # Choose a random walker and step.
    w = np.random.randint(sampler.chain.shape[0])
    n = np.random.randint(sampler.chain.shape[1])
    gp.set_parameter_vector(sampler.chain[w, n])

    # Plot a single sample.
    plt.plot(x, gp.sample_conditional(y, x), "g", alpha=0.1)

plt.plot(t, y, ".k")

plt.xlim(t.min(), t.max())
plt.xlabel("BJD")
plt.ylabel("Polarization [ppm]");
'''

##########
# Plot 2 #  
##########
x_pred = np.linspace(min(t), max(t), 250)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="b", alpha=0.2)
plt.plot(x_pred, pred, "b", lw=1.5, alpha=0.5)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD")
plt.ylabel("Polarization [ppm]");
plt.title("Fit with GP noise model");
plt.show()


#==============================================================================
# Corner plots
#==============================================================================
import corner

tri_cols = ['k1:k1:log_constant', 'k1:k2:metric:log_M_0_0', 'k2:gamma', 'k2:log_period']
#tri_labels = [r"$\n$", r"$\tau$", r"$\k$", r"$\w$", r"$\e_0$", r"$\offset$"]
tri_labels = ['k1:k1:log_constant', 'k1:k2:metric:log_M_0_0', 'k2:gamma', 'k2:log_period']
names = gp.get_parameter_names()
# inds = np.array([names.index("mean:"+k) for k in tri_cols])
corner.corner(sampler.flatchain[:, 1:5], labels=tri_labels)









