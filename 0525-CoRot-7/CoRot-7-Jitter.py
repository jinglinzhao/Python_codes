import numpy as np

BIN_DATA = 1

#############
# Read data #
#############

BJD 	= np.loadtxt('MJD_2012.txt')
Jitter 	= np.loadtxt('Jitter_model_2012.txt')
err  	= np.zeros(len(BJD)) + np.sqrt(2)

# bin the data of the same night - use a moving average # 

n_bin 	= int(max(BJD) - min(BJD) + 1)
t_bin 	= np.zeros(n_bin)
y_bin 	= np.zeros(n_bin)
err_bin = np.zeros(n_bin)

for i in range(n_bin):
	idx 	 = (BJD < (i+0.5)) & (BJD > (i-0.5))
	t_bin[i] = np.mean(BJD[idx])
	y_bin[i] = np.mean(Jitter[idx])
	err_bin[i]	 = np.sqrt(4/len(BJD[idx]))


##################
# Visualize data # 
##################
import matplotlib.pyplot as plt

plt.errorbar(BJD, Jitter, yerr = err, fmt="o", capsize=0)
# plt.errorbar(t_bin, y_bin, yerr = err_bin, fmt="*", capsize=0)
plt.title('Jitter model for CoRot-7 in 2012')
plt.xlabel('MJD')
plt.ylabel('Jitter model [m/s]')    


# convert between two scripts # 
if BIN_DATA: 
	t 	 = t_bin
	y 	 = y_bin
	yerr = err_bin
else:
	t 	 = BJD
	y 	 = Jitter
	yerr = err


################
# Optimization # 
################
from george import kernels

# kernel1 = 2.0 * kernels.Matern32Kernel(5)
kernel2 = 10 * kernels.ExpSine2Kernel(gamma=10.0, log_period=np.log(8))
kernel2 *= kernels.ExpSquaredKernel(5)
# kernel = kernel1 + kernel2
kernel = kernel2

import george
# gp = george.GP(kernel)
# y_test = gp.sample(BJD)
# gp.compute(BJD, yerr)
# plt.errorbar(t, y_test, yerr=yerr, fmt=".k", capsize=0)
# plt.show()


#gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)

# gp = george.GP(kernel)

if 0:
	gp.compute(t, yerr)
	print(gp.log_likelihood(y))
	print(gp.grad_log_likelihood(y))


if 0:
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



##############################
# Sampling & Marginalization # 
##############################

def lnprob(p):
    # Trivial uniform prior.
    if np.any((0 > p[1:]) + (p[1:] > 30)):
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
sampler.reset()

print("Running production chain")
sampler.run_mcmc(p0, 2000);


x_pred = np.linspace(min(t), max(t), 1000)
pred, pred_var = gp.predict(y, x_pred, return_var=True)


plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="b", alpha=0.2)
plt.plot(x_pred, pred, "b", lw=1.5, alpha=0.5)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD")
plt.ylabel("Jitter model [m/s]");
plt.title("Fit with GP noise model");
plt.show()

np.savetxt('x_gp.txt', x_pred)
if BIN_DATA:
	np.savetxt('y_gp_binned.txt', pred)
	np.savetxt('err_gp_binned.txt', pred_var)
	plt.savefig('GP_fit_binned.png')
else:
	np.savetxt('y_gp.txt', pred)
	np.savetxt('err_gp', pred_var)	
	plt.savefig('GP_fit.png')






