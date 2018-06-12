import numpy as np

BIN_DATA = 0

#==============================================================================
# Read data 
#==============================================================================

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

out_dir = './Output-CoRot-7-Jitter_original/'
np.savetxt(out_dir + 'MJD_BIN_2012.txt', t_bin)
np.savetxt(out_dir + 'Jitter_BIN_2012.txt', y_bin)
np.savetxt(out_dir + 'Jitter_err_BIN_2012.txt', err_bin)


import matplotlib.pyplot as plt
if 0:
	fig = plt.figure()
	plt.errorbar(BJD, Jitter, yerr = err, fmt="o", capsize=0)
	# plt.errorbar(t_bin, y_bin, yerr = err_bin, fmt="*", capsize=0)
	plt.title('Jitter model for CoRot-7 in 2012')
	plt.xlabel('MJD')
	plt.ylabel('Jitter model [m/s]')   
	plt.show() 


# convert between two scripts # 
if BIN_DATA: 
	t 	 = t_bin
	y 	 = y_bin
	yerr = err_bin
else:
	t 	 = BJD
	y 	 = Jitter
	yerr = err


#==============================================================================
# Gaussian Processes
#==============================================================================
import george
from george import kernels

# kernel1 =  kernels.Matern32Kernel(5)
kernel2 = np.var(y) * kernels.ExpSine2Kernel(gamma=2., log_period=np.log(23.81/3))
kernel2 *= kernels.ExpSquaredKernel(5)
# kernel = kernel1 + kernel2
kernel = kernel2
# kernel.freeze_parameter("k1:k2:log_period")


if 0:	# generate a test GP time series 
	gp = george.GP(kernel)
	y_test = gp.sample(BJD)
	plt.errorbar(t, y_test, yerr=yerr, fmt=".k", capsize=0)
	plt.show()


#gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
# gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
gp = george.GP(kernel)


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
    if np.any((-5 > p) + (p > 5)): # or np.any((p[:,3] > 5) + (p[:,3] < 1))
        return -np.inf

	# Update the kernel and compute the lnlikelihood.
    kernel.pars = np.exp(p) 
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)


########
# MCMC #
########

import emcee

gp.compute(t, yerr)

# Set up the sampler.
nwalkers, ndim = 32, len(gp)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Initialize the walkers.
p_0 = gp.get_parameter_vector()
p0 	= np.log(p_0) + 1e-4 * np.random.randn(nwalkers, ndim)

print("Running burn-in")
p0, prob, state = sampler.run_mcmc(p0, 1000)
sampler.reset()

print("Running second burn-in...")
p0 = [p0[np.argmax(prob)] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
p0, prob, state  = sampler.run_mcmc(p0, 1000)
sampler.reset()

print("Running production chain")
sampler.run_mcmc(p0, 2000);
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
samples = np.exp(samples)



x_pred = np.linspace(min(t), max(t), 1000)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="b", alpha=0.2)
plt.plot(x_pred, pred, "b", lw=1.5, alpha=0.5)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD")
plt.ylabel("Jitter model [m/s]");
plt.title("Fit with GP noise model");
# plt.savefig('GP_fit.png')
plt.show()

import corner 
fig = corner.corner(samples, labels = gp.get_parameter_names())
plt.show()



fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = gp.get_parameter_names()
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()


pred_t, pred_var_t = gp.predict(y, t, return_var=True)
if BIN_DATA: 
	np.savetxt('GP_y_2012_bin.txt', pred_t)
	np.savetxt('GP_err_2012_bin.txt', np.sqrt(pred_var_t))
else:
	np.savetxt('GP_y_2012.txt', pred_t)
	np.savetxt('GP_err_2012.txt', np.sqrt(pred_var_t))



###############
# Periodogram #
###############
if 0:
	import scipy.signal as signal
	nout = 100000
	f = np.linspace(0.01, 10, nout)
	pgram1 = signal.lombscargle(x_pred, pred, f, normalize=True)
	pgram2 = signal.lombscargle(t_bin, y_bin, f, normalize=True)
	pgram3 = signal.lombscargle(BJD, Jitter, f, normalize=True)
	plt.plot(1/f, pgram1, label='GP_binned')
	plt.plot(1/f, pgram2, label='Raw_binned')
	plt.plot(1/f, pgram3, label='Raw')
	plt.xlim([0.0, 4])
	plt.legend()
	plt.show()



print(gp.log_likelihood(y))
print(gp.get_parameter_names())
print(gp.get_parameter_vector())





