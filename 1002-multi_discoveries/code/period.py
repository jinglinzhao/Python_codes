import numpy as np

#==============================================================================
# Read and reformat data 
#==============================================================================

star    = 'HD128621'

t   = np.loadtxt('../data/'+star+'/MJD.dat')
XX  = np.loadtxt('../data/'+star+'/RV_HARPS.dat')
XX  = (XX - np.mean(XX)) * 1000
yerr= np.loadtxt('../data/'+star+'/RV_noise.dat')

YY  = np.loadtxt('../data/'+star+'/YY.txt')
xy  = np.loadtxt('../data/'+star+'/xy.txt')
ZZ  = np.loadtxt('../data/'+star+'/ZZ.txt')


# convert between two scripts # 
t 	 = t - min(t)
y = XX-YY
# idx = (t<130)
# # y 	 = XX - YY 
# t = t[idx]
# y = xy[idx]
# yerr = yerr[idx]
#==============================================================================
# GP 
#==============================================================================

from george import kernels

k1 = 1**2 * kernels.ExpSquaredKernel(metric=10**2)
k2 = 1**2 * kernels.ExpSquaredKernel(80**2) * kernels.ExpSine2Kernel(gamma=8, log_period=np.log(36.2))
# boundary doesn't seem to take effect
# k2 = 1**2 * kernels.ExpSquaredKernel(80**2) * kernels.ExpSine2Kernel(gamma=1, log_period=np.log(36.2),
#                             bounds=dict(gamma=(-3,30), log_period=(np.log(36.2-5),np.log(36.2+6))))
k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
k4 = 1**2 * kernels.ExpSquaredKernel(40**2)
# kernel = k1 + k2 + k3 + k4
kernel = k2

import george
# gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
# gp.freeze_parameter('kernel:k2:log_period')
# gp.freeze_parameter('kernel:k2:gamma')


#==============================================================================
# Optimization 
#==============================================================================
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

# print the rotation period 
print(np.exp(results.x[4]))

#==============================================================================
# MCMC
#==============================================================================

# Define the objective function (negative log-likelihood in this case).
def lnprob(p):
    if np.any((results.x-0.5*abs(results.x) > p) + (p > results.x+0.5*abs(results.x))):
        return -np.inf
    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)    


import emcee

initial = results.x
names = gp.get_parameter_names()
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, 5000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)

fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=names
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('ksiboo-2-Trace.png')

import corner
fig = corner.corner(real_samples, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('ksiboo-3-Corner.png')

#==============================================================================
# Output
#==============================================================================
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = map(lambda v: 
    (v[1], v[0], v[2], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
aa = np.zeros((12,5))
aa[0,:] = [a0[i] for i in range(5)]
aa[1,:] = [a1[i] for i in range(5)]
aa[2,:] = [a2[i] for i in range(5)]
aa[3,:] = [a3[i] for i in range(5)]
aa[4,:] = [a4[i] for i in range(5)]
aa[5,:] = [a5[i] for i in range(5)]
aa[6,:] = [a6[i] for i in range(5)]
aa[7,:] = [a7[i] for i in range(5)]
aa[8,:] = [a8[i] for i in range(5)]
aa[9,:] = [a9[i] for i in range(5)]
aa[10,:] = [a10[i] for i in range(5)]
aa[11,:] = [a11[i] for i in range(5)]
np.savetxt('ksiboo_hat_p.txt', aa, fmt='%.6f')

#==============================================================================
# Plots
#==============================================================================

gp.set_parameter_vector(aa[:,0])
gp.set_parameter_vector(results.x)
# Make the maximum likelihood prediction
x = np.linspace(min(t-1), max(t+1), 1000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)
gp_predict = np.transpose(np.vstack((x,mu,std)))
# np.savetxt('gp_predict_hat(p).txt', gp_predict, fmt='%.8f')
# np.savetxt('gp_predict.txt', gp_predict, fmt='%.8f')


# x: oversampled time (column 1)
# mu: Gaussian processes prediction of the most likely value (column 2)
# std: standard deivation of walkers in all runs in MCMC (column 3)
color = "#ff7f0e"
fig = plt.figure(figsize=(18,6))
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
# plt.errorbar(MJD[~idx], XY[~idx], yerr=YERR[~idx], fmt=".r", alpha=0.1, capsize=0)
plt.plot(x, mu, color=color)
# plt.xlim(55275, 55365)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.xlabel("JD - 2,400,000")
# plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
# plt.title("hat(p) - maximum likelihood prediction (MCMC)");
# plt.savefig('../output/'+star+'.png') 
plt.title('2010-03-23..2010-06-12')
# plt.title('2011-02-18..2011-05-15')
# plt.title('2009-02-15..2009-05-06')
plt.savefig(star+'.png')
# plt.title("Maximum likelihood prediction");
# plt.savefig('ksiboo-prediction-4.png') 
# plt.title("sqrt(p) - maximum likelihood prediction");
# plt.savefig('ksiboo-prediction-4-MCMC.png') 
plt.show()

np.savetxt('plot_t.txt', t)
np.savetxt('plot_RV_HARPS.txt', XX[idx])
np.savetxt('plot_y.txt', y)
np.savetxt('plot_yerr.txt', yerr)
np.savetxt('plot_x.txt', x)
np.savetxt('plot_mu.txt', mu)
np.savetxt('plot_std.txt', std)

#==============================================================================
# For use of lining up the plots only
#==============================================================================
# Make the maximum likelihood prediction
x1 = np.linspace(min(t), max(t), 1000)
mu1, var1 = gp.predict(y, x, return_var=True)
std1 = np.sqrt(var)
t1 = t
y1 = y
yerr1 = yerr
t_off1 = min(ALL[:,0])

# color = "#ff7f0e"
plt.errorbar(t + min(ALL[:,0]), y, yerr=yerr, fmt=".b", capsize=0, label='sqrt(p)')
plt.errorbar(t1 + t_off1, y1, yerr=yerr1, fmt=".r", capsize=0, label='BI')
plt.plot(x + min(ALL[:,0]), mu, color='blue')
plt.plot(x1 + t_off1, mu1, color='red')
plt.fill_between(x + min(ALL[:,0]), mu+std, mu-std, color='blue', alpha=0.3, edgecolor="none")
plt.fill_between(x1 + t_off1, mu1+std1, mu1-std1, color='red', alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.title("");
plt.savefig('ksiboo-Line-up.png') 
plt.show()