# based on HD85390-2_planet_0_jitter_2sets.py

import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn


#==============================================================================
# Import data 
#==============================================================================

all_rvs     = np.loadtxt('HD85390_quad.vels')
RV_HARPS    = np.loadtxt('RV_HARPS.dat')

x 		= all_rvs[:,0]
y       = (RV_HARPS-np.mean(RV_HARPS))*1000
yerr 	= all_rvs[:,2]

import time
import os
import shutil
time0   = time.time()
dir_name = str(time0) + 'celerite'
os.makedirs(dir_name)
shutil.copy('GP_85390_celerite_2_planets.py', dir_name +'/GP_85390_celerite_2_planets.py')  
os.chdir(dir_name)

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel("RV [m/s]")
plt.xlabel("Shifted JD [d]")
plt.savefig('HD85390-1-RV.png')
# plt.show()

#==============================================================================
# Model
#==============================================================================
import celerite
from celerite.modeling import Model

class Model(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'P2', 'tau2', 'k2', 'w2', 'e2', 'offset1', 'offset2')

    def get_value(self, t):

        # Planet 1
        M_anom1 = 2*np.pi/(100*self.P1) * (t - 1000*self.tau1)
        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))
        
        # Planet 2
        M_anom2 = 2*np.pi/(100*self.P2) * (t - 1000*self.tau2)
        e_anom2 = solve_kep_eqn(M_anom2, self.e2)
        f2      = 2*np.arctan( np.sqrt((1+self.e2)/(1-self.e2))*np.tan(e_anom2*.5) )
        rv2     = 100*self.k2*(np.cos(f2 + self.w2) + self.e2*np.cos(self.w2))

        offset      = np.zeros(len(t))
        idx         = t < 57300
        offset[idx] = self.offset1
        offset[~idx]= self.offset2

        return rv1 + rv2 + offset

mean_model = Model(P1=8., tau1=1., k1=np.std(y)/100, w1=0., e1=0.4, 
                   P2=100, tau2=1., k2=np.std(y)/100, w2=0., e2=0.4, offset1=0., offset2=0.)


#==============================================================================
# The fit
#==============================================================================
from scipy.optimize import minimize

import celerite
from celerite import terms

# Set up the GP model
kernel = terms.RealTerm(log_a=np.log(np.var(y)), log_c=-np.log(10.0))
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(x, yerr)
print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

# Define a cost function
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

# Fit for the maximum likelihood parameters
initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(soln.x)
print("Final log-likelihood: {0}".format(-soln.fun))

# Make the maximum likelihood prediction
t = np.linspace(-5, 5, 500)
mu, var = gp.predict(y, t, return_var=True)
std = np.sqrt(var)

# Plot the data
color = "#ff7f0e"
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(t, mu, color=color)
plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(-5, 5)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("maximum likelihood prediction");

#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(params):
    _, _, P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset1, offset2 = params
    if (7. < P1 < 9.0) and (0 < k1 < 0.1) and (-2*np.pi < w1 < 2*np.pi) and (0 < e1 < 0.9) and \
    (-50 < tau2 < 50) and (0. < k2 < 0.2) and (-2*np.pi < w2 < 2*np.pi) and (0 < e2 < 0.9):       
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
# def lnlike(theta):
#     P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset1, offset2 = theta
#     fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, 
#                         P2=P2, tau2=tau2, k2=k2, w2=w2, e2=e2, offset1=offset1, offset2=offset2)
#     y_fit       = fit_curve.get_value(x)
#     return -0.5*(np.sum( ((y-y_fit)/yerr)**2))

def lnprob(params):
    gp.set_parameter_vector(params)
    lp = lnprior(params)
    # lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y) + lp


import emcee
initial_params = gp.get_parameter_vector()
initial = np.array(initial_params)
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
pos = initial + 1e-8 * np.random.randn(nwalkers, ndim)
pos, prob, _ = sampler.run_mcmc(pos, 3000)


print("Running second burn-in...")
pos = pos[np.argmax(prob)] + 1e-8 * np.random.randn(nwalkers, ndim)
pos, prob, _  = sampler.run_mcmc(pos, 2000)

print("Running third burn-in...")
pos = pos[np.argmax(prob)] + 1e-8 * np.random.randn(nwalkers, ndim)
pos, prob, _  = sampler.run_mcmc(pos, 2000)

print("Running production...")
# pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
# pos, prob, state  = sampler.run_mcmc(pos, 3000)
sampler.run_mcmc(pos, 3000);

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, 5000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)
real_samples[:,3]   = 10*real_samples[:,3]
real_samples[:,8]   = 10*real_samples[:,8]
real_samples[:,2:5] = 100*real_samples[:,2:5]
real_samples[:,7:10] = 100*real_samples[:,7:10]
idx = real_samples[:,5] > 0
real_samples[idx,5] = real_samples[idx, 5] - 2*np.pi
idx = real_samples[:,8] < 1.5
real_samples[idx,10] = real_samples[idx, 10] + 2*np.pi

# idx_P2 = real_samples[:,5] < 60000


fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=["1", "2", r"$\frac{P_{1}}{100}$", r"$\frac{T_{1}}{100}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", 
            r"$\frac{P_{2}}{100}$", r"$\frac{T_{2}}{100}$", r"$\frac{K_{2}}{100}$", r"$\omega2$", r"$e2$", 
            "offset1", "offset2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('HD85390-2-Trace.png')
# plt.show()


import corner
labels=["1", "2", r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", "offset1", "offset2"]
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('HD85390-3-Corner.png')
# plt.show()


#==============================================================================
# Output
#==============================================================================

_, _, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((12,3))
aa[0,:] = [a0[i] for i in range(3)]
aa[1,:] = [a1[i] for i in range(3)]
aa[2,:] = [a2[i] for i in range(3)]
aa[3,:] = [a3[i] for i in range(3)]
aa[4,:] = [a4[i] for i in range(3)]
aa[5,:] = [a5[i] for i in range(3)]
aa[6,:] = [a6[i] for i in range(3)]
aa[7,:] = [a7[i] for i in range(3)]
aa[8,:] = [a8[i] for i in range(3)]
aa[9,:] = [a9[i] for i in range(3)]
aa[10,:]= [a10[i] for i in range(3)]
aa[11,:]= [a11[i] for i in range(3)]
np.savetxt('HD85390_fit.txt', aa, fmt='%.6f')


P1, tau1, k1, w1, e1, P2, tau2, k2, w2, e2, offset1, offset2 = aa[:,0]
fig = plt.figure(figsize=(10, 7))
frame1 = fig.add_axes((.15,.3,.8,.6))
frame1.axhline(y=0, color='k', ls='--', alpha=.3)
t_sample    = np.linspace(min(x), max(x), num=10001, endpoint=True)
# Planet 1 #
Planet1     = Model(P1=P1/100, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, 
                    P2=P2/100, tau2=tau2/1000, k2=0, w2=w2, e2=e2, offset1=offset1, offset2=0)
y1          = Planet1.get_value(t_sample)
plt.plot(t_sample, y1, 'b-.', alpha=.3, label='Planet1')
# Planet 2 #
Planet2     = Model(P1=P1/100, tau1=tau1/1000, k1=0, w1=w1, e1=e1, 
                    P2=P2/100, tau2=tau2/1000, k2=k2/100, w2=w2, e2=e2, offset1=0, offset2=offset2)
y2          = Planet2.get_value(t_sample)
plt.plot(t_sample, y2, 'b--', alpha=.3, label='Planet2')
# Planet1 + Planet2 #
y12         = y1 + y2
plt.plot(t_sample, y12, 'b-', alpha=.5, label='Planet1+Planet2')
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label='HARPS RV')
plt.legend()
plt.ylabel("Radial velocity [m/s]")

fit_curve   = Model(P1=P1/100, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, 
                    P2=P2/100, tau2=tau2/1000, k2=k2/100, w2=w2, e2=e2, offset1=offset1, offset2=offset2)
y_fit       = fit_curve.get_value(x)

residual    = y_fit - y
chi2        = sum(residual**2 / yerr**2)
rms         = np.sqrt(np.mean(residual**2))
wrms        = np.sqrt(sum((residual/yerr)**2) / sum(1/yerr**2))

frame2  = fig.add_axes((.15,.1,.8,.2))   
frame2.axhline(y=0, color='k', ls='--', alpha=.3)
plt.errorbar(x, residual, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("BJD - 2400000")
plt.ylabel('Residual [m/s]')
plt.savefig('HD85390-4-MCMC_fit.png')
plt.close("all")


# Make the maximum likelihood prediction
t = np.linspace(min(x), max(x), 10000)
mu, var = gp.predict(y, t, return_var=True)
std = np.sqrt(var)


# Plot the data
color = "#ff7f0e"
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(t, mu, color=color)
plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
# plt.xlim(-5, 5)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("maximum likelihood prediction");
plt.savefig('HD85390-5-prediction.png')


os.chdir('..')


















