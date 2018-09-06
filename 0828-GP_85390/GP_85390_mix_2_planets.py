# based on GP_85390_george_SE_2_planets.py

import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn

#==============================================================================
# Import data 
#==============================================================================

all_rvs     = np.loadtxt('HD85390_quad.vels')
RV_HARPS    = np.loadtxt('RV_HARPS.dat')

t 		= all_rvs[:,0]
x       = t
y       = (RV_HARPS-np.mean(RV_HARPS))*1000
yerr 	= all_rvs[:,2]

import time
import os
import shutil
time0   = time.time()
dir_name = 'george' + str(time0)
os.makedirs(dir_name)
shutil.copy('GP_85390_mix_2_planets.py', dir_name+'/GP_85390_mix_2_planets.py')  
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
import george
from george.modeling import Model

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

#==============================================================================
# GP
#==============================================================================
from george import kernels

from george import kernels

k1 = 1**2 * kernels.ExpSquaredKernel(metric=10**2)
k2 = 1**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=1, log_period=1.8)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)
kernel = k1 + k2 + k3 + k4

truth = dict(P1=8., tau1=1., k1=np.std(y)/100, w1=0., e1=0.4, 
            P2=100, tau2=1., k2=np.std(y)/100, w2=0., e2=0.4, offset1=0., offset2=0.)
kwargs = dict(**truth)
kwargs["bounds"] = dict(P1=(7.5,8.5), k1=(0,0.1), w1=(-2*np.pi,2*np.pi), e1=(0,0.9), 
					   tau2=(-50,50), k2=(0,0.2), w2=(-2*np.pi,2*np.pi), e2=(0,0.9))
mean_model = Model(**kwargs)
gp = george.GP(kernel, mean=mean_model, fit_mean=True)
gp.compute(t, yerr)

def lnprob2(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()           

#==============================================================================
# MCMC
#==============================================================================
import emcee

initial = gp.get_parameter_vector()
names = gp.get_parameter_names()
ndim, nwalkers = len(initial), 64
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 3000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, _, _ = sampler.run_mcmc(p0, 3000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, 6000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)
real_samples[:,1]   = 10*real_samples[:,1]
real_samples[:,6]   = 10*real_samples[:,6]
real_samples[:,0:3] = 100*real_samples[:,0:3]
real_samples[:,5:8] = 100*real_samples[:,5:8]
idx = real_samples[:,3] > 0
real_samples[idx,3] = real_samples[idx, 3] - 2*np.pi
idx = real_samples[:,8] < 3
real_samples[idx,8] = real_samples[idx, 8] + 2*np.pi


fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=[r"$\frac{P_{1}}{100}$", r"$\frac{T_{1}}{1000}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", 
            r"$\frac{P_{2}}{100}$", r"$\frac{T_{2}}{1000}$", r"$\frac{K_{2}}{100}$", r"$\omega2$", r"$e2$", 
            "offset1", "offset2", "1", "2", "3", "4", '5', '6', '7', '8', '9', '10', '11']
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
labels=[r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", 
		r"$P2$", r"$T_{2}$", r"$K2$", r"$\omega2$", r"$e2$", 
        "offset1", "offset2", "1", "2", "3", "4", '5', '6', '7', '8', '9', '10', '11']
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('HD85390-3-Corner.png')
# plt.show()


#==============================================================================
# Output
#==============================================================================
vv = np.zeros(11)
# aaa= np.zeros(12)
# np.hstack((vv, aaa)) = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))

a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11 = map(lambda v: 
	(v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
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

solution = np.zeros(len(gp))
solution[12] = v1[0]
solution[13] = v2[0]
solution[14] = v3[0]
solution[15] = v4[0]
solution[16] = v5[0]
solution[17] = v6[0]
solution[18] = v7[0]
solution[19] = v8[0]
solution[20] = v9[0]
solution[21] = v10[0]
solution[22] = v11[0]
solution[0:12] = aa[:,0]

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
gp.set_parameter_vector(solution)
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
plt.title("HD85390 - maximum likelihood prediction");
plt.savefig('HD85390-5-prediction.png')
plt.close('all')

# os.chdir('..')


















