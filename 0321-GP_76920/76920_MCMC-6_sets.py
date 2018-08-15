#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Combine 76920_MCMC.py and Demo_76920_celerite-5_sets.py
to introduce 5 different offsets for the MCMC fitting
'''


#==============================================================================
# Import data 
#==============================================================================
import numpy as np


# all_rvs 	= np.genfromtxt('all_rvs_N102_outlier_removed.dat', dtype = None)
all_rvs     = np.genfromtxt('all_rvs_N102_outlier_and_5MJ_removed.dat', dtype = None)

for i in range(len(all_rvs)):
    all_rvs[i][2]     = (all_rvs[i][2]**2 + 7**2)**0.5 

DATA_AAT 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'AAT']
DATA_CHIRON = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'CHIRON']
DATA_FEROS 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FEROS']
DATA_MJ1 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ1']
DATA_MJ3 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ3']
DATA_FIDEOS = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FIDEOS']


RV_AAT 		= np.zeros( (len(DATA_AAT), 3) )
RV_CHIRON 	= np.zeros( (len(DATA_CHIRON), 3) )
RV_FEROS 	= np.zeros( (len(DATA_FEROS), 3) )
RV_MJ1 		= np.zeros( (len(DATA_MJ1), 3) )
RV_MJ3 		= np.zeros( (len(DATA_MJ3), 3) )
RV_FIDEOS   = np.zeros( (len(DATA_FIDEOS), 3) )

for k in range(len(DATA_AAT)):
	RV_AAT[k, :] 	= [ DATA_AAT[k][i] for i in range(3) ]

for k in range(len(DATA_CHIRON)):
	RV_CHIRON[k, :]	= [ DATA_CHIRON[k][i] for i in range(3) ]

for k in range(len(DATA_FEROS)):
	RV_FEROS[k, :]	= [ DATA_FEROS[k][i] for i in range(3) ]

for k in range(len(DATA_MJ1)):
	RV_MJ1[k, :]	= [ DATA_MJ1[k][i] for i in range(3) ]

for k in range(len(DATA_MJ3)):
	RV_MJ3[k, :]	= [ DATA_MJ3[k][i] for i in range(3) ]

for k in range(len(DATA_FIDEOS)):
    RV_FIDEOS[k, :] = [ DATA_FIDEOS[k][i] for i in range(3) ]    


# Concatenate the five data sets # 
RV_ALL  = np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3, RV_FIDEOS))
RV_SORT = sorted(RV_ALL, key=lambda x: x[0])
x       = [RV_SORT[i][0] for i in range(len(RV_SORT))]
y       = [RV_SORT[i][1] for i in range(len(RV_SORT))]
yerr    = [RV_SORT[i][2] for i in range(len(RV_SORT))]


#==============================================================================
# Model
#==============================================================================

import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'off_aat', 'off_chiron', 'off_feros', 'off_mj1', 'off_mj3', 'off_fideos')

    def get_value(self, t):
        M_anom  = 2*np.pi/np.exp(self.P) * (t.flatten() - np.exp(self.tau))
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        rv      = np.exp(self.k)*(np.cos(f + self.w) + self.e0*np.cos(self.w))

        offset = np.zeros(len(t))
        for i in range(len(t)):
            if t[i] in RV_AAT[:,0]:
                offset[i] = self.off_aat
            elif t[i] in RV_CHIRON[:,0]:
                offset[i] = self.off_chiron
            elif t[i] in RV_FEROS[:,0]:
                offset[i] = self.off_feros           
            elif t[i] in RV_MJ1[:,0]:
                offset[i] = self.off_mj1
            elif t[i] in RV_MJ3[:,0]:
                offset[i] = self.off_mj3             
            elif t[i] in RV_FIDEOS[:,0]:
                offset[i] = self.off_fideos

        return rv + 100*offset


# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
OFFSET_AAT      = 3.0/100
OFFSET_CHIRON   = -70./100
OFFSET_FEROS    = -8.4/100
OFFSET_MJ1      = -12.8/100
OFFSET_MJ3      = -54.4/100
OFFSET_FIDEOS   = -83.2/100
truth   = dict(log_P=np.log(415.9), log_tau=np.log(4812), log_k=np.log(186.8), w=-0.06, e0=0.856,
                off_aat=OFFSET_AAT, off_chiron=OFFSET_CHIRON, off_feros=OFFSET_FEROS, off_mj1=OFFSET_MJ1, off_mj3=OFFSET_MJ3, off_fideos=OFFSET_FIDEOS)



#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P, tau, k, w, e0, off_aat, off_chiron, off_feros, off_mj1, off_mj3, off_fideos = theta
    if (5.8 < P < 6.1) and (4.6 < k < 5.7) and (-np.pi < w < np.pi) and (0.7 < e0 < 0.99):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P, tau, k, w, e0, off_aat, off_chiron, off_feros, off_mj1, off_mj3, off_fideos = theta
    fit_curve   = Model(P=P, tau=tau, k=k, w=w, e0=e0, off_aat=off_aat, off_chiron=off_chiron, 
                        off_feros=off_feros, off_mj1=off_mj1, off_mj3=off_mj3, off_fideos=off_fideos)
    y_fit       = fit_curve.get_value(np.array(x))
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2. ))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    


import emcee
ndim = len(truth)
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
pos = [[6., 8.5, 5.3, 0, 0.8, OFFSET_AAT, OFFSET_CHIRON, OFFSET_FEROS, OFFSET_MJ1, OFFSET_MJ3, OFFSET_FIDEOS] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
# pos = [[6., 8.5, 5.3, 0, 0.8, 0., 0., 0., 0., 0., 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
pos, prob, state  = sampler.run_mcmc(pos, 3000)

print("Running second burn-in...")
pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
pos, prob, state  = sampler.run_mcmc(pos, 3000)

print("Running production...")
sampler.run_mcmc(pos, 10000);


time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
log_samples         = sampler.chain[:, 6000:, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,0:3] = np.exp(real_samples[:,0:3])
real_samples[:,5:]  = real_samples[:,5:]*100


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$\log\ P$", r"$\log\ T_{0}$", r"$\log\ K$", r"$\omega$", r"$e$", 
            r"$\frac{\delta_{AAT}}{100}$", r"$\frac{\delta_{CHIRON}}{100}$", 
            r"$\frac{\delta_{FEROS}}{100}$", r"$\frac{\delta_{MJ1}}{100}$", 
            r"$\frac{\delta_{MJ3}}{100}$", r"$\frac{\delta_{FIDEOS}}{100}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('76920_MCMC_6sets-2-Trace.png')
# plt.show()


import corner
labels=[r"$P$", r"$T_{0}$", r"$K$", r"$\omega$", r"$e$", r"$\delta_{AAT}$", r"$\delta_{CHIRON}$", 
            r"$\delta_{FEROS}$", r"$\delta_{MJ1}$", r"$\delta_{MJ3}$", r"$\delta_{FIDEOS}$"]
fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('76920_MCMC_6sets-1-Corner.png')
# plt.show()


#==============================================================================
# Output
#==============================================================================

a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((len(truth),3))
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
np.savetxt('76920_MCMC_6sets_result.txt', aa, fmt='%.6f')


aa = np.genfromtxt('76920_MCMC_6sets_5MJ_removed_0710/76920_MCMC_6sets_result-5MJ_removed.txt', dtype = None)
P, tau, k, w, e0, off_aat, off_chiron, off_feros, off_mj1, off_mj3, off_fideos = aa[:,0]
fit_curve = Model(P=np.log(P), tau=np.log(tau), k=np.log(k), w=w, e0=e0, off_aat=off_aat/100, off_chiron=off_chiron/100, 
                        off_feros=off_feros/100, off_mj1=off_mj1/100, off_mj3=off_mj3/100, off_fideos=off_fideos/100)
t_fit   = np.linspace(min(RV_ALL[:,0])-300, max(RV_ALL[:,0]+300), num=10001, endpoint=True)
y_fit   = fit_curve.get_value(np.array(t_fit))

residual= fit_curve.get_value(np.array(x)) - np.array(y)
chi2    = sum(residual**2 / np.array(yerr)**2)
rms     = np.sqrt(np.mean(residual**2))
np.savetxt('residual_6set.txt', residual)



plt.figure()
ax = plt.subplot(111)
ax.axhline(y=0, color='k', ls='--', alpha=.5)
plt.plot(t_fit, y_fit, alpha=.5)
plt.errorbar(RV_AAT[:,0],   RV_AAT[:,1]-off_aat,        yerr=RV_AAT[:,2],   fmt=".", capsize=0, label='AAT')
plt.errorbar(RV_CHIRON[:,0],RV_CHIRON[:,1]-off_chiron,  yerr=RV_CHIRON[:,2],fmt=".", capsize=0, label='CHIRON')
plt.errorbar(RV_FEROS[:,0], RV_FEROS[:,1]-off_feros,    yerr=RV_FEROS[:,2], fmt=".", capsize=0, label='FEROS')
plt.errorbar(RV_MJ1[:,0],   RV_MJ1[:,1]-off_mj1,        yerr=RV_MJ1[:,2],   fmt=".", capsize=0, label='MJ1')
plt.errorbar(RV_MJ3[:,0],   RV_MJ3[:,1]-off_mj3,        yerr=RV_MJ3[:,2],   fmt=".", capsize=0, label='MJ3')
plt.errorbar(RV_FIDEOS[:,0],RV_FIDEOS[:,1]-off_fideos,  yerr=RV_FIDEOS[:,2],fmt=".", capsize=0, label='FIDEOS')
plt.ylabel("Radial velocity [m/s]")
plt.xlabel("BJD - 2450000")
plt.legend(loc="upper center")
plt.savefig('76920_MCMC_6sets-3-MCMC_fit.png')
plt.show()




# convert x-axis to year number #
year_AAT    = (RV_AAT[:,0] + 728956.75107372)/365.25
year_CHIRON = (RV_CHIRON[:,0] + 728956.75107372)/365.25
year_FEROS  = (RV_FEROS[:,0] + 728956.75107372)/365.25
year_MJ1    = (RV_MJ1[:,0] + 728956.75107372)/365.25
year_MJ3    = (RV_MJ3[:,0] + 728956.75107372)/365.25
year_FIDEOS = (RV_FIDEOS[:,0] + 728956.75107372)/365.25
plt.figure()
ax = plt.subplot(111)
ax.axhline(y=0, color='k', ls='--', alpha=.5)
plt.plot((t_fit+728956.75107372)/365.25, y_fit, alpha=.5, label='Model')
plt.errorbar(year_AAT,   RV_AAT[:,1]-off_aat,        yerr=RV_AAT[:,2],   fmt="o", capsize=2, label='AAT')
plt.errorbar(year_CHIRON,RV_CHIRON[:,1]-off_chiron,  yerr=RV_CHIRON[:,2],fmt="o", capsize=2, label='CHIRON')
plt.errorbar(year_FEROS, RV_FEROS[:,1]-off_feros,    yerr=RV_FEROS[:,2], fmt="o", capsize=2, label='FEROS')
plt.errorbar(year_MJ1,   RV_MJ1[:,1]-off_mj1,        yerr=RV_MJ1[:,2],   fmt="o", capsize=2, label='MJ1')
plt.errorbar(year_MJ3,   RV_MJ3[:,1]-off_mj3,        yerr=RV_MJ3[:,2],   fmt="o", capsize=2, label='MJ3')
plt.errorbar(year_FIDEOS,RV_FIDEOS[:,1]-off_fideos,  yerr=RV_FIDEOS[:,2],fmt="o", capsize=2, label='FIDEOS')
plt.ylabel("Radial velocity [m/s]")
plt.xlabel("Year")
plt.legend(loc="upper center")
plt.savefig('76920_MCMC_6sets-3-MCMC_fit2.png', dpi=600)
plt.show()



# Plot 2 #
day_AAT     = [((RV_AAT[i,0]-tau)/P-8)*P for i in range(len(RV_AAT[:,0]))]
day_CHIRON  = [((RV_CHIRON[i,0]-tau)/P-8)*P for i in range(len(RV_CHIRON[:,0]))]
day_FEROS   = [((RV_FEROS[i,0]-tau)/P-8)*P for i in range(len(RV_FEROS[:,0]))]
day_MJ1     = [((RV_MJ1[i,0]-tau)/P-8)*P for i in range(len(RV_MJ1[:,0]))]
day_MJ3     = [((RV_MJ3[i,0]-tau)/P-8)*P for i in range(len(RV_MJ3[:,0]))]
day_FIDEOS  = [((RV_FIDEOS[i,0]-tau)/P-8)*P for i in range(len(RV_FIDEOS[:,0]))]
day_fit     = ((t_fit-tau)/P-8)*P

plt.figure()
ax = plt.subplot(111)
ax.axhline(y=0, color='k', ls='--', alpha=.5)
plt.plot(day_fit, y_fit, alpha=.5, label='Model')
plt.errorbar(day_AAT,   RV_AAT[:,1]-off_aat,        yerr=RV_AAT[:,2],   fmt="o", capsize=2, label='')
plt.errorbar(day_CHIRON,RV_CHIRON[:,1]-off_chiron,  yerr=RV_CHIRON[:,2],fmt="o", capsize=2, label='CHIRON')
plt.errorbar(day_FEROS, RV_FEROS[:,1]-off_feros,    yerr=RV_FEROS[:,2], fmt="o", capsize=2, label='FEROS')
plt.errorbar(day_MJ1,   RV_MJ1[:,1]-off_mj1,        yerr=RV_MJ1[:,2],   fmt="o", capsize=2, label='MJ1')
plt.errorbar(day_MJ3,   RV_MJ3[:,1]-off_mj3,        yerr=RV_MJ3[:,2],   fmt="o", capsize=2, label='MJ3')
plt.errorbar(day_FIDEOS,RV_FIDEOS[:,1]-off_fideos,  yerr=RV_FIDEOS[:,2],fmt="o", capsize=2, label='FIDEOS')
plt.plot(day_fit_old, y_fit_old, '-.', alpha=.5, label='Old prediction')
plt.xlim(-45, 15)
plt.ylabel("Radial velocity [m/s]")
plt.xlabel("Day")
plt.legend(loc="upper left")
plt.savefig('76920_MCMC_6sets-3-MCMC_fit3.png', dpi=600)
plt.show()


fit_curve_old = Model(P=np.log(415.4), tau=np.log(4813.42), k=np.log(186.8), w=-0.1239183768915978, e0=0.856, off_aat=off_aat/100, off_chiron=off_chiron/100, 
                        off_feros=off_feros/100, off_mj1=off_mj1/100, off_mj3=off_mj3/100, off_fideos=off_fideos/100)
t_fit_old   = np.linspace(min(RV_ALL[:,0])-300, max(RV_ALL[:,0]+300), num=10001, endpoint=True)
y_fit_old   = fit_curve_old.get_value(np.array(t_fit_old))
day_fit_old = ((t_fit_old-tau)/P-8)*P







if 0:
    inds = np.random.randint(len(log_samples), size=100)
    plt.figure()
    for ind in inds:
        sample = log_samples[ind]
        fit_curve = Model(P=sample[0], tau=sample[1], k=sample[2], w=sample[3], e0=sample[4], off_aat=sample[5],
                     off_chiron=sample[6], off_feros=sample[7], off_mj1=sample[8], off_mj3=sample[9])
        y_fit   = fit_curve.get_value(np.array(t_fit))
        plt.plot(t_fit, y_fit, "g", alpha=0.1)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel("RV [m/s]")
    plt.xlabel("Shifted JD [d]")
    plt.savefig('76920_MCMC_6sets-4-MCMC_100_realizations.png')
    # plt.close()


#==============================================================================
# Periodogram
#==============================================================================

from astropy.stats import LombScargle

RV_AAT[:,1]     -= off_aat
RV_CHIRON[:,1]  -= off_chiron
RV_FEROS[:,1]   -= off_feros
RV_MJ1[:,1]     -= off_mj1
RV_MJ3[:,1]     -= off_mj3
RV_FIDEOS[:,1]  -= off_fideos

RV_ALL_LS   = np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3, RV_FIDEOS))
RV_SORT_LS  = sorted(RV_ALL_LS, key=lambda x: x[0])
x_LS        = [RV_SORT_LS[i][0] for i in range(len(RV_SORT_LS))]
y_LS        = [RV_SORT_LS[i][1] for i in range(len(RV_SORT_LS))]
yerr_LS     = [RV_SORT_LS[i][2] for i in range(len(RV_SORT_LS))]

if 1: # test plot
    plt.figure()
    # plt.errorbar(x_LS, y_LS, yerr_LS, fmt=".", capsize=0)
    plt.errorbar(x_LS, residual, yerr_LS, fmt=".", capsize=0)
    # plt.plot(x_LS, -res_MCMC, '.')
    # plt.plot(t_fit, y_fit, label='MCMC fit')
    plt.show()



x_LS    -= min(x_LS)
min_f   = 0.001
max_f   = 5
spp     = 1000  # spp=1000 will take a while; for quick results set spp = 10

frequency0, power0 = LombScargle(x_LS, y_LS, yerr_LS).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency1, power1 = LombScargle(x_LS, residual, yerr_LS).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)


ax = plt.subplot(111)
ax.axvline(x=415.862413, color='k', ls='-.')
plt.plot(1/frequency0, power0, label='Data', ls='-')
plt.plot(1/frequency1, power1, label='Residual', ls='--')
plt.xlabel("Period")
plt.ylabel("Power")
plt.legend()
plt.savefig('76920_MCMC_6sets-5-Periodogram.png')
plt.show()


#==============================================================================
# calculate semi-major axis and planet mass
#==============================================================================
'''
bestpars[0] = period [days]
bestpars[1] = T_0 [days]
bestpars[2] = eccentricity
bestpars[3] = omega in degrees
bestpars[4] = K [m/s]
bestpars[5] = gamma [m/s]
'''

G       = 6.67428e-11       # gravitational constant [m^3 kg^-1 s^-2]
msun    = 1.98892e30
mjup    = 1.8986e27
m_to_au = 149597870700.

P       = real_samples[:,0]
K       = real_samples[:,2]
e0      = real_samples[:,4]
a_semi  = (P**2 * 86400.**2 * G * 1.17 * msun / (4. * np.pi**2))**(1/3) / m_to_au
mpsini  = (P * 86400. / (2. * np.pi * G))**(1/3) * K * (1.17 * msun)**(2/3) * np.sqrt(1 - e0**2) / mjup 

a_semi_16, a_semi_50, a_semi_84 = np.percentile(a_semi, [16, 50, 84])
mpsini_16, mpsini_50, mpsini_84 = np.percentile(mpsini, [16, 50, 84])

a_semi_array = [a_semi_50, a_semi_84-a_semi_50, a_semi_50-a_semi_16]
mpsini_array = [mpsini_50, mpsini_84-mpsini_50, mpsini_50-mpsini_16]
np.savetxt('76920_MCMC_6sets_result2.txt', np.vstack((a_semi_array, mpsini_array)), fmt='%.6f')

if 0:
    np.histogram(a_semi, bins=20)
    plt.hist(P, bins='auto')
    plt.show()















