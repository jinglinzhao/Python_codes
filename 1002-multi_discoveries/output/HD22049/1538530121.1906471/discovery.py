import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing
from rv import solve_kep_eqn
import os

#==============================================================================
# Import data 
#==============================================================================
star    = 'HD22049'
print('*'*len(star))
print(star)
print('*'*len(star))


DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
t 		= np.loadtxt(DIR + '/MJD.dat')
XX 		= np.loadtxt(DIR + '/RV_HARPS.dat')
XX 		= (XX - np.mean(XX)) * 1000
yerr 	= np.loadtxt(DIR + '/RV_noise.dat') #m/s

YY  = np.loadtxt('../data/'+star+'/YY.txt')
ZZ  = np.loadtxt('../data/'+star+'/ZZ.txt')

XY 	= XX - YY
ZX 	= ZZ - XX

os.chdir('../output/'+star)

#==============================================================================
# Time Series
#==============================================================================

plt.figure()
plt.errorbar(t, XX, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, XY, yerr=yerr, fmt=".r", capsize=0, label='$RV_{HARPS} - RV_{FT,L}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV1.png')
# plt.show()

plt.figure()
plt.errorbar(t, XX, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, ZX, yerr=yerr, fmt=".r", capsize=0, label='$RV_{FT,H} - RV_{HARPS}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV2.png')
# plt.show()

plt.figure()
plt.errorbar(XX, XY, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.savefig('2-correlation_XY.png')
# plt.show()

plt.figure()
plt.errorbar(XX, ZX, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_ZX.png')
# plt.show()

plt.figure()
plt.errorbar(XY, ZX, yerr=yerr, fmt=".r")
plt.xlabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_XYZ.png')
# plt.show()

#==============================================================================
# Smoothing
#==============================================================================
sl      = 1         # smoothing length
xx 	 	= gaussian_smoothing(t, XX, t, sl)
xy      = gaussian_smoothing(t, XY, t, sl)
zx      = gaussian_smoothing(t, ZX, t, sl)

plt.figure()
plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, xy, yerr=yerr, fmt=".r", capsize=0, label='$RV_{HARPS} - RV_{FT,L}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('s1-RV1.png')
# plt.show()

plt.figure()
plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, zx, yerr=yerr, fmt=".r", capsize=0, label='$RV_{FT,H} - RV_{HARPS}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('s1-RV2.png')
# plt.show()

plt.figure()
plt.errorbar(xx, xy, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.savefig('s2-correlation_XY.png')
# plt.show()

plt.figure()
plt.errorbar(xx, zx, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_ZX.png')
# plt.show()

plt.figure()
plt.errorbar(xy, zx, yerr=yerr, fmt=".r")
plt.xlabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('s2-correlation_XYZ.png')
# plt.show()
plt.close('all')

if 0:
	# idx = (x > 53500) & (x < 53800)
	idx = (x > 54250) & (x < 54500)

	x = x[idx]
	XX = XX[idx]
	XY = XY[idx]
	yerr = yerr[idx]

#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/200
max_f   = 1
spp     = 10

frequency0, power0 = LombScargle(t, XX, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(t, ZX, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

frequency2, power2 = LombScargle(t, XY, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axhline(y=0, color='k')
plt.plot(1/frequency0, power0, 'b-', label=r'$RV_{Gaussian}$', linewidth=1.0)
plt.plot(1/frequency1, power1, 'r-.', label=r'$RV_{FT,H} - RV_{Gaussian}$', alpha=0.7)
plt.plot(1/frequency2, power2, 'g--', label=r'$RV_{Gaussian} - RV_{FT,L}$', alpha=0.7)
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('3-Periodogram.png')
plt.show()


frequency0, power0 = LombScargle(t, xx, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(t, zx, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

frequency2, power2 = LombScargle(t, xy, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axhline(y=0, color='k')
plt.plot(1/frequency0, power0, 'b-', label=r'$RV_{Gaussian}$', linewidth=1.0)
plt.plot(1/frequency1, power1, 'r-.', label=r'$RV_{FT,H} - RV_{Gaussian}$', alpha=0.7)
plt.plot(1/frequency2, power2, 'g--', label=r'$RV_{Gaussian} - RV_{FT,L}$', alpha=0.7)
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('s3-Periodogram.png')
plt.show()


#==============================================================================
#==============================================================================
# Discovery mode
#==============================================================================
#==============================================================================
from celerite.modeling import Model
import time
import shutil
time0   = time.time()
os.makedirs(str(time0))
shutil.copy('../../code/discovery.py', str(time0)+'/discovery.py')  
os.chdir(str(time0))


#==============================================================================
# Model
#==============================================================================
class Model(Model):
    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'offset1', 'alpha')

    def get_value(self, t):

        # Planet 1
        M_anom1 = 2*np.pi/(100*self.P1) * (t - 1000*self.tau1)
        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))

        # The last part is not "corrected" with jitter
        jitter      = np.zeros(len(t))
        jitter[idx] = self.alpha * xy

        return rv1 + self.offset + jitter


#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P1, tau1, k1, w1, e1, offset, alpha = theta
    if (-2*np.pi < w1 < 2*np.pi) and (0 < e1 < 0.9) and (alpha > 1):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P1, tau1, k1, w1, e1, offset, alpha = theta
    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, offset=offset, alpha=alpha)
    y_fit       = fit_curve.get_value(x)
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    



import emcee
ndim = 7
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)

import time
time_start  = time.time()

print("Running first burn-in...")
pos = [[20., 1., np.std(XX)/100, 0, 0.4, 0, 3] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
pos, prob, state  = sampler.run_mcmc(pos, 3000)

print("Running second burn-in...")
pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
pos, prob, state  = sampler.run_mcmc(pos, 2000)

# print("Running third burn-in...")
# pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
# pos, prob, state  = sampler.run_mcmc(pos, 2000)

print("Running production...")
sampler.run_mcmc(pos, 3000);

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))

































