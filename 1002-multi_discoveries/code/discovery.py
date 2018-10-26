import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing
from rv import solve_kep_eqn
import os

#==============================================================================
# Import data 
#==============================================================================
star    = 'HD128621'
print('*'*len(star))
print(star)
print('*'*len(star))

plt.rcParams.update({'font.size': 14})

DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
t 		= np.loadtxt(DIR + '/MJD.dat')
# MJD     = np.loadtxt(DIR + '/MJD.dat')
XX 		= np.loadtxt(DIR + '/RV_HARPS.dat')
XX 		= (XX - np.mean(XX)) * 1000
yerr 	= np.loadtxt(DIR + '/RV_noise.dat') #m/s
# YERR    = np.loadtxt(DIR + '/RV_noise.dat') #m/s
FWHM    = np.loadtxt(DIR + '/FWHM.dat')


# t  	= np.loadtxt('../data/'+star+'/MJD.dat')
# XX  = np.loadtxt('../data/'+star+'/RV_HARPS.dat')
# XX  = (XX - np.mean(XX)) * 1000
# yerr= np.loadtxt('../data/'+star+'/RV_noise.dat')
# FWHM  = np.loadtxt('../data/'+star+'/FWHM.dat')

YY  = np.loadtxt('../data/'+star+'/YY.txt')
ZZ  = np.loadtxt('../data/'+star+'/ZZ.txt')


XY 	= XX - YY
ZX 	= ZZ - XX

os.chdir('../output/'+star)

#==============================================================================
# Time Series
#==============================================================================

# present the pre-filetered data

plt.figure()
plt.errorbar(t, XX, yerr=yerr, fmt=".k", capsize=0, alpha=0.2, label='$RV_{HARPS}$')
plt.errorbar(t, XY, yerr=yerr, fmt=".r", capsize=0, alpha=0.2, label='$RV_{HARPS} - RV_{FT,L}$')
plt.errorbar(t, ZX, yerr=yerr, fmt=".b", capsize=0, alpha=0.2, label='$RV_{FT,H} - RV_{HARPS}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.show()

# Valid for data HD128621_2_2010-03-22..2010-06-12[PART]
# Filter the unwanted data
idx  = ~((XY>3) | ((t>55340) & (XX>0)))

plt.figure()
plt.errorbar(FWHM[idx], XY[idx], yerr=yerr[idx], fmt=".k")
plt.errorbar(FWHM[~idx], XY[~idx], yerr=yerr[~idx], fmt=".b")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.xlabel("FWHM")
plt.savefig('0-correlation_XY.png')
plt.show()

plt.figure()
plt.errorbar(FWHM[idx], ZX[idx], yerr=yerr[idx], fmt=".k")
plt.errorbar(FWHM[~idx], ZX[~idx], yerr=yerr[~idx], fmt=".b")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.xlabel("FWHM")
plt.savefig('0-correlation_ZX.png')
plt.show()


plt.figure()
plt.errorbar(t[idx], XX[idx], yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label='$RV_{HARPS}$')
plt.errorbar(t[~idx], XX[~idx], yerr=yerr[~idx], fmt="*r", capsize=0, alpha=0.2, label='$RV_{HARPS}$ outlier')
# plt.errorbar(t[idx], XY[idx], yerr=yerr[idx], fmt=".r", capsize=0, alpha=0.2, label='$RV_{HARPS} - RV_{FT,L}$')
# plt.errorbar(t[~idx], XY[~idx], yerr=yerr[~idx], fmt=".b", capsize=0, alpha=0.2, label='outlier?')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV1.png')
plt.show()

plt.figure()
# plt.errorbar(t, XX, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t[idx], ZX[idx], yerr=yerr[idx], fmt=".r", capsize=0, alpha=0.2, label='$RV_{FT,H} - RV_{HARPS}$')
plt.errorbar(t[~idx], ZX[~idx], yerr=yerr[~idx], fmt=".b", capsize=0, alpha=0.2, label='outlier?')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV2.png')
plt.show()

plt.figure()
plt.errorbar(XX[idx], XY[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
plt.errorbar(XX[~idx], XY[~idx], yerr=yerr[~idx], fmt="*r", alpha=0.2)
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.savefig('2-correlation_XY.png')
plt.show()

plt.figure()
plt.errorbar(XX[idx], ZX[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
plt.errorbar(XX[~idx], ZX[~idx], yerr=yerr[~idx], fmt="*r", alpha=0.2)
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_ZX.png')
plt.show()

plt.figure()
plt.errorbar(XY[idx], ZX[idx], yerr=yerr[idx], fmt=".r", alpha=0.2)
plt.errorbar(XY[~idx], ZX[~idx], yerr=yerr[~idx], fmt="*b", alpha=0.2)
plt.xlabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_XYZ.png')
plt.show()



left  = 0.05  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.plot(XX[idx], YY[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], YY[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.plot(XX[idx], ZZ[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], ZZ[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.plot(XX[idx], XY[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], XY[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{HARPS} - RV_{FT,L}$ [m/s]')
plt.title(r'$\alpha$' + ' Centauri B 2010-03-23..2010-06-12')


plt.subplot(154)
plt.plot(XX[idx], ZX[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], ZX[~idx], '*r', markersize=3, alpha=0.3)   
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{FT,H} - RV_{HARPS}$ [m/s]')

plt.subplot(155)
fit = np.polyfit(XY, ZX, 1)
x_sample = np.linspace(min(XY)*1.2, max(XY)*1.2, num=100, endpoint=True)
plt.plot(XY[idx], ZX[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XY[~idx], ZX[~idx], '*r', markersize=3, alpha=0.3)   
plt.xlabel('$RV_{HARPS} - RV_{FT,L}$ [m/s]')    
plt.ylabel('$RV_{FT,H} - RV_{HARPS}$ [m/s]')     
plt.savefig('Correlation.png')   
plt.show()

# t = t - min(t)
t = t[idx]
y = XY[idx]
yerr = yerr[idx]

#==============================================================================
# Smoothing
#==============================================================================
sl      = 0.5         # smoothing length
# xx 	 	= gaussian_smoothing(t, XX, t, sl)
xy      = gaussian_smoothing(t, y, t, sl, 1/yerr**2)
# zx      = gaussian_smoothing(t, ZX, t, sl)
np.savetxt('../../data/'+star+'/xy.txt', xy)


plt.figure()
# plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, xy, yerr=yerr, fmt=".r", capsize=0, label='$RV_{HARPS} - RV_{FT,L}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('s1-RV1.png')
plt.show()

plt.figure()
plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, zx, yerr=yerr, fmt=".r", capsize=0, label='$RV_{FT,H} - RV_{HARPS}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('s1-RV2.png')
plt.show()

plt.figure()
plt.errorbar(xx, xy, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.savefig('s2-correlation_XY.png')
plt.show()

plt.figure()
plt.errorbar(xx, zx, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_ZX.png')
plt.show()

plt.figure()
plt.errorbar(xy, zx, yerr=yerr, fmt=".r")
plt.xlabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('s2-correlation_XYZ.png')
plt.show()
plt.close('all')

#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/100
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
plt.figure()
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axvline(x=36.2, color='k')
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
plt.figure()
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axvline(x=36.2, color='k')
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

for kk in np.arange(50):

	os.chdir(str(time0))

	if 0:
	    idx = (t < 57161)

	    t = t[idx]
	    xx = xx[idx]
	    xy = xy[idx]
	    XX = XX[idx]
	    yerr = yerr[idx]

	#==============================================================================
	# Model
	#==============================================================================
	class Model(Model):
	    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'offset1', 'offset2', 'alpha')

	    def get_value(self, t):

	        # Planet 1
	        M_anom1 = 2*np.pi/np.exp(10*self.P1) * (t - 1000*self.tau1)
	        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
	        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
	        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))

	        # The last part is not "corrected" with jitter
	        jitter  = np.exp(self.alpha) * xy

	        offset      = np.zeros(len(t))
	        idx         = t < 57161
	        offset[idx] = self.offset1
	        offset[~idx]= self.offset2

	        return rv1 + offset + jitter

	class Model2(Model):
	    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'offset1', 'offset2')

	    def get_value(self, t):

	        # Planet 1
	        M_anom1 = 2*np.pi/np.exp(10*self.P1) * (t - 1000*self.tau1)
	        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
	        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
	        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))

	        offset      = np.zeros(len(t))
	        idx         = t < 57161
	        offset[idx] = self.offset1
	        offset[~idx]= self.offset2

	        return rv1 + offset

	#==============================================================================
	# MCMC
	#==============================================================================
	# Define the posterior PDF
	# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

	def lnprior(theta):
	    P1, tau1, k1, w1, e1, offset1, offset2, alpha = theta
	    if (-2*np.pi < w1 < 2*np.pi) and (0 < e1 < 0.9) and (alpha > 0):
	        return 0.0
	    return -np.inf

	# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
	def lnlike(theta, x, y, yerr):
	    P1, tau1, k1, w1, e1, offset1, offset2, alpha = theta
	    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, offset1=offset1, offset2=offset2, alpha=alpha)
	    y_fit       = fit_curve.get_value(x)
	    return -0.5*(np.sum( ((y-y_fit)/yerr)**2))

	def lnprob(theta, x, y, yerr):
	    lp = lnprior(theta)
	    if not np.isfinite(lp):
	        return -np.inf
	    return lp + lnlike(theta, x, y, yerr)



	import emcee
	ndim = 8
	nwalkers = 32
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, XX, yerr), threads=14)

	import time
	time_start  = time.time()

	INIT_P =  np.random.uniform(0,0.9)
	INIT_e =  np.random.uniform(0.01,0.89)
	print("Running first burn-in...")
	pos = [[INIT_P, 1., (max(XX)-min(XX))/100, 0, INIT_e, 0, 0, 1.5] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
	pos, prob, state  = sampler.run_mcmc(pos, 2000)

	print("Running second burn-in...")
	pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
	pos, prob, state  = sampler.run_mcmc(pos, 1000)

	print("Running third burn-in...")
	pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
	pos, prob, state  = sampler.run_mcmc(pos, 3000)

	print("Running production...")
	sampler.run_mcmc(pos, 3000);

	time_end    = time.time()
	print('\nRuntime = %.2f seconds' %(time_end - time_start))


	#==============================================================================
	# Trace and corner plots 
	#==============================================================================

	import copy
	raw_samples         = sampler.chain[:, -3000, :].reshape((-1, ndim))
	real_samples        = copy.copy(raw_samples)
	real_samples[:,0] = np.exp(10*real_samples[:,0])
	real_samples[:,1] = 1000*real_samples[:,1]
	real_samples[:,2] = 100*real_samples[:,2]
	idx = real_samples[:,3] < 0
	real_samples[idx,3] = real_samples[idx, 3] + 2*np.pi
	real_samples[:,7] = np.exp(real_samples[:,7])


	fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
	labels_log=[r"$\frac{\log P_{1}}{10}$", r"$\frac{T_{1}}{1000}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", "offset1", "offset2", r"$\log \alpha$"]
	for i in range(ndim):
	    ax = axes[i]
	    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
	    ax.set_xlim(0, sampler.chain.shape[1])
	    ax.set_ylabel(labels_log[i])
	    ax.yaxis.set_label_coords(-0.1, 0.5)

	axes[-1].set_xlabel("step number");
	plt.savefig('2-Trace.png')
	# plt.show()


	import corner
	labels=[r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", "offset1", "offset2", r"$\alpha$"]
	fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
	plt.savefig('3-Corner.png')
	# plt.show()

	#==============================================================================
	# Output
	#==============================================================================

	a0, a1, a2, a3, a4, a5, a6, a7 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
	aa = np.zeros((8,3))
	aa[0,:] = [a0[i] for i in range(3)]
	aa[1,:] = [a1[i] for i in range(3)]
	aa[2,:] = [a2[i] for i in range(3)]
	aa[3,:] = [a3[i] for i in range(3)]
	aa[4,:] = [a4[i] for i in range(3)]
	aa[5,:] = [a5[i] for i in range(3)]
	aa[6,:] = [a6[i] for i in range(3)]
	aa[7,:] = [a7[i] for i in range(3)]
	np.savetxt('parameter_fit.txt', aa, fmt='%.6f')


	P1, tau1, k1, w1, e1, offset1, offset2, alpha = aa[:,0]
	fig         = plt.figure(figsize=(10, 7))
	frame1      = fig.add_axes((.15,.3,.8,.6))
	frame1.axhline(y=0, color='k', ls='--', alpha=.3)
	t_sample    = np.linspace(min(t), max(t), num=10001, endpoint=True)

	# Planet 1 #
	Planet1     = Model2(P1=np.log(P1)/10, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, offset1=offset1, offset2=offset2)
	y1          = Planet1.get_value(t_sample)
	plt.plot(t_sample, y1, 'b-.', alpha=.3, label='Planet1')
	plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='HARPS RV')
	plt.legend()
	plt.ylabel("Radial velocity [m/s]")
	# Jitter#
	Jitter      = Model(P1=np.log(P1)/10, tau1=tau1/1000, k1=0, w1=w1, e1=e1, offset1=0, offset2=0, alpha=np.log(alpha))
	y_jitter    = Jitter.get_value(t)
	plt.plot(t, y_jitter, 'ro', alpha=.5, label='smoothed jitter')

	Fit         = Model(P1=np.log(P1)/10, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, offset1=offset1, offset2=offset1, alpha=np.log(alpha))
	y_fit       = Fit.get_value(t)
	plt.plot(t, y_fit, 'bo', alpha=.5, label='Planet 1 + smoothed jitter')
	# plt.plot(x[x<57300], alpha*jitter_smooth, 'ro', alpha=.5, label='smoothed jitter')
	plt.legend()
	plt.ylabel("Radial velocity [m/s]")

	residual    = y_fit - XX
	chi2        = sum(residual**2 / yerr**2)
	rms         = np.sqrt(np.mean(residual**2))
	wrms        = np.sqrt(sum((residual/yerr)**2) / sum(1/yerr**2))

	frame2  = fig.add_axes((.15,.1,.8,.2))   
	frame2.axhline(y=0, color='k', ls='--', alpha=.3)
	plt.errorbar(t, residual, yerr=yerr, fmt=".k", capsize=0)
	plt.xlabel("JD - 2,400,000")
	plt.ylabel('Residual [m/s]')
	plt.savefig('4-MCMC_fit.png')
	# plt.show()

	os.chdir(str('../'))


























