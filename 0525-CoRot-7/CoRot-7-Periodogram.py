
import numpy as np
from astropy.stats import LombScargle
import matplotlib.pyplot as plt






BJD 		= np.loadtxt('MJD_2012.txt')
Jitter 		= np.loadtxt('Jitter_model_2012.txt')
RV_HARPS 	= np.loadtxt('RV_HARPS_2012.txt')
RV_FT_2012 	= np.loadtxt('RV_FT_2012.txt')
RV_noise= np.loadtxt('RV_noise_2012.txt')



GP_y_2012 		= np.loadtxt('GP_y_2012.txt')
GP_err_2012		= np.loadtxt('GP_err_2012.txt')

MJD_BIN_2012 		= np.loadtxt('MJD_BIN_2012.txt')
Jitter_BIN_2012 	= np.loadtxt('Jitter_BIN_2012.txt')
Jitter_err_BIN_2012 = np.loadtxt('Jitter_err_BIN_2012.txt')

GP_y_2012_bin 		= np.loadtxt('GP_y_2012_bin.txt')
GP_err_2012_bin 	= np.loadtxt('GP_err_2012_bin.txt')


min_f 	= 0.04
max_f 	= 5
spp 	= 10

frequency0, power0 = LombScargle(BJD, Jitter, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency1, power1 = LombScargle(BJD, GP_y_2012, GP_err_2012).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency2, power2 = LombScargle(MJD_BIN_2012, Jitter_BIN_2012, Jitter_err_BIN_2012).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency3, power3 = LombScargle(MJD_BIN_2012, GP_y_2012_bin, GP_err_2012_bin).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

frequency4, power4 = LombScargle(BJD, RV_HARPS, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

# frequency5, power5 = LombScargle(BJD, RV_HARPS - 4*GP_y_2012, np.sqrt(1+16*GP_err_2012**2)).autopower(minimum_frequency=0.01,
#                                                    maximum_frequency=max_f,
#                                                    samples_per_peak=spp)

frequency6, power6 = LombScargle(BJD, RV_FT_2012, RV_noise).autopower(minimum_frequency=min_f,
                                                   maximum_frequency=max_f,
                                                   samples_per_peak=spp)

ax = plt.subplot(111)
ax.set_xscale('log')
ax.axhline(y=0, color='k')
ax.axvline(x=0.853591, color='k')
ax.axvline(x=3.70, color='k')
ax.axvline(x=23.81, color='k', ls='-.')
ax.axvline(x=23.81/2, color='k', ls='-.')
ax.axvline(x=23.81/3, color='k', ls='-.')
plt.plot(1/frequency0, power0, '--', label='Jitter Model (HARPS - FT)')
plt.plot(1/frequency1, power1, label='GP')
plt.plot(1/frequency2, -power2, '--', label='Jitter Model bin')
plt.plot(1/frequency3, -power3, label='GP bin')
plt.plot(1/frequency4, power4, label='RV_HARPS', linewidth=2.0)
# plt.plot(1/frequency5, power5, label='Correction', linewidth=2.0)
plt.plot(1/frequency6, power6, label='RV_FT', linewidth=2.0)
plt.xlim([0, 25])
plt.legend()
plt.show()




if 0: 

	plt.fill_between(x_pred, raw_pred - np.sqrt(raw_pred_var), pred + np.sqrt(raw_pred_var),
	                color="b", alpha=0.2)
	plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
	                color="r", alpha=0.2)
	plt.plot(x_pred, raw_pred, "b", lw=1.5, alpha=0.5)
	plt.plot(x_pred, pred, "r", lw=1.5, alpha=0.5)
	plt.xlabel("BJD")
	plt.ylabel("Jitter model [m/s]");
	plt.title("Fit with GP noise model");
	# plt.savefig('GP_fit_comparison.png')
	plt.show()






