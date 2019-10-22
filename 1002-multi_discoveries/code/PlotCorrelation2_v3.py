'''
Correlation
'''

# include FWHM, BIS, V_span @27/08

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from WeightedPearsonCorrelationCoefficient import wPearsonCoefficient

#==============================================================================
# Import data 
#==============================================================================

# directory 	= '/Volumes/DataSSD/MATLAB_codes/0615-FT-HD189733/'
# star 		= 'HD189733'
star 		= 'HD103720'
# star 		= 'HD36051'
# star 		= 'HD22049'
directory 	= '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
RV 			= np.loadtxt(directory + 'GG.txt')
x = RV
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
bi 			= np.loadtxt(directory + 'BI.txt')
# xy 			= 0 - y
# zx   		= z  
if 0:
	xy = RV - y 
	zx = z - RV
xy = -y
zx = z

DIR     	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
RV_noise	= np.loadtxt(DIR + '/RV_noise.dat')
MJD     	= np.loadtxt(DIR + '/MJD.dat')
FWHM 		= np.loadtxt(DIR + '/FWHM.dat')
dFWHM 		= np.loadtxt(DIR + '/dFWHM.dat')
BIS 		= np.loadtxt(DIR + '/BIS.dat')
dBIS 		= np.loadtxt(DIR + '/dBIS.dat')
V_span 		= np.loadtxt(DIR + '/V_span.dat')
dV_span 	= np.loadtxt(DIR + '/dV_span.dat')

if 0:
	from functions import gaussian_smoothing
	plt.errorbar(MJD, xy, yerr=RV_noise*3**0.5, fmt="r.", capsize=0, alpha=0.2)
	jitter_s = gaussian_smoothing(MJD, xy, MJD, 0.5, 1/RV_noise**2)
	plt.errorbar(MJD, jitter_s, yerr=RV_noise*3**0.5, fmt="b.", capsize=0, alpha=0.3)
	plt.show()

	plt.errorbar(MJD, x, yerr=RV_noise, fmt="k.", capsize=0, alpha=0.5)
	# plt.errorbar(MJD, xy, yerr=RV_noise*3**0.5, fmt="r.", capsize=0, alpha=0.2)
	plt.errorbar(MJD, zx, yerr=RV_noise*3**0.5, fmt="r.", capsize=0, alpha=0.2)
	plt.show()


#==============================================================================
# Correlation 1
#==============================================================================

left  	= 0.08  # the left side of the subplots of the figure
right 	= 0.95    # the right side of the subplots of the figure
bottom 	= 0.2   # the bottom of the subplots of the figure
top 	= 0.8      # the top of the subplots of the figure
wspace 	= 0.6   # the amount of width reserved for blank space between subplots
hspace 	= 0.2   # the amount of height reserved for white space between subplots
w 		= 1/RV_noise**2
alpha 	= 0.5
Nx 		= 3
Ny 		= 3
fontsize= 18


plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(20, 4))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star=='HD36051' or star=='HD22049':
	fig.suptitle('HD ' + star[2:] + ' (planet candidate not removed)', y=0.95)
elif star=='HD103720':
	fig.suptitle('HD ' + star[2:] + ' (planet not removed)', y=0.95)
elif star=='HD189733':
	fig.suptitle('HD ' + star[2:] + ' (companions not removed)', y=0.95)
else:
	fig.suptitle('HD ' + star[2:], y=0.95)	

axes_1 = plt.subplot(151)
plt.errorbar(RV, FWHM, xerr=RV_noise, yerr=dFWHM, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel('FWHM [km/s]')
r, delta_r = wPearsonCoefficient(RV, FWHM, w)
plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(152)
plt.errorbar(RV, BIS, xerr=RV_noise, yerr=dBIS, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel('BIS [m/s]')
r, delta_r = wPearsonCoefficient(RV, BIS, w)
plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(153)
plt.errorbar(RV, V_span, xerr=RV_noise, yerr=dV_span, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$V_{span}$ [m/s]')
r, delta_r = wPearsonCoefficient(RV, V_span, w)
plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_4 = plt.subplot(154)
plt.errorbar(RV, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
r, delta_r = wPearsonCoefficient(RV, xy, w)
fit, V 	= np.polyfit(RV, xy, 1, w=w, cov=True)
plt.title(r'$R = {0:.2f}±{1:.2f}; k={2:.2f}$'.format(r,delta_r,fit[0]), fontsize=fontsize)
axes_4.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_4.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_5 = plt.subplot(155)
plt.errorbar(RV, zx, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
r, delta_r = wPearsonCoefficient(RV, zx, w)
fit, V 	= np.polyfit(RV, zx, 1, w=w, cov=True)
plt.title(r'$R = {0:.2f}±{1:.2f}; k={2:.2f}$'.format(r,delta_r,fit[0]), fontsize=fontsize)
# plt.title(r'$R = {0:.2f};\ k={0:.2f}±{1:.2f}$'.format(r,fit[0],V[0,0]**0.5), fontsize=fontsize)
axes_5.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_5.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '_1.png')
plt.show()


#==============================================================================
# Keplerian orbit 
#==============================================================================
if star	== 'HD22049':

	from rv import solve_kep_eqn
	from celerite.modeling import Model

	class Model(Model):
	    parameter_names = ('P', 'tau', 'k', 'w0', 'e0', 'offset')

	    def get_value(self, t):
	        M_anom  = 2*np.pi/self.P * (t - self.tau)
	        e_anom  = solve_kep_eqn(M_anom, self.e0)
	        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
	        rv      = self.k * (np.cos(f + self.w0) + self.e0*np.cos(self.w0))

	        return rv + self.offset

	###################
	# Plots the model # 
	###################
	P       = 2691
	tau     = 2447213 - 2400000.5
	e0       = 0.071
	offset  = -4.5
	k       = 11.48
	w0       = 177 / 360 * 2 * np.pi

	left  = 0.1  # the left side of the subplots of the figure
	right = 0.95    # the right side of the subplots of the figure
	bottom = 0.12   # the bottom of the subplots of the figure
	top = 0.85      # the top of the subplots of the figure
	wspace = 0.55   # the amount of width reserved for blank space between subplots
	hspace = 0.1   # the amount of height reserved for white space between subplots
	fontsize = 24
	alpha = 0.3

# plt.rcParams.update({'font.size': 20})
# fig, axes = plt.subplots(figsize=(16, 5))
	plt.rcParams.update({'font.size': 24})  
	fig = plt.figure(figsize=(15, 7))  
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)	

	fig.suptitle(r'$\epsilon$ Eridani time series', y=0.95)		
	axes_1 = plt.subplot(211)
	axes_1.axhline(color="gray", ls='--')
	fit_curve 	= Model(P=P, tau=tau, k=k, w0=w0, e0=e0, offset=offset)
	plot_x      = np.linspace(min(MJD), max(MJD), num=10000)
	plot_y     	= fit_curve.get_value(plot_x)
	wrms1   	= np.sqrt( sum(((x-np.mean(x))/RV_noise)**2) / sum(1/RV_noise**2))
	rms1 		= np.var(x)**0.5
	plt.errorbar(MJD, x+7, yerr=RV_noise, fmt="k.", capsize=0, alpha=alpha, label=r'$RV_{HARPS}$')
	plt.plot(plot_x, plot_y, 'g-', linewidth=2.0, label='Model')
	plt.ylim(-21, 21)
	plt.ylabel('RV [m/s]')
	plt.legend(loc = 2, prop={'size': fontsize})
	axes_1.set_xticklabels([])
	plt.text(54000, 15, r'$\sigma_{HARPS}=%.1f$ m/s' %wrms1, fontsize=fontsize)

	axes_2 = plt.subplot(212)
	axes_2.axhline(color="gray", ls='--')
	model 	= fit_curve.get_value(MJD)
	res_2   = x + 7 - model
	plt.errorbar(MJD, res_2, yerr=RV_noise, fmt=".k", capsize=0, alpha=alpha)
	res_2 	= res_2 - np.mean(res_2)
	wrms2   = np.sqrt(sum((res_2/RV_noise)**2)/sum(1/RV_noise**2))
	rms2 	= np.var(res_2)**0.5
	plt.ylim(-21, 21)
	plt.xlabel('MJD')
	plt.ylabel('Residual [m/s]')
	axes_2.xaxis.set_major_locator(plt.MaxNLocator(5))
	plt.text(54000, 15, r'$\sigma_{residual}=%.1f$ m/s' %wrms2, fontsize=fontsize)
	plt.savefig('../output/TimeSeries_' + star + '.png')
	plt.show()
	plt.close('all')

	bi = model

	#######################
	# Plot jitter metrics #
	#######################
	plt.rcParams.update({'font.size': 24})  
	fig = plt.figure(figsize=(15, 7))  
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)	

	fig.suptitle(r'$\epsilon$ Eridani jitter analysis', y=0.95)		
	axes_1 = plt.subplot(211)
	axes_1.axhline(color="gray", ls='--')
	wrms1   	= np.sqrt( sum(((xy-np.mean(xy))/RV_noise)**2) / sum(1/RV_noise**2))
	rms1 		= np.var(xy)**0.5
	plt.errorbar(MJD, xy, yerr=RV_noise*3**0.5, fmt='k.', alpha=alpha)
	plt.ylim(-21, 21)
	plt.ylabel(r'$\Delta RV_{L}$ [m/s]')	
	axes_1.set_xticklabels([])
	plt.text(54000, 15, r'$\sigma_{L}=%.1f$ m/s' %wrms1, fontsize=fontsize)

	axes_2 = plt.subplot(212)
	axes_2.axhline(color="gray", ls='--')
	wrms2   	= np.sqrt( sum(((zx-np.mean(zx))/RV_noise)**2) / sum(1/RV_noise**2))
	rms2 		= np.var(zx)**0.5
	plt.errorbar(MJD, zx, yerr=RV_noise*3**0.5, fmt='k.', alpha=alpha)
	plt.xlabel('MJD')
	plt.ylim(-21, 21)
	plt.ylabel(r'$\Delta RV_{H}$ [m/s]')
	axes_2.xaxis.set_major_locator(plt.MaxNLocator(5))
	plt.text(54000, 15, r'$\sigma_{H}=%.1f$ m/s' %wrms2, fontsize=fontsize)
	plt.savefig('../output/JitterMetrics_' + star + '.png')
	plt.show()
	plt.close('all')


#==============================================================================
# Correlation 2: remove binary companion 
#==============================================================================
left  	= 0.08  # the left side of the subplots of the figure
right 	= 0.95    # the right side of the subplots of the figure
bottom 	= 0.2   # the bottom of the subplots of the figure
top 	= 0.8      # the top of the subplots of the figure
wspace 	= 0.6   # the amount of width reserved for blank space between subplots
hspace 	= 0.2   # the amount of height reserved for white space between subplots
w 		= 1/RV_noise**2
alpha 	= 0.5
Ny 		= 3
Nx 		= 4
fontsize= 18


plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(20, 4))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star=='HD36051' or star=='HD22049':
	fig.suptitle('HD ' + star[2:] + ' (planet candidate removed)', y=0.95)
if star=='HD103720':
	fig.suptitle('HD ' + star[2:] + ' (planet removed)', y=0.95)
if star=='HD189733':
	fig.suptitle('HD ' + star[2:] + ' (companions removed)', y=0.95)


axes_1 = plt.subplot(151)
plt.errorbar(RV-bi, FWHM, xerr=RV_noise, yerr=dFWHM, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
plt.ylabel('FWHM [km/s]')
r, delta_r = wPearsonCoefficient(RV-bi, FWHM, w)
plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r,delta_r), fontsize=fontsize)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(152)
plt.errorbar(RV-bi, BIS, xerr=RV_noise, yerr=dBIS, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
plt.ylabel('BIS [m/s]')
r, delta_r = wPearsonCoefficient(RV-bi, BIS, w)
plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r,delta_r), fontsize=fontsize)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(153)
plt.errorbar(RV-bi, V_span, xerr=RV_noise, yerr=dV_span, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
plt.ylabel(r'$V_{span}$ [m/s]')
r, delta_r = wPearsonCoefficient(RV-bi, V_span, w)
plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r,delta_r), fontsize=fontsize)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

# idx  = xy < 20
# axes_4 = plt.subplot(154)
# plt.errorbar(RV[idx]-bi[idx], xy[idx], xerr=RV_noise[idx], yerr=RV_noise[idx]*3**0.5, fmt="ko", capsize=0, alpha=alpha)
# plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
# plt.ylabel(r'$\Delta RV_L$ [m/s]')
# r, delta_r = wPearsonCoefficient(RV[idx]-bi[idx], xy[idx], w[idx])
# fit, V 	= np.polyfit(RV[idx]-bi[idx], xy[idx], 1, w=w[idx], cov=True)
# plt.title(r'$R = {0:.2f};\ k={1:.2f}$'.format(r,fit[0]), fontsize=fontsize)
# axes_4.xaxis.set_major_locator(plt.MaxNLocator(Nx))
# axes_4.yaxis.set_major_locator(plt.MaxNLocator(Ny))


axes_4 = plt.subplot(154)
plt.errorbar(RV-bi, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
plt.ylabel(r'$\Delta RV_L$ [m/s]')
r, delta_r = wPearsonCoefficient(RV-bi, xy, w)
fit, V 	= np.polyfit(RV-bi, xy, 1, w=w, cov=True)
plt.title(r'$R = {0:.2f}±{1:.2f};\ k={2:.2f}$'.format(r,delta_r,fit[0]), fontsize=fontsize)
axes_4.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_4.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_5 = plt.subplot(155)
plt.errorbar(RV-bi, zx, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
plt.ylabel(r'$\Delta RV_H$ [m/s]')
r, delta_r = wPearsonCoefficient(RV-bi, zx, w)
fit, V 	= np.polyfit(RV-bi, zx, 1, w=w, cov=True)
plt.title(r'$R = {0:.2f}±{1:.2f};\ k={2:.2f}$'.format(r,delta_r,fit[0]), fontsize=fontsize)
# plt.title(r'$R = {0:.2f};\ k={0:.2f}±{1:.2f}$'.format(r,fit[0],V[0,0]**0.5), fontsize=fontsize)
axes_5.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_5.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '_2.png')
plt.show()






if 0:

	left  	= 0.15  # the left side of the subplots of the figure
	right 	= 0.95    # the right side of the subplots of the figure
	bottom 	= 0.2   # the bottom of the subplots of the figure
	top 	= 0.8      # the top of the subplots of the figure
	wspace 	= 0.6   # the amount of width reserved for blank space between subplots
	hspace 	= 0.2   # the amount of height reserved for white space between subplots
	w 		= 1/RV_noise**2
	alpha 	= 0.5
	Nx 		= 3
	Ny 		= 3
	fontsize= 18

	plt.rcParams.update({'font.size': 20})
	fig, axes = plt.subplots(figsize=(8, 4))
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

	if star=='HD36051' or star=='HD22049':
		fig.suptitle('HD ' + star[2:] + ' (planet candidate removed)', y=0.95)
	if star=='HD103720':
		fig.suptitle('HD ' + star[2:] + ' (planet removed)', y=0.95)
	if star=='HD189733':
		fig.suptitle('HD ' + star[2:] + ' (companions removed)', y=0.95)

	axes_4 = plt.subplot(121)
	plt.errorbar(RV-bi, xy, xerr=RV_noise, yerr=RV_noise*(39/19)**0.5, fmt="ko", capsize=0, alpha=alpha)
	plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
	plt.ylabel(r'$\Delta RV_L$ [m/s]')
	r, delta_r = wPearsonCoefficient(RV-bi, xy, w)
	plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
	axes_4.xaxis.set_major_locator(plt.MaxNLocator(Nx))
	axes_4.yaxis.set_major_locator(plt.MaxNLocator(Ny))

	axes_5 = plt.subplot(122)
	plt.errorbar(RV-bi, zx, xerr=RV_noise, yerr=RV_noise*21**0.5, fmt="ko", capsize=0, alpha=alpha)
	plt.xlabel(r"$RV'_{HARPS}$ [m/s]")
	plt.ylabel(r'$\Delta RV_H$ [m/s]')
	r, delta_r = wPearsonCoefficient(RV-bi, zx, w)
	plt.title(r'$R = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
	axes_5.xaxis.set_major_locator(plt.MaxNLocator(Nx))
	axes_5.yaxis.set_major_locator(plt.MaxNLocator(Ny))

	plt.savefig('../output/Correlation_' + star + '_half_frequency.png')
	plt.show()




#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
if 0:

	from astropy.stats import LombScargle

	left  = 0.1  # the left side of the subplots of the figure
	right = 0.96    # the right side of the subplots of the figure
	bottom = 0.12   # the bottom of the subplots of the figure
	top = 0.9      # the top of the subplots of the figure
	wspace = 0.55   # the amount of width reserved for blank space between subplots
	hspace = 0.1   # the amount of height reserved for white space between subplots

	plt.rcParams.update({'font.size': 24})
	fig = plt.figure(figsize=(15, 7))  
	ax = fig.add_subplot(1, 1, 1)
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

	min_f   = 1/2000
	max_f   = 1
	spp     = 20

	frequency0, power0 = LombScargle(MJD, x, w).autopower(minimum_frequency=min_f,
	                                                        maximum_frequency=max_f,
	                                                        samples_per_peak=spp)

	frequency1, power1 = LombScargle(MJD, xy, w).autopower(minimum_frequency=min_f,
	                                                            maximum_frequency=max_f,
	                                                            samples_per_peak=spp)

	frequency2, power2 = LombScargle(MJD, zx, w*3).autopower(minimum_frequency=min_f,
	                                                        maximum_frequency=max_f,
	                                                        samples_per_peak=spp)


	plt.plot(1/frequency0, power0, 'k-', label=r'$RV_{HARPS}$')
	plt.plot(1/frequency1, power1, 'r--', label=r'$\Delta RV_{L}$', alpha=0.7)
	plt.plot(1/frequency2, power2, 'b-.', label=r'$\Delta RV_{H}$', alpha=0.7)
	plt.xlabel('Period [days]')
	plt.ylabel("Power")
	ax.axvline(x=11.45, color='orange', linewidth=3.0, alpha=0.5)
	ax.axvline(x=2.92*365, color='orange', linewidth=3.0, alpha=0.5)
	ax.axvline(x=3.17*365, color='orange', linewidth=3.0, alpha=0.5)
	ax.axvline(x=2.59*365, color='orange', linewidth=3.0, alpha=0.5)
	ax.set_xscale('log')
	plt.ylim(0, 0.6)   
	plt.xlim(0, 2000)   
	plt.legend()    
	plt.savefig('../output/' + star + 'Periodogram_1.png')
	plt.show()


	left  = 0.1  # the left side of the subplots of the figure
	right = 0.96    # the right side of the subplots of the figure
	bottom = 0.12   # the bottom of the subplots of the figure
	top = 0.9      # the top of the subplots of the figure
	wspace = 0.55   # the amount of width reserved for blank space between subplots
	hspace = 0.1   # the amount of height reserved for white space between subplots

	plt.rcParams.update({'font.size': 24})
	fig = plt.figure(figsize=(15, 7))  
	ax = fig.add_subplot(1, 1, 1)
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

	min_f   = 1/2000
	max_f   = 1
	spp     = 20

	frequency0, power0 = LombScargle(MJD, x, w).autopower(minimum_frequency=min_f,
	                                                        maximum_frequency=max_f,
	                                                        samples_per_peak=spp)

	frequency1, power1 = LombScargle(MJD, xy, w).autopower(minimum_frequency=min_f,
	                                                            maximum_frequency=max_f,
	                                                            samples_per_peak=spp)

	frequency2, power2 = LombScargle(MJD, zx, w*3).autopower(minimum_frequency=min_f,
	                                                        maximum_frequency=max_f,
	                                                        samples_per_peak=spp)


	plt.plot(frequency0, power0, 'k-', label=r'$RV_{HARPS}$')
	plt.plot(frequency1, power1, 'r-.', label=r'$\Delta RV_{L}$', alpha=0.7)
	plt.plot(frequency2, power2, 'b--', label=r'$\Delta RV_{H}$', alpha=0.7)
	plt.xlabel('Frequency [1/day]')
	plt.ylabel("Power")
	ax.axvline(x=1/11.45, color='orange', linewidth=3.0, alpha=0.5)
	ax.axvline(x=1/(2.92*365), color='orange', linewidth=3.0, alpha=0.5)
	ax.axvline(x=1/(3.17*365), color='orange', linewidth=3.0, alpha=0.5)
	ax.axvline(x=1/(2.59*365), color='orange', linewidth=3.0, alpha=0.5)
	plt.ylim(0, 0.6)   
	plt.xlim(0, 1)   
	plt.legend(loc=9)    
	plt.savefig('../output/' + star + 'Periodogram_2.png')
	plt.show()