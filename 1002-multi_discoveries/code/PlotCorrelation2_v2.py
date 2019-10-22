'''
Correlation
'''



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
x 			= np.loadtxt(directory + 'GG.txt')
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
bi 			= np.loadtxt(directory + 'BI.txt')
# xy 			= 0 - y
# zx   		= z  
xy = x - y 
zx = z - x


DIR     	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
RV_noise	= np.loadtxt(DIR + '/RV_noise.dat')
MJD     	= np.loadtxt(DIR + '/MJD.dat')


from functions import gaussian_smoothing
plt.errorbar(MJD, xy, yerr=RV_noise*3**0.5, fmt="r.", capsize=0, alpha=0.2)
jitter_s = gaussian_smoothing(MJD, xy, MJD, 0.5, 1/RV_noise**2)
plt.errorbar(MJD, jitter_s, yerr=RV_noise*3**0.5, fmt="b.", capsize=0, alpha=0.3)
plt.show()

plt.errorbar(MJD, x, yerr=RV_noise, fmt="k.", capsize=0, alpha=0.5)
# plt.errorbar(MJD, xy, yerr=RV_noise*3**0.5, fmt="r.", capsize=0, alpha=0.2)
plt.errorbar(MJD, zx, yerr=RV_noise*3**0.5, fmt="r.", capsize=0, alpha=0.2)
plt.show()


# smoothing 
# x = gaussian_smoothing(MJD, x, MJD, 1, 1/RV_noise**2)
# y = gaussian_smoothing(MJD, y, MJD, 1, 1/RV_noise**2)
# z = gaussian_smoothing(MJD, z, MJD, 1, 1/RV_noise**2)
# xy = gaussian_smoothing(MJD, xy, MJD, 2, 1/RV_noise**2)
# zx = gaussian_smoothing(MJD, zx, MJD, 2, 1/RV_noise**2)

# data  		= np.loadtxt('./HD103720/gp_predict.txt')
# jitter  	= data[:,1]

#==============================================================================
# Correlation 1
#==============================================================================

left  = 0.08  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.2   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.55   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
alpha = 0.3
markersize = 12

w = 1/RV_noise**2
Nx = 3
Ny = 3

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star=='HD36051' or star=='HD22049':
	fig.suptitle('HD ' + star[2:] + ' (planet candidate not removed)', y=0.9)
elif star=='HD103720':
	fig.suptitle('HD ' + star[2:] + ' (planet not removed)', y=0.9)
elif star=='HD189733':
	fig.suptitle('HD ' + star[2:] + ' (companions not removed)', y=0.9)
else:
	fig.suptitle('HD ' + star[2:], y=0.9)	

axes_1 = plt.subplot(131)
plt.errorbar(x, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
fit, V = np.polyfit(x, xy, 1, w=w, cov=True)
r1, p = stats.pearsonr(x, xy)
r = wPearsonCoefficient(x, xy, w)
xspacing = (max(x) - min(x)) * 0.15
yspacing 	= (max(zx) - min(zx)) * 0.1
x_up 		= max(x+RV_noise+0.5*xspacing)
x_lo		= min(x-RV_noise-0.5*xspacing)
y_up 		= max(zx+RV_noise*3**0.5+2.5*yspacing)
y_lo		= min(zx-RV_noise*3**0.5-0.5*yspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(132)
plt.errorbar(x, zx, yerr=RV_noise, xerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit = np.polyfit(x, zx, 1, w=w)
r1, p 		= stats.pearsonr(x, zx)
r 			= wPearsonCoefficient(x, zx, w)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(133)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
xspacing 	= (max(xy) - min(xy)) * 0.1
x_up 		= max(xy+RV_noise*3**0.5+0.5*xspacing)
x_lo		= min(xy-RV_noise*3**0.5-0.5*xspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
fit, V = np.polyfit(xy, zx, 1, w=w, cov=True)
r1, p 		= stats.pearsonr(xy, zx)
r = wPearsonCoefficient(xy, zx, w)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '_1.png')
plt.show()


plt.rcParams.update({'font.size': 14})
DIR 	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star + '/'
V_span 	= np.loadtxt(DIR + 'V_span.dat')
plt.figure()
plt.errorbar(x, V_span, xerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$V_{span}$ [m/s]')
fit, V 	= np.polyfit(x, V_span, 1, w=w, cov=True)
r1, p 	= stats.pearsonr(x, V_span)
r 		= wPearsonCoefficient(x, V_span, w)
xspacing= (max(x) - min(x)) * 0.10
yspacing= (max(V_span) - min(V_span)) * 0.1
y_up 	= max(V_span+RV_noise+2.5*yspacing)
y_lo	= min(V_span-RV_noise-0.5*yspacing)
plt.xlim(min(x)-xspacing, max(x)+xspacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=14)
plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=14)
plt.savefig('../output/Correlation_' + star + 'V_span_1.png')
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
left  = 0.08  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.2   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.55   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
alpha = 0.3
markersize = 12

w = 1/RV_noise**2
Ny = 3
Nx = 4

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star=='HD36051' or star=='HD22049':
	fig.suptitle('HD ' + star[2:] + ' (planet candidate removed)', y=0.9)
if star=='HD103720':
	fig.suptitle('HD ' + star[2:] + ' (planet removed)', y=0.9)
if star=='HD189733':
	fig.suptitle('HD ' + star[2:] + ' (companions removed)', y=0.9)

axes_1 = plt.subplot(131)
plt.errorbar(x-bi, xy, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ detrended [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
fit, V = np.polyfit(x-bi, xy, 1, w=w, cov=True)
r = wPearsonCoefficient(x-bi, xy, w)
r1, p = stats.pearsonr(x-bi, xy)
xspacing = (max(x) - min(x)) * 0.05
x_up	= max(x-bi+RV_noise*2**0.5+0.5*xspacing)
x_lo	= min(x-bi-RV_noise*2**0.5-0.5*xspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(132)
plt.errorbar(x-bi, zx, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ detrended [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit, V = np.polyfit(x-bi, zx, 1, w=w, cov=True)
r = wPearsonCoefficient(x-bi, zx, w)
r1, p = stats.pearsonr(x-bi, zx)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(133)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
xspacing 	= (max(xy) - min(xy)) * 0.1
x_up 		= max(xy+RV_noise*3**0.5+0.5*xspacing)
x_lo		= min(xy-RV_noise*3**0.5-0.5*xspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
fit, V = np.polyfit(xy, zx, 1, w=w, cov=True)
r1, p 		= stats.pearsonr(xy, zx)
r = wPearsonCoefficient(xy, zx, w)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '_2.png')
plt.show()


# V_span #
plt.rcParams.update({'font.size': 14})
plt.figure()
plt.errorbar(x-bi, V_span, xerr=RV_noise*2**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ detrended [m/s]')
plt.ylabel(r'$V_{span}$ [m/s]')
fit, V 	= np.polyfit(x-bi, V_span, 1, w=w, cov=True)
r1, p 	= stats.pearsonr(x-bi, V_span)
r 		= wPearsonCoefficient(x-bi, V_span, w)
xspacing = (max(x) - min(x)) * 0.05
x_up	= max(x-bi+RV_noise*2**0.5+0.5*xspacing)
x_lo	= min(x-bi-RV_noise*2**0.5-0.5*xspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
plt.text(0.9*x_lo+0.1*x_up, 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=14)
plt.text(0.9*x_lo+0.1*x_up, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=14)
plt.savefig('../output/Correlation_' + star + 'V_span_2.png')
plt.show()


#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
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