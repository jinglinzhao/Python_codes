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

directory 	= '/Volumes/DataSSD/MATLAB_codes/0615-FT-HD189733/'
star 		= 'HD189733'
# star 		= 'HD103720'
# star 		= 'HD36051'
# directory 	= '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
x 			= np.loadtxt(directory + 'GG.txt')
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
bi 			= np.loadtxt(directory + 'BI.txt')
xy 			= 0 - y
zx   		= z  

DIR     	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
RV_noise	= np.loadtxt(DIR + '/RV_noise.dat')
MJD     	= np.loadtxt(DIR + '/MJD.dat')


from functions import gaussian_smoothing
plt.errorbar(MJD, xy, yerr=RV_noise*3**0.5, fmt="r.", capsize=0)
jitter_s = gaussian_smoothing(MJD, xy, MJD, 0.008, 1/RV_noise**2)
plt.errorbar(MJD, jitter_s, yerr=RV_noise*3**0.5, fmt="b.", capsize=0)
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
Ny = 4

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star=='HD36051':
	fig.suptitle('HD ' + star[2:] + ' (planet candidate not removed)', y=0.9)
if star=='HD103720':
	fig.suptitle('HD ' + star[2:] + ' (planet not removed)', y=0.9)
if star=='HD189733':
	fig.suptitle('HD ' + star[2:] + ' (companions not removed)', y=0.9)

axes_1 = plt.subplot(131)
plt.errorbar(x, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
fit, V = np.polyfit(x, xy, 1, w=w, cov=True)
r1, p = stats.pearsonr(x, xy)
r = wPearsonCoefficient(x, xy, w)
xspacing = (max(x) - min(x)) * 0.05
yspacing 	= (max(zx) - min(zx)) * 0.1
y_up 		= max(zx+RV_noise+2.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.xlim(min(x)-xspacing, max(x)+xspacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(132)
plt.errorbar(x, zx, yerr=RV_noise, xerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit = np.polyfit(x, zx, 1, w=w)
r1, p 		= stats.pearsonr(x, zx)
r 			= wPearsonCoefficient(x, zx, w)
plt.xlim(min(x)-xspacing, max(x)+xspacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(133)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
xspacing 	= (max(xy) - min(xy)) * 0.1
x_up 		= max(xy+RV_noise+0.5*xspacing)
x_lo		= min(xy-RV_noise-0.5*xspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
fit, V = np.polyfit(xy, zx, 1, w=w, cov=True)
r1, p 		= stats.pearsonr(xy, zx)
r = wPearsonCoefficient(xy, zx, w)
plt.text(min(xy), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(xy), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '_1.png')
plt.show()

#==============================================================================
# Correlation 2: remove binary companion 
#==============================================================================

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star=='HD36051':
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
plt.xlim(min(x-bi)-xspacing, max(x-bi)+xspacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-bi), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-bi), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(132)
plt.errorbar(x-bi, zx, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ detrended [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit, V = np.polyfit(x-bi, zx, 1, w=w, cov=True)
r = wPearsonCoefficient(x-bi, zx, w)
r1, p = stats.pearsonr(x-bi, zx)
plt.xlim(min(x-bi)-xspacing, max(x-bi)+xspacing)
plt.ylim(y_lo, y_up)
if star=='HD103720' or star=='HD36051' or star=='HD189733':
	plt.text(min(x-bi), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
	plt.text(min(x-bi), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
else: 
	plt.xlim(min(x)-xspacing, max(x)+xspacing)
	plt.text(min(x), 0.95*max(zx)+0.05*min(zx), 'R={:.3f}'.format(r), fontsize=20)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(133)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
xspacing 	= (max(xy) - min(xy)) * 0.1
x_up 		= max(xy+RV_noise+0.5*xspacing)
x_lo		= min(xy-RV_noise-0.5*xspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
fit, V = np.polyfit(xy, zx, 1, w=w, cov=True)
r1, p 		= stats.pearsonr(xy, zx)
r = wPearsonCoefficient(xy, zx, w)
plt.text(min(xy), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
# plt.text(min(xy), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r), fontsize=20)
plt.text(min(xy), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '_2.png')
plt.show()
