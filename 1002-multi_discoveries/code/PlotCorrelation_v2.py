'''
Correlation
'''

# only 3 panels 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from WeightedPearsonCorrelationCoefficient import wPearsonCoefficient

#==============================================================================
# Import data 
#==============================================================================

star 		= 'HD224789'
# star 		= 'HD200143'
# star 		= 'BD-213153'
# star 		= 'HD216770'
# star 		= 'HD36051'

directory 	= '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
x 			= np.loadtxt(directory + 'GG.txt')
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
xy 			= x - y
zx   		= z - x 

DIR     	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
RV_noise	= np.loadtxt(DIR + '/RV_noise.dat')

#==============================================================================
# Correlation 1
#==============================================================================

left  = 0.08  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.2   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.55   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
alpha = 0.5
w = 1/RV_noise**2
Nx = 3
Ny = 3

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

if star == 'BD-213153':
	fig.suptitle(star, y=0.9)
elif star == 'HD216770':
	fig.suptitle('HD ' + star[2:] + ' (G2 template, after fibre upgrade)', y=0.9)
	# fig.suptitle('HD ' + star[2:] + ' (K5 template, before fibre upgrade)', y=0.9)
else:
	fig.suptitle('HD ' + star[2:], y=0.9)

axes_1 = plt.subplot(131)
plt.errorbar(x, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
fit, V = np.polyfit(x, xy, 1, w=w, cov=True)
r1, p = stats.pearsonr(x, xy)
r = wPearsonCoefficient(x, xy, w)
xspacing = (max(x) - min(x)) * 0.10
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
plt.errorbar(x, zx, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit, V = np.polyfit(x, zx, 1, w=w, cov=True)
r1, p 		= stats.pearsonr(x, zx)
r 			= wPearsonCoefficient(x, zx, w)
plt.xlim(min(x)-xspacing, max(x)+xspacing)
plt.ylim(y_lo, y_up)
# if star == 'HD224789' or star == 'HD200143':
plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
# else:
# 	plt.ylim(y_lo, y_up)
# 	plt.text(min(x), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(133)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.ylim(y_lo, y_up)
spacing_xy = (max(xy) - min(xy)) * 0.15
plt.xlim(min(xy)-spacing_xy, max(xy)+spacing_xy)
fit = np.polyfit(xy, zx, 1, w=w)
r1, p = stats.pearsonr(xy, zx)
r = wPearsonCoefficient(xy, zx, w)
plt.text(min(xy), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(xy), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=20)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '.png')
plt.show()

# V_span
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
plt.show()

