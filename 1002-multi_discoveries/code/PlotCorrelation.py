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

# if star == 'HD36051' or star=='HD216770':
left = 0.07  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
# if star == 'HD36051' or star=='HD216770':
wspace = 0.5  # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
alpha = 0.5
markersize = 12
w = 1/RV_noise**2

plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


plt.subplot(151)
plt.errorbar(x, y, xerr=RV_noise, yerr=RV_noise*2**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]') 
spacing = (max(x) - min(x)) * 0.15
plt.xlim(min(x)-spacing, max(x)+spacing)
fit = np.polyfit(x, y, 1, w=w)
r = wPearsonCoefficient(x, y, w)
plt.text(min(x-RV_noise), 0.95*max(y+RV_noise)+0.05*min(y-RV_noise), 'R={0:.2f}'.format(r), fontsize=20)
plt.text(min(x-RV_noise), 0.85*max(y+RV_noise)+0.15*min(y-RV_noise), 'k={0:.2f}'.format(fit[0]), fontsize=20)


plt.subplot(152)
plt.errorbar(x, z, xerr=RV_noise, yerr=RV_noise*2**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')   
plt.xlim(min(x)-spacing, max(x)+spacing)
fit = np.polyfit(x, z, 1, w=w)
r = wPearsonCoefficient(x, z, w)
plt.text(min(x-RV_noise), 0.95*max(z+RV_noise)+0.05*min(z-RV_noise), 'R={0:.2f}'.format(r), fontsize=20)
plt.text(min(x-RV_noise), 0.85*max(z+RV_noise)+0.15*min(z-RV_noise), 'k={0:.2f}'.format(fit[0]), fontsize=20)


plt.subplot(153)
plt.errorbar(x, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.title('HD ' + star[2:])
# plt.title('HD ' + star[2:] + ' (MJD > 57161)')
fit, V = np.polyfit(x, xy, 1, w=w, cov=True)
r1, p = stats.pearsonr(x, xy)
r = wPearsonCoefficient(x, xy, w)
yspacing 	= (max(xy) - min(xy)) * 0.1
y_up 		= max(xy+RV_noise+1.5*yspacing)
y_lo		= min(xy-RV_noise-0.5*yspacing)
plt.xlim(min(x)-spacing, max(x)+spacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)


plt.subplot(154)
plt.errorbar(x, zx, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit, V = np.polyfit(x, zx, 1, w=w, cov=True)
r1, p 		= stats.pearsonr(x, zx)
r 			= wPearsonCoefficient(x, zx, w)
yspacing 	= (max(zx) - min(zx)) * 0.1
y_up 		= max(zx+RV_noise+1.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.xlim(min(x)-spacing, max(x)+spacing)
plt.ylim(y_lo, y_up)
if star == 'HD224789':
	plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
	plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)
else:
	plt.ylim(y_lo, y_up)
	plt.text(min(x), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)


plt.subplot(155)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
spacing_xy = (max(xy) - min(xy)) * 0.1
plt.xlim(min(xy)-spacing_xy, max(xy)+spacing_xy)
fit = np.polyfit(xy, zx, 1, w=w)
r = wPearsonCoefficient(xy, zx, w)
plt.text(min(xy), 0.95*max(zx)+0.05*min(zx), 'R={0:.2f}'.format(r), fontsize=20)
plt.text(min(xy), 0.85*max(zx)+0.15*min(zx), 'k={0:.2f}'.format(fit[0]), fontsize=20)   

plt.savefig('../output/Correlation_' + star + '.png')
plt.show()


def wrms(x, w):
    mean = np.sum(x*w)/np.sum(w)
    return np.sqrt(np.sum((x-mean)**2*w) / np.sum(w))


wrms_x = wrms(x, w)
wrms_l = wrms(xy, w)
wrms_h = wrms(zx, w)
print(wrms_l, wrms_h, wrms_x, wrms_l/wrms_x, wrms_h/wrms_x)

wrms_x = (wrms(x, w)**2 - (2*2**0.5)**2)**0.5
wrms_l = (wrms(xy, w)**2 - (2*2**0.5)**2)**0.5
wrms_h = (wrms(zx, w)**2 - (2*2**0.5)**2)**0.5
wrms_x2 = (wrms(x-bi, w)**2 - 2**2)**0.5