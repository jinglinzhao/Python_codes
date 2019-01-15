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
# star 		= 'HD103720'
star 		= 'HD36051'
directory 	= '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
x 			= np.loadtxt(directory + 'GG.txt')
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
bi 			= np.loadtxt(directory + 'BI.txt')
xy 			= x - y
zx   		= z - x 

DIR     	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
RV_noise	= np.loadtxt(DIR + '/RV_noise.dat')

# data  		= np.loadtxt('./HD103720/gp_predict.txt')
# jitter  	= data[:,1]

#==============================================================================
# Correlation 1
#==============================================================================

if star=='HD103720' or star=='HD36051':
	left = 0.07
else:
	left  = 0.06  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
if star=='HD103720' or star=='HD36051':
	wspace = 0.5
else: 
	wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
alpha = 0.3
markersize = 12

plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.errorbar(x, y, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]') 
spacing = (max(x) - min(x)) * 0.1
plt.xlim(min(x)-spacing, max(x)+spacing)

plt.subplot(152)
plt.errorbar(x, z, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')   
plt.xlim(min(x)-spacing, max(x)+spacing)

plt.subplot(153)
plt.errorbar(x, xy, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
# plt.title('HD 189733 (companions not removed)')
if star=='HD36051':
	plt.title('HD ' + star[2:] + ' (planet candidate not removed)')
else:
	plt.title('HD ' + star[2:] + ' (planet not removed)')
plt.xlim(min(x)-spacing, max(x)+spacing)
r1, p = stats.pearsonr(x, xy)
r = wPearsonCoefficient(x, xy, 1/RV_noise**2)
plt.text(min(x), 0.95*max(xy+RV_noise)+0.05*min(xy-RV_noise), 'R={0:.3f} ({1:.3f})'.format(r,r1), fontsize=20)


plt.subplot(154)
plt.errorbar(x, zx, yerr=RV_noise, xerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
plt.xlim(min(x)-spacing, max(x)+spacing)
r1, p = stats.pearsonr(x, zx)
r = wPearsonCoefficient(x, zx, 1/RV_noise**2)
yspacing 	= (max(zx) - min(zx)) * 0.1
y_up 		= max(zx+RV_noise+1.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.ylim(y_lo, y_up)
plt.text(min(x), 0.9*y_up+0.1*y_lo, 'R={0:.3f} ({1:.3f})'.format(r,r1), fontsize=20)


plt.subplot(155)
plt.errorbar(xy, zx, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
spacing_xy = (max(xy) - min(xy)) * 0.1
plt.xlim(min(xy)-spacing_xy, max(xy)+spacing_xy)
plt.savefig('../output/Correlation_' + star + '_1.png')
plt.show()

#==============================================================================
# Correlation 2: remove binary companion 
#==============================================================================

plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.errorbar(x-bi, y-bi, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]')    
if star=='HD103720' or star=='HD36051':
	pass
else:
	plt.xlim(min(x)-spacing, max(x)+spacing)


plt.subplot(152)
plt.errorbar(x-bi, z-bi, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')
if star=='HD103720' or star=='HD36051':
	pass
else:
	plt.xlim(min(x)-spacing, max(x)+spacing)


plt.subplot(153)
plt.errorbar(x-bi, xy, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
r = wPearsonCoefficient(x-bi, xy, 1/RV_noise**2)
r1, p = stats.pearsonr(x-bi, xy)
if star=='HD103720' or star=='HD36051':
	if star=='HD36051':
		plt.title('HD ' + star[2:] + ' (planet candidate removed)')
	else:
		plt.title('HD ' + star[2:] + ' (planet removed)')
	# plt.xlim(min(x-bi)-spacing, max(x-bi)+spacing)
	plt.text(min(x-bi), 0.95*max(xy+RV_noise)+0.05*min(xy-RV_noise), 'R={0:.3f} ({1:.3f})'.format(r,r1), fontsize=20)
else: 
	plt.title('HD 189733 (companions removed)')
	plt.xlim(min(x)-spacing, max(x)+spacing)
	plt.text(min(x), 0.95*max(xy)+0.05*min(xy), 'R={:.3f}'.format(r), fontsize=20)


plt.subplot(154)
plt.errorbar(x-bi, zx, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
r = wPearsonCoefficient(x-bi, zx, 1/RV_noise**2)
r1, p = stats.pearsonr(x-bi, zx)

from numpy.polynomial import polynomial as P
c = P.polyfit(x-bi,xy,1,w=0/RV_noise**2+1)
print(c)
print(stats)

if star=='HD103720' or star=='HD36051':
	plt.ylim(y_lo, y_up)
	plt.text(min(x-bi), 0.9*y_up+0.1*y_lo, 'R={0:.3f} ({1:.3f})'.format(r,r1), fontsize=20)
else: 
	plt.xlim(min(x)-spacing, max(x)+spacing)
	plt.text(min(x), 0.95*max(zx)+0.05*min(zx), 'R={:.3f}'.format(r), fontsize=20)


plt.subplot(155)
plt.errorbar(xy, zx, xerr=RV_noise, yerr=RV_noise, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.xlim(min(xy)-spacing_xy, max(xy)+spacing_xy)
plt.savefig('../output/Correlation_' + star + '_2.png')
plt.show()