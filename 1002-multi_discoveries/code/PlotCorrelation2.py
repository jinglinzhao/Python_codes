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
xy 			= x - y
zx   		= z - x 

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
fit, V = np.polyfit(x, y, 1, w=w, cov=True)
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
# plt.title('HD 189733 (companions not removed)')
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

if star=='HD36051':
	plt.title('HD ' + star[2:] + ' (planet candidate not removed)')
if star=='HD103720':
	plt.title('HD ' + star[2:] + ' (planet not removed)')
if star=='HD189733':
	plt.title('HD ' + star[2:] + ' (companions not removed)')

plt.subplot(154)
plt.errorbar(x, zx, yerr=RV_noise, xerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
plt.xlim(min(x)-spacing, max(x)+spacing)
fit = np.polyfit(x, zx, 1, w=w)
r1, p 		= stats.pearsonr(x, zx)
r 			= wPearsonCoefficient(x, zx, w)
yspacing 	= (max(zx) - min(zx)) * 0.1
y_up 		= max(zx+RV_noise+1.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.xlim(min(x)-spacing, max(x)+spacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)


plt.subplot(155)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
xspacing 	= (max(xy) - min(xy)) * 0.1
yspacing 	= (max(zx) - min(zx)) * 0.1
x_up 		= max(xy+RV_noise+0.5*xspacing)
x_lo		= min(xy-RV_noise-0.5*xspacing)
y_up 		= max(zx+RV_noise+1.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
fit = np.polyfit(xy, zx, 1, w=w)
r = wPearsonCoefficient(xy, zx, w)
plt.text(min(xy), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r), fontsize=20)
plt.text(min(xy), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)   



plt.savefig('../output/Correlation_' + star + '_1.png')
plt.show()

#==============================================================================
# Correlation 2: remove binary companion 
#==============================================================================

plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.errorbar(x-bi, y-bi, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]')    
if star=='HD103720' or star=='HD36051' or star=='HD189733':
	spacing = (max(x-bi) - min(x-bi)) * 0.15
	plt.xlim(min(x-bi)-spacing, max(x-bi)+spacing)
	fit = np.polyfit(x-bi, y-bi, 1, w=w)
	r = wPearsonCoefficient(x-bi, y-bi, w)
	plt.text(min(x-bi-RV_noise), 0.95*max(y-bi+RV_noise)+0.05*min(y-bi-RV_noise), 'R={0:.2f}'.format(r), fontsize=20)
	plt.text(min(x-bi-RV_noise), 0.85*max(y-bi+RV_noise)+0.15*min(y-bi-RV_noise), 'k={0:.2f}'.format(fit[0]), fontsize=20)
else:
	plt.xlim(min(x)-spacing, max(x)+spacing)


plt.subplot(152)
plt.errorbar(x-bi, z-bi, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')
if star=='HD103720' or star=='HD36051' or star=='HD189733':
	fit = np.polyfit(x-bi, z-bi, 1, w=w)
	r = wPearsonCoefficient(x-bi, z-bi, w)
	plt.text(min(x-bi-RV_noise), 0.95*max(z-bi+RV_noise)+0.05*min(z-bi-RV_noise), 'R={0:.2f}'.format(r), fontsize=20)
	plt.text(min(x-bi-RV_noise), 0.85*max(z-bi+RV_noise)+0.15*min(z-bi-RV_noise), 'k={0:.2f}'.format(fit[0]), fontsize=20)
else:
	plt.xlim(min(x)-spacing, max(x)+spacing)


plt.subplot(153)
plt.errorbar(x-bi, xy, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
fit, V = np.polyfit(x-bi, xy, 1, w=w, cov=True)
r = wPearsonCoefficient(x-bi, xy, w)
r1, p = stats.pearsonr(x-bi, xy)
yspacing 	= (max(xy) - min(xy)) * 0.1
y_up 		= max(xy+RV_noise+1.5*yspacing)
y_lo		= min(xy-RV_noise-0.5*yspacing)
plt.xlim(min(x-bi)-spacing, max(x-bi)+spacing)
plt.ylim(y_lo, y_up)
plt.text(min(x-bi-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
plt.text(min(x-bi-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)
if star=='HD36051':
	plt.title('HD ' + star[2:] + ' (planet candidate removed)')
if star=='HD103720':
	plt.title('HD ' + star[2:] + ' (planet removed)')
if star=='HD189733':
	plt.title('HD ' + star[2:] + ' (companions removed)')


plt.subplot(154)
plt.errorbar(x-bi, zx, xerr=RV_noise*2**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
fit, V = np.polyfit(x-bi, zx, 1, w=w, cov=True)
r = wPearsonCoefficient(x-bi, zx, w)
r1, p = stats.pearsonr(x-bi, zx)
yspacing 	= (max(zx) - min(zx)) * 0.1
y_up 		= max(zx+RV_noise+1.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.xlim(min(x-bi)-spacing, max(x-bi)+spacing)
plt.ylim(y_lo, y_up)
# from numpy.polynomial import polynomial as P
# c = P.polyfit(x-bi,xy,1,w=0/RV_noise**2+1)
# print(c)
# print(stats)
if star=='HD103720' or star=='HD36051' or star=='HD189733':
	plt.text(min(x-bi-RV_noise), 0.9*y_up+0.1*y_lo, 'R={0:.2f} ({1:.2f})'.format(r,r1), fontsize=20)
	plt.text(min(x-bi-RV_noise), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)
else: 
	plt.xlim(min(x)-spacing, max(x)+spacing)
	plt.text(min(x), 0.95*max(zx)+0.05*min(zx), 'R={:.3f}'.format(r), fontsize=20)


plt.subplot(155)
plt.errorbar(xy, zx, xerr=RV_noise*3**0.5, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
xspacing 	= (max(xy) - min(xy)) * 0.1
yspacing 	= (max(zx) - min(zx)) * 0.1
x_up 		= max(xy+RV_noise+0.5*xspacing)
x_lo		= min(xy-RV_noise-0.5*xspacing)
y_up 		= max(zx+RV_noise+1.5*yspacing)
y_lo		= min(zx-RV_noise-0.5*yspacing)
plt.xlim(x_lo, x_up)
plt.ylim(y_lo, y_up)
fit = np.polyfit(xy, zx, 1, w=w)
r = wPearsonCoefficient(xy, zx, w)
plt.text(min(xy), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r), fontsize=20)
plt.text(min(xy), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=20)   

plt.savefig('../output/Correlation_' + star + '_2.png')
plt.show()



def wrms(x, w):
    mean = np.sum(x*w)/np.sum(w)
    return np.sqrt(np.sum((x-mean)**2*w) / np.sum(w))

wrms1 = wrms(x, w)
wrms2 = wrms(xy, w)
wrms3 = wrms(x-bi, w)
print(wrms1, wrms2, wrms3, wrms2/wrms1)

wrms_x = (wrms(x, w)**2 - (2*2**0.5)**2)**0.5
wrms_l = (wrms(xy, w)**2 - (2*2**0.5)**2)**0.5
wrms_h = (wrms(zx, w)**2 - (2*2**0.5)**2)**0.5
wrms_x2 = (wrms(x-bi, w)**2 - 2**2)**0.5

print(wrms_x, wrms_l, wrms_h, wrms_l/wrms_x, wrms_h/wrms_x)
print(wrms_x2, wrms_l/wrms_x2, wrms_h/wrms_x2)

