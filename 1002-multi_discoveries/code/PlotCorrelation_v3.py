'''
Correlation
'''

# include FWHM, BIS, V_span @22/08

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from WeightedPearsonCorrelationCoefficient import wPearsonCoefficient
# uncertainty https://stats.stackexchange.com/questions/226380/derivation-of-the-standard-error-for-pearsons-correlation-coefficient


#==============================================================================
# Import data 
#==============================================================================

star 		= 'HD224789'
# star 		= 'HD200143'
# star 		= 'BD-213153'
# star 		= 'HD216770'
# star 		= 'HD36051'

directory 	= '/Volumes/DataSSD/MATLAB_codes/0816-FT-multiple_stars/' + star + '/'
RV 			= np.loadtxt(directory + 'GG.txt')
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
xy 			= RV - y
zx   		= z - RV 

DIR     	= '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
RV_noise	= np.loadtxt(DIR + '/RV_noise.dat')
FWHM 		= np.loadtxt(DIR + '/FWHM.dat')
dFWHM 		= np.loadtxt(DIR + '/dFWHM.dat')
BIS 		= np.loadtxt(DIR + '/BIS.dat')
dBIS 		= np.loadtxt(DIR + '/dBIS.dat')
V_span 		= np.loadtxt(DIR + '/V_span.dat')
dV_span 	= np.loadtxt(DIR + '/dV_span.dat')
MJD 		= np.loadtxt(DIR + '/MJD.dat')

#==============================================================================
# Correlation 
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

if star == 'BD-213153':
	fig.suptitle(star, y=0.95)
elif star == 'HD216770':
	# fig.suptitle('HD ' + star[2:] + ' (G2 template, after fibre upgrade)', y=0.95)
	fig.suptitle('HD ' + star[2:] + ' (K5 template, before fibre upgrade)', y=0.95)
else:
	fig.suptitle('HD ' + star[2:], y=0.95)

axes_1 = plt.subplot(151)
plt.errorbar(RV, FWHM, xerr=RV_noise, yerr=dFWHM, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel('FWHM [km/s]')
r, delta_r = wPearsonCoefficient(RV, FWHM, w)
plt.title(r'$\rho = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
axes_1.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_1.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_2 = plt.subplot(152)
plt.errorbar(RV, BIS, xerr=RV_noise, yerr=dBIS, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel('BIS [m/s]')
r, delta_r = wPearsonCoefficient(RV, BIS, w)
plt.title(r'$\rho = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
axes_2.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_2.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_3 = plt.subplot(153)
plt.errorbar(RV, V_span, xerr=RV_noise, yerr=dV_span, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$V_{span}$ [m/s]')
r, delta_r = wPearsonCoefficient(RV, V_span, w)
plt.title(r'$\rho = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
axes_3.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_3.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_4 = plt.subplot(154)
plt.errorbar(RV, xy, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
r, delta_r = wPearsonCoefficient(RV, xy, w)
fit, V 	= np.polyfit(RV, xy, 1, w=w, cov=True)
plt.title(r'$\rho = {0:.2f}±{1:.2f};\ k={2:.2f}$'.format(r,delta_r,fit[0]), fontsize=fontsize)
axes_4.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_4.yaxis.set_major_locator(plt.MaxNLocator(Ny))

axes_5 = plt.subplot(155)
plt.errorbar(RV, zx, xerr=RV_noise, yerr=RV_noise*3**0.5, fmt="ko", capsize=0, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
r, delta_r = wPearsonCoefficient(RV, zx, w)
fit, V 	= np.polyfit(RV, zx, 1, w=w, cov=True)
plt.title(r'$\rho = {0:.2f}±{1:.2f};\ k={2:.2f}$'.format(r,delta_r,fit[0]), fontsize=fontsize)
axes_5.xaxis.set_major_locator(plt.MaxNLocator(Nx))
axes_5.yaxis.set_major_locator(plt.MaxNLocator(Ny))

plt.savefig('../output/Correlation_' + star + '.png')
plt.show()


#############################
# Plot the improved version #
#############################
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

	if star == 'BD-213153':
		fig.suptitle(star, y=0.95)
	elif star == 'HD216770':
		# fig.suptitle('HD ' + star[2:] + ' (G2 template, after fibre upgrade)', y=0.95)
		fig.suptitle('HD ' + star[2:] + ' (K5 template, before fibre upgrade)', y=0.95)
	else:
		fig.suptitle('HD ' + star[2:], y=0.95)

	axes_4 = plt.subplot(121)
	plt.errorbar(RV, xy, xerr=RV_noise, yerr=RV_noise*(39/19)**0.5, fmt="ko", capsize=0, alpha=alpha)
	plt.xlabel(r'$RV_{HARPS}$ [m/s]')
	plt.ylabel(r'$\Delta RV_L$ [m/s]')
	r, delta_r = wPearsonCoefficient(RV, xy, w)
	plt.title(r'$\rho = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
	axes_4.xaxis.set_major_locator(plt.MaxNLocator(Nx))
	axes_4.yaxis.set_major_locator(plt.MaxNLocator(Ny))

	axes_5 = plt.subplot(122)
	plt.errorbar(RV, zx, xerr=RV_noise, yerr=RV_noise*21**0.5, fmt="ko", capsize=0, alpha=alpha)
	plt.xlabel(r'$RV_{HARPS}$ [m/s]')
	plt.ylabel(r'$\Delta RV_H$ [m/s]')
	r, delta_r = wPearsonCoefficient(RV, zx, w)
	plt.title(r'$\rho = {0:.2f}±{1:.2f}$'.format(r, delta_r), fontsize=fontsize)
	axes_5.xaxis.set_major_locator(plt.MaxNLocator(Nx))
	axes_5.yaxis.set_major_locator(plt.MaxNLocator(Ny))

	plt.savefig('../output/Correlation_' + star + '_half_frequency.png')
	plt.show()


if 0: 
	plt.rcParams.update({'font.size': 20})
	fig, axes = plt.subplots(figsize=(20, 8))
	if star == 'HD224789':
		axes = plt.subplot(311)
		plt.errorbar(MJD, RV, RV_noise, fmt=".k", capsize=0)
		plt.ylabel('RV [m/s]')
		plt.ylim([-40,40])
		plt.title(star)

		axes = plt.subplot(312)
		plt.errorbar(MJD, xy, RV_noise*3**0.5, fmt=".k", capsize=0)
		plt.ylim([-40,40])
		plt.ylabel(r'$\Delta RV_L$ [m/s]')

		axes = plt.subplot(313)
		plt.errorbar(MJD, zx, RV_noise*3**0.5, fmt=".k", capsize=0)
		plt.xlabel('MJD')
		plt.ylim([-40,40])
		plt.ylabel(r'$\Delta RV_H$ [m/s]')

		plt.savefig('../output/Time_series_' + star + '.png')
		plt.show()


