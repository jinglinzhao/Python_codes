import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# import data
#==============================================================================

DIR = '/Volumes/DataSSD/Python_codes/1002-multi_discoveries/data/HD128621/plot/'

t2009 		= np.loadtxt(DIR + '2009/plot_t.txt')
y2009 		= np.loadtxt(DIR + '2009/plot_RV_HARPS.txt')
yerr2009 	= np.loadtxt(DIR + '2009/plot_yerr.txt')

t2010 		= np.loadtxt(DIR + '2010/plot_t.txt')
y2010 		= np.loadtxt(DIR + '2010/plot_RV_HARPS.txt')
yerr2010 	= np.loadtxt(DIR + '2010/plot_yerr.txt')

t2011 		= np.loadtxt(DIR + '2011/plot_t.txt')
y2011 		= np.loadtxt(DIR + '2011/plot_RV_HARPS.txt')
yerr2011 	= np.loadtxt(DIR + '2011/plot_yerr.txt')

t 		= np.hstack([t2009, t2010, t2011]) + 0.5
RV_HARPS= np.hstack([y2009, y2010, y2011]) * 1000
yerr 	= np.hstack([yerr2009, yerr2010, yerr2011])

#==============================================================================
# Fitting
#==============================================================================

from numpy.polynomial import polynomial as P
c, stats    = P.polyfit(t, RV_HARPS, 2, full=True, w = 1/yerr**2)

if 0: # visualize the fitting 
	x_fit       = np.linspace(min(t-1), max(t+1), 10000)
	y_fit       = P.polyval(x_fit, c)
	plt.errorbar(t, RV_HARPS, yerr=yerr, fmt=".k", capsize=0, alpha=0.2)
	plt.plot(x_fit, y_fit)
	plt.show()

y_fitt = P.polyval(t, c)
plt.errorbar(t, RV_HARPS-y_fitt, yerr=yerr, fmt=".k", capsize=0, alpha=0.2)
plt.show()