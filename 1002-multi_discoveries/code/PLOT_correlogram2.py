# Change the year accordingly 

import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing

#==============================================================================
# Binary orbit 
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

t 			= np.hstack([t2009, t2010, t2011]) + 0.5
RV_HARPS 	= np.hstack([y2009, y2010, y2011]) * 1000
yerr 		= np.hstack([yerr2009, yerr2010, yerr2011])

idx1_HARPS	= (t < 54962) & (t > 54872)
idx2_HARPS	= (t < 55363) & (t > 55273)
idx3_HARPS	= (t < 55698) & (t > 55608)

from numpy.polynomial import polynomial as P
c, stats    = P.polyfit(t, RV_HARPS, 2, full=True, w = 1/yerr**2)

# Binary orbit (without fitting planet) # 
# email correspondence with Dumusque
lin0 = -22700.1747
lin1 = -0.5307
lin2 = -1.83e-5
BJD0 = 55279.109840075726
def trend(x):
    return lin0 + lin1 * (x-BJD0) + lin2 * (x-BJD0)**2

if 0: # visualize the fitting 
	x_fit       = np.linspace(min(t-1), max(t+1), 10000)
	y_fit       = P.polyval(x_fit, c)
	plt.errorbar(t, RV_HARPS, yerr=yerr, fmt=".k", capsize=0, alpha=0.2)
	plt.plot(x_fit, y_fit)
	plt.show()

y_fitt = P.polyval(t, c)
if 0: 
    plt.errorbar(t, RV_HARPS-y_fitt, yerr=yerr, fmt=".k", capsize=0, alpha=0.2)
    plt.show()

    plt.errorbar(t, RV_HARPS-trend(t), yerr=yerr, fmt=".k", capsize=0, alpha=0.2)
    plt.show()


#==============================================================================
# import my jitter metric 
#==============================================================================
YEAR = 2010

t 		= np.loadtxt('../data/HD128621/plot/' + str(YEAR) + '/plot_t.txt') + 0.5 	# JD - 2,400,000
y 		= np.loadtxt('../data/HD128621/plot/' + str(YEAR) + '/plot_y.txt')
yerr 	= np.loadtxt('../data/HD128621/plot/' + str(YEAR) + '/plot_yerr.txt')


#==============================================================================
# import the wise data 
#==============================================================================
JD_wise = np.loadtxt('../data/HD128621/plot/Activity_indicators_from_Wise/JDs.txt')
JD_wise = JD_wise - 2400000
fe4376  = np.loadtxt('../data/HD128621/plot/Activity_indicators_from_Wise/fe4376.txt')
fe5250  = np.loadtxt('../data/HD128621/plot/Activity_indicators_from_Wise/fe5250.txt')
idx1    = (JD_wise < 54962) & (JD_wise > 54872)
idx2    = (JD_wise < 55363) & (JD_wise > 55273)
idx3    = (JD_wise < 55698) & (JD_wise > 55608)

#==============================================================================
# Alpha_Cen_B_supplementary_data
#==============================================================================
# data format #
# jdb (days)  RV (km/s)   RV error (km/s) realistic RV error (km/s)   Bisector (km/s) FWHM (km/s) Photon noise (km/s) log(R'hk)   log(R'hk) error Low pass filter log(R'hk)   Alpha (hours)   Delta (degrees) Seeing
# ----------  ---------   --------------- -------------------------   --------------- ----------- ------------------- ---------   --------------- -------------------------   -------------   --------------- ------
file    = '../data/HD128621/Alpha_Cen_B_supplementary_data.txt'
data    = np.loadtxt(file)
BJD     = data[:,0]
BIS     = data[:,4] * 1000 # m/s
FWHM    = data[:,5] # km/s
log_RHK = data[:,7]
log_RHK_err = data[:,8]
RHK_l   = data[:,9]
iddx1   = (BJD < 54962) & (BJD > 54872)
iddx2   = (BJD < 55363) & (BJD > 55273)
iddx3   = (BJD < 55698) & (BJD > 55608)

#==============================================================================
# Align the data using a moving average
#==============================================================================

if YEAR == 2009:
    idx     = idx1
    iddx    = iddx1
    idx_HARPS 	= idx1_HARPS
if YEAR == 2010:
    idx 	= idx2 
    iddx 	= iddx2
    idx_HARPS 	= idx2_HARPS
if YEAR == 2011:
    idx 	= idx3
    iddx 	= iddx3
    idx_HARPS 	= idx3_HARPS

sl      = 0.05
fe4376_s    = gaussian_smoothing(JD_wise[idx], fe4376[idx], t, sl, np.ones(sum(idx)))
fe5250_s    = gaussian_smoothing(JD_wise[idx], fe5250[idx], t, sl, np.ones(sum(idx)))
log_RHK_s   = gaussian_smoothing(BJD[iddx], log_RHK[iddx], t, sl, np.ones(sum(iddx)))
BIS_s       = gaussian_smoothing(BJD[iddx], BIS[iddx], t, sl, np.ones(sum(iddx)))
FWHM_s      = gaussian_smoothing(BJD[iddx], FWHM[iddx], t, sl, np.ones(sum(iddx)))
FTL_s       = gaussian_smoothing(t, y, t, sl, yerr)
# RV_no_binary= RV_HARPS[idx_HARPS] - P.polyval(t, c)

# remove binary 
# RV_no_binary= RV_HARPS[idx_HARPS] - trend(t)
# RV_s        = gaussian_smoothing(t, RV_no_binary, t, sl, yerr)

# remove a linear trend
fit, V  = np.polyfit(t, RV_HARPS[idx_HARPS], 1, w=1/yerr**2, cov=True)
RV_s    = RV_HARPS[idx_HARPS] - fit[0]*t - fit[1]

if 0: # visualize the binary removal
    # plt.plot(t, RV_HARPS[idx_HARPS] - np.mean(RV_HARPS[idx_HARPS]), 'k.', alpha = 0.1)
    plt.plot(t, RV_s - np.mean(RV_s), 'b.')

    idxx1 = t<55310
    idxx2 = (t>55310) & (t<55343)
    idxx3 = t>55343    

    RV_nbm = RV_no_binary - np.mean(RV_no_binary)
    plt.plot(t[idxx1], RV_nbm[idxx1], 'k.', alpha=0.1)
    plt.plot(t[idxx2], RV_nbm[idxx2], 'b.', alpha=0.1)
    plt.plot(t[idxx3], RV_nbm[idxx3], 'r.', alpha=0.1)
    plt.plot(t, y, 'g.', alpha=0.05)
    plt.show()

# plt.plot(t, RV_s, '.')
# plt.show()

#==============================================================================
# Correlogram
#==============================================================================

left  = 0.1  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.07   # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.1   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots

alpha   = 0.02
markersize = 5

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(figsize=(14, 14))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(661)
plt.plot(RV_s, FTL_s, '.k', markersize=markersize, alpha=alpha)
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.xticks([])
#
plt.subplot(667)
plt.plot(RV_s, log_RHK_s, '.k', markersize=markersize, alpha=alpha)
plt.ylabel('log $(R^{\'}_{HK})$')
plt.xticks([])
#
fig.add_subplot(6,6,13)
plt.plot(RV_s, fe4376_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.ylabel('Line core flux') # Fe $4375.9 \AA$
#
fig.add_subplot(6,6,19)
plt.plot(RV_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.ylabel('Half-depth range') #  Fe $5250.2 \AA$
#
fig.add_subplot(6,6,25)
plt.plot(RV_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.ylabel('FWHM [km/s]')
#
fig.add_subplot(6,6,31)
plt.plot(RV_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel('Bisector [m/s]')

plt.subplot(668)
plt.plot(FTL_s, log_RHK_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,14)
plt.plot(FTL_s, fe4376_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,20)
plt.plot(FTL_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,26)
plt.plot(FTL_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,32)
plt.plot(FTL_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')
plt.yticks([])

fig.add_subplot(6,6,15)
plt.plot(log_RHK_s, fe4376_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,21)
plt.plot(log_RHK_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,27)
plt.plot(log_RHK_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#

axes_2 = fig.add_subplot(6,6,33)
plt.plot(log_RHK_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('log $(R^{\'}_{HK})$')
plt.yticks([])
axes_2.xaxis.set_major_locator(plt.MaxNLocator(2))


fig.add_subplot(6,6,22)
plt.plot(fe4376_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,28)
plt.plot(fe4376_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,34)
plt.plot(fe4376_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('Line core flux')
plt.yticks([])

fig.add_subplot(6,6,29)
plt.plot(fe5250_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(6,6,35)
plt.plot(fe5250_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('Half-depth range')
plt.yticks([])

fig.add_subplot(6,6,36)
plt.plot(FWHM_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('FWHM [km/s]')
plt.yticks([])

plt.savefig('../output/HD128621/Correlogram2_indicator_' + str(YEAR) + '.png')
plt.show()
