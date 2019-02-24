# Change the year accordingly 

import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing

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

sl 		= 0.1
if YEAR == 2009:
    idx     = idx1
    iddx    = iddx1
if YEAR == 2010:
    idx 	= idx2 
    iddx 	= iddx2
if YEAR == 2011:
    idx 	= idx3
    iddx 	= iddx3

fe4376_s    = gaussian_smoothing(JD_wise[idx], fe4376[idx], t, sl, np.ones(sum(idx)))
fe5250_s    = gaussian_smoothing(JD_wise[idx], fe5250[idx], t, sl, np.ones(sum(idx)))
log_RHK_s   = gaussian_smoothing(BJD[iddx], log_RHK[iddx], t, sl, np.ones(sum(iddx)))
BIS_s       = gaussian_smoothing(BJD[iddx], BIS[iddx], t, sl, np.ones(sum(iddx)))
FWHM_s      = gaussian_smoothing(BJD[iddx], FWHM[iddx], t, sl, np.ones(sum(iddx)))
FTL_s       = gaussian_smoothing(t, y, t, sl, yerr)

#==============================================================================
# Correlogram
#==============================================================================

left  = 0.1  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.07   # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.12   # the amount of width reserved for blank space between subplots
hspace = 0.12   # the amount of height reserved for white space between subplots

alpha   = 0.02
markersize = 15

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(figsize=(14, 14))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(551)
plt.plot(FTL_s, log_RHK_s, '.k', markersize=markersize, alpha=alpha)
plt.ylabel('log $(R^{\'}_{HK})$')
plt.xticks([])
#
fig.add_subplot(5,5,6)
plt.plot(FTL_s, fe4376_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.ylabel('Line core flux') # Fe $4375.9 \AA$
#
fig.add_subplot(5,5,11)
plt.plot(FTL_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.ylabel('Half-depth range') #  Fe $5250.2 \AA$
#
fig.add_subplot(5,5,16)
plt.plot(FTL_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.ylabel('FWHM [km/s]')
#
fig.add_subplot(5,5,21)
plt.plot(FTL_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')
plt.ylabel('Bisector [m/s]')

fig.add_subplot(5,5,7)
plt.plot(log_RHK_s, fe4376_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(5,5,12)
plt.plot(log_RHK_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(5,5,17)
plt.plot(log_RHK_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(5,5,22)
plt.plot(log_RHK_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('log $(R^{\'}_{HK})$')
plt.yticks([])

fig.add_subplot(5,5,13)
plt.plot(fe4376_s, fe5250_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(5,5,18)
plt.plot(fe4376_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(5,5,23)
plt.plot(fe4376_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('Line core flux')
plt.yticks([])

fig.add_subplot(5,5,19)
plt.plot(fe5250_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
plt.xticks([])
plt.yticks([])
#
fig.add_subplot(5,5,24)
plt.plot(fe5250_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('Half-depth range')
plt.yticks([])

fig.add_subplot(5,5,25)
plt.plot(FWHM_s, BIS_s, '.k', markersize=markersize, alpha=alpha)
plt.xlabel('FWHM [km/s]')
plt.yticks([])

plt.savefig('../output/HD128621/Correlogram_indicator_' + str(YEAR) + '.png')
plt.show()
