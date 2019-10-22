'''
Jitter metrics of 2009 - 2011
'''

import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing
from GlobalFit import GlobalFit

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
#
#
#
star    = 'HD128621'
DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
t       = np.loadtxt(DIR + '/MJD.dat') + 0.5
XX      = np.loadtxt(DIR + '/RV_HARPS.dat')
XX      = XX * 1000
#
#
#
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


if 0: 
    plt.errorbar(BJD, log_RHK, yerr=log_RHK_err, fmt=".k", capsize=0, alpha=0.2)
    plt.show()

    plt.plot(BJD, BIS, '.k', alpha=0.2)
    plt.show()


globalfit = GlobalFit()
# globalfit.activity2009(BJD)
# globalfit.fit(BJD)

iddx1   = (BJD < 54962) & (BJD > 54872)
iddx2   = (BJD < 55363) & (BJD > 55273)
iddx3   = (BJD < 55698) & (BJD > 55608)
BJD_plot = np.hstack((BJD[iddx1], BJD[iddx2], BJD[iddx3]))
plt.plot(BJD_plot, globalfit.fit(t=BJD, RHK_l=RHK_l) - np.mean(globalfit.activity2010(t=BJD,RHK_l=RHK_l)), '.', alpha=0.5)
plt.plot(t, XX, 'r.', alpha=0.1)
plt.show()


if 0:
    RV_activity2009 = globalfit.activity2009(BJD[iddx1])
    RV_activity2009 = RV_activity2009 - np.mean(RV_activity2009)

    RV_activity2010 = activity2010(BJD[iddx2])
    RV_activity2010 = RV_activity2010 - np.mean(RV_activity2010)

    RV_activity2011 = globalfit.activity2011(BJD[iddx3])
    RV_activity2011 = RV_activity2011 - np.mean(RV_activity2011)




    #==============================================================================
    # Plot #
    #==============================================================================

    left  = 0.1  # the left side of the subplots of the figure
    right = 0.95    # the right side of the subplots of the figure
    bottom = 0.05   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.4   # the amount of width reserved for blank space between subplots
    hspace = 0.3   # the amount of height reserved for white space between subplots

    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(figsize=(14, 14))
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # fig.add_subplot(111)    # The big subplot
    # plt.axis('off')     # hide frame
    # plt.xticks([])                        # don't want to see any ticks on this axis
    plt.yticks([])
    # plt.xlabel("JD - 2,400,000")
    # plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")

    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    color = "#ff7f0e"

    ####################
    # Subset 2009 data # 
    ####################

    t 	= np.loadtxt('../data/HD128621/plot/2009/plot_t.txt') + 0.5
    y 	= np.loadtxt('../data/HD128621/plot/2009/plot_y.txt')
    yerr = np.loadtxt('../data/HD128621/plot/2009/plot_yerr.txt')
    x 	= np.loadtxt('../data/HD128621/plot/2009/plot_x.txt') + 0.5
    mu 	= np.loadtxt('../data/HD128621/plot/2009/plot_mu.txt')
    std = np.loadtxt('../data/HD128621/plot/2009/plot_std.txt')

    plt.subplot(311)
    plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
    # plt.errorbar(BJD[iddx1], (log_RHK[iddx1] - np.mean(log_RHK[iddx1]))*50, yerr=log_RHK_err[iddx1], fmt=".b", capsize=0, alpha=0.2)
    # plt.plot(BJD[iddx1], RV_activity2009 / 4, 'r.')
    # plt.plot(JD_wise[idx1], (fe4376[idx1]- np.mean(fe4376[idx1])) *300, 'b.', alpha=0.1)
    # plt.plot(JD_wise[idx1], (fe5250[idx1]- np.mean(fe5250[idx1])) *800, 'r.', alpha=0.1)
    plt.plot(x, mu, color=color)
    plt.xlim(54872, 54962)
    plt.ylim(-3.5, 3.1)
    plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
    plt.title('Epoch 1: 2009-02-15..2009-05-06')
    plt.ylabel(r"$\Delta RV_L$ [m/s]")

    ####################
    # Subset 2010 data # 
    ####################

    t 	= np.loadtxt('../data/HD128621/plot/2010/plot_t.txt') + 0.5
    y 	= np.loadtxt('../data/HD128621/plot/2010/plot_y.txt')
    yerr = np.loadtxt('../data/HD128621/plot/2010/plot_yerr.txt')
    x 	= np.loadtxt('../data/HD128621/plot/2010/plot_x.txt') + 0.5
    mu 	= np.loadtxt('../data/HD128621/plot/2010/plot_mu.txt')
    std = np.loadtxt('../data/HD128621/plot/2010/plot_std.txt')

    plt.subplot(312)
    plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
    # plt.errorbar(BJD[iddx2], (log_RHK[iddx2] - np.mean(log_RHK[iddx2]))*50, yerr=log_RHK_err[iddx2], fmt=".b", capsize=0, alpha=0.2)
    # plt.plot(BJD[iddx2], RV_activity2010 / 4, 'r.')
    # plt.plot(JD_wise[idx2], (fe4376[idx2]- np.mean(fe4376[idx2])) *300, 'b.', alpha=0.1)
    # plt.plot(JD_wise[idx2], (fe5250[idx2]- np.mean(fe5250[idx2])) *800, 'r.', alpha=0.1)
    plt.plot(x, mu, color=color)
    plt.xlim(55273, 55363)
    plt.ylim(-3.5, 3.1)
    plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
    plt.title('Epoch 2: 2010-03-23..2010-06-12')
    plt.ylabel(r"$\Delta RV_L$ [m/s]")


    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # ###########################
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(x, y1)
    # ax1.set_ylabel('y1')

    # ax2 = ax1.twinx()
    # ax2.plot(x, y2, 'r-')
    # ax2.set_ylabel('y2', color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')
    # ###########################

    ####################
    # Subset 2011 data # 
    ####################

    t 	= np.loadtxt('../data/HD128621/plot/2011/plot_t.txt') + 0.5
    y 	= np.loadtxt('../data/HD128621/plot/2011/plot_y.txt')
    yerr = np.loadtxt('../data/HD128621/plot/2011/plot_yerr.txt')
    x 	= np.loadtxt('../data/HD128621/plot/2011/plot_x.txt') + 0.5
    mu 	= np.loadtxt('../data/HD128621/plot/2011/plot_mu.txt')
    std = np.loadtxt('../data/HD128621/plot/2011/plot_std.txt')

    plt.subplot(313)
    plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
    # plt.errorbar(BJD[iddx3], (log_RHK[iddx3] - np.mean(log_RHK[iddx3]))*50, yerr=log_RHK_err[iddx3], fmt=".b", capsize=0, alpha=0.2)
    # plt.plot(BJD[iddx3], RV_activity2011 / 4, 'r.')
    # plt.plot(JD_wise[idx3], (fe4376[idx3]- np.mean(fe4376[idx3])) *300, 'b.', alpha=0.1)
    # plt.plot(JD_wise[idx3], (fe5250[idx3]- np.mean(fe5250[idx3])) *800, 'r.', alpha=0.1)
    plt.plot(x, mu, color=color)
    plt.xlim(55608, 55698)
    plt.ylim(-3.5, 3.1)
    plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
    plt.title('Epoch 3: 2011-02-18..2011-05-15')
    plt.xlabel("JD - 2,400,000")
    plt.ylabel(r"$\Delta RV_L$ [m/s]")

    # plt.savefig('../output/HD128621/final_plot_FE4376.png')
    # plt.savefig('../output/HD128621/final_plot_FEfe5250.png')
    # plt.savefig('../output/HD128621/final_plot_RHK.png')
    # plt.savefig('../output/HD128621/final_plot.png')
    plt.show()



# Periodogram # 
from astropy.stats import LombScargle
min_f   = 1/90
max_f   = 10
spp     = 50

# 2009 #
t       = np.loadtxt('../data/HD128621/plot/2009/plot_t.txt') + 0.5
y       = np.loadtxt('../data/HD128621/plot/2009/plot_y.txt')
yerr    = np.loadtxt('../data/HD128621/plot/2009/plot_yerr.txt')

frequency2009, power2009 = LombScargle(t, y, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

# 2010 #
t       = np.loadtxt('../data/HD128621/plot/2010/plot_t.txt') + 0.5
y       = np.loadtxt('../data/HD128621/plot/2010/plot_y.txt')
yerr    = np.loadtxt('../data/HD128621/plot/2010/plot_yerr.txt')

frequency2010, power2010 = LombScargle(t, y, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

# 2011 #
t       = np.loadtxt('../data/HD128621/plot/2011/plot_t.txt') + 0.5
y       = np.loadtxt('../data/HD128621/plot/2011/plot_y.txt')
yerr    = np.loadtxt('../data/HD128621/plot/2011/plot_yerr.txt')

frequency2011, power2011 = LombScargle(t, y, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

plt.rcParams.update({'font.size': 14})
ax = plt.subplot(111)
# ax.set_xscale('log')
plt.plot(1/frequency2009, power2009, 'b--', label='2009')
period2009 = 1/frequency2009[power2009 == max(power2009)]
ax.axvline(x=period2009, color='b', linestyle='--', linewidth=2, alpha = 0.5)

plt.plot(1/frequency2010, power2010, 'k-', label='2010')
period2010 = 1/frequency2010[power2010 == max(power2010)]
ax.axvline(x=period2010, color='k', linestyle='-', linewidth=2, alpha = 0.5)

plt.plot(1/frequency2011, power2011, 'r-.', label='2011')
period2011 = 1/frequency2011[power2011 == max(power2011)]
ax.axvline(x=period2011, color='r', linestyle='-.', linewidth=2, alpha = 0.5)

plt.title('Periodogram')
plt.xlabel('day')
plt.ylabel("Power")
plt.ylim(0, 0.9)   
plt.xlim(0, 90)   
plt.legend()
plt.savefig('../output/Periodogram.png')
plt.show()

print(period2009, period2010, period2011)

#
#
#
#
#
#
#
#
#==============================================================================
# test #
#==============================================================================
if 0:
    # Create some mock data
    t = np.arange(0.01, 10.0, 0.01)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
