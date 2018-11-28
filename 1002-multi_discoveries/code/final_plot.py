import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing

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
log_RHK = data[:,7]
log_RHK_err = data[:,8]
RHK_l   = data[:,9]
BIS     = data[:,4] * 1000 # m/s
FWHM    = data[:,5] # km/s
iddx1   = (BJD < 54962) & (BJD > 54872)
iddx2   = (BJD < 55363) & (BJD > 55273)
iddx3   = (BJD < 55698) & (BJD > 55608)

if 0: 
    plt.errorbar(BJD, log_RHK, yerr=log_RHK_err, fmt=".k", capsize=0, alpha=0.2)
    plt.show()

    plt.plot(BJD, BIS, '.k', alpha=0.2)
    plt.show()

lin0        = -22700.1747
lin1        = -0.5307
lin2        = -1.83e-5
A_RV_Rhk    = 66.1781

####################
# Subset 2009 data # 
####################

P1          = 39.7572
A11s        = 0.5022
A11c        = 1.3878
A12s        = 0.7684
A12c        = 0.2643
RHK_l_2009  = RHK_l[iddx1]

def Activity2009(x):
    return A_RV_Rhk*RHK_l_2009 + \
    A11s*np.sin(2*np.pi/P1*x) + A11c*np.cos(2*np.pi/P1*x) + \
    A12s*np.sin(2*np.pi*2/P1*x) + A12c*np.cos(2*np.pi*2/P1*x) 

RV_activity2009 = Activity2009(BJD[iddx1])
RV_activity2009 = RV_activity2009 - np.mean(RV_activity2009)

####################
# Subset 2010 data # 
####################

P2          = 37.8394
A21s        = -1.0757
A21c        = 1.1328 
A23s        = -1.3124
A23c        = -1.0487
A24s        = -0.1096
A24c        = -1.3694
RHK_l_2010  = RHK_l[iddx2]

def Activity2010(x):
    return A_RV_Rhk*RHK_l_2010 + \
    A21s*np.sin(2*np.pi/P2*x) + A21c*np.cos(2*np.pi/P2*x) + \
    A23s*np.sin(2*np.pi*3/P2*x) + A23c*np.cos(2*np.pi*3/P2*x) + \
    A24s*np.sin(2*np.pi*4/P2*x) + A24c*np.cos(2*np.pi*4/P2*x)

RV_activity2010 = Activity2010(BJD[iddx2])
RV_activity2010 = RV_activity2010 - np.mean(RV_activity2010)

####################
# Subset 2011 data # 
####################

P3          = 36.7549
A31s        = 1.1029
A31c        = -0.9084
A32s        = -0.7422
A32c        = -0.3392
A33s        = -1.2984
A33c        = 0.707
RHK_l_2011  = RHK_l[iddx3]

def Activity2011(x):
    return A_RV_Rhk*RHK_l_2011 + \
    A31s*np.sin(2*np.pi/P3*x) + A31c*np.cos(2*np.pi/P3*x) + \
    A32s*np.sin(2*np.pi*2/P3*x) + A32c*np.cos(2*np.pi*2/P3*x) + \
    A33s*np.sin(2*np.pi*3/P3*x) + A33c*np.cos(2*np.pi*3/P3*x)

RV_activity2011 = Activity2011(BJD[iddx3])
RV_activity2011 = RV_activity2011 - np.mean(RV_activity2011)

# plt.plot(BJD[iddx3], RV_activity, '.'); plt.show()

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
x 	= np.loadtxt('../data/HD128621/plot/2009/plot_x.txt')
mu 	= np.loadtxt('../data/HD128621/plot/2009/plot_mu.txt')
std = np.loadtxt('../data/HD128621/plot/2009/plot_std.txt')

sl = 1
fe4376_s    = gaussian_smoothing(JD_wise[idx1], fe4376[idx1], t, sl, np.ones(sum(idx1)))
log_RHK_s   = gaussian_smoothing(BJD[iddx1], log_RHK[iddx1], t, sl, np.ones(sum(iddx1)))
BIS_s   = gaussian_smoothing(BJD[iddx1], BIS[iddx1], t, sl, np.ones(sum(iddx1)))


plt.subplot(311)
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
# plt.errorbar(BJD[iddx1], (log_RHK[iddx1] - np.mean(log_RHK[iddx1]))*50, yerr=log_RHK_err[iddx1], fmt=".b", capsize=0, alpha=0.2)
# plt.plot(BJD[iddx1], RV_activity2009 / 4, 'r.')
plt.plot(JD_wise[idx1], (fe4376[idx1]- np.mean(fe4376[idx1])) *300, 'b.', alpha=0.1)
plt.plot(t, (fe4376_s - np.mean(fe4376_s)) *300, 'r.')
# plt.plot(JD_wise[idx1], (fe5250[idx1]- np.mean(fe5250[idx1])) *800, 'r.', alpha=0.1)
plt.plot(x, mu, color=color)
plt.xlim(54872, 54962)
plt.ylim(-3.5, 3.1)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.title('Epoch 1: 2009-02-15..2009-05-06')
plt.ylabel(r"$\Delta RV_L$ [m/s]")

if 0:
    plt.errorbar(fe4376_s, y, yerr, fmt=".k", alpha=0.1)
    plt.show()


####################
# Subset 2010 data # 
####################

t 	= np.loadtxt('../data/HD128621/plot/2010/plot_t.txt') + 0.5
y 	= np.loadtxt('../data/HD128621/plot/2010/plot_y.txt')
yerr = np.loadtxt('../data/HD128621/plot/2010/plot_yerr.txt')
x 	= np.loadtxt('../data/HD128621/plot/2010/plot_x.txt')
mu 	= np.loadtxt('../data/HD128621/plot/2010/plot_mu.txt')
std = np.loadtxt('../data/HD128621/plot/2010/plot_std.txt')

plt.subplot(312)
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
# plt.errorbar(BJD[iddx2], (log_RHK[iddx2] - np.mean(log_RHK[iddx2]))*50, yerr=log_RHK_err[iddx2], fmt=".b", capsize=0, alpha=0.2)
# plt.plot(BJD[iddx2], RV_activity2010 / 4, 'r.')
plt.plot(JD_wise[idx2], (fe4376[idx2]- np.mean(fe4376[idx2])) *300, 'b.', alpha=0.1)
plt.plot(t, (fe4376_s - np.mean(fe4376_s)) *300, 'r.')
# plt.plot(JD_wise[idx2], (fe5250[idx2]- np.mean(fe5250[idx2])) *800, 'r.', alpha=0.1)
plt.plot(x, mu, color=color)
plt.xlim(55273, 55363)
plt.ylim(-3.5, 3.1)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.title('Epoch 2: 2010-03-23..2010-06-12')
plt.ylabel(r"$\Delta RV_L$ [m/s]")

if 0:
    plt.errorbar(fe4376_s, y, yerr, fmt=".k", alpha=0.1)
    plt.show()

    plt.plot(fe4376_s, y, ".k", alpha=0.1)
    plt.show()    

    plt.plot(BIS_s, y, ".k", alpha=0.1)
    plt.show()    

    plt.errorbar(log_RHK_s, y, yerr, fmt=".k", alpha=0.1)
    plt.show()

    plt.plot(log_RHK_s, y, ".k", alpha=0.1)
    plt.show()

    plt.plot(log_RHK_s, fe4376_s, '.k', alpha=0.1)
    plt.xlabel('log_{RHK}_s')
    plt.ylabel('fe4376_s')
    # plt.savefig('Correlation-log_{RHK}_s-fe4376_s.png')
    plt.show()

fe4376_s    = gaussian_smoothing(JD_wise[idx2], fe4376[idx2], t, sl, np.ones(sum(idx2)))
fe5250_s    = gaussian_smoothing(JD_wise[idx2], fe5250[idx2], t, sl, np.ones(sum(idx2)))
log_RHK_s   = gaussian_smoothing(BJD[iddx2], log_RHK[iddx2], t, sl, np.ones(sum(iddx2)))
BIS_s       = gaussian_smoothing(BJD[iddx2], BIS[iddx2], t, sl, np.ones(sum(iddx2)))
FWHM_s      = gaussian_smoothing(BJD[iddx2], FWHM[iddx2], t, sl, np.ones(sum(iddx2)))
FTL_s       = gaussian_smoothing(t, y, t, sl, yerr)

#==============================================================================
# Correlogram
#==============================================================================

left  = 0.1  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.07   # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.05   # the amount of width reserved for blank space between subplots
hspace = 0.05   # the amount of height reserved for white space between subplots

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

plt.savefig('Correlogram_indicator.png')
plt.show()

# plt.plot(fe4376_s, log_RHK_s, '.k', markersize=markersize, alpha=alpha)
# plt.xlabel('Line core flux Fe $4375.9 \AA$')
# plt.ylabel('log $R^{\'}_{HK}$')    

plt.plot(fe4376_s, FWHM_s, '.k', markersize=markersize, alpha=alpha)
    

plt.subplot(557)




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
x 	= np.loadtxt('../data/HD128621/plot/2011/plot_x.txt')
mu 	= np.loadtxt('../data/HD128621/plot/2011/plot_mu.txt')
std = np.loadtxt('../data/HD128621/plot/2011/plot_std.txt')

plt.subplot(313)
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
plt.errorbar(BJD[iddx3], (log_RHK[iddx3] - np.mean(log_RHK[iddx3]))*50, yerr=log_RHK_err[iddx3], fmt=".b", capsize=0, alpha=0.2)
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

# plt.savefig('final_plot_FE4376.png')
# plt.savefig('final_plot_FEfe5250.png')
# plt.savefig('final_plot_RHK.png')
plt.show()
