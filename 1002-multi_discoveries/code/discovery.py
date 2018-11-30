import numpy as np
import matplotlib.pyplot as plt
from functions import gaussian_smoothing
from rv import solve_kep_eqn
import os

#==============================================================================
# Import data 
#==============================================================================
star    = 'HD128621'
print('*'*len(star))
print(star)
print('*'*len(star))

plt.rcParams.update({'font.size': 14})

DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
# DIR     = '/run/user/1000/gvfs/sftp:host=durufle.phys.unsw.edu.au,user=jzhao/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
t 		= np.loadtxt(DIR + '/MJD.dat')
XX      = np.loadtxt(DIR + '/RV_HARPS.dat')
XX 		= (XX - np.mean(XX)) * 1000
yerr 	= np.loadtxt(DIR + '/RV_noise.dat') #m/s
FWHM    = np.loadtxt(DIR + '/FWHM.dat')

# t  	= np.loadtxt('../data/'+star+'/MJD.dat')
# XX  = np.loadtxt('../data/'+star+'/RV_HARPS.dat')
# XX  = (XX - np.mean(XX)) * 1000
# yerr= np.loadtxt('../data/'+star+'/RV_noise.dat')
# FWHM  = np.loadtxt('../data/'+star+'/FWHM.dat')

YY  = np.loadtxt('../data/'+star+'/YY.txt')
ZZ  = np.loadtxt('../data/'+star+'/ZZ.txt')

XY 	= XX - YY
ZX 	= ZZ - XX

os.chdir('../output/'+star)

#==============================================================================
# Visualizing
#==============================================================================

# present the pre-filetered data

plt.figure()
# plt.errorbar(t, XX, yerr=yerr, fmt=".k", capsize=0, alpha=0.2, label='$RV_{HARPS}$')
plt.errorbar(t, XY, yerr=yerr, fmt=".r", capsize=0, alpha=0.2, label=r'$\Delta RV_L$')
# plt.errorbar(t, ZX, yerr=yerr, fmt=".b", capsize=0, alpha=0.2, label=r'$\Delta RV_H$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.show()

# Filter the unwanted data #
if 1: # Valid for data HD128621_2_2010-03-22..2010-06-12[PART]
    # idx  = ~((XY>3) | ((t>55340) & (XX>0)))
    # the following does the job equally good 
    idx = (FWHM<6.3) &  (FWHM>6.24)
if 0: # valid for part 3 [part] (i.e. 2011-02-18..2011-05-15)
    idx = ~ ((XY>1.5) | (XX>30) | ((t>55658) & (t<55659) & (XX>5)) | ((t>55679) & (t<55680) & (XX>-10)) | ((t>55692) & (t<55693) & (XY<0)) | (FWHM<6.22))
if 0: # valid for part 1 [part]
    idx = (FWHM>6.23) & (t<54975)

# present the filetered data

plt.figure()
plt.errorbar(FWHM[idx], XY[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
plt.errorbar(FWHM[~idx], XY[~idx], yerr=yerr[~idx], fmt=".b", alpha=0.2)
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.xlabel("FWHM")
plt.savefig('0-correlation_XY.png')
plt.show()

if 0:
    plt.figure()
    plt.plot(FWHM[idx], XY[idx], ".k", alpha=0.2)
    plt.plot(FWHM[~idx], XY[~idx], ".b", alpha=0.2)
    plt.ylabel(r'$\Delta RV_L$ [m/s]')
    plt.xlabel("FWHM")
    plt.show()

plt.figure()
plt.errorbar(FWHM[idx], ZX[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
plt.errorbar(FWHM[~idx], ZX[~idx], yerr=yerr[~idx], fmt=".b", alpha=0.2)
plt.ylabel(r'$\Delta RV_H$ [m/s]')
plt.xlabel("FWHM")
plt.savefig('0-correlation_ZX.png')
plt.show()


# Binary orbit (without fitting planet) # 
lin0 = -22700.1747+22720.110791117946
lin1 = -0.5307
lin2 = -1.83e-5
BJD0 = 55278.739366
def trend(x):
    return lin0 + lin1 * (x-BJD0) + lin2 * (x-BJD0)**2

# Binary orbit (with fitting planet) # 
lin0 = -22700.1678+22720.110791117946
lin1 = -0.5305
def trend(x):
    return lin0 + lin1 * (x-BJD0)


plt.figure()
# plt.errorbar(t[idx], XX[idx], yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label='$RV_{HARPS}$')
# plt.errorbar(t[idx], trend(t[idx]), yerr=yerr[idx], fmt=".b", capsize=0, alpha=0.2, label='model')
# plt.errorbar(t[~idx], XX[~idx], yerr=yerr[~idx], fmt="*r", capsize=0, alpha=0.2, label='$RV_{HARPS}$ outlier')
plt.errorbar(t[idx], XX[idx]-trend(t[idx]), yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label=r'$RV_{HARPS}$')
plt.errorbar(t[idx], YY[idx]-trend(t[idx]),  yerr=yerr[idx], fmt=".r", capsize=0, alpha=0.1)
plt.errorbar(t[idx], ZZ[idx]-trend(t[idx]),  yerr=yerr[idx], fmt=".b", capsize=0, alpha=0.1)
# plt.errorbar(t[~idx], XX[~idx]-trend(t[~idx]), yerr=yerr[~idx], fmt=".r", capsize=0, alpha=0.2, label=r'$RV_{HARPS}$ outlier')
plt.legend()
plt.savefig('1-RV0.png')
plt.show()


# correlation # 
plt.figure()
plt.errorbar(FWHM[idx], XX[idx]-trend(t[idx]),  yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2)
# plt.errorbar(FWHM[~idx], XX[~idx]-trend(t[~idx]),  yerr=yerr[~idx], fmt=".r", capsize=0, alpha=0.2)
plt.ylabel("RV [m/s]")
plt.xlabel("FWHM")
plt.savefig('Corr_FHHM.png')
plt.show()


plt.figure()
plt.errorbar(t[idx], XY[idx], yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label=r'$\Delta RV_L$')
plt.errorbar(t[~idx], XY[~idx], yerr=yerr[~idx], fmt=".r", capsize=0, alpha=0.2, label='outlier')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV1.png')
plt.show()

plt.figure()
plt.errorbar(t[idx], ZX[idx], yerr=yerr[idx], fmt=".k", capsize=0, alpha=0.2, label=r'$\Delta RV_H$')
plt.errorbar(t[~idx], ZX[~idx], yerr=yerr[~idx], fmt=".r", capsize=0, alpha=0.2, label='outlier')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('1-RV2.png')
plt.show()

if 0:
    plt.figure()
    plt.errorbar(XX[idx], XY[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
    plt.errorbar(XX[~idx], XY[~idx], yerr=yerr[~idx], fmt="*r", alpha=0.2)
    plt.xlabel("$RV_{HARPS}$ [m/s]")
    plt.ylabel(r"$\Delta RV_L$ [m/s]")
    plt.savefig('2-correlation_XY.png')
    plt.show()

    plt.figure()
    plt.errorbar(XX[idx], ZX[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
    plt.errorbar(XX[~idx], ZX[~idx], yerr=yerr[~idx], fmt="*r", alpha=0.2)
    plt.xlabel("$RV_{HARPS}$ [m/s]")
    plt.ylabel(r"$\Delta RV_H$ [m/s]")
    plt.savefig('2-correlation_ZX.png')
    plt.show()

    plt.figure()
    plt.errorbar(XY[idx], ZX[idx], yerr=yerr[idx], fmt=".k", alpha=0.2)
    plt.errorbar(XY[~idx], ZX[~idx], yerr=yerr[~idx], fmt="*r", alpha=0.2)
    plt.xlabel(r"$\Delta RV_L$ [m/s]")
    plt.ylabel(r"$\Delta RV_H$ [m/s]")
    # plt.savefig('2-correlation_XYZ.png')
    plt.show()


#==============================================================================
# Correlation 1
#==============================================================================
left  = 0.05  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.plot(XX[idx]-P.polyval(t[idx], c)-np.mean(XX[idx]-P.polyval(t[idx], c)), YY[idx]-P.polyval(t[idx], c)-np.mean(YY[idx]-P.polyval(t[idx], c)), '.k', markersize=3, alpha=0.3)
# plt.plot(XX[~idx]-trend(t[~idx]), YY[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.plot(XX[idx]-P.polyval(t[idx], c)-np.mean(XX[idx]-P.polyval(t[idx], c)), ZZ[idx]-P.polyval(t[idx], c)-np.mean(ZZ[idx]-P.polyval(t[idx], c)), '.k', markersize=3, alpha=0.3)
# plt.plot(XX[~idx]-trend(t[~idx]), ZZ[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.plot(XX[idx]-P.polyval(t[idx], c)-np.mean(XX[idx]-P.polyval(t[idx], c)), XY[idx], '.k', markersize=3, alpha=0.3)
# plt.plot(XX[~idx]-trend(t[~idx]), XY[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.title(r'$\alpha$' + ' Centauri B 2010-03-23..2010-06-12')
# plt.title(r'$\alpha$' + ' Centauri B 2011-02-18..2011-05-15')
# plt.title(r'$\alpha$' + ' Centauri B 2009-02-15..2009-05-06')

plt.subplot(154)
plt.plot(XX[idx]-P.polyval(t[idx], c)-np.mean(XX[idx]-P.polyval(t[idx], c)), ZX[idx], '.k', markersize=3, alpha=0.3)
# plt.plot(XX[~idx]-trend(t[~idx]), ZX[~idx], '*r', markersize=3, alpha=0.3)   
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')

plt.subplot(155)
fit = np.polyfit(XY, ZX, 1)
x_sample = np.linspace(min(XY)*1.2, max(XY)*1.2, num=100, endpoint=True)
plt.plot(XY[idx], ZX[idx], '.k', markersize=3, alpha=0.3)
# plt.plot(XY[~idx], ZX[~idx], '*r', markersize=3, alpha=0.3)   
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
# plt.savefig('Correlation.png')   
plt.show()




left  = 0.05  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

idxx1 = t[idx] < 55310
idxx2 = (t[idx] > 55310) & (t[idx] < 55343)
idxx3 = t[idx] > 55343

xxx = XX[idx]-P.polyval(t[idx], c) - np.mean(XX[idx]-P.polyval(t[idx], c))
yyy = YY[idx]-P.polyval(t[idx], c) - np.mean(YY[idx]-P.polyval(t[idx], c))
zzz = ZZ[idx]-P.polyval(t[idx], c) - np.mean(ZZ[idx]-P.polyval(t[idx], c))

plt.subplot(151)
plt.plot(xxx[idxx1], yyy[idxx1], '.k', markersize=3, alpha=0.1)
plt.plot(xxx[idxx2], yyy[idxx2], '.b', markersize=3, alpha=0.1)
plt.plot(xxx[idxx3], yyy[idxx3], '.r', markersize=3, alpha=0.1)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.plot(xxx[idxx1], zzz[idxx1], '.k', markersize=3, alpha=0.1)
plt.plot(xxx[idxx2], zzz[idxx2], '.b', markersize=3, alpha=0.1)
plt.plot(xxx[idxx3], zzz[idxx3], '.r', markersize=3, alpha=0.1)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.plot(xxx[idxx1], XY[idx][idxx1], '.k', markersize=3, alpha=0.1)
plt.plot(xxx[idxx2], XY[idx][idxx2], '.b', markersize=3, alpha=0.1)
plt.plot(xxx[idxx3], XY[idx][idxx3], '.r', markersize=3, alpha=0.1)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.title(r'$\alpha$' + ' Centauri B 2010-03-23..2010-06-12')
# plt.title(r'$\alpha$' + ' Centauri B 2011-02-18..2011-05-15')
# plt.title(r'$\alpha$' + ' Centauri B 2009-02-15..2009-05-06')

plt.subplot(154)
plt.plot(xxx[idxx1], ZX[idx][idxx1], '.k', markersize=3, alpha=0.1)
plt.plot(xxx[idxx2], ZX[idx][idxx2], '.b', markersize=3, alpha=0.1)
plt.plot(xxx[idxx3], ZX[idx][idxx3], '.r', markersize=3, alpha=0.1)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')

plt.subplot(155)
# fit = np.polyfit(XY, ZX, 1)
# x_sample = np.linspace(min(XY)*1.2, max(XY)*1.2, num=100, endpoint=True)
plt.plot(XY[idx][idxx1], ZX[idx][idxx1], '.k', markersize=3, alpha=0.1)
plt.plot(XY[idx][idxx2], ZX[idx][idxx2], '.b', markersize=3, alpha=0.1)
plt.plot(XY[idx][idxx3], ZX[idx][idxx3], '.r', markersize=3, alpha=0.1)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
# plt.savefig('Correlation.png')   
plt.show()




#==============================================================================
# Correlation 2
#==============================================================================
left  = 0.05  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.plot(XX[idx], YY[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], YY[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.plot(XX[idx], ZZ[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], ZZ[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.plot(XX[idx], XY[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], XY[~idx], '*r', markersize=3, alpha=0.3)
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{HARPS} - RV_{FT,L}$ [m/s]')
plt.title(r'$\alpha$' + ' Centauri B 2010-03-23..2010-06-12')
# plt.title(r'$\alpha$' + ' Centauri B 2011-02-18..2011-05-15')
# plt.title(r'$\alpha$' + ' Centauri B 2009-02-15..2009-05-06')

plt.subplot(154)
plt.plot(XX[idx], ZX[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XX[~idx], ZX[~idx], '*r', markersize=3, alpha=0.3)   
plt.xlabel('$RV_{HARPS}$ [m/s]')
plt.ylabel('$RV_{FT,H} - RV_{HARPS}$ [m/s]')

plt.subplot(155)
fit = np.polyfit(XY, ZX, 1)
x_sample = np.linspace(min(XY)*1.2, max(XY)*1.2, num=100, endpoint=True)
plt.plot(XY[idx], ZX[idx], '.k', markersize=3, alpha=0.3)
plt.plot(XY[~idx], ZX[~idx], '*r', markersize=3, alpha=0.3)   
plt.xlabel('$RV_{HARPS} - RV_{FT,L}$ [m/s]')    
plt.ylabel('$RV_{FT,H} - RV_{HARPS}$ [m/s]')     
# plt.savefig('Correlation.png')   
plt.show()


#==============================================================================
# GP 
#==============================================================================

t   = t[idx]
y   = XY[idx]
yerr = yerr[idx]

from george import kernels

# k1 = 1**2 * kernels.ExpSquaredKernel(metric=10**2)
# k2 = 1**2 * kernels.ExpSquaredKernel(80**2) * kernels.ExpSine2Kernel(gamma=8, log_period=np.log(36.2))
# boundary doesn't seem to take effect
k2 = 1**2 * kernels.ExpSquaredKernel(80**2) * kernels.ExpSine2Kernel(gamma=11, log_period=np.log(36.2),
                            bounds=dict(gamma=(-3,30), log_period=(np.log(36.2-5),np.log(36.2+6))))
# k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
# k4 = 1**2 * kernels.ExpSquaredKernel(40**2)
# kernel = k1 + k2 + k3 + k4
kernel = k2

import george
# gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
# gp.freeze_parameter('kernel:k2:log_period')
# gp.freeze_parameter('kernel:k2:gamma')

#==============================================================================
# Optimization 
#==============================================================================
import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t, yerr)

# Print the initial ln-likelihood.
print(gp.log_likelihood(y))

# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="Newton-CG")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(y))

# print the rotation period 
print(np.exp(results.x[4]))



#==============================================================================
# MCMC
#==============================================================================

# Define the objective function (negative log-likelihood in this case).
def lnprob(p):
    # if np.any((results.x-0.5*abs(results.x) > p) + (p > results.x+0.5*abs(results.x))):
    #     return -np.inf
    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)    


import emcee

initial = results.x
# initial = gp.get_parameter_vector()
names = gp.get_parameter_names()
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

import time
time_start  = time.time()

print("Running first burn-in...")
p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running third burn-in...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 3000)

print("Running production...")
p0 = p0[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 3000);    

time_end    = time.time()
print('\nRuntime = %.2f seconds' %(time_end - time_start))


#==============================================================================
# Trace and corner plots 
#==============================================================================

import copy
raw_samples         = sampler.chain[:, 5000:, :].reshape((-1, ndim))
real_samples        = copy.copy(raw_samples)

fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
labels_log=names
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('ksiboo-2-Trace.png')

import corner
fig = corner.corner(real_samples, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('ksiboo-3-Corner.png')

#==============================================================================
# Output
#==============================================================================
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = map(lambda v: 
    (v[1], v[0], v[2], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(raw_samples, [16, 50, 84], axis=0)))
aa = np.zeros((12,5))
aa[0,:] = [a0[i] for i in range(5)]
aa[1,:] = [a1[i] for i in range(5)]
aa[2,:] = [a2[i] for i in range(5)]
aa[3,:] = [a3[i] for i in range(5)]
aa[4,:] = [a4[i] for i in range(5)]
aa[5,:] = [a5[i] for i in range(5)]
aa[6,:] = [a6[i] for i in range(5)]
aa[7,:] = [a7[i] for i in range(5)]
aa[8,:] = [a8[i] for i in range(5)]
aa[9,:] = [a9[i] for i in range(5)]
aa[10,:] = [a10[i] for i in range(5)]
aa[11,:] = [a11[i] for i in range(5)]
np.savetxt('ksiboo_hat_p.txt', aa, fmt='%.6f')

#==============================================================================
# Plots
#==============================================================================

gp.set_parameter_vector(aa[:,0])
gp.set_parameter_vector(results.x)
# Make the maximum likelihood prediction
x = np.linspace(min(t-1), max(t+1), 1000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)
gp_predict = np.transpose(np.vstack((x,mu,std)))
# np.savetxt('gp_predict_hat(p).txt', gp_predict, fmt='%.8f')
# np.savetxt('gp_predict.txt', gp_predict, fmt='%.8f')


# x: oversampled time (column 1)
# mu: Gaussian processes prediction of the most likely value (column 2)
# std: standard deivation of walkers in all runs in MCMC (column 3)
color = "#ff7f0e"
fig = plt.figure(figsize=(18,6))
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
# plt.errorbar(MJD[~idx], XY[~idx], yerr=YERR[~idx], fmt=".r", alpha=0.1, capsize=0)
plt.plot(x, mu, color=color)
# plt.xlim(55275, 55365)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.xlabel("JD - 2,400,000")
# plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
# plt.title("hat(p) - maximum likelihood prediction (MCMC)");
# plt.savefig('../output/'+star+'.png') 
plt.title('2010-03-23..2010-06-12')
# plt.title('2011-02-18..2011-05-15')
# plt.title('2009-02-15..2009-05-06')
# plt.savefig(star+'.png')
plt.show()

np.savetxt('plot_t.txt', t)
np.savetxt('plot_RV_HARPS.txt', XX[idx])
np.savetxt('plot_y.txt', y)
np.savetxt('plot_yerr.txt', yerr)
np.savetxt('plot_x.txt', x)
np.savetxt('plot_mu.txt', mu)
np.savetxt('plot_std.txt', std)
























#==============================================================================
# Smoothing
#==============================================================================
sl      = 0.5         # smoothing length
# xx 	 	= gaussian_smoothing(t, XX, t, sl)
xy      = gaussian_smoothing(t, y, t, sl, 1/yerr**2)
# zx      = gaussian_smoothing(t, ZX, t, sl)
np.savetxt('../../data/'+star+'/xy.txt', xy)

plt.figure()
# plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, xy, yerr=yerr, fmt=".r", capsize=0, label='$RV_{HARPS} - RV_{FT,L}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('s1-RV1.png')
plt.show()

plt.figure()
plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='$RV_{HARPS}$')
plt.errorbar(t, zx, yerr=yerr, fmt=".r", capsize=0, label='$RV_{FT,H} - RV_{HARPS}$')
plt.ylabel("RV [m/s]")
plt.xlabel("JD - 2,400,000")
plt.legend()
plt.savefig('s1-RV2.png')
plt.show()

plt.figure()
plt.errorbar(xx, xy, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.savefig('s2-correlation_XY.png')
plt.show()

plt.figure()
plt.errorbar(xx, zx, yerr=yerr, fmt=".k")
plt.xlabel("$RV_{HARPS}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('2-correlation_ZX.png')
plt.show()

plt.figure()
plt.errorbar(xy, zx, yerr=yerr, fmt=".r")
plt.xlabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")
plt.ylabel("$RV_{FT,H} - RV_{HARPS}$ [m/s]")
plt.savefig('s2-correlation_XYZ.png')
plt.show()
plt.close('all')

#==============================================================================
# Lomb-Scargle periodogram 
#==============================================================================
from astropy.stats import LombScargle
min_f   = 1/100
max_f   = 1
spp     = 10

frequency0, power0 = LombScargle(t, XX, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(t, ZX, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

frequency2, power2 = LombScargle(t, XY, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)
plt.figure()
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axvline(x=36.2, color='k')
ax.axhline(y=0, color='k')
plt.plot(1/frequency0, power0, 'b-', label=r'$RV_{Gaussian}$', linewidth=1.0)
plt.plot(1/frequency1, power1, 'r-.', label=r'$RV_{FT,H} - RV_{Gaussian}$', alpha=0.7)
plt.plot(1/frequency2, power2, 'g--', label=r'$RV_{Gaussian} - RV_{FT,L}$', alpha=0.7)
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('3-Periodogram.png')
plt.show()


frequency0, power0 = LombScargle(t, xx, yerr).autopower(minimum_frequency=min_f,
                                                        maximum_frequency=max_f,
                                                        samples_per_peak=spp)

frequency1, power1 = LombScargle(t, zx, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)

frequency2, power2 = LombScargle(t, xy, yerr).autopower(minimum_frequency=min_f,
                                                            maximum_frequency=max_f,
                                                            samples_per_peak=spp)
plt.figure()
ax = plt.subplot(111)
# ax.set_xscale('log')
ax.axvline(x=36.2, color='k')
ax.axhline(y=0, color='k')
plt.plot(1/frequency0, power0, 'b-', label=r'$RV_{Gaussian}$', linewidth=1.0)
plt.plot(1/frequency1, power1, 'r-.', label=r'$RV_{FT,H} - RV_{Gaussian}$', alpha=0.7)
plt.plot(1/frequency2, power2, 'g--', label=r'$RV_{Gaussian} - RV_{FT,L}$', alpha=0.7)
plt.title('Lomb-Scargle Periodogram')
plt.xlabel("Period [d]")
plt.ylabel("Power")
plt.legend()
plt.savefig('s3-Periodogram.png')
plt.show()














#==============================================================================
#==============================================================================
# Discovery mode
#==============================================================================
#==============================================================================
from celerite.modeling import Model
import time
import shutil
time0   = time.time()
os.makedirs(str(time0))
shutil.copy('../../code/discovery.py', str(time0)+'/discovery.py')  

for kk in np.arange(50):

	os.chdir(str(time0))

	if 0:
	    idx = (t < 57161)

	    t = t[idx]
	    xx = xx[idx]
	    xy = xy[idx]
	    XX = XX[idx]
	    yerr = yerr[idx]

	#==============================================================================
	# Model
	#==============================================================================
	class Model(Model):
	    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'offset1', 'offset2', 'alpha')

	    def get_value(self, t):

	        # Planet 1
	        M_anom1 = 2*np.pi/np.exp(10*self.P1) * (t - 1000*self.tau1)
	        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
	        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
	        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))

	        # The last part is not "corrected" with jitter
	        jitter  = np.exp(self.alpha) * xy

	        offset      = np.zeros(len(t))
	        idx         = t < 57161
	        offset[idx] = self.offset1
	        offset[~idx]= self.offset2

	        return rv1 + offset + jitter

	class Model2(Model):
	    parameter_names = ('P1', 'tau1', 'k1', 'w1', 'e1', 'offset1', 'offset2')

	    def get_value(self, t):

	        # Planet 1
	        M_anom1 = 2*np.pi/np.exp(10*self.P1) * (t - 1000*self.tau1)
	        e_anom1 = solve_kep_eqn(M_anom1, self.e1)
	        f1      = 2*np.arctan( np.sqrt((1+self.e1)/(1-self.e1))*np.tan(e_anom1*.5) )
	        rv1     = 100*self.k1*(np.cos(f1 + self.w1) + self.e1*np.cos(self.w1))

	        offset      = np.zeros(len(t))
	        idx         = t < 57161
	        offset[idx] = self.offset1
	        offset[~idx]= self.offset2

	        return rv1 + offset

	#==============================================================================
	# MCMC
	#==============================================================================
	# Define the posterior PDF
	# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

	def lnprior(theta):
	    P1, tau1, k1, w1, e1, offset1, offset2, alpha = theta
	    if (-2*np.pi < w1 < 2*np.pi) and (0 < e1 < 0.9) and (alpha > 0):
	        return 0.0
	    return -np.inf

	# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
	def lnlike(theta, x, y, yerr):
	    P1, tau1, k1, w1, e1, offset1, offset2, alpha = theta
	    fit_curve   = Model(P1=P1, tau1=tau1, k1=k1, w1=w1, e1=e1, offset1=offset1, offset2=offset2, alpha=alpha)
	    y_fit       = fit_curve.get_value(x)
	    return -0.5*(np.sum( ((y-y_fit)/yerr)**2))

	def lnprob(theta, x, y, yerr):
	    lp = lnprior(theta)
	    if not np.isfinite(lp):
	        return -np.inf
	    return lp + lnlike(theta, x, y, yerr)



	import emcee
	ndim = 8
	nwalkers = 32
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, XX, yerr), threads=14)

	import time
	time_start  = time.time()

	INIT_P =  np.random.uniform(0,0.9)
	INIT_e =  np.random.uniform(0.01,0.89)
	print("Running first burn-in...")
	pos = [[INIT_P, 1., (max(XX)-min(XX))/100, 0, INIT_e, 0, 0, 1.5] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)] 
	pos, prob, state  = sampler.run_mcmc(pos, 2000)

	print("Running second burn-in...")
	pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
	pos, prob, state  = sampler.run_mcmc(pos, 1000)

	print("Running third burn-in...")
	pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
	pos, prob, state  = sampler.run_mcmc(pos, 3000)

	print("Running production...")
	sampler.run_mcmc(pos, 3000);

	time_end    = time.time()
	print('\nRuntime = %.2f seconds' %(time_end - time_start))


	#==============================================================================
	# Trace and corner plots 
	#==============================================================================

	import copy
	raw_samples         = sampler.chain[:, -3000, :].reshape((-1, ndim))
	real_samples        = copy.copy(raw_samples)
	real_samples[:,0] = np.exp(10*real_samples[:,0])
	real_samples[:,1] = 1000*real_samples[:,1]
	real_samples[:,2] = 100*real_samples[:,2]
	idx = real_samples[:,3] < 0
	real_samples[idx,3] = real_samples[idx, 3] + 2*np.pi
	real_samples[:,7] = np.exp(real_samples[:,7])


	fig, axes = plt.subplots(ndim, figsize=(20, 14), sharex=True)
	labels_log=[r"$\frac{\log P_{1}}{10}$", r"$\frac{T_{1}}{1000}$", r"$\frac{K_{1}}{100}$", r"$\omega1$", r"$e1$", "offset1", "offset2", r"$\log \alpha$"]
	for i in range(ndim):
	    ax = axes[i]
	    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
	    ax.set_xlim(0, sampler.chain.shape[1])
	    ax.set_ylabel(labels_log[i])
	    ax.yaxis.set_label_coords(-0.1, 0.5)

	axes[-1].set_xlabel("step number");
	plt.savefig('2-Trace.png')
	# plt.show()


	import corner
	labels=[r"$P1$", r"$T_{1}$", r"$K1$", r"$\omega1$", r"$e1$", "offset1", "offset2", r"$\alpha$"]
	fig = corner.corner(real_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
	plt.savefig('3-Corner.png')
	# plt.show()

	#==============================================================================
	# Output
	#==============================================================================

	a0, a1, a2, a3, a4, a5, a6, a7 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
	aa = np.zeros((8,3))
	aa[0,:] = [a0[i] for i in range(3)]
	aa[1,:] = [a1[i] for i in range(3)]
	aa[2,:] = [a2[i] for i in range(3)]
	aa[3,:] = [a3[i] for i in range(3)]
	aa[4,:] = [a4[i] for i in range(3)]
	aa[5,:] = [a5[i] for i in range(3)]
	aa[6,:] = [a6[i] for i in range(3)]
	aa[7,:] = [a7[i] for i in range(3)]
	np.savetxt('parameter_fit.txt', aa, fmt='%.6f')


	P1, tau1, k1, w1, e1, offset1, offset2, alpha = aa[:,0]
	fig         = plt.figure(figsize=(10, 7))
	frame1      = fig.add_axes((.15,.3,.8,.6))
	frame1.axhline(y=0, color='k', ls='--', alpha=.3)
	t_sample    = np.linspace(min(t), max(t), num=10001, endpoint=True)

	# Planet 1 #
	Planet1     = Model2(P1=np.log(P1)/10, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, offset1=offset1, offset2=offset2)
	y1          = Planet1.get_value(t_sample)
	plt.plot(t_sample, y1, 'b-.', alpha=.3, label='Planet1')
	plt.errorbar(t, xx, yerr=yerr, fmt=".k", capsize=0, label='HARPS RV')
	plt.legend()
	plt.ylabel("Radial velocity [m/s]")
	# Jitter#
	Jitter      = Model(P1=np.log(P1)/10, tau1=tau1/1000, k1=0, w1=w1, e1=e1, offset1=0, offset2=0, alpha=np.log(alpha))
	y_jitter    = Jitter.get_value(t)
	plt.plot(t, y_jitter, 'ro', alpha=.5, label='smoothed jitter')

	Fit         = Model(P1=np.log(P1)/10, tau1=tau1/1000, k1=k1/100, w1=w1, e1=e1, offset1=offset1, offset2=offset1, alpha=np.log(alpha))
	y_fit       = Fit.get_value(t)
	plt.plot(t, y_fit, 'bo', alpha=.5, label='Planet 1 + smoothed jitter')
	# plt.plot(x[x<57300], alpha*jitter_smooth, 'ro', alpha=.5, label='smoothed jitter')
	plt.legend()
	plt.ylabel("Radial velocity [m/s]")

	residual    = y_fit - XX
	chi2        = sum(residual**2 / yerr**2)
	rms         = np.sqrt(np.mean(residual**2))
	wrms        = np.sqrt(sum((residual/yerr)**2) / sum(1/yerr**2))

	frame2  = fig.add_axes((.15,.1,.8,.2))   
	frame2.axhline(y=0, color='k', ls='--', alpha=.3)
	plt.errorbar(t, residual, yerr=yerr, fmt=".k", capsize=0)
	plt.xlabel("JD - 2,400,000")
	plt.ylabel('Residual [m/s]')
	plt.savefig('4-MCMC_fit.png')
	# plt.show()

	os.chdir(str('../'))


























