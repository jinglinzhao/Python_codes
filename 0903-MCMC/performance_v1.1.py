# This aims to test the performance of the FIESTA jitter metrics for different planetary orbital amplitudes # 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


RV_jitter   = np.loadtxt('/Volumes/DataSSD/SOAP_2/outputs/02.01/RV.dat')
RV_jitter   = RV_jitter - np.mean(RV_jitter)
RV_jitter   = np.hstack((RV_jitter,RV_jitter, RV_jitter, RV_jitter))


left  = 0.10  # the left side of the subplots of the figure
right = 0.96    # the right side of the subplots of the figure
bottom = 0.10   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
alpha = 0.5

# Plot 1: K = 0 # 
DIR = '/Volumes/DataSSD/MATLAB_codes/Project180316-shfit_in_FT/end-to-end/SN10000_p0+j/'
GG  = np.loadtxt(DIR+'GG.txt')
XX  = np.loadtxt(DIR+'XX.txt')
YY  = np.loadtxt(DIR+'YY.txt')
ZZ  = np.loadtxt(DIR+'ZZ.txt')

ax = plt.subplot(131)
fit1 = np.polyfit(RV_jitter, GG-YY, 1)
fit2 = np.polyfit(RV_jitter, ZZ-GG, 1)
r1, p = stats.pearsonr(RV_jitter, GG-YY)
r2, p= stats.pearsonr(RV_jitter, ZZ-GG)
plt.plot(RV_jitter, GG-YY, 'd', label=r'$\Delta RV_{L} (R=%.2f, k=%.2f)$' %(r1, fit1[0]), alpha=alpha)
plt.plot(RV_jitter, ZZ-GG, 'P', label=r'$\Delta RV_{H} (R=%.2f, k=%.2f)$' %(r2, fit2[0]), alpha=alpha)
plt.axis('scaled')
plt.ylim(-1.8, 2.8)
plt.title(r'$K = 0$ m/s')
plt.xlabel('Jitter ' + '($RV_{Gaussian})$')
plt.ylabel(r'$\Phi$ESTA jitter metrics')
plt.legend()

# Plot 2: K = 2 # 
DIR = '/Volumes/DataSSD/MATLAB_codes/Project180316-shfit_in_FT/end-to-end/SN10000_p2+j/'
GG  = np.loadtxt(DIR+'GG.txt')
XX  = np.loadtxt(DIR+'XX.txt')
YY  = np.loadtxt(DIR+'YY.txt')
ZZ  = np.loadtxt(DIR+'ZZ.txt')

ax = plt.subplot(132)
fit1 = np.polyfit(RV_jitter, GG-YY, 1)
fit2 = np.polyfit(RV_jitter, ZZ-GG, 1)
r1, p = stats.pearsonr(RV_jitter, GG-YY)
r2, p= stats.pearsonr(RV_jitter, ZZ-GG)
plt.plot(RV_jitter, GG-YY, 'd', label=r'$\Delta RV_{L} (R=%.2f, k=%.2f)$' %(r1, fit1[0]), alpha=alpha)
plt.plot(RV_jitter, ZZ-GG, 'P', label=r'$\Delta RV_{H} (R=%.2f, k=%.2f)$' %(r2, fit2[0]), alpha=alpha)
plt.axis('scaled')
plt.ylim(-1.8, 2.8)
plt.title(r'$K = 2$ m/s')
plt.xlabel('Jitter ' + '($RV_{Gaussian})$')
plt.ylabel(r'$\Phi$ESTA jitter metrics')
plt.legend()


# Plot 3: K = 10 # 
DIR = '/Volumes/DataSSD/MATLAB_codes/Project180316-shfit_in_FT/end-to-end/SN10000_p10+j/'
GG  = np.loadtxt(DIR+'GG.txt')
XX  = np.loadtxt(DIR+'XX.txt')
YY  = np.loadtxt(DIR+'YY.txt')
ZZ  = np.loadtxt(DIR+'ZZ.txt')

ax = plt.subplot(133)
fit1 = np.polyfit(RV_jitter, GG-YY, 1)
fit2 = np.polyfit(RV_jitter, ZZ-GG, 1)
r1, p = stats.pearsonr(RV_jitter, GG-YY)
r2, p= stats.pearsonr(RV_jitter, ZZ-GG)
plt.plot(RV_jitter, GG-YY, 'd', label=r'$\Delta RV_{L} (R=%.2f, k=%.2f)$' %(r1, fit1[0]), alpha=alpha)
plt.plot(RV_jitter, ZZ-GG, 'P', label=r'$\Delta RV_{H} (R=%.2f, k=%.2f)$' %(r2, fit2[0]), alpha=alpha)
plt.axis('scaled')
plt.ylim(-1.8, 2.8)
plt.title(r'$K = 10$ m/s')
plt.xlabel('Jitter ' + '($RV_{Gaussian})$')
plt.ylabel(r'$\Phi$ESTA jitter metrics')
plt.legend()
# plt.savefig('Performance2.png')
plt.show()
