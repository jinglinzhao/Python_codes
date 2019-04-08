# Back to the tripple panel layout

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

left  = 0.08  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.2   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.55   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
linewidth = 10
fontsize = 18
alpha = 0.2
s = 8

############################################################
# Plot 1 												   #
############################################################

# Read Data 
DIR = 'Paper10000/p2+j'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 4.5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

range_x = (max(G) - min(G))
range_y = (max(Z) - min(Z))
x_lo = min(G)-0.1*range_x
x_up = max(G)+0.1*range_x
y_up = max(Z)+0.2*range_y
y_lo = min(Z)-0.1*range_y

fig.suptitle('Jitter amplitude' + r'$\approx$' + 'planetary amplitude', y=0.95)

axes_1 = plt.subplot(131)
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Y, 1, cov=True)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')
axes_1.xaxis.set_major_locator(plt.MaxNLocator(3))


axes_2 = plt.subplot(132)
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Z, 1, cov=True)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')
axes_2.xaxis.set_major_locator(plt.MaxNLocator(3))

axes_3 = plt.subplot(133)
fit = np.polyfit(Y, Z, 1)
x_sample = np.linspace(min(Y)*1.5, max(Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.xlim([-1.1, 1.1])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(Y, Z, 1, cov=True)
r1, p = stats.pearsonr(Y, Z)
plt.text(-1.1+0.1*2.2, 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(-1.1+0.1*2.2, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')  
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))


plt.savefig('Correlation_2pj.png')   
plt.show()


############################################################
# Plot 2 												   #
############################################################

# Read Data 
DIR = 'Paper10000/p10+j'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

range_x = (max(G) - min(G))
x_lo = min(G)-0.1*range_x
x_up = max(G)+0.1*range_x

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 4.5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

fig.suptitle('Planetary signal dominates', y=0.95)

axes_1 = plt.subplot(131)
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Y, 1, cov=True)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')

axes_2 = plt.subplot(132)
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Z, 1, cov=True)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')
axes_2.xaxis.set_major_locator(plt.MaxNLocator(3))

axes_3 = plt.subplot(133)
fit = np.polyfit(Y, Z, 1)
x_sample = np.linspace(min(Y)*1.5, max(Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.xlim([-1.1, 1.1])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(Y, Z, 1, cov=True)
r1, p = stats.pearsonr(Y, Z)
plt.text(-1.1+0.1*2.2, 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(-1.1+0.1*2.2, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)         
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')  
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))

plt.savefig('Correlation_10pj.png')   
plt.show()


############################################################
# Plot 3 												   #
############################################################

# Read Data 
DIR = 'Paper10000/p0+j'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

range_x = (max(G) - min(G))
x_lo = min(G)-0.1*range_x
x_up = max(G)+0.1*range_x

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 4.5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

fig.suptitle('Jitter only; no planet', y=0.95)

axes_1 = plt.subplot(131)
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Y, 1, cov=True)
x_sample = np.linspace(min(G)*2, max(G)*2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')
axes_1.xaxis.set_major_locator(plt.MaxNLocator(3))

axes_2 = plt.subplot(132)
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Z, 1, cov=True)
x_sample = np.linspace(min(G)*2, max(G)*2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')
axes_2.xaxis.set_major_locator(plt.MaxNLocator(3))

axes_3 = plt.subplot(133)
fit = np.polyfit(Y, Z, 1)
x_sample = np.linspace(min(Y)*1.5, max(Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.xlim([-1.1, 1.1])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(Y, Z, 1, cov=True)
r1, p = stats.pearsonr(Y, Z)
plt.text(-1.1+0.1*2.2, 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(-1.1+0.1*2.2, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)         
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')     
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))

plt.savefig('Correlation_null.png')   
plt.show()

# print(np.std(XY), np.std(XY)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))


############################################################
# Plot 4 												   #
############################################################

# Read Data 
DIR = 'Paper10000/p2'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

range_x = (max(G) - min(G))
x_lo = min(G)-0.1*range_x
x_up = max(G)+0.1*range_x

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(16, 4.5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

fig.suptitle('Planet only; no jitter', y=0.95)

axes_1 = plt.subplot(131)
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Y, 1, cov=True)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')
axes_1.xaxis.set_major_locator(plt.MaxNLocator(3))

axes_2 = plt.subplot(132)
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.xlim([x_lo, x_up])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(G, Z, 1, cov=True)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')
axes_2.xaxis.set_major_locator(plt.MaxNLocator(3))

axes_3 = plt.subplot(133)
fit = np.polyfit(Y, Z, 1)
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.xlim([-1.1, 1.1])
plt.ylim([y_lo, y_up])
fit, V = np.polyfit(Y, Z, 1, cov=True)
r1, p = stats.pearsonr(Y, Z)
plt.text(-1.1+0.1*2.2, 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(-1.1+0.1*2.2, 0.8*y_up+0.2*y_lo, 'k={0:.2f}±{1:.2f}'.format(fit[0],V[0,0]**0.5), fontsize=fontsize)   
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')  
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))

plt.savefig('Correlation_0jitter.png')   
plt.show()