import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

left  = 0.05  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.2   # the bottom of the subplots of the figure
top = 0.8      # the top of the subplots of the figure
wspace = 0.1   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
linewidth = 10
fontsize = 18
alpha = 0.2
s = 8

RV_jitter   = np.loadtxt('RV_jitter.txt')
RV_jitter   = RV_jitter - np.mean(RV_jitter)
J   		= np.hstack((RV_jitter,RV_jitter))

############################################################
# Plot 1 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p2+j'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(14, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

Grid = gridspec.GridSpec(1, 3)

range_x = (max(Y) - min(Y))
range_y = (max(Z) - min(Z))
# y_up = max(Z)+0.1*range_y
# y_lo = min(Z)-0.1*range_y
y_up = 5.8
y_lo = -5.5

fig.suptitle('Jitter amplitude' + r'$\approx$' + 'plantary amplitude', y=0.9)

axes_1 = plt.subplot(Grid[0, 0])
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Y, 1)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')

axes_2 = plt.subplot(Grid[0, 1])
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Z, 1)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')

axes_3 = plt.subplot(Grid[0, 2])
fit = np.polyfit(Y, Z, 1)
x_sample = np.linspace(min(Y)*2, max(Y)*2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.axis('scaled')
plt.ylim([y_lo, y_up])
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.xlim([-3, 4])
r1, p = stats.pearsonr(Y, Z)
plt.text(min(Y), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(Y), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)         
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')     
axes_3.xaxis.set_major_locator(plt.MaxNLocator(3))

plt.savefig('Correlation_2pj2000.png')   
plt.show()

# print(np.std(XY), np.std(XY)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))

############################################################
# Plot 2 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p10+j'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(21, 6))
plt.subplots_adjust(left=0.02, bottom=bottom, right=1, top=top, wspace=0.00, hspace=hspace)

Grid = gridspec.GridSpec(1, 44)

fig.suptitle('Planetary signal dominates', y=0.9)

axes_1 = plt.subplot(Grid[0,0:19])
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Y, 1)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')

axes_2 = plt.subplot(Grid[0,19:38])
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Z, 1)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')

axes_3 = plt.subplot(Grid[0,38:])
fit = np.polyfit(Y, Z, 1)
x_sample = np.linspace(min(Y)*2, max(Y)*2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.xlim([-3, 4])
plt.ylim([y_lo, y_up])
r1, p = stats.pearsonr(Y, Z)
plt.text(min(Y), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(Y), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)         
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')  
# axes_3.yaxis.set_major_locator(plt.MaxNLocator(3))

plt.savefig('Correlation_10pj2000.png')   
plt.show()

# print(np.std(XY), np.std(XY)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))


############################################################
# Plot 3 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p0+j'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

Grid = gridspec.GridSpec(1, 3)

fig.suptitle('Jitter only; no planet', y=0.9)

axes_1 = plt.subplot(Grid[0,0])
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Y, 1)
x_sample = np.linspace(min(G)*1.1, max(G)*1.1, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')

axes_2 = plt.subplot(Grid[0,1])
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Z, 1)
x_sample = np.linspace(min(G)*2, max(G)*2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')

axes_3 = plt.subplot(Grid[0,2])
fit = np.polyfit(Y, Z, 1)
x_sample = np.linspace(min(Y)*1.5, max(Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.axis('scaled')
plt.xlim([-3, 4])
plt.ylim([y_lo, y_up])
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
r1, p = stats.pearsonr(Y, Z)
plt.text(min(Y), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(Y), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)         
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')   

plt.savefig('Correlation_null2000.png')   
plt.show()

# print(np.std(XY), np.std(XY)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))


############################################################
# Plot 4 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p2'
G  = np.loadtxt(DIR + '/GG.txt')
X  = np.loadtxt(DIR + '/XX.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

Grid = gridspec.GridSpec(1, 3)

# range_x = (max(Y) - min(Y))
# range_y = (max(Z) - min(Z))
# y_up = max(Z)+0.1*range_y
# y_lo = min(Z)-0.1*range_y

fig.suptitle('Planet only; no jitter', y=0.9)

axes_1 = plt.subplot(Grid[0,0])
plt.scatter(G, Y, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Y, 1)
r1, p = stats.pearsonr(G, Y)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)              
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_L$')

axes_2 = plt.subplot(Grid[0,1])
plt.scatter(G, Z, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.ylim([y_lo, y_up])
fit = np.polyfit(G, Z, 1)
r1, p = stats.pearsonr(G, Z)
plt.text(min(G), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(G), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)    
plt.xlabel('$RV_{Gaussian}$')
plt.ylabel(r'$\Delta RV_H$')
# axes_2.xaxis.set_major_locator(plt.MaxNLocator(3))

# x_lo = min(Y)-0.05*range_x
# x_up = max(Y)+0.05*range_x
# y_lo = min(Z)-0.05*range_y
# y_up = max(Z)+0.05*range_y
axes_3 = plt.subplot(Grid[0,2])
fit = np.polyfit(Y, Z, 1)
plt.scatter(Y, Z, color='k', alpha=0.5, s=s)
plt.axis('scaled')
plt.xlim([-3, 4])
plt.ylim([y_lo, y_up])
# plt.xlim([-0.6, 0.6])
r1, p = stats.pearsonr(Y, Z)
plt.text(min(Y), 0.9*y_up+0.1*y_lo, 'R={0:.2f}'.format(r1), fontsize=fontsize)
plt.text(min(Y), 0.8*y_up+0.2*y_lo, 'k={0:.2f}'.format(fit[0]), fontsize=fontsize)         
plt.xlabel(r'$\Delta RV_L$')    
plt.ylabel(r'$\Delta RV_H$')  
# axes_3.yaxis.set_major_locator(plt.MaxNLocator(3))

plt.savefig('Correlation_0jitter2000.png')   
plt.show()