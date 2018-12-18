'''
Correlation
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#==============================================================================
# Import data 
#==============================================================================

directory 	= '/Volumes/DataSSD/MATLAB_codes/0615-FT-HD189733/'
x 			= np.loadtxt(directory + 'GG.txt')
y 			= np.loadtxt(directory + 'YY.txt')
z 			= np.loadtxt(directory + 'ZZ.txt')
bi 			= np.loadtxt(directory + 'BI.txt')
xy 			= x - y
# xy  		= np.loadtxt(directory + 'XY.txt')
zx   		= z - x 

#==============================================================================
# Correlation 1
#==============================================================================

left  = 0.06  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
alpha = 0.5
markersize = 20

plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.plot(x, y, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]') 
spacing = (max(x) - min(x)) * 0.1
plt.xlim(min(x)-spacing, max(x)+spacing)

plt.subplot(152)
plt.plot(x, y, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')   
plt.xlim(min(x)-spacing, max(x)+spacing)

plt.subplot(153)
plt.plot(x, xy, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.title('HD 189733 (companions not removed)')
plt.xlim(min(x)-spacing, max(x)+spacing)
r, p = stats.pearsonr(x, xy)
plt.text(min(x), 0.9*max(xy)+0.1*min(xy), 'R={:.2f}'.format(r), fontsize=20)

plt.subplot(154)
plt.plot(x, zx, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
plt.xlim(min(x)-spacing, max(x)+spacing)
r, p = stats.pearsonr(x, zx)
plt.text(min(x), 0.9*max(zx)+0.1*min(zx), 'R={:.2f}'.format(r), fontsize=20)

plt.subplot(155)
plt.plot(xy, zx, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
spacing_xy = (max(xy) - min(xy)) * 0.1
plt.xlim(min(xy)-spacing_xy, max(xy)+spacing_xy)
plt.savefig('Correlation_HD189733_1.png')   
plt.show()

#==============================================================================
# Correlation 2: remove binary companion 
#==============================================================================

plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.plot(x-bi, y-bi, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,L}$ [m/s]')    
plt.xlim(min(x)-spacing, max(x)+spacing)

plt.subplot(152)
plt.plot(x-bi, y-bi, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$RV_{FT,H}$ [m/s]')        
plt.xlim(min(x)-spacing, max(x)+spacing)

plt.subplot(153)
plt.plot(x-bi, xy, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')
plt.title('HD 189733 (companions removed)')
plt.xlim(min(x)-spacing, max(x)+spacing)
np.corrcoef(x-bi, xy)
r, p = stats.pearsonr(x-bi, xy)
plt.text(min(x), 0.9*max(xy)+0.1*min(xy), 'R={:.2f}'.format(r), fontsize=20)

plt.subplot(154)
plt.plot(x-bi, zx, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$RV_{HARPS}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')
plt.xlim(min(x)-spacing, max(x)+spacing)
r, p = stats.pearsonr(x-bi, zx)
plt.text(min(x), 0.9*max(zx)+0.1*min(zx), 'R={:.2f}'.format(r), fontsize=20)

plt.subplot(155)
plt.plot(xy, zx, '.k', markersize=markersize, alpha=alpha)
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.xlim(min(xy)-spacing_xy, max(xy)+spacing_xy)
plt.savefig('Correlation_HD189733_2.png')   
plt.show()