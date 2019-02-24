import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

left  = 0.06  # the left side of the subplots of the figure
right = 0.98    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.55   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
linewidth = 10
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
X  = np.loadtxt(DIR + '/GG.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.scatter(X, Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Y, 1)
r1, p = stats.pearsonr(X, Y)
plt.text(min(X), 0.95*max(Y)+0.05*min(Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Y)+0.15*min(Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.scatter(X, Z, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z, 1)
r1, p = stats.pearsonr(X, Z)
plt.text(min(X), 0.95*max(Z)+0.05*min(Z), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z)+0.15*min(Z), 'k={0:.2f}'.format(fit[0]), fontsize=20)        
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.scatter(X, X-Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, X-Y, 1)
r1, p = stats.pearsonr(X, X-Y)
plt.text(min(X), 0.95*max(X-Y)+0.05*min(X-Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(X-Y)+0.15*min(X-Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)              
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')

plt.title('Jitter amplitude' + r'$\approx$' + 'plantary amplitude')

plt.subplot(154)
plt.scatter(X, Z-X, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z-X, 1)
r1, p = stats.pearsonr(X, Z-X)
plt.text(min(X), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)          
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')


plt.subplot(155)
range_x = (max(X-Y) - min(X-Y))
range_y = (max(Z-X) - min(Z-X))
fit = np.polyfit(X-Y, Z-X, 1)
x_sample = np.linspace(min(X-Y)*1.5, max(X-Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(X-Y, Z-X, color='k', alpha=0.5, s=s)
plt.xlim([min(X-Y)-0.1*range_x, max(X-Y)+0.1*range_x])
plt.ylim([min(Z-X)-0.1*range_y, max(Z-X)+0.1*range_y])
r1, p = stats.pearsonr(X-Y, Z-X)
plt.text(min(X-Y), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X-Y), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)         
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.savefig('Correlation_2pj2000.png')
plt.show()

print(np.std(X-Y), np.std(X-Y)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))

############################################################
# Plot 2 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p10+j'
X  = np.loadtxt(DIR + '/GG.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.scatter(X, Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Y, 1)
r1, p = stats.pearsonr(X, Y)
plt.text(min(X), 0.95*max(Y)+0.05*min(Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Y)+0.15*min(Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.scatter(X, Z, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z, 1)
r1, p = stats.pearsonr(X, Z)
plt.text(min(X), 0.95*max(Z)+0.05*min(Z), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z)+0.15*min(Z), 'k={0:.2f}'.format(fit[0]), fontsize=20)        
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.scatter(X, X-Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, X-Y, 1)
r1, p = stats.pearsonr(X, X-Y)
plt.text(min(X), 0.95*max(X-Y)+0.05*min(X-Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(X-Y)+0.15*min(X-Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)              
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')

plt.title('Planetary signal dominates')

plt.subplot(154)
plt.scatter(X, Z-X, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z-X, 1)
r1, p = stats.pearsonr(X, Z-X)
plt.text(min(X), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)          
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')


plt.subplot(155)
range_x = (max(X-Y) - min(X-Y))
range_y = (max(Z-X) - min(Z-X))
fit = np.polyfit(X-Y, Z-X, 1)
x_sample = np.linspace(min(X-Y)*1.5, max(X-Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(X-Y, Z-X, color='k', alpha=0.5, s=s)
plt.xlim([min(X-Y)-0.1*range_x, max(X-Y)+0.1*range_x])
plt.ylim([min(Z-X)-0.1*range_y, max(Z-X)+0.1*range_y])
r1, p = stats.pearsonr(X-Y, Z-X)
plt.text(min(X-Y), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X-Y), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)         
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.savefig('Correlation_10pj2000.png')   
plt.show()

print(np.std(X-Y), np.std(X-Y)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))


############################################################
# Plot 3 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p0+j'
X  = np.loadtxt(DIR + '/GG.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.scatter(X, Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Y, 1)
x_sample = np.linspace(min(X)*1.1, max(X)*1.1, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(X, Y)
plt.text(min(X), 0.95*max(Y)+0.05*min(Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Y)+0.15*min(Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.scatter(X, Z, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z, 1)
x_sample = np.linspace(min(X)*1.1, max(X)*1.1, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(X, Z)
plt.text(min(X), 0.95*max(Z)+0.05*min(Z), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z)+0.15*min(Z), 'k={0:.2f}'.format(fit[0]), fontsize=20)        
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.scatter(X, X-Y, color='k', alpha=0.5, s=s)
fit, V = np.polyfit(X, X-Y, 1, cov=True)
x_sample = np.linspace(min(X)*1.2, max(X)*1.2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)      
r1, p = stats.pearsonr(X, X-Y)
plt.text(min(X), 0.95*max(X-Y)+0.05*min(X-Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(X-Y)+0.15*min(X-Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)              
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')

plt.title('Jitter only; no planet')

plt.subplot(154)
plt.scatter(X, Z-X, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z-X, 1)
x_sample = np.linspace(min(X)*1.2, max(X)*1.2, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)      
r1, p = stats.pearsonr(X, Z-X)
plt.text(min(X), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)          
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')

plt.subplot(155)
range_x = (max(X-Y) - min(X-Y))
range_y = (max(Z-X) - min(Z-X))
fit = np.polyfit(X-Y, Z-X, 1)
x_sample = np.linspace(min(X-Y)*1.5, max(X-Y)*1.5, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)    
plt.scatter(X-Y, Z-X, color='k', alpha=0.5, s=s)
plt.xlim([min(X-Y)-0.1*range_x, max(X-Y)+0.1*range_x])
plt.ylim([min(Z-X)-0.1*range_y, max(Z-X)+0.1*range_y])
r1, p = stats.pearsonr(X-Y, Z-X)
plt.text(min(X-Y), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X-Y), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)         
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.savefig('Correlation_null2000.png')   
plt.show()

print(np.std(X-Y), np.std(X-Y)/np.std(J), np.std(Z-X), np.std(Z-X)/np.std(J))


############################################################
# Plot 4 												   #
############################################################

# Read Data 
DIR = 'Paper2000/p2'
X  = np.loadtxt(DIR + '/GG.txt')
Y  = np.loadtxt(DIR + '/YY.txt')
Z  = np.loadtxt(DIR + '/ZZ.txt')

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.subplot(151)
plt.scatter(X, Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Y, 1)
x_sample = np.linspace(min(X)*1.1, max(X)*1.1, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(X, Y)
plt.text(min(X), 0.95*max(Y)+0.05*min(Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Y)+0.15*min(Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,L}$ [m/s]')    

plt.subplot(152)
plt.scatter(X, Z, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z, 1)
x_sample = np.linspace(min(X)*1.1, max(X)*1.1, num=100, endpoint=True)
y_sample = fit[0]*x_sample + fit[1]        
plt.plot(x_sample, y_sample, 'g-', linewidth=linewidth, alpha=alpha)
r1, p = stats.pearsonr(X, Z)
plt.text(min(X), 0.95*max(Z)+0.05*min(Z), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z)+0.15*min(Z), 'k={0:.2f}'.format(fit[0]), fontsize=20)        
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel('$RV_{FT,H}$ [m/s]')        

plt.subplot(153)
plt.scatter(X, X-Y, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, X-Y, 1)
r1, p = stats.pearsonr(X, X-Y)
plt.text(min(X), 0.95*max(X-Y)+0.05*min(X-Y), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(X-Y)+0.15*min(X-Y), 'k={0:.2f}'.format(fit[0]), fontsize=20)              
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_L$ [m/s]')

plt.title('Planet only; no jitter')

plt.subplot(154)
plt.scatter(X, Z-X, color='k', alpha=0.5, s=s)
fit = np.polyfit(X, Z-X, 1)
r1, p = stats.pearsonr(X, Z-X)
plt.text(min(X), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)          
plt.xlabel('$RV_{Gaussian}$ [m/s]')
plt.ylabel(r'$\Delta RV_H$ [m/s]')

plt.subplot(155)
range_x = (max(X-Y) - min(X-Y))
range_y = (max(Z-X) - min(Z-X))
fit = np.polyfit(X-Y, Z-X, 1)
plt.scatter(X-Y, Z-X, color='k', alpha=0.5, s=s)
plt.xlim([min(X-Y)-0.1*range_x, max(X-Y)+0.1*range_x])
plt.ylim([min(Z-X)-0.1*range_y, max(Z-X)+0.1*range_y])
r1, p = stats.pearsonr(X-Y, Z-X)
plt.text(min(X-Y), 0.95*max(Z-X)+0.05*min(Z-X), 'R={0:.2f}'.format(r1), fontsize=20)
plt.text(min(X-Y), 0.85*max(Z-X)+0.15*min(Z-X), 'k={0:.2f}'.format(fit[0]), fontsize=20)         
plt.xlabel(r'$\Delta RV_L$ [m/s]')    
plt.ylabel(r'$\Delta RV_H$ [m/s]')     
plt.savefig('Correlation_0jitter2000.png')   
plt.show()

