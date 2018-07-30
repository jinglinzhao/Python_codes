# THIS IS A PYTHON VERSION OF FT_SOAP.m


import numpy as np
import matplotlib.pyplot as plt



##############
# Parameters #
##############
SN          = 5000
N_FILE      = 200
t           = np.arange(N_FILE)
grid_size   = 0.1
Fs          = 1/grid_size
v0          = np.arange(-20, 20.1, grid_size)
dir1        = '/Volumes/DataSSD/SOAP_2/outputs/02.01/'
dir2        = '/Volumes/DataSSD/SOAP_2/outputs/02.01/CCF_dat/'
# dir1      = '/Volumes/DataSSD/SOAP_2/outputs/HERMIT_2spot/'
# dir2      = '/Volumes/DataSSD/SOAP_2/outputs/HERMIT_2spot/fits/CCF_dat/'
jitter      = np.loadtxt(dir1 + 'RV.dat') / 1000      # activity induced RV [km/s]
jitter      = np.hstack((jitter, jitter))
idx         = (v0 >= -10) & (v0 < 10.1)
v1          = v0[idx]

# window function #
window  = v1 * 0 + 1
bound   = 9
idx_w   = abs(v1) >= bound
window[idx_w]   = (np.cos((abs(v1[idx_w])-bound)/(10-bound)*np.pi) + 1) /2

h = plt.figure()
plt.plot(v1, window)
plt.title('Window function')
plt.xlabel('Wavelength in RV [km/s]')
plt.ylabel('Window function')
plt.savefig('0-Window_function.png')
plt.show()
plt.close(h)



# estimate the size of array FFT_power
filename    = dir2 + 'CCF'+ str(1) + '.dat'
A           = 1 - np.loadtxt(filename)
A           = A[idx]
A1          = A
A_FT 		= np.fft.fft(A * window)
power 		= np.abs(A_FT)**2
angle 		= np.angle(A_FT)

plt.plot(v1, power, '.')
plt.show()

# size1       = length(bb)
# FFT_power   = zeros(size1, N_FILE)
# Y           = zeros(size1, N_FILE)
# RV_noise    = zeros(1,N_FILE)
# # v_planet_array  = linspace(-3,3,101) / 1000.
# v_planet_array  = 4 * sin(t/100.*1.8*2*pi + 1) * 0.001
# RV_gauss        = zeros(N_FILE,1)