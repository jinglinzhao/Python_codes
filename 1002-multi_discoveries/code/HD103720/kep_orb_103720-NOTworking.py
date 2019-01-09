import numpy as np
from rv import solve_kep_eqn
from kep_orb import kep_orb
import matplotlib.pyplot as plt

P 		= 4.5557
tau 	= 1387.46
e  		= 0.086
offset 	= 0.
k 		= 89
w 		= 262 / 360 * 2 * np.pi

star 	= 'HD103720'
DIR     = '/Volumes/DataSSD/OneDrive - UNSW/Hermite_Decomposition/ESO_HARPS/' + star
MJD     = np.loadtxt(DIR + '/MJD.dat')
RV_HARPS=    np.loadtxt(DIR + '/RV_HARPS.dat') * 1000
RV_noise= np.loadtxt(DIR + '/RV_noise.dat')

# convert to x, y, yerr
x       = MJD
y       = RV_HARPS - np.mean(RV_HARPS)
yerr    = RV_noise

# y 		= kep_orb(MJD+2400000.5, P, tau, k, w, e, offset)

if 0:
	plt.plot(MJD, y, '.')
	plt.plot(MJD, RV_HARPS, '.')
	plt.show()

if 0: 
	plt.plot(RV_HARPS, y, '.')
	plt.show()
 # MJD = JD - 2400000.5

# test 
if 0: 
	x = np.linspace(0,100,10000)
	y = kep_orb(x, P, tau, k, w, e, offset)
	plt.plot(x,y, '-')
	plt.show()

bnds = ((4, 5), (1380, 1390), (80, 100), (-np.pi, np.pi), (0,0.1), (-50, 50))

def LeastSquare(x):
    y_fit = kep_orb(MJD, x[0], x[1], x[2], x[3], x[4], x[5])
    return np.sum( ((y-y_fit)/yerr)**2. )

p0 = np.array([P, tau, k, w, e, offset])
import scipy.optimize as op
results = op.minimize(LeastSquare, p0, method="BFGS", bounds=bnds)
