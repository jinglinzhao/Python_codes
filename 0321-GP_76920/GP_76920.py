import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn


#==============================================================================
# Import data 
#==============================================================================

# all_rvs 	= np.genfromtxt('all_rvs.dat', dtype = None)
all_rvs 	= np.genfromtxt('all_rvs_1outlier_removed.dat', dtype = None)


DATA_AAT 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'AAT']
DATA_CHIRON = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'CHIRON']
DATA_FEROS 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FEROS']
DATA_MJ1 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ1']
DATA_MJ3 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ3']


#==============================================================================
# apply the offset
#==============================================================================

OFFSET_CHIRON 	= -73.098470
OFFSET_FEROS	= -6.5227172
OFFSET_MJ1 		= -14.925970
OFFSET_MJ3 		= -56.943472

RV_AAT 		= np.zeros( (len(DATA_AAT), 3) )
RV_CHIRON 	= np.zeros( (len(DATA_CHIRON), 3) )
RV_FEROS 	= np.zeros( (len(DATA_FEROS), 3) )
RV_MJ1 		= np.zeros( (len(DATA_MJ1), 3) )
RV_MJ3 		= np.zeros( (len(DATA_MJ3), 3) )


for k in range(len(DATA_AAT)):
	RV_AAT[k, :] 	= [ DATA_AAT[k][i] for i in range(3) ]

for k in range(len(DATA_CHIRON)):
	RV_CHIRON[k, :]	= [ DATA_CHIRON[k][i] for i in range(3) ]
	RV_CHIRON[k, 1] = RV_CHIRON[k, 1] - OFFSET_CHIRON

for k in range(len(DATA_FEROS)):
	RV_FEROS[k, :]	= [ DATA_FEROS[k][i] for i in range(3) ]
	RV_FEROS[k, 1] 	= RV_FEROS[k, 1] - OFFSET_FEROS

for k in range(len(DATA_MJ1)):
	RV_MJ1[k, :]	= [ DATA_MJ1[k][i] for i in range(3) ]
	RV_MJ1[k, 1] 	= RV_MJ1[k, 1] - OFFSET_MJ1

for k in range(len(DATA_MJ3)):
	RV_MJ3[k, :]	= [ DATA_MJ3[k][i] for i in range(3) ]
	RV_MJ3[k, 1] 	= RV_MJ3[k, 1] - OFFSET_MJ3


if 1:
    plt.errorbar(RV_AAT[:,0], 	RV_AAT[:,1], 	yerr=RV_AAT[:,2], 	fmt=".", capsize=0, label='AAT')
    plt.errorbar(RV_CHIRON[:,0],RV_CHIRON[:,1], yerr=RV_CHIRON[:,2],fmt=".", capsize=0, label='CHIRON')
    plt.errorbar(RV_FEROS[:,0], RV_FEROS[:,1], 	yerr=RV_FEROS[:,2], fmt=".", capsize=0, label='FEROS')
    plt.errorbar(RV_MJ1[:,0], 	RV_MJ1[:,1], 	yerr=RV_MJ1[:,2], 	fmt=".", capsize=0, label='MJ1')
    plt.errorbar(RV_MJ3[:,0], 	RV_MJ3[:,1], 	yerr=RV_MJ3[:,2], 	fmt=".", capsize=0, label='MJ3')
    plt.ylabel(r"$RV [m/s]$")
    plt.xlabel(r"$JD$")
    plt.title("Adjusted RV time series")
    plt.legend()
    # plt.show()


# Concatenate the five data sets # 
RV_ALL 	= np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3))

if 0: 
	plt.errorbar(RV_ALL[:,0], RV_ALL[:,1], yerr=RV_ALL[:,2], fmt=".", capsize=0)
	plt.ylabel(r"$RV [m/s]$")
	plt.xlabel(r"$JD$")
	plt.title("RV time series")
	plt.show()


#==============================================================================
# Test Fit 
#==============================================================================    
import george
from george.modeling import Model


class Model(Model):
    parameter_names = ('n', 'tau', 'k', 'w', 'e0', 'offset')

    def get_value(self, t):
         e_anom 	= solve_kep_eqn(self.n*(t.flatten()-self.tau),self.e0)
         f 		= 2*np.arctan2(np.sqrt(1+self.e0)*np.sin(e_anom*.5),np.sqrt(1-self.e0)*np.cos(e_anom*.5))
         return self.k*(np.cos(f + self.w) + self.e0*np.cos(self.w)) + self.offset
        # return self.n+self.tau+self.k+self.w+self.e0+self.offset*t.flatten()


# initial guess of parameters # 
'''
# Period (days)
P0 = 415.4 / 6.3
n = 1/P0

# Eccentricity 
e0 = 0.856

# omega ?
w = 0

# tau = time of pericenter passage ? 
tau = 62.19
	
# k = amplitude of radial velocity (m/s)
K = 186.8

# offset 
offset = 0
'''

t 		= np.linspace(min(RV_ALL[:,0]), max(RV_ALL[:,0]), num=10000, endpoint=True)
truth 	= dict(n=0.0151661049, tau=62.19, k=186.8, w=0, e0=0.856, offset=0) 
#truth       = dict(amp=5, P=25*0.31, phase=0.1) 

# y		= Model(**truth).get_value(RV_ALL[:,0])
y		= Model(**truth).get_value(t)

# Plot the data using the star.plot function
# plt.plot(RV_ALL[:,0], y, '-')
plt.plot(t, y, '-')
plt.ylabel('RV' r"$[m/s]$")
plt.xlabel('t')
plt.title('Simulated RV')
plt.show()



#==============================================================================
# GP Modelling 
#==============================================================================    

from george.modeling import Model
from george import kernels

k1  	= kernels.ExpSine2Kernel(gamma = 1, log_period = np.log(415.4))
k2  	= np.var(y) * kernels.ExpSquaredKernel(1)
kernel 	= k1 * k2

gp  = george.GP(kernel, mean=Model(**truth), white_noise = np.log(1), fit_white_noise = True)                                         

# gp  	= george.GP(kernel, mean=Model(**truth))                                         
gp.compute(RV_ALL[:,0], RV_ALL[:,2])   

def lnprob2(p):
    # Set the parameter values to the given vector
    gp.set_parameter_vector(p)                                                  
    # Compute the logarithm of the marginalized likelihood of a set of observations under the Gaussian process model. 
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()                    




#==============================================================================
# run MCMC on this model
#==============================================================================

import emcee
# Get an array of the parameter values in the correct order. len(initial) = 5. 
initial = gp.get_parameter_vector()                                            
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

print("Running first burn-in...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 2000)

print("Running second burn-in...")
p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
sampler.reset()
p0, _, _ = sampler.run_mcmc(p0, 1000)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 1000);








#==============================================================================
# Check the data sets
#==============================================================================

if 0: 

	# check the completeness of data 
	(len(DATA_AAT) + len(DATA_CHIRON) + len(DATA_FEROS) + len(DATA_MJ1) + len(DATA_MJ3)) == len(all_rvs)

	x 	= [all_rvs[k][0] for k in range(len(all_rvs))]
	y 	= [all_rvs[k][1] for k in range(len(all_rvs))]
	yerr= [all_rvs[k][2] for k in range(len(all_rvs))]
	yerr= [(i**2 + 7**2)**0.5 for i in yerr]

	# Plot the whole raw time series

	plt.errorbar(x, y, yerr=yerr, fmt=".", capsize=0)
	plt.ylabel(r"$RV [m/s]$")
	plt.xlabel(r"$JD$")
	plt.title("Raw RV time series")
	plt.show()



