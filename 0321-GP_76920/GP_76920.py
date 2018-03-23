
import numpy as np
import matplotlib.pyplot as plt
from rv import *
from rv_fit import * 


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
OFFSET_FEROS 	= -6.5227172
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
    # plt.legend(['AAT', 'CHIRON', 'FEROS', 'MJ1', 'MJ3'])
    plt.legend()
    plt.show()


# Concatenate the five data sets # 
RV_ALL 	= np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3))

plt.errorbar(RV_ALL[:,0], RV_ALL[:,1], yerr=RV_ALL[:,2], fmt=".", capsize=0)
plt.ylabel(r"$RV [m/s]$")
plt.xlabel(r"$JD$")
plt.title("RV time series")
plt.show()


#==============================================================================
# Model
#==============================================================================    

import celerite
celerite.__version__
from celerite import terms














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



