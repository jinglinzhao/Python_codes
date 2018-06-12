'''
Based on Demo_76920_celerite.py.
Change to celerite
'''

#==============================================================================
# Simulated Dataset
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from rv import solve_kep_eqn
from celerite.modeling import Model


class Model(Model):
    parameter_names = ('P', 'tau', 'k', 'w', 'e0', 'offset')

    def get_value(self, t):
        M_anom  = 2*np.pi/np.exp(self.P) * (t.flatten() - np.exp(self.tau))
        e_anom  = solve_kep_eqn(M_anom, self.e0)
        f       = 2*np.arctan( np.sqrt((1+self.e0)/(1-self.e0))*np.tan(e_anom*.5) )
        return np.exp(self.k)*(np.cos(f + self.w) + self.e0*np.cos(self.w)) - self.offset


# The dict() constructor builds dictionaries directly from sequences of key-value pairs:
truth 	= dict(log_P=np.log(415.9), log_tau=np.log(4812), log_k=np.log(186.8), w=-0.06, e0=0.856, offset=0)        


#==============================================================================
# Import data 
#==============================================================================

all_rvs 	= np.genfromtxt('all_rvs_1outlier_removed.dat', dtype = None)

for i in range(len(all_rvs)):
    all_rvs[i][2]     = (all_rvs[i][2]**2 + 7**2)**0.5 

DATA_AAT 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'AAT']
DATA_CHIRON = [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'CHIRON']
DATA_FEROS 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'FEROS']
DATA_MJ1 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ1']
DATA_MJ3 	= [all_rvs[k] for k in range(len(all_rvs)) if all_rvs[k][3] == b'MJ3']


#==============================================================================
# apply the offset
#==============================================================================

OFFSET_CHIRON   = -73.098470
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



# Concatenate the five data sets # 
RV_ALL  = np.concatenate((RV_AAT, RV_CHIRON, RV_FEROS, RV_MJ1, RV_MJ3))
RV_SORT = sorted(RV_ALL, key=lambda x: x[0])
x       = [RV_SORT[i][0] for i in range(len(RV_SORT))]
y       = [RV_SORT[i][1] for i in range(len(RV_SORT))]
yerr    = [RV_SORT[i][2] for i in range(len(RV_SORT))]


#==============================================================================
# MCMC
#==============================================================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    P, tau, k, w, e0, offset = theta
    # if (350 < P < 450) and (100 < k < 300) and (-np.pi < w < np.pi) and (0.7 < e0 < 0.99):
    if (5.8 < P < 6.1) and (4.6 < k < 5.7) and (-np.pi < w < np.pi) and (0.7 < e0 < 0.99):
        return 0.0
    return -np.inf

# As likelihood, we assume the chi-square. Note: we do not even need to normalize it.
def lnlike(theta, x, y, yerr):
    P, tau, k, w, e0, offset = theta
    fit_curve   = Model(P=P, tau=tau, k=k, w=w, e0=e0, offset=offset)
    y_fit       = fit_curve.get_value(np.array(x))
    return -0.5*(np.sum( ((y-y_fit)/yerr)**2. ))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)    



import emcee
ndim = 6
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), threads=14)

print("Running first burn-in...")
pos = [[6., 8.5, 5.3, 0, 0.8, 0.] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] 
pos, prob, state  = sampler.run_mcmc(pos, 1000)

print("Running second burn-in...")
pos = pos[np.argmax(prob)] + 1e-4 * np.random.randn(nwalkers, ndim)
sampler.reset()
pos, _, _ = sampler.run_mcmc(pos, 1000)
sampler.reset()

print("Running production...")
sampler.run_mcmc(pos, 10000);


import copy
log_samples         = sampler.chain[:, :, :].reshape((-1, ndim))
real_samples        = copy.copy(log_samples)
real_samples[:,0:3] = np.exp(real_samples[:,0:3])


import corner
labels=[r"$P$", r"$\tau$", r"$k$", r"$\omega$", r"$e_{0}$", "$offset$"]
# fig = corner.corner(samples, labels=["$P$", "$tau$", "$k$", "$w$", "$e0$", "$offset$"])
fig = corner.corner(real_samples, labels=labels, # truths=[415.4, 4800, 186.8, (352.9/360-1)*2*np.pi, 0.856, 0],
                quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('76920_MCMC-2-Corner.png')
# plt.show()



#==============================================================================
# Trace
#==============================================================================

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels_log=[r"$\log\ P$", r"$\log\ \tau$", r"$\log\ k$", r"$\omega$", r"$e_{0}$", "$offset$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot( np.rot90(sampler.chain[:, :, i], 3), "k", alpha=0.3)
    ax.set_xlim(0, sampler.chain.shape[1])
    ax.set_ylabel(labels_log[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.savefig('76920_MCMC-3-Trace.png')
# plt.show()


a0, a1, a2, a3, a4, a5= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(real_samples, [16, 50, 84], axis=0)))
aa = np.zeros((6,3))
aa[0,:] = [a0[i] for i in range(3)]
aa[1,:] = [a1[i] for i in range(3)]
aa[2,:] = [a2[i] for i in range(3)]
aa[3,:] = [a3[i] for i in range(3)]
aa[4,:] = [a4[i] for i in range(3)]
aa[5,:] = [a5[i] for i in range(3)]
np.savetxt('76920_MCMC_result.txt', aa, fmt='%.6f')


P, tau, k, w, e0, offset = aa[:,0]
fit_curve = Model(P=np.log(P), tau=np.log(tau), k=np.log(k), w=w, e0=e0, offset=offset)
t_fit   = np.linspace(min(RV_ALL[:,0]), max(RV_ALL[:,0]), num=10001, endpoint=True)
y_fit   = fit_curve.get_value(np.array(t_fit))
plt.figure()
plt.plot(t_fit, y_fit, label='MCMC fit')
plt.errorbar(RV_AAT[:,0],   RV_AAT[:,1],    yerr=RV_AAT[:,2],   fmt=".", capsize=0, label='AAT')
plt.errorbar(RV_CHIRON[:,0],RV_CHIRON[:,1], yerr=RV_CHIRON[:,2],fmt=".", capsize=0, label='CHIRON')
plt.errorbar(RV_FEROS[:,0], RV_FEROS[:,1],  yerr=RV_FEROS[:,2], fmt=".", capsize=0, label='FEROS')
plt.errorbar(RV_MJ1[:,0],   RV_MJ1[:,1],    yerr=RV_MJ1[:,2],   fmt=".", capsize=0, label='MJ1')
plt.errorbar(RV_MJ3[:,0],   RV_MJ3[:,1],    yerr=RV_MJ3[:,2],   fmt=".", capsize=0, label='MJ3')
plt.title("Adjusted RV time series")
plt.ylabel("RV [m/s]")
plt.xlabel("Shifted JD [d]")
plt.legend()
plt.savefig('76920_MCMC-4-MCMC_fit.png')
# plt.show()


residual = fit_curve.get_value(np.array(x)) - np.array(y)
chi2 = sum(residual**2 / np.array(yerr)**2)
rms = np.sqrt(np.mean(residual**2))


inds = np.random.randint(len(log_samples), size=100)
plt.figure()
for ind in inds:
    sample = log_samples[ind]
    fit_curve = Model(P=sample[0], tau=sample[1], k=sample[2], w=sample[3], e0=sample[4], offset=sample[5])
    y_fit   = fit_curve.get_value(np.array(t_fit))
    plt.plot(t_fit, y_fit, "g", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel("RV [m/s]")
plt.xlabel("Shifted JD [d]")
plt.savefig('76920_MCMC-5-MCMC_100_realizations.png')



plt.show()







