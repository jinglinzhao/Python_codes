import numpy as np

##########################
# Read and reformat data #
##########################

BJD 	= np.loadtxt('ksiboo_BJD_sqrt_p.dat')
BJD 	= BJD.reshape((BJD.shape[0], 1))
# data 	= np.loadtxt('ksiboo_sqrt_p.txt')
data    = np.loadtxt('ksiboo_hat_p.txt')
ALL 	= np.concatenate((BJD, data), axis=1)

# BJD 	= np.loadtxt('ksiboo_BJD_BI.dat')
# BJD 	= BJD.reshape((BJD.shape[0], 1))
# data 	= np.loadtxt('ksiboo_BI.txt')
# ALL 	= np.concatenate((BJD, data), axis=1)

##################
# Visualize data # 
##################
import matplotlib.pyplot as plt

if 0:
    plt.errorbar(ALL[:,0], 	ALL[:,1], 	yerr=ALL[:,2], 	fmt="o", capsize=0, label='sqrt(q)')
    plt.ylabel('Polarization [ppm]')
    plt.xlabel(r"$JD$")
    plt.legend()
    plt.show()

# convert between two scripts # 
t 	 = ALL[:,0] - min(ALL[:,0])
y 	 = ALL[:,1]
yerr = ALL[:,2]



################
# Optimization # 
################
from george import kernels

k1 = 1**2 * kernels.ExpSquaredKernel(metric=10**2)
k2 = 1**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=1, log_period=1.9)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)
kernel = k1 + k2 + k3 + k4
#kernel = k1  +  k2

import george
#gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
#               white_noise=np.log(0.19**2), fit_white_noise=True)
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
gp.compute(t, yerr)
print(gp.log_likelihood(y))
print(gp.grad_log_likelihood(y))


import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t, yerr)

# Print the initial ln-likelihood.
print(gp.log_likelihood(y))

# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="Newton-CG")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(y))
print(np.exp(results.x[6]))


# Make the maximum likelihood prediction
x = np.linspace(min(t), max(t), 1000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)


# Make the maximum likelihood prediction
x1 = np.linspace(min(t), max(t), 1000)
mu1, var1 = gp.predict(y, x, return_var=True)
std1 = np.sqrt(var)
t1 = t
y1 = y
yerr1 = yerr
t_off1 = min(ALL[:,0])



# Plot the data
# plt.figure()
color = "#ff7f0e"
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, mu, color=color)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("hat(p) - maximum likelihood prediction");
plt.savefig('ksiboo-prediction-6.png') 
# plt.title("BI - maximum likelihood prediction");
# plt.savefig('ksiboo-prediction-4.png') 
# plt.title("sqrt(p) - maximum likelihood prediction");
# plt.savefig('ksiboo-prediction-5.png') 
plt.show()


# 6.50581
# 6.49663


# color = "#ff7f0e"
plt.errorbar(t + min(ALL[:,0]), y, yerr=yerr, fmt=".b", capsize=0, label='sqrt(p)')
plt.errorbar(t1 + t_off1, y1, yerr=yerr1, fmt=".r", capsize=0, label='BI')
plt.plot(x + min(ALL[:,0]), mu, color='blue')
plt.plot(x1 + t_off1, mu1, color='red')
plt.fill_between(x + min(ALL[:,0]), mu+std, mu-std, color='blue', alpha=0.3, edgecolor="none")
plt.fill_between(x1 + t_off1, mu1+std1, mu1-std1, color='red', alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.title("");
plt.savefig('ksiboo-Line-up.png') 
plt.show()