import pymc3 as pm
from theano import shared
from pymc3.distributions.timeseries import GaussianRandomWalk
from scipy import optimize
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats

x   = np.arange(200)
XX  = np.loadtxt('XX.txt')
YY  = np.loadtxt('YY.txt')
y   = XX-YY
jitter = np.loadtxt('RV.dat')
jitter = np.hstack((jitter, jitter))
jitter = jitter - np.mean(jitter)

plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Jitter proto")
plt.savefig('Proto_jitter.png')


LARGE_NUMBER = 1e5

model = pm.Model()
with model:
    smoothing_param = shared(0.9)
    mu = pm.Normal("mu", sd=LARGE_NUMBER)
    tau = pm.Exponential("tau", 1.0/LARGE_NUMBER)
    z = GaussianRandomWalk("z",
                           mu=mu,
                           tau=tau / (1.0 - smoothing_param),
                           shape=y.shape)
    obs = pm.Normal("obs",
                    mu=z,
                    tau=tau / smoothing_param,
                    observed=y)

def infer_z(smoothing):
    with model:
        smoothing_param.set_value(smoothing)
        res = pm.find_MAP(vars=[z], fmin=optimize.fmin_l_bfgs_b)
        return res['z']


# allocate 50% variance to the noise #                             
smoothing = 0.9
z_val = infer_z(smoothing)

plt.figure(figsize=(20, 14))
plt.plot(x, y);
plt.plot(x, jitter/5)
plt.plot(x, z_val);
plt.title("Smoothing={}".format(smoothing));
plt.savefig('Jitter_YY_Smoothing.png')

np.savetxt('Jitter_y_val.txt', z_val, fmt='%.6f')
