import numpy as np
import matplotlib.pyplot as plt

from celerite.modeling import Model

# Define the model
class MeanModel(Model):
    parameter_names = ("alpha", "ell", "log_sigma2")

    def get_value(self, t):
        return self.alpha * np.exp(-0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2))

    # This method is optional but it can be used to compute the gradient of the
    # cost function below.
    def compute_gradient(self, t):
        e = 0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2)
        dalpha = np.exp(-e)
        dell = self.alpha * dalpha * (t-self.ell) * np.exp(-self.log_sigma2)
        dlog_s2 = self.alpha * dalpha * e
        return np.array([dalpha, dell, dlog_s2])

mean_model = MeanModel(alpha=-1.0, ell=0.1, log_sigma2=np.log(0.4))
true_params = mean_model.get_parameter_vector()

# Simuate the data
np.random.seed(42)
x = np.sort(np.random.uniform(-5, 5, 50))
yerr = np.random.uniform(0.05, 0.1, len(x))
K = 0.1*np.exp(-0.5*(x[:, None] - x[None, :])**2/10.5)
K[np.diag_indices(len(x))] += yerr**2
y = np.random.multivariate_normal(mean_model.get_value(x), K)

# Plot the data
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(-5, 5)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("simulated data");