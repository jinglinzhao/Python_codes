# x: oversampled time (column 1)
# mu: Gaussian processes prediction of the most likely value (column 2)
# std: standard deivation of walkers in all runs in MCMC (column 3)
color = "#ff7f0e"
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, mu, color=color)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.title("BI - maximum likelihood prediction (MCMC)");
plt.savefig('ksiboo-prediction-4-MCMC.png') 
plt.show()