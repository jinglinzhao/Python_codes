import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.optimize import curve_fit,fsolve
from rv import *
from rv_fit import * 



plt.close('all')
# Load the data file in here
# Pass the file name of the data file to the load_single_star function
# star= load_single_star('hd10442.dat')
# star= load_single_star('hd5319.dat')
star= load_single_star('HD103720.txt')

# You now have a star object
# Print out the star's name and mass
# Hint: star is a class defined in rv.py with attributes name and mass
print star.name
print star.mass

# Plot the data using the star.plot function
plt.errorbar(star.t, star.vr, yerr=star.vr_err, fmt=".", capsize=0)
plt.ylabel('RV' r"$[m/s]$")
plt.xlabel('JD')
plt.title(star.name)
# plt.show()


# Now fit the data with the fit_data function defined above

fit_data(star,fitting_function)
plt.show()


# Now repeat these steps for the other stars.
