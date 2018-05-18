# defind group random number generator 

import numpy as np
import random


def gran_gen(n_group, n_obs):

	ave_sample_per_group 	= int(n_obs / n_group) + 1
	sample_per_group 		= int(1.5 * ave_sample_per_group)


	# generate a starting array in which elements are minimum spaced by 9 
	while True:
		x_start = np.sort(random.sample(range(200-sample_per_group-1), n_group))

		if not any(np.diff(x_start) < (sample_per_group + 1)):
			break

	# fill in the elements per group
	x 	= np.hstack([i + np.sort(random.sample(range(sample_per_group), random.randint(ave_sample_per_group, sample_per_group))) for i in x_start])

	# remove excessive elements and truncate the array size to 20 
	idx =  np.sort(random.sample(range(np.size(x)), n_obs))

	return x[idx]




