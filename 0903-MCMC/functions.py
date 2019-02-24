# defind group random number generator 

import numpy as np
import random


# This is the old version:
def gran_gen_old(n_group, n_obs):

	min_sample_per_group 	= int(n_obs / n_group) + 1
	max_sample_per_group 		= int(1.5 * min_sample_per_group)


	# generate a starting array in which elements are minimum spaced by 9 
	while True:
		x_start = np.sort(random.sample(range(400-max_sample_per_group-1), n_group))

		if not any(np.diff(x_start) < (max_sample_per_group + 1)):
			break

	# fill in the elements per group
	x 	= np.hstack([i + np.sort(random.sample(range(max_sample_per_group), random.randint(min_sample_per_group, max_sample_per_group))) for i in x_start])

	# remove excessive elements and truncate the array size to 20 
	idx =  np.sort(random.sample(range(np.size(x)), n_obs))

	return x[idx]



def gran_gen(n_group, n_obs):


	min_sample_per_group 	= 0
	max_sample_per_group 	= 5


	# generate a starting array in which elements are minimum spaced by 9 
	while True:
		x_start = np.sort(random.sample(range(400-max_sample_per_group-1), n_group))

		if not any(np.diff(x_start) < (max_sample_per_group*2)):
			break

	# fill in the elements per group
	
	
	while True:
		x 	= np.hstack([i + np.sort(random.sample(range(1, max_sample_per_group), random.randint(min_sample_per_group, max_sample_per_group-1))) for i in x_start])
		if not len(x) < n_obs:
			break

	x_diff = np.setdiff1d(x, x_start)

	# remove excessive elements and truncate the array size to 24
	idx =  np.sort(random.sample(range(np.size(x)), n_obs-n_group))
	x_diff[idx]


	return np.union1d(x_start, x_diff[idx]).astype(int)



def gaussian_smoothing(t, y, t_smooth, len_smooth):

	y_smooth = np.zeros(len(t_smooth))

	for i in range(len(t_smooth)):
	    weight = np.exp(-(t_smooth[i]-t)**2/(2*len_smooth**2))
	    y_smooth[i] = sum(y * weight) / sum(weight)

	return y_smooth
