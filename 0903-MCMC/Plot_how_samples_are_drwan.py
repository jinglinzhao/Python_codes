import numpy as np
import matplotlib.pyplot as plt
import random
from functions import gran_gen

mode    = 5; 
n_group = 12
n_obs   = 36


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


cluster = x_start
sample 	= np.union1d(x_start, x_diff[idx])
y = np.zeros(len(sample))


plt.rcParams.update({'font.size': 20})
left  = 0.02  # the left side of the subplots of the figure
right = 0.98    # the right side of the subplots of the figure
bottom = 0.35   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
fig = plt.subplots(figsize=(20, 2.5))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
# plt.axhline(color="gray", ls='-')
plt.plot(sample, y, 'ko', alpha = 0.5)
for i in np.arange(n_group):
	if i < n_group-1:
		idx = (sample >= cluster[i]) & (sample < cluster[i+1])
	else:
		idx = (sample >= cluster[i]) & (sample <= 400)
	left 	= min(sample[idx]) - 1
	right 	= max(sample[idx]) + 1
	plt.fill([left, right, right, left], [-0.1, -0.1, 0.1, 0.1], 'k', alpha=0.2, edgecolor='k')
	mid = (left + right)/2
	if i == 0:
		mid0 = mid
	if i == n_group-1:
		mid1 = mid
	plt.plot([mid, mid], [0.1, 0.15], 'k-', alpha=0.5, linewidth=2)
plt.plot([mid0, mid1], [0.15, 0.15], 'k-', alpha=0.5, linewidth=2)
plt.plot([200, 200], [0.15, 0.25], 'k-', alpha=0.5, linewidth=2)
plt.text(200, 0.28, '36 samples clustered in 12 groups', horizontalalignment='center') 
plt.xlabel('Serial number')
plt.xlim([0, 400])
plt.ylim([-0.2, 0.5])	
plt.yticks([])
plt.savefig('Sampling_demo.png') 
plt.show()