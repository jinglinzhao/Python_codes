import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 30})

x 	= np.linspace(-30,30,201,endpoint=True)
N 	= 100

def output(shift):
	return 1 - np.exp(-((x-shift*2)/5)**2)

plot_y = []
for i in range(N+1):
	fig, axes = plt.subplots(figsize=(16, 9))

	# Part 1 #
	plt.subplot(211)
	t 		= i/N 												# ranging from 0 to 1
	shift 	= np.sin(t*2*np.pi)
	y_se    = output(shift)
	plt.plot(x, y_se, 'k-', linewidth=2.0)

	alpha = 0.5
	if t > 0:
		plt.plot(x, output(0), 'b--', linewidth=3.0, alpha=alpha)	
	if t > 0.25:
		plt.plot(x, output(1), 'b--', linewidth=3.0, alpha=alpha)	
	if t > 0.5:
		plt.plot(x, output(0), 'b--', linewidth=3.0, alpha=alpha)
	if t > 0.75:
		plt.plot(x, output(-1), 'b--', linewidth=3.0, alpha=alpha)
	if t > 1:
		plt.plot(x, output(0), 'b--', linewidth=3.0, alpha=alpha)		

	plt.box(on=None)
	plt.axis('off')
	plt.xlim(-15,15)
	plt.title('Line shift (cross-correlation function)')


	# Part 2 #
	ax = plt.subplot(212)
	plot_x 		= np.arange(i+1)/N
	plot_y.append(shift)
	plt.plot(plot_x, plot_y, 'k-', linewidth=2.0)
	plt.xlim(-0.01,1.01)
	plt.ylim(-1.1,1.1)
	plt.xlabel('Time')
	plt.ylabel('RV [m/s]')
	ax.set_xticks([])
	ax.set_yticks([])

	s = 150
	if t > 0:
		plt.scatter(0, 0, s=s, c='b', marker='o', alpha=alpha)
	if t > 0.25:
		plt.scatter(0.25, 1, s=s, c='b', marker='o', alpha=alpha)	
	if t > 0.5:
		plt.scatter(0.5, 0, s=s, c='b', marker='o', alpha=alpha)
	if t > 0.75:
		plt.scatter(0.75, -1, s=s, c='b', marker='o', alpha=alpha)	
	if t >= 1:
		plt.scatter(1, 0, s=s, c='b', marker='o', alpha=alpha)			

	plt.savefig('./Figure_RV_LineShift/shift' + str(i) + '.png')
	plt.close('all')



# plt.show()