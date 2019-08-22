import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits


plt.rcParams.update({'font.size': 30})


# Files #
DIR     = '/Volumes/DataSSD/SOAP_2/outputs/0704-SingleSpot/'
FILE    = glob.glob(DIR + 'fits/*fits')
N       = len(FILE)
x 		= np.linspace(-20,20,401,endpoint=True)

hdulist     = fits.open(FILE[60])   
template    = hdulist[0].data   # template
RV = np.loadtxt(DIR + 'RV.dat')


for n in range(51):

	fig, axes = plt.subplots(figsize=(16, 9))

	# Part 1 #
	plt.subplot(211)	
	hdulist  = fits.open(FILE[n])
	CCF      = hdulist[0].data
	CCF_d    = CCF - template

	plt.plot(x, template + CCF_d*100, 'k-', linewidth=2.0)
	plt.box(on=None)
	plt.axis('off')
	plt.xlim(-20,20)
	plt.ylim(0.4,1)
	plt.title('Line deformation (cross-correlation function)')

	
	if n == 0:
		plot_CCF0 = template + CCF_d*100
	if RV[n] == max(RV):
		plot_CCF1 = template + CCF_d*100
	if RV[n] == min(RV):
		plot_CCF2 = template + CCF_d*100	
	if n == 27:
		plot_CCF3 = template + CCF_d*100			

	alpha = 0.5
	index = np.arange(len(RV))
	if n > 0:
		plt.plot(x, plot_CCF0, 'r--', linewidth=3.0, alpha=alpha)	
	n1 = index[RV == max(RV)]
	if n > n1[0]:
		plt.plot(x, plot_CCF1, 'r--', linewidth=3.0, alpha=alpha)
	n2 = index[RV == min(RV)]
	if n > n2[0]:
		plt.plot(x, plot_CCF2, 'r--', linewidth=3.0, alpha=alpha)	   
	if n > 27:
		plt.plot(x, plot_CCF3, 'r--', linewidth=3.0, alpha=alpha)

	# Part 2 #
	ax = plt.subplot(212)
	plot_x 	= np.arange(n+1)
	plot_y 	= RV[0:n+1]
	plt.plot(plot_x, plot_y, 'k-', linewidth=2.0)
	plt.xlim(-1, 51)
	plt.ylim(min(RV)-0.5, max(RV)+0.5)
	plt.xlabel('Time')
	plt.ylabel('RV [m/s]')
	ax.set_xticks([])
	ax.set_yticks([])	

	s = 150
	if n > 0:
		plt.scatter(0, RV[0], s=s, c='r', marker='o', alpha=alpha)
	if n > n1[0]:
		plt.scatter(n1[0], RV[n1], s=s, c='r', marker='o', alpha=alpha)	
	if n > n2[0]:
		plt.scatter(n2[0], RV[n2], s=s, c='r', marker='o', alpha=alpha)
	if n > 27:
		plt.scatter(27, RV[27], s=s, c='r', marker='o', alpha=alpha)	
	if n == 50:
		plt.scatter(50, RV[50], s=s, c='r', marker='o', alpha=alpha)	

	plt.savefig('./Figure_RV_LineDeformation/defo' + str(n) + '.png')
	plt.close('all')

plt.show()


