import numpy as np

def wPearsonCoefficient(x, y, w):

	def cov(x, y, w):

		def wmean(x, w):
			return np.sum(x*w) / np.sum(w)

		return np.sum(w * (x-wmean(x, w)) * (y-wmean(y, w)))  /  np.sum(w)

	return cov(x, y, w) / ( cov(x, x, w) * cov(y, y, w) )**0.5