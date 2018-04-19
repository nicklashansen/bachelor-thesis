from numpy import *
from sklearn.preprocessing import quantile_transform
from scipy.signal import medfilt

"""
WRITTEN BY
Nicklas Hansen
"""

def median_filt(X, kernel = 3):
	return operation(X, medfilt, (3))

def quantile_norm(X, quantiles = 10):
	return operation(X, quantile_transform, (0,10))

def operation(X, op=quantile_transform, args=None):
	_X = transpose(X)
	for i in range(1,len(_X)-1):
		if (len(args) > 1):
			_X[i] = squeeze(op(reshape(_X[i], (-1,1)), args[0], args[1]))
		else:
			_X[i] = squeeze(op(reshape(_X[i], (-1,1)), args[0]))
	return transpose(_X)