'''
WRITTEN BY
Nicklas Hansen

Simple helper module responsible for formatting filters.
'''

from numpy import *
from sklearn.preprocessing import quantile_transform
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler

def median_filt(X, kernel = 3):
	'''
	Applies 1D median filter to X using specified kernel size.
	'''
	return operation_1D(X, medfilt, (kernel,0))

def quantile_norm(X, quantiles = 1000):
	'''
	Applies a quantile normalisation to X using specified number of quantiles.
	'''
	return operation_2D(X, quantile_transform, (0,quantiles))

def operation_1D(X, op=median_filt, args=None):
	'''
	Applies a 1D operation on X.
	'''
	_X = transpose(X)
	for i in range(1,len(_X)-1):
		if (args == None):
			_X[i] = op(_X[i])
		else:
			_X[i] = op(_X[i], args[0])
	return transpose(_X)

def operation_2D(X, op=quantile_transform, args=None):
	'''
	Applies a 2D operation on X.
	'''
	_X = transpose(X)
	for i in range(1,len(_X)-1):
		if (args == None):
			_X[i] = squeeze(op(reshape(_X[i], (-1,1))))
		elif (args[0] == 0):
			_X[i] = squeeze(op(reshape(_X[i], (-1,1)), args[0], args[1]))
		else:
			_X[i] = squeeze(op(reshape(_X[i], (-1,1)), args[0]))
	return transpose(_X)