from numpy import *
from sklearn.preprocessing import quantile_transform

"""
WRITTEN BY
Nicklas Hansen
"""

def quantile_norm(X):
	_X = transpose(X)
	for i in range(1,len(_X)):
		_X[i] = squeeze(quantile_transform(reshape(_X[i], (-1,1)), n_quantiles=10))
	return transpose(_X)