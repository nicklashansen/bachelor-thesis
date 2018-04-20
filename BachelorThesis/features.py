from numpy import *
from filters import quantile_norm, median_filt
from scipy.interpolate import CubicSpline
from log import Log
from stopwatch import stopwatch
from plots import plot_data

"""
WRITTEN BY
Micheal Kirkegaard,
Nicklas Hansen
"""

def make_features(X, y):
	masklist, mask = make_masks(X)
	X = data_fix(X, masklist)
	X,y = sleep_removal(X, y)
	X = median_filt(X)
	X = quantile_norm(X, 10)
	return X, y, mask

def make_masks(X):
	Xt = transpose(X)

	# Outlier detection
	def threeSigmaRule(data):
		_data = [x for x in data if x != 1]
		mu = mean(_data)
		std3 = std(_data)*3
		# if datapoint's distance from mu is above 3 times the standard deviation
		# or value is -1 (i.e. no PTT value)
		return [(1 if x < 0 or abs(mu-x) > std3 else 0) for x in data]

	masklist = [threeSigmaRule(x_feat) for x_feat in Xt[1:5]] # DR, RPA, PTT, PWA
	mask = [sum(tub) for tub in zip(*masklist)]

	return masklist, mask

def data_fix(X, masks):
	Xt = transpose(X)

	def spline(maskid, data):
		mask = masks[maskid]
		x = [i for i,m in enumerate(mask) if m == 0]
		datamask = [data[i] for i in x]
		cs = CubicSpline(x,datamask)
		xs = range(len(mask))
		datacs = cs(xs)
		#plot_data([data, datacs], labels=['Signal','Correction'], normalization=False)
		return array(datacs)

	Xt = array([Xt[0]] + [spline(id,x) for id,x in enumerate(Xt[1:5])] + [Xt[5]]) # Spline DR,RPA,PTT,PWA
	X = transpose(Xt)
	return X

def sleep_removal(X, y):
	_X = transpose(X)
	keep = [i for i,state in enumerate(_X[5]) if state >= 0]
	X = array([X[i] for i in keep])
	y = array([y[i] for i in keep])
	return X,y