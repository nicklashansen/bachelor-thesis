from numpy import *
from filters import quantile_norm, median_filt
from scipy.interpolate import CubicSpline
from log import getLog
from stopwatch import stopwatch
from plots import plot_data

"""
WRITTEN BY
Micheal Kirkegaard,
Nicklas Hansen
"""

def make_features(X, y, sample_rate, removal = True):
	masklist, mask = make_masks(X)
	X = cubic_spline(X, masklist)
	if removal:
		X,y,mask = sleep_removal(X, y, mask, sample_rate)
	X = median_filt(X)
	X = quantile_norm(X, 10)
	return X, y, mask

def make_masks(X):
	Xt = transpose(X)

	# Outlier detection
	def threeSigmaRule(data):
		_data = [x for x in data if x != -1]
		mu = mean(_data)
		std3 = std(_data)*3
		# if datapoint's distance from mu is above 3 times the standard deviation
		# or value is -1 (i.e. no PTT value)
		return [(1 if x == -1 or abs(mu-x) > std3 else 0) for x in data]

	masklist = [threeSigmaRule(x_feat) for x_feat in Xt[1:5]] # DR, RPA, PTT, PWA
	mask = [sum(tub) for tub in zip(*masklist)]

	return masklist, mask

def cubic_spline(X, masks, plot=False):
	Xt = transpose(X)

	def spline(maskid, data):
		mask = masks[maskid]
		datamask = [data[i] for i,m in enumerate(mask) if m == 0]
		cs = CubicSpline(range(len(datamask)),datamask)
		datacs = cs(range(len(mask)))
		if plot:
			plot_data([data, datacs], labels=['Signal','Spline Correction'])
		return array(datacs)

	Xt = array([Xt[0]] + [spline(id,x) for id,x in enumerate(Xt[1:5])] + [Xt[5]]) # Spline DR,RPA,PTT,PWA
	X = transpose(Xt)
	return X

def sleep_removal(X, y, mask, sample_rate):
	_X = transpose(X)
	keep = [i for i,state in enumerate(_X[5]) if state >= 0]
	X = array([X[i] for i in keep])
	y = array([y[i] for i in keep])
	return X,y, mask

#def sleep_removal(X, y, mask, sample_rate):
#	Xt = transpose(X)
#	indexes = Xt[0]
#	sleep = Xt[5]
#	n = len(sleep)

#	keep = []
#	i = 0
#	while(i < n):
#		k = None
#		j = i+1
#		if sleep[i] >= 0:
#			while(j < n and sleep[j] >= 0):
#				j += 1
#		else:
#			while(j < n and sleep[j] == -1 and indexes[j]-indexes[i] < 30*sample_rate):
#				j += 1
#			if j < n and sleep[j] == -1:
#				cuti = j
#				cutj = i
#				while(j < n and sleep[j] == -1):
#					j += 1
#					cutj += 1 
#				if(cutj - cuti > 0):
#					k = list(range(i,cuti)) + list(range(cutj,j))
#		if not k:
#			k = list(range(i,j))
#		keep += k
#		i = j

#	X = array([X[i] for i in keep])
#	y = array([y[i] for i in keep])
#	mask = [mask[i] for i in keep]
#	return X,y,mask