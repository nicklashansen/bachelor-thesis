import numpy as np
from epoch import *
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

def make_masks(X, y):
	Xt = np.transpose(X)
	index = Xt[0]
	m_DR = maskDR(Xt[1])
	m_RPA = maskRPA(Xt[2])
	m_PTT = maskPTT(Xt[3])
	m_PWA = maskPTT(Xt[4])
	m_SS = maskSS(Xt[5])
	m_AA = maskAA(y)
	masklist = [m_DR, m_RPA, m_PTT, m_PWA, m_SS, m_AA]
	return masklist, mergeMasks(masklist)

def make_features(X, y):
	masklist, mask = make_masks(X, y)
	#X, y = data_fix(X, y, masklist)
	X,y = sleep_removal(X, y)
	X = median_filt(X)
	X = quantile_norm(X, 10)
	return X,y,mask

# Makes Feature Masks

# DR
def maskDR(x_DR):
	mask = threeSigmaRule(x_DR)
	for index in mask:
		if (x_DR[index] > 2.0 or x_DR[index] < 0.4):
			mask[index] = 1
	return mask

# RPA
def maskRPA(x_RPA):
	mask = threeSigmaRule(x_RPA)
	return mask

# PTT
def maskPTT(x_PTT):
	mask = threeSigmaRule(x_PTT)
	for i,ppt in enumerate(x_PTT):
		if ppt == -1:
			mask[i] = 1
	return mask

# PWA
def maskPWA(x_PWA):
	mask = threeSigmaRule(x_PWA)
	for i,pwa in enumerate(x_PWA):
		if pwa == -1:
			mask[i] = 1
	return mask

# SS
def maskSS(x_SS):
	mask = [0]*len(x_SS)
	
	# Wake States
	for i,state in enumerate(x_SS):
		if state == -1:
			mask[i] = -1

	return mask

# AA
def maskAA(y_AA):
	mask = [0]*len(y_AA)
	# No Mask, but leaving in, in case of future changes
	return mask 

# Outlier detection
def threeSigmaRule(data):
	_data = [x for x in data if x != 1]
	mu = np.mean(_data)
	std3 = np.std(_data)*3

	#(if datapoint's distance from mu is above 3 times the standard deviation, it's an outlier)
	return [(1 if abs(mu-x) > std3 else 0) for x in data]

def mergeMasks(masks):
	return [sum(tub) for tub in zip(*masks)]

def sleep_removal(X, y):
	_X = transpose(X)
	keep = [i for i,state in enumerate(_X[5]) if state >= 0]
	X = array([X[i] for i in keep])
	y = array([y[i] for i in keep])
	return X,y


def data_fix(X, y, masks):
	Xt = np.transpose(X)
	index = Xt[0]
	x_SS = Xt[5]
	sleepCut = [i for i,state in enumerate(Xt[5]) if state >= 0]

	def spline(maskid, data):
		mask = masks[maskid]
		x = [i for i,m in enumerate(mask) if m == 0]
		datamask = [data[i] for i in x]
		cs = CubicSpline(x,datamask)
		xs = range(len(mask))
		datacs = cs(xs)
		plot_data(data, datacs, [i for i,m in enumerate(mask) if m > 0])
		return datacs

	#Xt = array([spline(id,x) for id,x in enumerate(Xt[1:5])]) # Spline DR,RPA,PTT,PWA
	Xt = np.insert(Xt, 0, index)
	Xt = np.append(Xt, x_SS)
	X = transpose(Xt)
	X = array([X[i] for i in sleepCut])
	y = array([y[i] for i in sleepCut])
	return X, y 