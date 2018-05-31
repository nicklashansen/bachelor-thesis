'''
AUTHOR(S):
Nicklas Hansen

Module is responsible for time series manipulation used by other modules.
Primary purpose is to ready signals for plots in the GUI.
'''

from numpy import *
from epoch import epoch
import settings

def timeseries(epochs, epoch_length = settings.EPOCH_LENGTH, overlap_factor = settings.OVERLAP_FACTOR, sample_rate = settings.SAMPLE_RATE):
	'''
	Takes a list of epochs as argument and computes binary sequences denoting sleep stages and splined areas.
	'''
	window = int(epoch_length - ( epoch_length / overlap_factor))
	length = int(epochs[-1].index_stop/sample_rate)
	wake, nrem, rem, illegal = zeros(length), zeros(length), zeros(length), zeros(length)
	for i,obj in enumerate(epochs):
		features = transpose(obj.X)
		wake = modify_timeseries(wake, features[-3], 1, obj.timecol, window, sample_rate)
		nrem = modify_timeseries(nrem, features[-2], 1, obj.timecol, window, sample_rate)
		rem = modify_timeseries(rem, features[-1], 1, obj.timecol, window, sample_rate)
		illegal = modify_timeseries(illegal, obj.mask, 1, obj.timecol, window, sample_rate)
	for i in range(len(wake)):
		if illegal[i] == 1 and wake[i] == 1:
			illegal[i] = 0
	return wake, nrem, rem, illegal

def modify_timeseries(ts, values, criteria, timecol, window, sample_rate):
	'''
	Sub-function of the timeseries function that checks each index in a sequence for a specified criterion.
	'''
	for i,y in enumerate(values[window:]):
		enum = [int(timecol[window+i-1]/sample_rate),int(timecol[window+i]/sample_rate)]
		if enum[0] > enum[1]:
			enum[0] = 0
		if y == criteria:
			for j in range(enum[0],enum[1]):
				ts[j] = 1
	return ts

def region(array, count = False):
	'''
	Takes an array as input and returns a new array containing all '1'-sequences stored as start and end indices.
	'''
	regions, start, bin, n = [], 0, False, 0
	for i,val in enumerate(array):
		if val == 1:
			if not bin:
				start, bin = i, True
		elif bin:
			bin = False
			n += 1
			regions.append([start, i-1])
	if bin:
		regions.append([start, i])
		n += 1
	if count:
		return regions, n
	return regions