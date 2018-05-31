from numpy import *
from epoch import epoch
import settings

"""
WRITTEN BY:
Nicklas Hansen
"""

def timeseries(epochs, epoch_length = settings.EPOCH_LENGTH, overlap_factor = settings.OVERLAP_FACTOR, sample_rate = settings.SAMPLE_RATE):
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
	for i,y in enumerate(values[window:]):
		enum = [int(timecol[window+i-1]/sample_rate),int(timecol[window+i]/sample_rate)]
		if enum[0] > enum[1]:
			enum[0] = 0
		if y == criteria:
			for j in range(enum[0],enum[1]):
				ts[j] = 1
	return ts

def region(array, count = False):
	regions, start, bin, n = [], 0, False, 0
	for i,val in enumerate(array):
		if val == 1:
			if not bin:
				start, bin = i, True
		elif bin:
			bin = False
			n += 1
			regions.append([start, i-1])
			#if i-1-start <= 3 and start > 2:
			#	regions.append([start-2,i-1])
			#else:
			#	regions.append([start, i-1])
	if bin:
		regions.append([start, i])
		n += 1
	if count:
		return regions, n
	return regions

def add_ECG_overhead(epoch, illegal):
	illegal.append([0, int(epoch.index_start/sample_rate)])
	return illegal