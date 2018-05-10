from numpy import *
from filters import quantile_norm
import filesystem as fs
from log import *

"""
WRITTEN BY
Nicklas Hansen
"""

EPOCH_LENGTH = 120
OVERLAP_FACTOR = 2
SAMPLE_RATE = 256
MASK_THRESHOLD = 0.125

def get_epochs(X, y, mask, epoch_length = EPOCH_LENGTH, overlap_factor = OVERLAP_FACTOR, filter = True):
	return generate_epochs(X, y, mask, epoch_length, overlap_factor, filter)

def extract_timecol(X):
	timecol = transpose(X)[0]
	X = delete(X, 0, axis=1)
	return X, timecol

def generate_epochs(X, y, mask, epoch_length, overlap_factor, filter):
	epochs, index, length = [], int(0), len(y)-epoch_length
	X, timecol = extract_timecol(X)
	#a=b=c=overlap_factor*(length/epoch_length)
	while (index < length):
		index = int(index)
		end = int(index+epoch_length)
		e = epoch(X[index:end], y[index:end], timecol[index:end], mask[index:end])
		#if (e.continuous()):
		#	a -= 1
		#if (e.acceptable()):
		#	b -= 1
		#if (e.no_cut()):
		#	c -= 1
		if not filter or (e.continuous() and e.acceptable() and e.no_cut()):
			epochs.append(e)
		index += epoch_length/overlap_factor
	#print(a,b,c)
	#print(len(epochs))
	return epochs

def save_epochs(epochs):
	fs.write_epochs(epochs)

class epoch(object):
	def __init__(self, X, y, timecol, mask = None):
		self.X, self.y, self.yhat = X, y, None
		self.timecol, self.mask, self.timesteps, self.features = timecol, mask, X.shape[0], X.shape[1]
		self.index_start, self.index_stop = int(timecol[0]), int(timecol[len(timecol)-1])

	def continuous(self):
		for i in range(1, len(self.timecol)):
			if (self.timecol[i] - self.timecol[i-1]) >= 1280:
				return False
		return True

	def acceptable(self):
		if (self.mask == None):
			return False
		num = sum(self.mask)
		if (num > MASK_THRESHOLD * EPOCH_LENGTH):
			return False
		return True

	def no_cut(self):
		start,stop = self.y[0], self.y[-1]
		if start or stop:
			return False
		return True