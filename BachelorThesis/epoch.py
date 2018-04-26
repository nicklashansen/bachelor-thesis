from numpy import *
from filters import quantile_norm
import pickle as pck
import filesystem as fs
from log import *

"""
WRITTEN BY
Nicklas Hansen
"""

EPOCH_LENGTH = 120
OVERLAP_FACTOR = 2
MASK_THRESHOLD = 0.125

def get_epochs(X, y, mask):
	epochs = generate_epochs(quantile_norm(X), y, mask)
	return filter_epochs(epochs)

def generate_epochs(X, y, mask):
	epochs, index, length = [], int(0), len(y)-EPOCH_LENGTH
	timecol = transpose(X)[0]
	X = delete(X, 0, axis=1)
	a=b=c=OVERLAP_FACTOR*(length/EPOCH_LENGTH)
	while (index < length):
		index = int(index)
		end = int(index+EPOCH_LENGTH)
		e = epoch(X[index:end], y[index:end], timecol[index:end], mask[index:end])
		#if (e.continuous()):
		#	a -= 1
		#if (e.acceptable()):
		#	b -= 1
		#if (e.no_cut()):
		#	c -= 1
		if e.continuous() and e.acceptable() and e.no_cut():
			epochs.append(e)
		index += EPOCH_LENGTH/OVERLAP_FACTOR
	#print(a,b,c)
	#print(len(epochs))
	return epochs

def filter_epochs(epochs):
	filtered = []
	for i,e in enumerate(epochs):
		if (e.index_start > 0):
			filtered.append(e)
	return filtered

def save_epochs(epochs):
	os.makedirs(fs.Filepaths.SaveEpochs, exist_ok=True)
	file = fs.Filepaths.SaveEpochs + 'epochs.pickle'
	with open(file, 'wb') as handle:
		pck.dump(epochs, handle, protocol=pck.HIGHEST_PROTOCOL)

class epoch(object):
	def __init__(self, X, y, timecol, mask = None):
		self.X, self.y = X, y
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
			#print('unacceptable')
			return False
		return True

	def no_cut(self):
		start,stop = self.y[0], self.y[-1]
		if start or stop:
			return False
		return True