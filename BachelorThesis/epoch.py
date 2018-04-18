from numpy import *

"""
WRITTEN BY
Nicklas Hansen
"""

EPOCH_LENGTH = 120
OVERLAP_FACTOR = 2

def generate_epochs(X, y, mask):
	epochs, index, length = [], int(0), len(y)-EPOCH_LENGTH
	timecol = transpose(X)[0]
	X = delete(X, 0, axis=1)
	while (index < length):
		index = int(index)
		end = int(index+EPOCH_LENGTH-1)
		e = epoch(X[index:end], y[index:end], int(timecol[index]), int(timecol[end]), mask[index:end])
		epochs.append(e)
		index += EPOCH_LENGTH/OVERLAP_FACTOR
	return epochs

def filter_epochs(epochs):
	filtered = []
	for i,e in enumerate(epochs):
		# Filter criterions
		if (e.index_start > 0):
			filtered.append(e)
	return filtered

class epoch(object):
	def __init__(self, X, y, index_start, index_stop, mask = None):
		self.X = X
		self.y = y
		self.index_start = index_start
		self.index_stop = index_stop
		self.mask = mask