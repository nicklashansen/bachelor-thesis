from numpy import *

EPOCH_LENGTH = 120
OVERLAP_FACTOR = 2

def generate_epochs(X, y, mask):
	epochs, index, length = [], 0, len(y)-EPOCH_LENGTH
	while (index < length):
		end = index + EPOCH_LENGTH - 1
		epochs.append(epoch(X[index:end], y[index:end], index, EPOCH_LENGTH, mask[index:end]))
		index += EPOCH_LENGTH/OVERLAP_FACTOR
	return epochs

def filter_epochs(epochs):
	filtered = []
	for i,e in enumerate(epochs):
		# Filter criterions
		if (e.duration > 0):
			filtered.append(e)
	return filtered

class epoch(object):
	def __init__(self, X, y, time_start, duration, mask = None):
		self.X = X
		self.y = y
		self.time_start = time_start
		self.duration = duration