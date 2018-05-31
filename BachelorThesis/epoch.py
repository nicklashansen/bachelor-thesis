'''
AUTHOR(S):
Nicklas Hansen

Module is responsible for the creation, filtering and manipulation of epochs through high-level functions and a dedicated epoch object.
'''

from numpy import *
from filters import quantile_norm
import filesystem as fs
from log import Log, get_log
import settings

def get_epochs(X, y, mask, epoch_length = settings.EPOCH_LENGTH, overlap_factor = settings.OVERLAP_FACTOR, filter = True):
	'''
	High-level call for the epoch generation to be used by other modules.
	'''
	return generate_epochs(X, y, mask, epoch_length, overlap_factor, filter)

def extract_timecol(X):
	'''
	Takes a feature matrix X and extracts its time axis.
	'''
	timecol = transpose(X)[0]
	X = delete(X, 0, axis=1)
	return X, timecol

def generate_epochs(X, y, mask, epoch_length, overlap_factor, filter):
	'''
	Main function responsible for epoch generation.
	Takes arguments X, y and mask from pre-processing and returns a list of epochs generated from that record.
	Epochs are generated based on the inputted parameters.
	If filter is true, epochs are filtered based on the continuous, acceptable and uncut criteria.
	'''
	epochs, index, length = [], int(0), X.shape[0]-epoch_length
	X, timecol = extract_timecol(X)
	while (index < length):
		index = int(index)
		end = int(index+epoch_length)
		if y is not None:
			e = epoch(X[index:end], y[index:end], timecol[index:end], mask[index:end])
		else:
			e = epoch(X[index:end], None, timecol[index:end], mask[index:end])
		if not filter or (e.continuous() and e.acceptable() and e.no_cut()):
			epochs.append(e)
		index += epoch_length/overlap_factor
	return epochs

def save_epochs(epochs, name = 'epochs'):
	'''
	Saves a list of epochs as a local file using the filesystem module.
	'''
	fs.write_epochs(epochs, name)

class epoch(object):
	'''
	Class responsible for storing data related to a generated epoch.
	In addition to storing its features and targets, start and end indices are kept for future signal reconstruction.
	'''
	def __init__(self, X, y, timecol, mask = None):
		'''
		Creates an instance of the epoch object.
		'''
		self.X, self.y, self.yhat = X, y, None
		self.timecol, self.mask, self.timesteps, self.features = timecol, mask, X.shape[0], X.shape[1]
		self.index_start, self.index_stop = int(timecol[0]), int(timecol[len(timecol)-1])

	def continuous(self):
		'''
		Evaluates whether or not the epoch passes the continuous criteria.
		'''
		for i in range(1, len(self.timecol)):
			if (self.timecol[i] - self.timecol[i-1]) >= settings.SAMPLE_RATE * 5:
				return False
		return True

	def acceptable(self):
		'''
		Evaluates whether or not the epoch passes the acceptable criteria.
		'''
		if (self.mask == None):
			return False
		num = sum(self.mask)
		if (num > settings.MASK_THRESHOLD * settings.EPOCH_LENGTH):
			return False
		return True

	def no_cut(self):
		'''
		Evaluates whether or not the epoch passes the uncut criteria.
		'''
		if self.y is not None:
			start,stop = self.y[0], self.y[-1]
			if start or stop:
				return False
		return True