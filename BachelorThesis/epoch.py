
EPOCH_LENGTH = 120

def generate_epochs(X, y, masks):
	X_epochs, y_epochs, mask_epochs = [],[],[]
	index, length = 0, len(y)-EPOCH_LENGTH
	while (index < length):
		end = index + EPOCH_LENGTH - 1
		X_epochs.append(X[index:end])
		y_epochs.append(y[index:end])
		index += EPOCH_LENGTH/2
	return X_epochs, y_epochs

class epoch(object):
	def __init__(self, X, y, time_start, duration):
		self.X = X
		self.y = y
		self.time_start = time_start
		self.duration = duration