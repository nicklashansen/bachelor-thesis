from numpy import *
from random import random, shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.preprocessing import sequence
from epoch import *
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
"""

class dataset:
	def __init__(self, epochs = None):
		self.X = []
		self.y = []
		#if epochs is not None:
		#	for epoch in epochs:
		#		self.X.append(epoch.X)
		#		self.y.append(epoch.y)
		#	self.size = len(self.X)
		#	self.timesteps = len(self.X[0])
		#	self.features = len(self.X[0,0])
		#else:
		#	self.size = -1
		#	self.timesteps = -1
		#	self.features = -1

	def holdout(self, split = 0.67):
		train_size = int(self.size * split)
		test_size = self.size - train_size
		trainX, trainY, testX, testY = self.X[:train_size], self.y[:train_size], self.X[train_size:], self.y[train_size:]
		return trainX, trainY, testX, testY

	def kfold(self, folds = 5):
		kf = KFold(folds)
		trainX, trainY, testX, testY = [],[],[],[]
		for train_index, test_index in kf.split(self.X):
			trainX.append(self.X[train_index[0]:train_index[len(train_index)-1]])
			trainY.append(self.y[train_index[0]:train_index[len(train_index)-1]])
			testX.append(self.X[test_index[0]:test_index[len(test_index)-1]])
			testY.append(self.y[test_index[0]:test_index[len(test_index)-1]])
		return trainX, trainY, testX, testY

	def load_mesa(self, summary = False):
		setlist,_ = fs.getAllSubjectFilenames(preprocessed=True)
		self.load_external(setlist, summary)

	'''
	def load_physionet(self, summary = False):
		#'0005', '0029', '0052', '0061', 
		setlist = ['0078', '0079', '0083', '0086', '0087', '0092', '0100', '0103', '0134', '0135', '0141', '0146', '0152', '0166', '0167', '0179', '0184', '0198']
		for i in range(len(setlist)):
			setlist[i] = 'tr03-' + setlist[i]
		self.load_external(setlist, summary)
	'''

	def load_external(self, setlist, summary = False):
		shuffle(setlist)
		for set in setlist:
			# Load Data
			X,y = fs.load_csv(set)

			# Count Arousals
			c = 0
			b = False
			for a in y:
				if a == 1.0:
					if not b:
						b = True
						c += 1
				else:
					b = False
			print(set, ' contains ', c, ' arousals!')

			# Transform Data
			X,y = self.transform(X, y)
			self.X.append(X)
			self.y.append(y)
			
			
		self.size = len(self.X)
		if (summary):
			print('Count: ', self.size, ' Timesteps: ', self.timesteps, ' Features: ', self.features)
		self.X = self.pad_tensors(self.X)
		self.y = self.pad_tensors(self.y)

	def transform(self, X, y):
		self.features = X.shape[1]
		scaler = MinMaxScaler()
		X = scaler.fit_transform(X.reshape(X.shape[0], self.features))
		for i in range(len(y)):
			if y[i] == -1:
				y[i] = 0
		if (len(X) > self.timesteps):
			self.timesteps = len(X)
		X = X.reshape(1, len(X), self.features)
		y = y.reshape(1, len(y), 1)
		return X,y

	def pad_tensors(self, tensor):
		for i in range(len(tensor)):
			seq = tensor[i]
			npad = [(0,0), (0, self.timesteps-seq.shape[1]), (0,0)]
			seq = pad(seq, pad_width=npad, mode='constant', constant_values=0)
			tensor[i] = seq
		return tensor

	'''
	def load_example(self, count = 1000, timesteps = 80):
		for i in range(count):
			X,y = self.get_sequence(timesteps)
			self.X.append(X)
			self.y.append(y)
		self.timesteps = timesteps
		self.size = len(self.X)
		print('Size: ', self.size, ' Timesteps: ', self.timesteps)

	def get_sequence(self, timesteps):
		X = array([random() for _ in range(timesteps)])
		limit = timesteps/5.0
		y = zeros(timesteps)
		xsum = cumsum(X)
		for i in range(timesteps):
			x = xsum[i]
			if x > limit:
				#if x > limit*1.2:
				#	limit *= 3
				y[i] = 1
			else:
				y[i] = 0
		X = X.reshape(1, timesteps, 1)
		y = y.reshape(1, timesteps, 1)
		return X, y
	'''