from numpy import *
from random import random, shuffle
from sklearn.model_selection import KFold
import random

SEED = 22

class dataset:
	def __init__(self, epochs, shuffle=True, balance=False, only_arousal=False, exclude_ptt=False):
		self.epochs = epochs
		self.size = len(epochs)
		self.timesteps = epochs[0].timesteps
		self.features = epochs[0].features
		if only_arousal:
			self.only_arousal()
		elif balance:
			self.balance()
		if shuffle:
			self.shuffle()
		if exclude_ptt:
			self.exclude_ptt()
		

	def shuffle(self, seed = SEED):
		random.Random(seed).shuffle(self.epochs)

	def shuffle_list(self, list, seed = SEED):
		random.Random(seed).shuffle(list)
		return list

	def balance(self):
		list = []
		for _,obj in enumerate(self.epochs):
			if sum(obj.y) == 0:
				self.epochs.remove(obj)
				list.append(obj)
		list = self.shuffle_list(list)[:len(self.epochs)]
		self.epochs.extend(list)
		self.size = len(self.epochs)

	def only_arousal(self):
		for _,obj in enumerate(self.epochs):
			if sum(obj.y) == 0:
				self.epochs.remove(obj)
		self.size = len(self.epochs)
	
	#def exclude_ptt(self): # ONLY RR
	#	for i,obj in enumerate(self.epochs):
	#		obj.X = delete(obj.X, 1, 1)
	#		obj.X = delete(obj.X, 1, 1)
	#		obj.X = delete(obj.X, 1, 1)
	#		self.epochs[i].X = obj.X
	#	self.features = self.epochs[0].X.shape[1]

	def exclude_ptt(self): # ONLY RWA
		for i,obj in enumerate(self.epochs):
			obj.X = delete(obj.X, 0, 1)
			obj.X = delete(obj.X, 1, 1)
			obj.X = delete(obj.X, 1, 1)
			self.epochs[i].X = obj.X
		self.features = self.epochs[0].X.shape[1]

	#def exclude_ptt(self): # ONLY PTT
	#	for i,obj in enumerate(self.epochs):
	#		obj.X = delete(obj.X, 0, 1)
	#		obj.X = delete(obj.X, 0, 1)
	#		self.epochs[i].X = obj.X
	#	self.features = self.epochs[0].X.shape[1]

	#def exclude_ptt(self): # OVLY ECG
	#	for i,obj in enumerate(self.epochs):
	#		obj.X = delete(obj.X, 2, 1)
	#		obj.X = delete(obj.X, 2, 1)
	#		self.epochs[i].X = obj.X
	#	self.features = self.epochs[0].X.shape[1]

	def get_split(self):
		split = 0.9
		if self.size * split >= 10 ** 5:
			return (self.size - 2.5 * (10 ** 4)) / self.size
		return split

	def holdout(self, split = 0.9):
		cut = int(self.size * split)
		return self.epochs[:cut], self.epochs[cut:]

	def kfold(self, folds = 10):
		kf = KFold(folds)
		train, test = [], []
		for train_index, test_index in kf.split(self.epochs):
			train.append(self.epochs[train_index[0]:train_index[len(train_index)-1]])
			test.append(self.epochs[test_index[0]:test_index[len(test_index)-1]])
		return train, test