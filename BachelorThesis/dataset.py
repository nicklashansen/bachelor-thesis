from numpy import *
from random import random, shuffle
from sklearn.model_selection import KFold
import random

class dataset:
	def __init__(self, epochs, shuffle=True, adjust_sleep=True, balance=False):
		self.epochs = epochs
		self.size = len(epochs)
		self.timesteps = epochs[0].timesteps
		self.features = epochs[0].features
		if balance:
			self.balance()
		if shuffle:
			self.shuffle()
		if adjust_sleep:
			self.adjust_sleep()

	def shuffle(self, seed = 22):
		random.Random(seed).shuffle(self.epochs)

	def balance(self):
		for _,obj in enumerate(self.epochs):
			if sum(obj.y) == 0:
				self.epochs.remove(obj)
		self.size = len(self.epochs)

	def adjust_sleep(self):
		index = self.features - 1
		for j,e in enumerate(self.epochs):
			for i,val in enumerate(transpose(e.X)[index]):
				self.epochs[j].X[i, index] = val + 1

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