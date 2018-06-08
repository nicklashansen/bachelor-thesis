'''
AUTHOR(S):
NICKLAS HANSEN

Module is responsible for data-set manipulation using a dedicated object.
'''

from numpy import *
from random import random, shuffle
from sklearn.model_selection import KFold
import random

SEED = 22

class dataset:
	'''
	Class responsible for handling data-set related activities.
	'''
	def __init__(self, epochs, shuffle=True, balance=False, only_arousal=False, exclude_ptt=False, only_rwa=False):
		'''
		Creates a new data-set object from given array of epochs.
		Unless disabled, all epochs are shuffled using a fixed seed for reproducibility.
		Optionally removes epochs that do not contain arousals.
		Optionally removes PPG features from the feature matrix of each epoch.
		'''
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
		if only_rwa:
			self.only_rwa()

	def shuffle(self, seed = SEED):
		'''
		Shuffles the list of epochs in-place.
		'''
		random.Random(seed).shuffle(self.epochs)

	def shuffle_list(self, list, seed = SEED):
		'''
		Shuffles a list passed as argument.
		'''
		random.Random(seed).shuffle(list)
		return list

	def balance(self):
		'''
		Legacy function that balances the data-set such that the number of epochs with and without arousals is equal.
		'''
		list = []
		for _,obj in enumerate(self.epochs):
			if sum(obj.y) == 0:
				self.epochs.remove(obj)
				list.append(obj)
		list = self.shuffle_list(list)[:len(self.epochs)]
		self.epochs.extend(list)
		self.size = len(self.epochs)

	def only_arousal(self):
		'''
		Removes all epochs that do not contain arousals from the data-set. Useful for data-trimming.
		'''
		for _,obj in enumerate(self.epochs):
			if sum(obj.y) == 0:
				self.epochs.remove(obj)
		self.size = len(self.epochs)

	def exclude_ptt(self):
		'''
		Excludes the PPG features by matrix operations.
		'''
		for i,obj in enumerate(self.epochs):
			obj.X = delete(obj.X, 2, 1)
			obj.X = delete(obj.X, 2, 1)
			obj.X = delete(obj.X, 0, 1)
			self.epochs[i].X = obj.X
		self.features = self.epochs[0].X.shape[1]

	def only_rwa(self):
		'''
		Includes only RWA feature with matrix operations
		'''
		for i,obj in enumerate(self.epochs):
			obj.X = delete(obj.X, 0, 1)
			obj.X = delete(obj.X, 2, 1)
			obj.X = delete(obj.X, 2, 1)
			self.epochs[i].X = obj.X
		self.features = self.epochs[0].X.shape[1]