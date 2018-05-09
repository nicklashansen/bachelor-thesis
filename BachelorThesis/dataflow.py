from numpy import *
from random import random, shuffle
from sklearn.model_selection import KFold
from keras.models import load_model
from features import *
from epoch import *
from gru import *
from preprocessing import prepSingle
import random
import filesystem as fs
import epoch

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

epoch_length, overlap_factor, overlap_score, sample_rate = 120, 2, 3, 256

def test():
	files = fs.load_splits()[1]
	TP=FP=TN=FN=0
	for file in files:
		try:
			y, yhat = predict_file(file)
			tp, fp, tn, fn = metrics.cm_overlap(y, yhat, overlap_score, sample_rate)
			TP += tp
			FP += fp
			TN += tn
			FN += fn
		except Exception as e:
			print(e)
	results = metrics.compute_score(y, yhat)

def predict_file(filename):
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=True)
	model = gru(dataset(epochs))
	model.graph = load_model('gru.h5')
	epochs = model.predict(epochs)
	full = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=False)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	full.sort(key=lambda x: x.index_start, reverse=False)
	y, yhat, wake, rem, illegal = timeseries(epochs, full, epoch_length, overlap_factor, 1)
	return y, yhat

def dataflow(filename = 'mesa-sleep-2084'):
	#X,y = prepSingle(filename, save=False)
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=True)
	full = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=False)
	model = gru(dataset(epochs))
	model.graph = load_model('gru.h5')
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	full.sort(key=lambda x: x.index_start, reverse=False)
	ya, yhat, wake, rem, illegal = timeseries(epochs, full, epoch_length, overlap_factor, sample_rate)

	results = metrics.compute_score(ya, yhat)['cm_overlap']
	del results['mcc']
	del results['precision']
	del results['specificity']
	del results['TP_FP_TN_FN']
	del results['accuracy']
	print(results)

	print('Evaluated from', int(epochs[0].index_start/sample_rate), ' s to', int(epochs[-1].index_stop/sample_rate), 's')
	ill = region(illegal)
	ill.append([0, int(full[0].index_start/sample_rate)])
	X = transpose(X)
	plot_results(X[0]/sample_rate, [X[1], y], ['RR', 'arousal'], region(wake), region(rem), ill, region(yhat), int(full[-1].index_stop/sample_rate))

def timeseries(epochs, full, epoch_length, overlap_factor, sample_rate):
	window = int(epoch_length - ( epoch_length / overlap_factor))
	length = int(full[-1].index_stop/sample_rate)
	y, yhat, wake, rem, illegal = zeros(length), zeros(length), zeros(length), zeros(length), zeros(length)
	for i,obj in enumerate(epochs):
		y = modify_timeseries(y, obj.y, 1, obj.timecol, window, sample_rate)
		yhat = modify_timeseries(yhat, obj.yhat, 1, obj.timecol, window, sample_rate)
	for i,obj in enumerate(full):
		sleep = transpose(obj.X)[-1]
		wake = modify_timeseries(wake, sleep, -1, obj.timecol, window, sample_rate)
		rem = modify_timeseries(rem, sleep, 1, obj.timecol, window, sample_rate)
		illegal = modify_timeseries(illegal, obj.mask, 1, obj.timecol, window, sample_rate)
	for i in range(len(wake)):
		#if yhat[i] == 1 and wake[i] == 1:
		#	yhat[i] = 0
		if illegal[i] == 1 and wake[i] == 1:
			illegal[i] = 0
	return y, yhat, wake, rem, illegal

def modify_timeseries(ts, values, criteria, timecol, window, sample_rate):
	for i,y in enumerate(values[window:]):
		enum = [int(timecol[window+i-3]/sample_rate),int(timecol[window+i]/sample_rate)]
		if y == criteria:
			for j in range(enum[0],enum[1]):
				ts[j] = 1
	return ts

def region(array):
	regions, start, bin = [], 0, False
	for i,val in enumerate(array):
		if val == 1:
			if not bin:
				start, bin = i, True
		elif bin:
			bin = False
			regions.append([start, i-1])
	if bin:
		regions.append([start, i-1])
	return regions

def fit_eval(gpu = True):
	batch_size = 2 ** 10 if gpu else 2 ** 7
	model, test = fit(batch_size)
	model.save()
	#results = eval(model, test)

def fit(batch_size = 100):
	data = dataset(fs.load_epochs(), balance = False)
	#fs.write_epochs(data.epochs, 'epochs_new')
	train,test = data.epochs, []
	#train,test = data.holdout(data.get_split())
	model = gru(data, batch_size)
	model.fit(train)
	return model, test

#def eval():
#	model = gru(dataset(epochs))
#	model.graph = load_model('gru.h5')
#	results = model.evaluate(epochs)
#	print(results)
#	return results

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