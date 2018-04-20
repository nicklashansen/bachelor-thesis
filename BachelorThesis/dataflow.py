from numpy import *
from random import random, shuffle
from sklearn.model_selection import KFold
from features import *
from epoch import *
from gru import *
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
"""

def flow_all():
	log, clock = Log('Features', echo=True), stopwatch()
	files,_ = fs.getAllSubjectFilenames(preprocessed=True)
	log.print('Total files: {0}'.format(len(files)))
	clock.round()
	epochs = compile_epochs(files)
	log.print('Initiating dataflow...')
	dataflow(epochs)
	log.print('Successfully completed full dataflow.')

def compile_epochs(files):
	epochs = []
	for i, filename in enumerate(files):
		try:
			X,y = fs.load_csv(filename)
			X,y,mask = make_features(X, y)
			epochs.extend(get_epochs(X, y, mask))
		except Exception as e:
			continue
	return epochs

def dataflow(epochs):
	print('Generated a total of {0} epochs'.format(len(epochs)))
	data = dataset(epochs)
	train,test = data.holdout(0.9)
	print('train:', len(train), ' test:', len(test))
	n_cells = epochs[0].timesteps
	model = gru(data, n_cells)
	print('Fitting...')
	model.fit(train, 60)
	print('Evaluating...')
	score = model.evaluate(test)
	print(score)
	#print(metrics.compute_score(score, metrics.TPR_FNR).items())

class dataset:
	def __init__(self, epochs, shuffle=True):
		self.epochs = epochs
		self.size = len(epochs)
		self.timesteps = epochs[0].timesteps
		self.features = epochs[0].features
		if shuffle:
			self.shuffle_epochs()

	def shuffle_epochs(self):
		shuffle(self.epochs)

	def holdout(self, split = 0.67):
		cut = int(self.size * split)
		return self.epochs[:cut], self.epochs[cut:]

	def kfold(self, folds = 10):
		kf = KFold(folds)
		train, test = [], []
		for train_index, test_index in kf.split(self.epochs):
			train.append(self.epochs[train_index[0]:train_index[len(train_index)-1]])
			test.append(self.epochs[test_index[0]:test_index[len(test_index)-1]])
		return train, test