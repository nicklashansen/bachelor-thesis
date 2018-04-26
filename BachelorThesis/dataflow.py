from numpy import *
from random import random, shuffle
from sklearn.model_selection import KFold
from features import *
from epoch import *
from gru import *
import filesystem as fs
import epoch

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

def flow_fit():
	file = fs.Filepaths.SaveEpochs + 'epochs.pickle'
	with open(file, 'rb') as f:
		epochs = pck.load(f)
	return epochs

def flow_all():
	log, clock = Log('Features', echo=True), stopwatch()
	files = fs.getAllSubjectFilenames(preprocessed=True)
	log.print('Total files:         {0}'.format(len(files)))
	files = reliableFiles(files)
	log.print('Files after removal: {0}'.format(len(files)))
	clock.round()
	epochs = compile_epochs(files)
	log.print('Initiating dataflow...')
	clock.round()
	#dataflow(epochs)
	clock.stop()
	log.print('Successfully completed full dataflow.')

def reliableFiles(files):
	log = Log('Discard', echo=False)
	datasetCsv = fs.getDataset_csv()

	def isReliable(filename):
		mesaid = int(filename[-4:])
		filter = ['ai_all5','overall5','slewake5',]
		# ai_all5  = arousal index
		# overall5 = overall study quality 
		# slewake5 = poor quailty EEG (sleep stage)
		df = datasetCsv[datasetCsv['mesaid'] == mesaid][filter].iloc[0]
		criteria = [
			df[0] > 10.0,	# low ai index
			df[1] > 3.0,	# low overall quality
			df[2] == 0.0	# poor EEG (sleep stage scoring)
			]	
		return criteria

	reliable = [isReliable(fn) for fn in files]
	reliableFiles = [files[i] for i,r in enumerate(reliable) if all(r)]
	
	# Log Status
	arr = array(reliable)
	a = list(arr[:,0]).count(False)
	b = list(arr[:,1]).count(False)
	c = list(arr[:,2]).count(False)

	log.print('Total files:               {0}'.format(len(files)))
	log.print('Reliable files:            {0}'.format(len(reliableFiles)))
	log.print('Removed by ai_all5 > 10.0: {0}'.format(a))
	log.print('Removed by overall5 > 3.0: {0}'.format(b))
	log.print('Removed by slewake5 = 1.0: {0}'.format(c))
	log.print('-'*35)
	for fn in [f for f in files if f not in reliableFiles]:
		log.print(fn)

	return reliableFiles

def compile_epochs(files, save = True):
	log = Log('Epochs', True)

	epochs = []
	for i, filename in enumerate(files):
		try:
			X,y = fs.load_csv(filename)
			X,y,mask = make_features(X, y)
			eps = get_epochs(X, y, mask)
			log.print('{0} created {1} epochs'.format(filename, len(eps)))
			epochs.extend(eps)
			if save: #Save new version each step
				save_epochs(epochs)
		except Exception as e:
			log.print('{0} Exception: {1}'.format(filename, str(e)))
	if save:
		save_epochs(epochs)
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