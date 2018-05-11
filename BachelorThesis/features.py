from numpy import *
from filters import quantile_norm, median_filt
from scipy.interpolate import CubicSpline
from log import getLog
from stopwatch import stopwatch
from plots import plot_data
import epoch
import filesystem as fs

"""
WRITTEN BY
Michael Kirkegaard,
Nicklas Hansen
"""

def process_epochs():
	#files = fs.getAllSubjectFilenames(preprocessed=True)
	#files = reliableFiles(files)
	train = fs.load_splits()[0]
	#train, _, _ = train_test_eval_split(files) # = train,test,eval
	epochs = compile_epochs(train)

def reliableFiles(files):
	log = getLog('Discard', echo=False)
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

	log.print('Preprocessed files:        {0}'.format(len(files)))
	log.print('Reliable files:            {0}'.format(len(reliableFiles)))
	log.print('Removed by ai_all5 > 10.0: {0}'.format(a))
	log.print('Removed by overall5 > 3.0: {0}'.format(b))
	log.print('Removed by slewake5 = 1.0: {0}'.format(c))
	log.printHL()
	for fn in [f for f in files if f not in reliableFiles]:
		log.print(fn)

	return reliableFiles

def train_test_eval_split(files, testsize=0.05, evalsize=0.05):
	shuffle(files)
	te = int(len(files)*(1.0-testsize))
	tt = int(len(files)*(1.0-testsize-evalsize))
	train,test,eval =  files[:tt], files[tt:te], files[te:]
	fs.write_splits(train,test,eval)
	return train,test,eval

def compile_epochs(files, save = True):
	log = getLog('Epochs', True)

	log.print('Total files: {0}'.format(len(files)))
	log.printHL()

	p = int(len(files)/14)
	epochs = []
	for i, filename in enumerate(files):
		try:
			X,y = fs.load_csv(filename)
			eps = epochs_from_prep(X, y)
			epochs.extend(eps)
			log.print('{0} created {1} epochs'.format(filename, len(eps)))
			if save and i > 0 and i % p == 0: # Backup saves
				epoch.save_epochs(epochs)
				log.printHL()
				log.print('Backup save of {0} epochs'.format(len(epochs)))
				log.printHL()
		except Exception as e:
			log.print('{0} Exception: {1}'.format(filename, str(e)))
	if save:
		epoch.save_epochs(epochs)
		log.printHL()
		log.print('Final save of {0} epochs'.format(len(epochs)))
		log.printHL()
	return epochs

def epochs_from_prep(X, y, epoch_length=epoch.EPOCH_LENGTH, overlap_factor=epoch.OVERLAP_FACTOR, sample_rate = epoch.SAMPLE_RATE, filter = True, removal = True):
	X,y,mask = make_features(X, y, sample_rate, removal)
	return epoch.get_epochs(X, y, mask, epoch_length, overlap_factor, filter)

def make_features(X, y, sample_rate, removal = True):
	masklist, mask = make_masks(X)
	X = cubic_spline(X, masklist)
	if removal:
		X,y,mask = sleep_removal(X, y, mask, sample_rate)
	X = median_filt(X)
	X = quantile_norm(X, 10)
	return X, y, mask

def make_masks(X):
	Xt = transpose(X)

	# Outlier detection
	def threeSigmaRule(data):
		_data = [x for x in data if x != -1]
		mu = mean(_data)
		std3 = std(_data)*3
		# if datapoint's distance from mu is above 3 times the standard deviation
		# or value is -1 (i.e. no PTT value)
		return [(1 if x == -1 or abs(mu-x) > std3 else 0) for x in data]

	masklist = [threeSigmaRule(x_feat) for x_feat in Xt[1:5]] # DR, RPA, PTT, PWA
	mask = [sum(tub) for tub in zip(*masklist)]

	return masklist, mask

def cubic_spline(X, masks, plot=False):
	Xt = transpose(X)

	def spline(maskid, data):
		mask = masks[maskid]
		datamask = [data[i] for i,m in enumerate(mask) if m == 0]
		if(len(datamask) >= 2):
			cs = CubicSpline(range(len(datamask)),datamask)
			datacs = cs(range(len(mask)))
			if plot:
				plot_data([data, datacs], labels=['Signal','Spline Correction'])
			return array(datacs)
		return data

	Xt = array([Xt[0]] + [spline(id,x) for id,x in enumerate(Xt[1:5])] + [Xt[5]]) # Spline DR,RPA,PTT,PWA
	X = transpose(Xt)
	return X

#def sleep_removal(X, y, mask, sample_rate):
#	_X = transpose(X)
#	keep = [i for i,state in enumerate(_X[5]) if state >= 0]
#	X = array([X[i] for i in keep])
#	if y is not None:
#		y = array([y[i] for i in keep])
#	if mask is not None:
#		mask = [mask[i] for i in keep]
#	return X,y, mask

def sleep_removal(X, y, mask, sample_rate):
	Xt = transpose(X)
	indexes = Xt[0]
	sleep = Xt[5]
	n = len(sleep)

	keep = []
	i = 0
	while(i < n):
		k = None
		j = i+1
		if sleep[i] >= 0:
			while(j < n and sleep[j] >= 0):
				j += 1
		else:
			while(j < n and sleep[j] == -1 and indexes[j]-indexes[i] < 30*sample_rate):
				j += 1
			if j < n and sleep[j] == -1:
				cuti = j
				cutj = i
				while(j < n and sleep[j] == -1):
					j += 1
					cutj += 1 
				if(cutj - cuti > 0):
					k = list(range(i,cuti)) + list(range(cutj,j))
		if not k:
			k = list(range(i,j))
		keep += k
		i = j

	X = array([X[i] for i in keep])
	if y is not None:
		y = array([y[i] for i in keep])
	if mask is not None:
		mask = [mask[i] for i in keep]
	return X,y,mask