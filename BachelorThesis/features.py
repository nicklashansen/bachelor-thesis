from numpy import *
from filters import quantile_norm, median_filt
from scipy.interpolate import CubicSpline
from log import get_log
from stopwatch import stopwatch
from plots import plot_data
import epoch
import settings
import filesystem as fs

"""
WRITTEN BY
Michael Kirkegaard,
Nicklas Hansen
"""

GOOD = ['mesa-sleep-0064','mesa-sleep-2685','mesa-sleep-2260','mesa-sleep-6176','mesa-sleep-6419','mesa-sleep-5221','mesa-sleep-3267','mesa-sleep-2821','mesa-sleep-4588']
BAD = ['mesa-sleep-1035','mesa-sleep-6614','mesa-sleep-5339','mesa-sleep-4379','mesa-sleep-4178','mesa-sleep-5734','mesa-sleep-0718','mesa-sleep-3692','mesa-sleep-2472']

def test_reliableFiles():
	before = fs.getAllSubjectFilenames(preprocessed=True)
	#before = GOOD + BAD
	after = reliableFiles(before)
	#tr,ev,te = train_vali_test_split(after, save=False)
	None

def process_epochs():
	train = fs.load_splits()[0]
	epochs = compile_epochs(train)

def reliableFiles(files, ai_all5=15.0, overall5=4.0, slewake5=0.0, maskThreshold_all=1/16, maskTreshhold_single=1/32):
	log = get_log('Discard', echo=False)
	datasetCsv = fs.getDataset_csv()

	def isReliable(filename):
		# Target file
		mesaid = int(filename[-4:])
		X,y = fs.load_csv(filename)
		criteria = []

		# MESA variables
		filter = ['ai_all5','overall5','slewake5',]
		# ai_all5  = arousal index
		# overall5 = overall study quality 
		# slewake5 = poor quailty EEG (sleep stage)
		df = datasetCsv[datasetCsv['mesaid'] == mesaid][filter].iloc[0]
		criteria += [
			df[0] >= ai_all5,	# low ai index (events per hour)
			df[1] >= overall5,	# low overall quality
			df[2] == slewake5,	# poor EEG (sleep stage scoring)
			]

		# Doublecheck arousals
		criteria += [sum(y) > 0]

		# Mask threshhold
		X,_,_ = sleep_removal_new(X,y,None,256) # TODO: REMOVE HARDCODE -----------------------------
		masklist, mask = make_masks(X)
		criteria += [sum(m)/len(m) <= maskTreshhold_single for m in masklist]
		criteria += [sum(mask)/len(mask) <= maskThreshold_all]

		sums = [sum(m)/len(m) for m in masklist]
		msum = sum(mask)/len(mask)

		if filename in ['mesa-sleep-1035','mesa-sleep-3267','mesa-sleep-2821','mesa-sleep-5734']:
			None
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

def train_vali_test_split(files, valisize=0.05, testsize=0.05, save=True):
	random.shuffle(files)
	va = int(len(files)*(1.0-valisize))
	tr = int(len(files)*(1.0-valisize-testsize))
	train,vali,test =  files[:tr], files[tr:va], files[va:]
	if save:
		fs.write_splits(train,vali,test)
	return train,vali,test

def compile_epochs(files, save = True):
	log = get_log('Epochs', True)

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

def epochs_from_prep(X, y, epoch_length=settings.EPOCH_LENGTH, overlap_factor=settings.OVERLAP_FACTOR, sample_rate = settings.SAMPLE_RATE, filter = True, removal = True):
	X,y,mask = make_features(X, y, sample_rate, removal)
	return epoch.get_epochs(X, y, mask, epoch_length, overlap_factor, filter)

def make_features(X, y, sample_rate, removal = True, old_removal = False, onehot=True):
	masklist, mask = make_masks(X)
	X = cubic_spline(X, masklist)
	if removal:
		if not old_removal:
			X,y,mask = sleep_removal_new(X, y, mask, sample_rate)
		else: 
			X,y,mask = sleep_removal_old(X, y, mask, sample_rate)
	#X = median_filt(X)
	X = quantile_norm(X, 1000)
	if onehot:
		X = sleep_onehot(X)
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

	masklist = [threeSigmaRule(x_feat) for x_feat in Xt[1:5]] # RR, RPA, PTT, PWA
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

def sleep_removal_old(X, y, mask, sample_rate):
	_X = transpose(X)
	keep = [i for i,state in enumerate(_X[5]) if state >= 0]
	X = array([X[i] for i in keep])
	if y is not None:
		y = array([y[i] for i in keep])
	if mask is not None:
		mask = [mask[i] for i in keep]
	return X,y, mask

def sleep_removal_new(X, y, mask, sample_rate):
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

def sleep_onehot(X):
	Xt = transpose(X)

	def onehot_arr(data):
		n = len(data)
		wake,nrem,rem = zeros(n), zeros(n), zeros(n) 
		arr = [wake,nrem,rem]

		for i,val in enumerate(data):
			arr[int(val+1)][i] = 1

		return [wake,nrem,rem]


	Xt = array([f for f in Xt[0:5]] + onehot_arr(Xt[5])) # Spline DR,RPA,PTT,PWA
	return transpose(Xt)

if __name__ == '__main__':
	test_reliableFiles()