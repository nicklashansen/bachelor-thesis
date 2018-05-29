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

def hours_of_sleep_files(files=[f for f in fs.getAllSubjectFilenames(preprocessed=True) if f not in fs.load_splits()[0]+fs.load_splits()[1]+fs.load_splits()[2]]):
	total = 0.0
	log = get_log("SleepHour", echo=True)
	for file in files:
		X,y = fs.load_csv(file)
		X,_,_ = make_features(X,y,settings.SAMPLE_RATE)
		timecol = transpose(X)[0]
		t = count_hours_of_sleep(timecol)
		log.print(file + ' -- {0} hours'.format(t))
		total += t
	log.printHL()
	log.print('total -- {0} hours'.format(total))
	return total

def count_hours_of_sleep(timecol): # Assumes sleep-removal
	t = 0.0
	j = 0
	for i in range(1, len(timecol)+1):
		if i == len(timecol) or (timecol[i] - timecol[i-1]) >= settings.SAMPLE_RATE * 5:
			t += timecol[i-1] - timecol[j] 
			j = i
	return t/settings.SAMPLE_RATE/60/60 # samples / simple_rate / 60 seconds / 60 minutes => amount in hours

def make_splits():
	before = fs.getAllSubjectFilenames(preprocessed=True)
	after = reliableFiles(before)
	tr,ev,te = train_vali_test_split(after)

def process_epochs():
	train = fs.load_splits()[0]
	epochs = compile_epochs(train)

def reliableFiles(files, ai_all5=10.0, overall5=4.0, slewake5=0.0, maskThreshold_all=0.1, maskTreshhold_single=0.05):
	log = get_log('Discard', echo=True)
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
		X,_,_ = sleep_removal_new(X,y,None,settings.SAMPLE_RATE)
		masklist, mask = make_masks(X)
		criteria += [sum(m)/len(m) <= maskTreshhold_single for m in masklist]
		criteria += [sum(mask)/len(mask) <= maskThreshold_all]

		sums = [sum(m)/len(m) for m in masklist]
		msum = sum(mask)/len(mask)

		return criteria

	reliable = [isReliable(fn) for fn in files]
	reliableFiles = [files[i] for i,r in enumerate(reliable) if all(r)]
	
	# Log Status
	arr = array(reliable)
	labels = ['ai_all5 ', 'overall5', 'slewake5', 'sum(y)=0', 'mask RR ', 'mask RPA', 'mask PTT', 'mask PWA', 'mask all']

	log.print('Preprocessed files:  {0}'.format(len(files)))
	log.print('Removed files:       {0}'.format(len(files) - len(reliableFiles)))
	log.print('Reliable files:      {0}'.format(len(reliableFiles)))
	for i,l in enumerate(labels):
		a = list(arr[:,i]).count(False)
		log.print('Removed by {0}: {1}'.format(l,a))
	log.printHL()
	for i,rl in enumerate(reliable):
		if not all(rl):
			log.print(files[i] + ' -- ' + ', '.join([labels[j] for j,r in enumerate(rl) if not r]))

	return reliableFiles

def train_vali_test_split(files, valisize=0.05, testsize=0.05, save=True, name=None):
	random.shuffle(files)
	va = int(len(files)*(1.0-valisize))
	tr = int(len(files)*(1.0-valisize-testsize))
	train,vali,test =  files[:tr], files[tr:va], files[va:]
	if save:
		fs.write_splits(train,vali,test,name='splits' + '' if not name else name)
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
		std4 = std(_data)*4
		# if datapoint's distance from mu is above 4 times the standard deviation
		# or value is -1 (i.e. no PTT value)
		return [(1 if x == -1 or abs(mu-x) > std4 else 0) for x in data]

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
			datacs = [d if mask[i] == 0 else datacs[i] for i,d in enumerate(data)]
			if plot:
				plot_data([data, datacs], labels=['Signal','Spline Correction'], indice=(0,30000))
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