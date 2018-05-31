'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

Finilise preprocessed files into feature matrix X and score vector y,
filter off unreliable files taking MESA variable and masks as criteria
and serve as wrapper for generating epochs used for model training.
'''
from numpy import *
from filters import quantile_norm, median_filt
from scipy.interpolate import CubicSpline
from log import get_log
from stopwatch import stopwatch
from plots import plot_data
import epoch
import settings
import filesystem as fs

def hours_of_sleep_files(files):
	'''
	Calculated the hours of sleep for a list of files,
	the results are logged for testing purposes.
	'''
	total = 0.0
	log = get_log("SleepHour", echo=True)
	for file in files:
		X,_ = fs.load_csv(file)
		X,_,_ = wake_removal_endpoints(X,None,None,settings.SAMPLE_RATE)
		t = count_hours_of_sleep(transpose(X)[0])
		log.print(file + ' -- {0} hours'.format(t))
		total += t
	log.printHL()
	log.print('total -- {0} hours'.format(total))
	return total
 
def count_hours_of_sleep(timecol):
	'''
	calculates hours of sleep from time series assuming wake periods are removed
	'''
	t = 0.0
	j = 0
	for i in range(1, len(timecol)+1):
		# finds the duration of all non-seperated periods
		if i == len(timecol) or (timecol[i] - timecol[i-1]) >= settings.SAMPLE_RATE * 5:
			t += timecol[i-1] - timecol[j] 
			j = i
	return t/settings.SAMPLE_RATE/60/60 # samples / simple_rate / 60 seconds / 60 minutes => amount in hours

def make_splits():
	'''
	split files into training, evaluation and testing after trimming away unreliable files
	'''
	before = fs.getAllSubjectFilenames(preprocessed=True)
	after = reliableFiles(before)
	tr,ev,te = train_vali_test_split(after)

def process_epochs():
	'''
	loads training data and generated epoch file
	'''
	train = fs.load_splits()[0]
	epochs = compile_epochs(train)


def reliableFiles(files, ai_all5=10.0, overall5=4.0, slewake5=0.0, maskThreshold_all=0.1, maskTreshhold_single=0.05):
	'''
	Given a list of files, this method returns only the reliable files
	as determined by MESA variable criteria cuts and mask thresholds.
	results are logged for testing and evaluation purposes.
	'''
	log = get_log('Discard', echo=True)
	datasetCsv = fs.getDataset_csv()

	# determines list of booleans, one for each critera
	def isReliable(filename):
		# Target file
		mesaid = int(filename[-4:])
		X,y = fs.load_csv(filename)
		criteria = []

		# MESA variables
		filter = ['ai_all5','overall5','slewake5',]
		df = datasetCsv[datasetCsv['mesaid'] == mesaid][filter].iloc[0]
		criteria += [
			df[0] >= ai_all5,	# low ai index (events per hour)
			df[1] >= overall5,	# low overall quality
			df[2] == slewake5,	# poor EEG (no sleep stage / arousal scoring)
			]

		# Doublecheck arousals
		criteria += [sum(y) > 0]

		# Mask threshhold
		X,_,_ = wake_removal_endpoints(X,None,None,settings.SAMPLE_RATE)
		masklist, mask = make_masks(X)
		criteria += [sum(m)/len(m) <= maskTreshhold_single for m in masklist]
		criteria += [sum(mask)/len(mask) <= maskThreshold_all]

		return criteria

	# Extract criteria arrays for all files, and filters fileslist
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
	'''
	Randomly splits a set of files into training, validation and test data partitions
	'''
	random.shuffle(files)
	te = int(len(files)*(1.0-testsize))
	tr = int(len(files)*(1.0-valisize-testsize))
	train,vali,test =  files[:tr], files[tr:te], files[te:]
	if save:
		fs.write_splits(train,vali,test,name='splits' + '' if not name else name)
	return train,vali,test

def compile_epochs(files, save = True):
	'''
	Compiles a single list of all epochs from all files given. this
	can be stored in a file for later use in modeltraining. Process,
	errors and amounts are logged for test and evaluations purposes.
	'''
	# initilise log
	log = get_log('Epochs', True)
	log.print('Total files: {0}'.format(len(files)))
	log.printHL()

	# run through list with try/catch in case of errors
	epochs = []
	for i, filename in enumerate(files):
		try:
			X,y = fs.load_csv(filename)
			eps = epochs_from_prep(X, y)
			epochs.extend(eps)
			log.print('{0} created {1} epochs'.format(filename, len(eps)))
			# backup saves if saveing is optionally on
			if save and i > 0 and i % int(len(files)/14) == 0:
				epoch.save_epochs(epochs)
				log.printHL()
				log.print('Backup save of {0} epochs'.format(len(epochs)))
				log.printHL()
		except Exception as e:
			log.print('{0} Exception: {1}'.format(filename, str(e)))
	# optionally stores the epochs
	if save:
		epoch.save_epochs(epochs)
		log.printHL()
		log.print('Final save of {0} epochs'.format(len(epochs)))
		log.printHL()
	return epochs

def epochs_from_prep(X, y, epoch_length=settings.EPOCH_LENGTH, overlap_factor=settings.OVERLAP_FACTOR, sample_rate = settings.SAMPLE_RATE, filter = True, removal = True):
	'''
	Finalises features and generates epochs
	'''
	X,y,mask = make_features(X, y, sample_rate, removal)
	return epoch.get_epochs(X, y, mask, epoch_length, overlap_factor, filter)


def make_features(X, y, sample_rate, removal = True, full_removal = False, onehot=True):
	'''
	compiles the final features, removing wake periods and correcting extreme outliers
    and invalid values as well as normalising featues and onehotting sleep stage signal
	'''
	# Mask are created
	masklist, mask = make_masks(X)
	
	# datapoints corrected
	X = cubic_spline(X, masklist)
	
	# wake periods optionally removed
	if removal:
		# removed either by full or by keeping first and last 30 seconds
		if not full_removal:
			X,y,mask = wake_removal_endpoints(X, y, mask, sample_rate)
		else: 
			X,y,mask = wake_removal_full(X, y, mask, sample_rate)

	# Normalisation to counter inter-patient variablity
	X = quantile_norm(X, 1000)

	# converts sleep stage from a single signal into wake, nrem and rem
	if onehot:
		X = sleep_onehot(X)

	return X, y, mask

def make_masks(X):
	'''
	Creates binary error mask using three sigma rule outlier detection
	'''
	Xt = transpose(X)

	# Outlier detection on each signal not including invalid ones
	def threeSigmaRule(data):
		_data = [x for x in data if x != -1]
		mu = mean(_data)
		std3 = std(_data)*3
		# if datapoint's distance from mu is above 3 times the standard deviation
		# or value is -1 (i.e. no PTT value)
		return [(1 if x == -1 or abs(mu-x) > std3 else 0) for x in data]

	# compiles mask for each file, and a summarised mask for X
	masklist = [threeSigmaRule(x_feat) for x_feat in Xt[1:5]] # RR, RPA, PTT, PWA
	mask = [sum(tub) for tub in zip(*masklist)]

	return masklist, mask

def cubic_spline(X, masks, plot=False):
	'''
	Spline corrects featues where mask index shows errors
	'''
	Xt = transpose(X)

	def spline(maskid, data):
		# datapoint of non-error indexes
		mask = masks[maskid]
		datamask = [data[i] for i,m in enumerate(mask) if m == 0]
		if(len(datamask) >= 2):
			# Creates spline and interpolates points inbetween 
			cs = CubicSpline(range(len(datamask)),datamask)
			datacs = cs(range(len(mask)))
			
			# plottin is an option
			if plot:
				plot_data([data, datacs], labels=['Signal','Spline Correction'])
			return array(datacs)
		return data

	# Spline for DR,RPA,PTT,PWA
	Xt = array([Xt[0]] + [spline(id,x) for id,x in enumerate(Xt[1:5])] + [Xt[5]])
	X = transpose(Xt)
	return X

def wake_removal_full(X, y, mask, sample_rate):
	'''
	Legacy function - removes all wake state datapoints from X
	'''
	_X = transpose(X)
	keep = [i for i,state in enumerate(_X[5]) if state >= 0]
	X = array([X[i] for i in keep])
	if y is not None:
		y = array([y[i] for i in keep])
	if mask is not None:
		mask = [mask[i] for i in keep]
	return X,y, mask


def wake_removal_endpoints(X, y, mask, sample_rate, keep_seconds=30):
	'''
	removes all wake state regions greater than 60 seconds except for the
	first and last 30 seconds of the region. Mesa and shhs are scored in
	30 seconds windows, which keeps first and last windows of a wake region.
	'''
	# extract timestep indexes and sleep values
	Xt = transpose(X)
	indexes = Xt[0]
	sleep = Xt[5]
	n = len(sleep)

	# loop through timesteps and sleep signal
	keep = []
	i = 0
	while(i < n):
		k = None
		j = i+1
		# find wake period from i to j 
		if sleep[i] >= 0:
			while(j < n and sleep[j] >= 0):
				j += 1
		else:
			# use timestep indexes to add 30 seconds to j or discover sleep regions is less than 30
			while(j < n and sleep[j] == -1 and indexes[j]-indexes[i] < keep_seconds*sample_rate):
				j += 1
			if j < n and sleep[j] == -1:
				# save range i:cuti for first 30 seconds
				cuti = j # cuti = i + 30 seconds
				cutj = i # cutj = j - 30 seconds
				# find end of wake period
				while(j < n and sleep[j] == -1):
					j += 1
					cutj += 1
				# if wake period is longer than 60 seconds, dont add region between cuti and cutj
				if(cutj - cuti > 0):
					k = list(range(i,cuti)) + list(range(cutj,j))
		if not k: # add period i:j
			k = list(range(i,j))
		keep += k
		i = j

	# filters X,y and mask by index in keep
	X = array([X[i] for i in keep])
	if y is not None:
		y = array([y[i] for i in keep])
	if mask is not None:
		mask = [mask[i] for i in keep]
	return X,y,mask

def sleep_onehot(X):
	'''
	one hot for sleep stage, transforming signal of {-1,0,1} into three binary arrays
	'''
	Xt = transpose(X)

	# onehot
	def onehot_arr(data):
		n = len(data)
		wake,nrem,rem = zeros(n), zeros(n), zeros(n) 
		arr = [wake,nrem,rem]

		# add binary value to sleep signal {-1,0,1} to index in arr[3] {0,1}
		for i,val in enumerate(data):
			arr[int(val+1)][i] = 1

		return [wake,nrem,rem]

	# Xt = timecol, DR,RPA,PTT,PWA,wake,nrem,rem
	Xt = array([f for f in Xt[0:5]] + onehot_arr(Xt[5])) 
	return transpose(Xt)