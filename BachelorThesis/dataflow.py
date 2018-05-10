from numpy import *
from sklearn.model_selection import KFold
from keras.models import load_model
from features import *
from epoch import *
from gru import *
from timeseries import *
from preprocessing import prepSingle
from dataset import *
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
	#files = ['mesa-sleep-2084']
	TP=FP=TN=FN=0
	model = None
	for file in files[:5]:
		y, yhat, timecol, model = predict_file(file, model)
		print(sum(y), sum(yhat))
		tp, fp, tn, fn = metrics.cm_overlap(y, yhat, overlap_score, sample_rate)
		TP += tp
		FP += fp
		TN += tn
		FN += fn
	results = metrics.compute_cm_score(TP, FP, TN, FN)['cm_overlap']
	del results['mcc']
	del results['specificity']
	del results['TP_FP_TN_FN']
	del results['accuracy']
	print(results)

def predict_file(filename, model):
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=False)
	if not model:
		model = gru(dataset(epochs))
		model.graph = load_model('gru.h5')
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat, timecol = reconstruct(X, y, epochs)
	return y, yhat, timecol, model

def reconstruct(X, y, epochs):
	timecol = transpose(X)[0]
	yhat, t = zeros(y.size), zeros(y.size)
	for _,e in enumerate(epochs):
		index = where(timecol == e.index_start)[0][0]
		for i,val in enumerate(e.yhat):
			yhat[index + i] = val
			t[index + i] = index
	return yhat, t

# god fil: 'mesa-sleep-2084'
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

	print('Evaluated from', int(epochs[0].index_start/sample_rate), 's to', int(epochs[-1].index_stop/sample_rate), 's')
	ill = region(illegal)
	ill.append([0, int(full[0].index_start/sample_rate)])
	X = transpose(X)
	plot_results(X[0]/sample_rate, [X[1], y], ['RR', 'arousal'], region(wake), region(rem), ill, region(yhat), int(full[-1].index_stop/sample_rate))

def epochs_from_prep(X, y, epoch_length=epoch_length, overlap_factor=overlap_factor, sample_rate = sample_rate, filter = True, removal = True):
	X,y,mask = make_features(X, y, sample_rate, removal)
	return get_epochs(X, y, mask, epoch_length, overlap_factor, filter)

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


### MICHAEL

def train_test_eval_split(files, testsize=0.05, evalsize=0.05):
	shuffle(files)
	te = int(len(files)*(1.0-testsize))
	tt = int(len(files)*(1.0-testsize-evalsize))
	train,test,eval =  files[:tt], files[tt:te], files[te:]
	fs.write_splits(train,test,eval)
	return train,test,eval

def process_epochs():
	files = fs.getAllSubjectFilenames(preprocessed=True)
	files = reliableFiles(files)
	train, _, _ = train_test_eval_split(files) # = train,test,eval
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

def compile_epochs(files, save = True):
	log = getLog('Epochs', True)

	log.print('Total files: {0}'.format(len(files)))
	log.printHL()

	p = int(len(files)/15)
	epochs = []
	for i, filename in enumerate(files):
		try:
			X,y = fs.load_csv(filename)
			eps = epochs_from_prep(X, y)
			epochs.extend(eps)
			log.print('{0} created {1} epochs'.format(filename, len(eps)))
			if save and i > 0 and i % p == 0: # Backup saves
				save_epochs(epochs)
				log.printHL()
				log.print('Backup save of {0} epochs'.format(len(epochs)))
				log.printHL()
		except Exception as e:
			log.print('{0} Exception: {1}'.format(filename, str(e)))
	if save:
		save_epochs(epochs)
		log.printHL()
		log.print('Final save of {0} epochs'.format(len(epochs)))
		log.printHL()
	return epochs