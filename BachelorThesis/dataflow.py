from numpy import *
from sklearn.model_selection import KFold
from keras.models import load_model
from features import *
from epoch import *
from gru import *
from timeseries import *
from preprocessing import prep_X
from dataset import *
from log import *
import filesystem as fs
import epoch

"""
WRITTEN BY:
Nicklas Hansen
"""

epoch_length, overlap_factor, overlap_score, sample_rate = 120, 2, 10, 256

def fit_validate(gpu = True, balance = False, only_arousal = False):
	batch_size = 2 ** 11 if gpu else 2 ** 7
	model = fit(batch_size, balance, only_arousal)
	model.save()
	evaluate(model, validation)

def fit(batch_size, balance, only_arousal):
	data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal)
	model = gru(data, batch_size)
	model.fit(data.epochs)
	return model

def evaluate(model = None, validation = True):
	set = 1 if validation else 2
	if model is None:
		model = gru(load_graph=True)
	files = fs.load_splits()[set]
	results = validate(model, files)
	log_results(results, validation=validation)

def validate(model, files):
	TP=FP=TN=FN=0
	count = len(files)
	for file in files:
		try:
			tp, fp, tn, fn = validate_file(file, model, overlap_score, sample_rate)
			TP += tp ; FP += fp ; TN += tn ; FN += fn
		except Exception as e:
			print(e)
	return metrics.compute_cm_score(TP, FP, TN, FN)

def validate_file(file, model, overlap_score, sample_rate):
	y, yhat, timecol = predict_file(file, model)
	print(file, '--', int(sum(y)), int(sum(yhat)))
	TP, FP, TN, FN = metrics.cm_overlap(y, yhat, timecol, overlap_score, sample_rate)
	return TP, FP, TN, FN

def predict_file(filename, model, filter = False, removal = True):
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, epoch_length, overlap_factor, sample_rate, filter, removal)
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat, timecol = reconstruct(X, y, epochs)
	return y, yhat, timecol

def reconstruct(X, y, epochs):
	timecol = transpose(X)[0]
	yhat = zeros(y.size)
	for _,e in enumerate(epochs):
		index = where(timecol == e.index_start)[0][0]
		for i,val in enumerate(e.yhat):
			if val == 1:
				yhat[index + i] = val
	return yhat, timecol

def log_results(results, validation = True):
	filename = 'Validation' if validation else 'Evaluation'
	log = getLog(filename, echo=True)
	for k,d in results.items():
		log.print(str(k))
		for key,val in d.items():
			log.print(str(key)+':'+str(val))

def add_predictions(yhat1, yhat2):
	assert len(yhat1) == len(yhat2)
	for i in range(len(yhat1)):
		if yhat2[i] == 1:
			yhat1[i] = 1
	return yhat1

def dataflow(edf = 'C:\\Users\\nickl\\Source\\Repos\\a000de373e6449ea8c29d5622ccbfcc6\\BachelorThesis\\Files\\Data\\mesa\\polysomnography\\edfs\\mesa-sleep-2084.edf', anno = 'C:\\Users\\nickl\\Source\\Repos\\a000de373e6449ea8c29d5622ccbfcc6\\BachelorThesis\\Files\\Data\\mesa\\polysomnography\\annotations-events-nsrr\\mesa-sleep-2084-nsrr.xml'):
	path1 = 'gru.h5'
	path2 = 'gru-new.h5'

	#X = prep_X(edf, anno)
	X,y= fs.load_csv('mesa-sleep-2084')
	epochs = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=False)
	full = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=False)
	model = gru(load_graph=True, path=path1)
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	full.sort(key=lambda x: x.index_start, reverse=False)
	_, yhat, wake, rem, illegal = timeseries(epochs, full, epoch_length, overlap_factor, sample_rate)
	ill = region(illegal)
	ill.append([0, int(full[0].index_start/sample_rate)])

	epochs2 = epochs_from_prep(X, y, epoch_length, overlap_factor, filter = False, removal=False)
	model2 = gru(load_graph=True, path=path2)
	epochs2 = model2.predict(epochs2)
	epochs2.sort(key=lambda x: x.index_start, reverse=False)
	_, yhat2, wake, rem, illegal = timeseries(epochs2, full, epoch_length, overlap_factor, sample_rate)

	# Modellerne lægges sammen til plottet
	yhat = add_predictions(yhat, yhat2)

	# Reconstruction af yhat for hver model
	_yhat, timecol = reconstruct(X, y, epochs)
	_yhat2, timecol2 = reconstruct(X, y, epochs2)

	# Modellerne lægges sammen til metrics
	_yhat = add_predictions(_yhat, _yhat2)

	# For at få time axis
	X = transpose(X)

	# Tjek forskelle i sum, bare for sjov
	print('y = ', sum(y))
	print('_yhat = ', sum(_yhat))

	# Her bruges _yhat i stedet, da yhat til plottet er resampled
	TP, FP, TN, FN = metrics.cm_overlap(y, _yhat, timecol, overlap_score, sample_rate)
	results = metrics.compute_cm_score(TP, FP, TN, FN)
	print(results)

	# Plot
	plot_results(X[0]/sample_rate, [X[1], y], ['RR', 'arousals'], region(wake), None, None, region(yhat), int(full[-1].index_stop/sample_rate))
	return X[0]/sample_rate, [X[1]], ['RR'], region(wake), region(rem), ill, region(yhat), int(full[-1].index_stop/sample_rate)