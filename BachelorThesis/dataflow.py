from numpy import *
from sklearn.model_selection import KFold
from keras.models import load_model
from features import *
from epoch import *
from gru import *
from timeseries import *
from preprocessing import prepSingle
from dataset import *
from log import *
import filesystem as fs
import epoch

"""
WRITTEN BY:
Nicklas Hansen
"""

epoch_length, overlap_factor, overlap_score, sample_rate = 120, 2, 3, 256

def fit_validate(gpu = True, balance = False, only_arousal = False):
	batch_size = 2 ** 11 if gpu else 2 ** 7
	model = fit(batch_size, balance, only_arousal)
	model.save()
	files = fs.load_splits()[1]
	results = validate(model, files, balance, only_arousal)
	log_results(results)

def log_results(results, validation = True):
	filename = 'Validation' if validation else 'Evaluation'
	log = getLog(filename, echo=True)
	for k,d in results.items():
		log.print(str(k))
		for key,val in d.items():
			log.print(str(key)+':'+str(val))

def fit(batch_size, balance, only_arousal):
	data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal)
	model = gru(data, batch_size)
	model.fit(data.epochs)
	return model

def validate(model, files, balance, only_arousal):
	TP=FP=TN=FN=0
	count = len(files)
	for file in files:
		tp, fp, tn, fn = validate_file(file, model, overlap_score, sample_rate)
		TP += tp ; FP += fp ; TN += tn ; FN += fn
	return metrics.compute_cm_score(TP, FP, TN, FN)

def validate_file(file, model, overlap_score, sample_rate):
	y, yhat, timecol = predict_file(file, model)
	print(file, '--', int(sum(y)), int(sum(yhat)))
	TP, FP, TN, FN = metrics.cm_overlap(y, yhat, timecol, overlap_score, sample_rate)
	return TP, FP, TN, FN

def predict_file(filename, model, balance = False, only_arousal = False, filter = False, removal = True):
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, epoch_length, overlap_factor, sample_rate, filter, removal)
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat, timecol = reconstruct(X, y, epochs)
	return y, yhat, timecol

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