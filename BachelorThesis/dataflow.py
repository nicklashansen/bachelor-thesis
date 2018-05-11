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
	batch_size = 2 ** 8 if gpu else 2 ** 6
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
		model = gru(load_graph=True, path = 'gru-newsleep-arousals.h5')
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

def get_timeseries_prediction(X, model):
	epochs = epochs_from_prep(X, None, epoch_length, overlap_factor, filter = False, removal=False)
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	_, yhat, wake, rem, illegal = timeseries(epochs, epochs, epoch_length, overlap_factor, sample_rate)
	return epochs, yhat, wake, rem, illegal

def add_predictions(yhat1, yhat2):
	assert len(yhat1) == len(yhat2)
	for i in range(len(yhat1)):
		if yhat2[i] == 1:
			yhat1[i] = 1
	return yhat1

def add_ECG_overhead(epoch, illegal):
	illegal.append([0, int(epoch.index_start/sample_rate)])
	return illegal

def dataflow(edf = 'D:\\BachelorThesis\\Files\\Data\\mesa\\polysomnography\\edfs\\mesa-sleep-0001.edf', anno = 'D:\\BachelorThesis\\Files\\Data\\mesa\\polysomnography\\annotations-events-nsrr\\mesa-sleep-0001-nsrr.xml', cmd_plot = False):
	#X = prep_X(edf, anno)
	X,_ = fs.load_csv(edf[-19:-4])
	epochs, yhat, wake, rem, illegal = get_timeseries_prediction(X, gru(load_graph=True))

	## if multiple models
	#yhat2, wake2, rem2, illegal2 = get_timeseries_prediction(X, gru(load_graph=True, path='gru2.h5'))
	#yhat = add_predictions(yhat, yhat2)

	X = transpose(X)
	if cmd_plot:
		plot_results(X[0]/sample_rate, [X[1]], ['RR'], region(wake), region(rem), add_ECG_overhead(epochs[0], region(illegal)), region(yhat), int(epochs[-1].index_stop/sample_rate))

	plot_data = (X[0]/sample_rate, [X[1]], ['RR'], region(wake), region(rem), add_ECG_overhead(epochs[0], region(illegal)), region(yhat), int(epochs[-1].index_stop/sample_rate))
	prop_dict = [(0,0),(1,1)]
	return plot_data, prop_dict