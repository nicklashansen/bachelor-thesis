'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

Module contains functions developed for the model and feature selection procedure,
including training, validation and testing phases.
'''

from numpy import *
from features import epochs_from_prep, make_features
from epoch import epoch, save_epochs
from gru import gru, gru_config
from dataset import dataset
from log import Log, get_log
import filesystem as fs
import metrics
import settings
import os, matlab.engine, preprocessing

def evaluate_LR():
	'''
	Evaluate dataset using LR model, results are logged for evaluation and testing purposes.
	'''
	eng = matlab.engine.start_matlab()
	os.makedirs(fs.Filepaths.Matlab, exist_ok=True)
	eng.cd(fs.Filepaths.Matlab)

	def transform_yhat(yhat,timecol_hat,timecol_removal,timecol):
		n = len(timecol_hat)
		yhat_new = zeros(len(timecol))
		j = 1
		for i,t in enumerate(timecol_removal):
			while(j < n and timecol_hat[j] < t):
				j += 1
			jx = j if j < n and abs(timecol_hat[j] - t) > abs(t - timecol_hat[j-1]) else j-1
			jx_t = list(timecol).index(t)
			yhat_new[jx_t] = yhat[jx]
		return yhat_new

	log = get_log('LR-evaluation', echo=True)
	TP=FP=TN=FN = 0
	for file in fs.load_splits()[2] if not settings.SHHS else fs.getAllSubjectFilenames(preprocessed=True):
		try:
			X_,y = fs.load_csv(file)
			X,_,_ = make_features(X_, None, settings.SAMPLE_RATE, removal=True)
			if not settings.SHHS:
				yhat, timecol_hat = eng.LR_classify_shhs(fs.directory(), file+'.edf', float(settings.SAMPLE_RATE), nargout=2)
			else:
				yhat, timecol_hat = eng.LR_classify(fs.directory(), file+'.edf', float(settings.SAMPLE_RATE), nargout=2)
			timecol_hat = array([t[0] for t in timecol_hat])
			yhat = transform_yhat([1. if yh[0] else 0. for yh in yhat], timecol_hat, transpose(X)[0], transpose(X_)[0])

			tp,fp,tn,fn = metrics.cm_overlap(y, yhat, transpose(X_)[0], settings.OVERLAP_SCORE, settings.SAMPLE_RATE)
			TP += tp ; FP += fp ; TN += tn ; FN += fn
			file_score = metrics.compute_cm_score(tp, fp, tn, fn)
			log.print(file + ' -- Se: ' + '{0:.2f}'.format(file_score['score']['sensitivity']) + ',  P+: ' + '{0:.2f}'.format(file_score['score']['precision']))
		except Exception as e:
			print(e)
	log.printHL()
	log.print('TOTAL')
	score = metrics.compute_cm_score(TP, FP, TN, FN)
	for k,d in score.items():
		log.print(str(k))
		for key,val in d.items():
			log.print(str(key)+':'+str(val))

def get_batch_size(gpu = True):
	'''
	Returns appropriate batch size depending on whether the model is trained on a GPU or CPU.
	'''
	return 2 ** 8 if gpu else 2 ** 6

def train_validate_test(gpu = True, config = None, rnn_layers = 1, evaluate_model = True, balance = False, only_arousal = False, exclude_ptt = False):
	'''
	Trains, validates and tests a model configuration with or without the full feature set, and automatically saves models and all results locally.
	'''
	data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal, exclude_ptt=exclude_ptt)
	if config is None:
		config = gru_config('bidir_' + str(rnn_layers) + 'layer', rnn_layers=rnn_layers, bidirectional=True, bidirectional_mode='sum')
	model = gru(batch_size=get_batch_size(), config=config)
	model = fit(model=model, data=data)
	model.save()
	if evaluate_model:
		evaluate(model, validation=False, log_filename = config.name)

def fit(batch_size = None, balance = False, only_arousal = False, model = None, load_path = None, data = None):
	'''
	Function responsible for fitting of a new model.
	If no model is inputted, a default model is created.
	If no data is specified, the default training data is loaded.
	'''
	if data is None:
		data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal)
		if balance or only_arousal:
			save_epochs(data.epochs)
	if model is None:
		model = gru(data, batch_size=batch_size)
	else:
		model.fit(data.epochs)
	return model

def evaluate(model = None, validation = False, path = 'gru.h5', log_filename = None, return_probabilities = False):
	'''
	Evaluate a trained model on a given set.
	Model can either be passed as parameter or by path.
	'''
	set = 1 if validation else 2
	if model is None:
		model = gru(load_graph=True, path=path)
	files = fs.load_splits()[set] if not settings.SHHS else fs.getAllSubjectFilenames(preprocessed=True)
	results = validate(model, files, log_results=True, validation=validation, return_probabilities=return_probabilities)
	log_results(results, validation=validation, filename=log_filename)
	return results

def q_threshold_cv(start = 0.1, stop = 0.9, step = 0.1):
	'''
	Automated cross-validation of prediction threshold (q).
	'''
	vals = arange(start, stop, step)
	for i,val in enumerate(vals):
		evaluate_q_threshold(val, validation=False)

def evaluate_q_threshold(threshold = settings.PREDICT_THRESHOLD, validation = False):
	'''
	Runs evaluation based on generated probability outputs from file (p-files).
	'''
	log = get_log('Thresholds', echo=True)
	set = 1 if validation else 2
	files = fs.load_splits()[set]
	TP=FP=TN=FN=0
	for file in files:
		try:
			arr = genfromtxt('p-' + file + '.csv', delimiter=',')
			yhat = [1 if val >= threshold else 0 for i,val in enumerate(arr[1])]
			tp, fp, tn, fn = metrics.cm_overlap(arr[0], yhat, arr[2], settings.OVERLAP_SCORE, settings.SAMPLE_RATE)
			TP += tp ; FP += fp ; TN += tn ; FN += fn
		except Exception as e:
			print(e)
	results = metrics.compute_cm_score(TP, FP, TN, FN)
	log.print('\nThreshold: ' + str(threshold))
	log_results(results, validation=validation, filename='Thresholds')

def validate(model, files, log_results = False, validation = True, return_probabilities = False):
	'''
	Runs performance test on all of the provided files and computes metrics with overlap.
	'''
	if log_results:
		filename = 'Validation' if validation else 'Evaluation'
		log = get_log(filename, echo=True)
	TP=FP=TN=FN=0
	count = len(files)
	for file in files:
		try:
			if return_probabilities:
				validate_file(file, model, settings.OVERLAP_SCORE, settings.SAMPLE_RATE, return_probabilities=return_probabilities)
			else:
				tp, fp, tn, fn = validate_file(file, model, settings.OVERLAP_SCORE, settings.SAMPLE_RATE)
				TP += tp ; FP += fp ; TN += tn ; FN += fn
				file_score = metrics.compute_cm_score(tp, fp, tn, fn)
				log.print(file + ', ' + '{0:.2f}'.format(file_score['score']['sensitivity']) + ', ' + '{0:.2f}'.format(file_score['score']['precision']))
		except Exception as e:
			print(e)
	return metrics.compute_cm_score(TP, FP, TN, FN)

def validate_file(file, model, overlap_score, sample_rate, return_probabilities = False):
	'''
	Sub-function of the validate() function.
	Runs a prediction on a given file and returns its yhat sequence as TP, FP, TN, FN metrics for that file.
	'''
	y, yhat, timecol = predict_file(file, model, return_probabilities=return_probabilities)
	#_, yhat2, __ = predict_file(file, gru(load_graph=True, path = 'ppg_2layer.h5'))
	#yhat = add_predictions(yhat, yhat2)
	#_, yhat3, __ = predict_file(file, gru(load_graph=True, path = 'gru_val_loss.h5'))
	#yhat = majority_vote(yhat, yhat2, yhat3)
	#yhat, n = postprocess(timecol, yhat, combine=False, remove=True)
	if return_probabilities:
		arr = zeros((3, y.size))
		arr[0] = y
		arr[1] = yhat
		arr[2] = timecol
		savetxt('p-' + file + '.csv', arr, delimiter=',')
	else:
		TP, FP, TN, FN = metrics.cm_overlap(y, yhat, timecol, overlap_score, sample_rate)
		return TP, FP, TN, FN

def predict_file(filename, model = None, filter = False, removal = True, return_probabilities = False):
	'''
	Performs predictions on a given file using a given model.
	Loaded file is already preprocessed.
	If no model is specified, a model is is loaded from default path.
	'''
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter, removal)
	if model == None:
		model = gru(load_graph=True, path = 'gru.h5')
	epochs = dataset(epochs, shuffle=False, exclude_ptt=True, only_arousal = True).epochs
	epochs = model.predict(epochs, return_probabilities=return_probabilities)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat, timecol = reconstruct(X, epochs, settings.PREDICT_THRESHOLD)
	return y, yhat, timecol

def reconstruct(X, epochs, threshold = 1):
	'''
	Takes feature matrix X and list of epochs as input and returns a reconstructed yhat sequence based on the time axis of X.
	'''
	timecol = transpose(X)[0]
	yhat = zeros(timecol.size)
	for _,e in enumerate(epochs):
		index = where(timecol == e.index_start)[0][0]
		for i,val in enumerate(e.yhat):
			if val == 1 or val >= threshold:
				yhat[index + i] = max(yhat[index + i], val)
	return yhat, timecol

def add_predictions(yhat1, yhat2):
	'''
	Adds together two predictions using a bit-wise OR operator.
	'''
	assert len(yhat1) == len(yhat2)
	for i in range(len(yhat1)):
		if yhat2[i] == 1:
			yhat1[i] = 1
	return yhat1

def majority_vote(yhat1, yhat2, yhat3):
	'''
	Adds together three predictions using a majority vote format.
	'''
	assert len(yhat1) == len(yhat2)
	assert len(yhat1) == len(yhat3)
	return [1 if sum([yhat1[i], yhat2[i], yhat3[i]]) > 1 else 0 for i in range(len(yhat1))]

def log_results(results, validation = True, filename = None):
	'''
	Takes a results dictionary as input and prints its contents to a log file.
	'''
	if filename is None:
		filename = 'Validation' if validation else 'Evaluation'
	log = get_log(filename, echo=True)
	for k,d in results.items():
		log.print(str(k))
		for key,val in d.items():
			log.print(str(key)+':'+str(val))