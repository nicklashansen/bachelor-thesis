from numpy import *
from features import epochs_from_prep, make_features
from epoch import epoch, save_epochs
from gru import gru, gru_config
from dataset import dataset
from log import Log, get_log
import filesystem as fs
import metrics
import settings

"""
WRITTEN BY:
Nicklas Hansen
"""

def get_batch_size(gpu = True):
	return 2 ** 8 if gpu else 2 ** 6

def parameter_tuning(gpu = True, evaluate_model = True, balance = False, only_arousal = False):
	# 0: 1 + 0 + 1
	# 1: 1 + 0 + 2
	# 2: 1 + 1 + 1
	# 3: 1 + 1 + 2
	# 4: 2 + 0 + 1
	# 5: 2 + 0 + 2
	# 6: 2 + 1 + 1
	# 7: 2 + 1 + 2 
	params = [[1,2], [0,1], [1,2]]
	batch_size = get_batch_size(gpu)
	data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal)
	step = 0
	for i,ix in enumerate(params[0]):
		for j,jx in enumerate(params[1]):
			for k,kx in enumerate(params[2]):
				print('Running configuration', step, '...')
				config = gru_config('param' + str(step), ix, jx, kx)
				model = gru(batch_size=batch_size, config=config)
				model = fit(model=model, data=data)
				model.save()
				if evaluate_model:
					evaluate(model, validation=True, log_filename = config.name)
				step += 1

def test_bidirectional(gpu = True, config = None, rnn_layers = 1, evaluate_model = True, balance = False, only_arousal = False, exclude_ptt = False):
	data = dataset(fs.load_epochs()[:12], balance=balance, only_arousal=only_arousal, exclude_ptt=exclude_ptt)
	if config is None:
		config = gru_config('bidir_' + str(rnn_layers) + 'layer', rnn_layers=rnn_layers, bidirectional=True, bidirectional_mode='sum')
	model = gru(batch_size=get_batch_size(), config=config)
	model = fit(model=model, data=data)
	model.save()
	if evaluate_model:
		evaluate(model, validation=True, log_filename = config.name)

def fit_validate_test(gpu = True, validation = True, balance = False, only_arousal = False, load_path = None, save_path = None):
	batch_size = get_batch_size(gpu)
	model = fit(batch_size, balance, only_arousal, validate=True)
	model.save()
	evaluate(model, validation=False)

def fit(batch_size = None, balance = False, only_arousal = False, model = None, load_path = None, data = None, validate = True):
	if data is None:
		data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal)
		if balance or only_arousal:
			save_epochs(data.epochs)
	if model is None:
		model = gru(data, batch_size=batch_size)
	if validate:
		val_epochs = fs.load_epochs('validation')
		valset = dataset(val_epochs, balance=balance, only_arousal=only_arousal, exclude_ptt=True)
		if balance or only_arousal:
			save_epochs(valset.epochs, 'validation')
		model.fit(data.epochs, valset.epochs)
	else:
		model.fit(data.epochs)
	return model

def evaluate(model = None, validation = True, log_filename = None):
	set = 1 if validation else 2
	if model is None:
		model = gru(load_graph=True)
	files = fs.load_splits()[set]
	results = validate(model, files, log_results = True, validation = True)
	log_results(results, validation=validation, filename=log_filename)
	return results

def validate(model, files, log_results = False, validation = True):
	if log_results:
		filename = 'Validation' if validation else 'Evaluation'
		log = get_log(filename, echo=True)
	TP=FP=TN=FN=0
	count = len(files)
	for file in files:
		try:
			tp, fp, tn, fn = validate_file(file, model, settings.OVERLAP_SCORE, settings.SAMPLE_RATE)
			TP += tp ; FP += fp ; TN += tn ; FN += fn
			file_score = metrics.compute_cm_score(tp, fp, tn, fn)
			log.print(file + ' -- Se: ' + '{0:.2f}'.format(file_score['score']['sensitivity']) + ',  P+: ' + '{0:.2f}'.format(file_score['score']['precision']))
		except Exception as e:
			print(e)
	return metrics.compute_cm_score(TP, FP, TN, FN)

def validate_file(file, model, overlap_score, sample_rate):
	y, yhat, timecol = predict_file(file, model)
	TP, FP, TN, FN = metrics.cm_overlap(y, yhat, timecol, overlap_score, sample_rate)
	return TP, FP, TN, FN

def predict_file(filename, model, filter = False, removal = True):
	X,y = fs.load_csv(filename)
	epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter, removal)
	epochs = dataset(epochs, shuffle=False, exclude_ptt=True).epochs
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat, timecol = reconstruct(X, y, epochs)
	return y, yhat, timecol

def reconstruct(X, epochs):
	timecol = transpose(X)[0]
	yhat = zeros(timecol.size)
	for _,e in enumerate(epochs):
		index = where(timecol == e.index_start)[0][0]
		for i,val in enumerate(e.yhat):
			if val == 1:
				yhat[index + i] = val
	return yhat, timecol

def add_predictions(yhat1, yhat2):
	assert len(yhat1) == len(yhat2)
	for i in range(len(yhat1)):
		if yhat2[i] == 1:
			yhat1[i] = 1
	return yhat1

def log_results(results, validation = True, filename = None):
	if filename is None:
		filename = 'Validation' if validation else 'Evaluation'
	log = get_log(filename, echo=True)
	for k,d in results.items():
		log.print(str(k))
		for key,val in d.items():
			log.print(str(key)+':'+str(val))