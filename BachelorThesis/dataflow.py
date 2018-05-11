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

def parameter_tuning(gpu = True, evaluate_model = True, balance = False, only_arousal = False):
	params = [[1,2], [0,1], [1,2], [0, 0.2]]
	batch_size = 2 ** 8 if gpu else 2 ** 6
	step = 0
	for i,ix in enumerate(params[0]):
		for j,jx in enumerate(params[1]):
			for k,kx in enumerate(params[2]):
				for h,hx in enumerate(params[3]):
					print('Running configuration', step, '...')
					config = gru_config('param' + str(step), ix, jx, kx, hx)
					model = gru(batch_size=batch_size, config=config)
					model = fit(model=model)
					model.save()
					if evaluate_model:
						evaluate(model, validation=True, log_filename = config.name)
					step += 1

def fit_validate(gpu = True, balance = False, only_arousal = False, load_path = None, save_path = None):
	batch_size = 2 ** 8 if gpu else 2 ** 6
	model = fit(batch_size, balance, only_arousal)
	model.save()
	evaluate(model, validation)

def fit(batch_size = None, balance = False, only_arousal = False, model = None, load_path = None):
	data = dataset(fs.load_epochs(), balance=balance, only_arousal=only_arousal)
	if model is None:
		model = gru(data, batch_size)
	model.fit(data.epochs)
	return model

def evaluate(model = None, validation = True, log_filename = None):
	set = 1 if validation else 2
	if model is None:
		model = gru(load_graph=True)
	files = fs.load_splits()[set]
	results = validate(model, files)
	log_results(results, validation=validation, log_filename=filename)
	return results

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

def log_results(results, validation = True, filename = None):
	if filename is None:
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

def summary_statistics(X, epochs, yhat, wake, rem, illegal):
	timecol = transpose(X)[0]
	rec_dur_float = ((timecol[-1]-timecol[0])/sample_rate)/60
	rec_dur = str(int(rec_dur_float)) + ' min'
	_, n_wake = region(wake, count = True)
	p_wake = n_wake/len(wake)
	pct_wake = str(int(p_wake*100)) + '%'
	_, n_rem = region(rem, count = True)
	pct_rem = str(int((n_rem/len(rem))*100)) + '%'
	_, n_ill = region(illegal, count = True)
	ill_score = str(int((n_ill/len(illegal))*(10**5)))
	arousals, n = region(yhat, count = True)
	n_arousals = len(arousals)
	arousals_hr = '{0:.1f}'.format(n_arousals/(rec_dur_float/60)*(1-p_wake))
	arousal_dur = []
	for arousal in arousals:
		arousal_dur.append(arousal[1] - arousal[0])
	return	[('rec_dur', rec_dur)
			,('pct_wake', pct_wake)
			,('pct_rem', pct_rem)
			,('n_arousals', str(n_arousals))
			,('arousals_hr', arousals_hr)
			,('avg_arousal', '{0:.1f}'.format(mean(arousal_dur)))
			,('med_arousal', '{0:.1f}'.format(median(arousal_dur)))
			,('std_arousal', '{0:.1f}'.format(std(arousal_dur)))
			,('ill_score', ill_score)
			]

def test_dataflow(edf = 'C:\\Users\\nickl\\Source\\Repos\\a000de373e6449ea8c29d5622ccbfcc6\\BachelorThesis\\Files\\Data\\mesa\\polysomnography\\edfs\\mesa-sleep-2084.edf', anno = 'C:\\Users\\nickl\\Source\\Repos\\a000de373e6449ea8c29d5622ccbfcc6\\BachelorThesis\\Files\\Data\\mesa\\polysomnography\\annotations-events-nsrr\\mesa-sleep-2084-nsrr.xml'):
	X,_ = fs.load_csv('mesa-sleep-2084')
	data, summary = dataflow(X, cmd_plot=True)
	print(summary)

def dataflow(X, cmd_plot = False):
	epochs, yhat, wake, rem, illegal = get_timeseries_prediction(X, gru(load_graph=True))

	## if multiple models
	#epochs2, yhat2, wake2, rem2, illegal2 = get_timeseries_prediction(X, gru(load_graph=True, path='gru2.h5'))
	#yhat = add_predictions(yhat, yhat2)

	summary = summary_statistics(X, epochs, yhat, wake, rem, illegal)
	X = transpose(X)
	if cmd_plot:
		plot_results(X[0]/sample_rate, [X[1]], ['RR'], region(wake), region(rem), add_ECG_overhead(epochs[0], region(illegal)), region(yhat), int(epochs[-1].index_stop/sample_rate))
	return (X[0]/sample_rate, [X[1]], ['RR'], region(wake), region(rem), add_ECG_overhead(epochs[0], region(illegal)), region(yhat), int(epochs[-1].index_stop/sample_rate)), summary