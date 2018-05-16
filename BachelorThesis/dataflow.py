from numpy import *
from features import epochs_from_prep, make_features
from epoch import epoch
from gru import gru, gru_config
from timeseries import timeseries, region, add_ECG_overhead
from dataset import dataset
from model_selection import add_predictions, reconstruct
from plots import plot_results
from log import Log, get_log
import filesystem as fs
import settings

"""
WRITTEN BY:
Nicklas Hansen
"""

def test_dataflow():
	X,y = fs.load_csv('mesa-sleep-6789')
	epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=False, removal=True)
	epochs = gru(load_graph=True).predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat, _ = reconstruct(X, y, epochs)
	X,_,mask = make_features(X, None, settings.SAMPLE_RATE, removal=False)
	X = transpose(X)
	ill = [1 if x >= 1 and X[5,i] == 0 else 0 for i,x in enumerate(mask)]
	
	ss = X[6].copy()
	for i,_ in enumerate(ss):
		if X[7,i]:
			ss[i] = 2
		elif X[5,i]:
			ss[i] = 0

	plot_results(X[0]/settings.SAMPLE_RATE, [X[1], X[3], ss, yhat, y*(-1)], ['RR interval', 'PTT', 'Sleep stage', 'yhat', 'y'], region(X[5]), region(X[7]), None, None, int(X[0,-1]/settings.SAMPLE_RATE))

def dataflow(X, cmd_plot = False):
	epochs,yhat,wake,rem,illegal = get_timeseries_prediction(X, gru(load_graph=True))
	summary = summary_statistics(X, epochs, yhat, wake, rem, illegal)
	X = transpose(X)
	if cmd_plot:
		plot_results(X[0]/settings.SAMPLE_RATE, [X[1], X[3], ss, yhat, y*(-1)], ['RR interval', 'PTT', 'Sleep stage', 'yhat', 'y'], region(X[5]), region(X[7]), None, None, int(X[0,-1]/settings.SAMPLE_RATE))
	return X[0]/settings.SAMPLE_RATE, [X[1], X[3], ss, yhat, y*(-1)], ['RR interval', 'PTT', 'Sleep stage', 'yhat', 'y'], region(X[5]), region(X[7]), None, None, int(X[0,-1]/settings.SAMPLE_RATE), summary

def get_timeseries_prediction(X, model, y=None):
	epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, filter = False, removal=True)
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	y, yhat, wake, rem, illegal, timecol = timeseries(epochs, epochs, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE)
	if y is not None:
		return epochs, y, yhat, wake, rem, illegal, timecol
	return epochs, yhat, wake, rem, illegal, timecol

def summary_statistics(X, epochs, yhat, wake, rem, illegal):
	timecol = transpose(X)[0]
	rec_dur_float = ((timecol[-1]-timecol[0])/settings.SAMPLE_RATE)/60
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