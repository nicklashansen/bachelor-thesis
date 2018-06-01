'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

Module is responsible for everything related to the dataflow concerning GUI output.
'''

from numpy import *
from features import epochs_from_prep, make_features
from epoch import epoch
from gru import gru, gru_config
from timeseries import timeseries, region
from dataset import dataset
from model_selection import add_predictions, reconstruct
from plots import plot_results
from log import Log, get_log
import filesystem as fs
import settings

def dataflow(X, cmd_plot = False):
	'''
	Primary function responsible for predictions and GUI output from a pre-processed file.
	Returns signals used for plotting of features as well as generated summary statistics.
	'''
	epochs = epochs_from_prep(X.copy(), None, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=False, removal=True)
	epochs = dataset(epochs, shuffle=False, exclude_ptt=False, only_arousal = True, only_rwa = True).epochs
	epochs = gru(load_graph=True, path=settings.BEST_MODEL).predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat,timecol = reconstruct(X, epochs)
	full = epochs_from_prep(X, None, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=False, removal=False)
	full.sort(key=lambda x: x.index_start, reverse=False)
	wake, nrem, rem, illegal = timeseries(full)
	summary = summary_statistics(timecol, yhat, wake, nrem, rem, illegal)
	X,_,mask = make_features(X, None, settings.SAMPLE_RATE, removal=False)
	X = transpose(X)
	ss = X[6].copy()
	for i,_ in enumerate(ss):
		if X[7,i]:
			ss[i] = 2
		elif X[5,i]:
			ss[i] = 0
	data = X[0]/settings.SAMPLE_RATE, [X[1], X[2], X[3], X[4], ss, yhat], ['RR', 'RWA', 'PTT', 'PWA', 'Sleep stage', 'Arousals'], region(X[5]), region(X[7]), None, None, int(X[0,-1]/settings.SAMPLE_RATE)
	if cmd_plot:
		plot_results(*list(data))
	return data, summary

def postprocess(timecol, yhat, combine = False, remove = False):
	'''
	Takes a prediction signal yhat and its corresponding time axis and conditionally alters yhat such that:
	1) If combine, predicted arousals that are within 3s (minimum arousal length) of each other are combined.
	2) If remove, predicted arousals that last for less than 3s (minimum arousal length) are removed.
	These two actions are not mutually exclusive.
	'''
	prev, start, bin, n = None, 0, False, 0
	for i,p in enumerate(yhat):
		if p:
			if not bin:
				start, bin = i, True
		elif bin:
			bin = False
			curr = [start, i-1]
			if remove:
				timecol, yhat, n = conditional_remove(timecol, yhat, curr, n)
			if combine and prev is not None:
				timecol, yhat, n = conditional_combine(timecol, yhat, curr, prev, n)
			prev = [start, i-1]
	return yhat, n

def conditional_remove(timecol, yhat, curr, n):
	'''
	Predicted arousals that last for less than 3s (minimum arousal length) are removed.
	'''
	dur = timecol[curr[1]] - timecol[curr[0]]
	if dur < 3 * settings.SAMPLE_RATE:
		n += 1
		for j in range(curr[0], curr[1]+1):
			yhat[j] = 0
	return timecol, yhat, n
		

def conditional_combine(timecol, yhat, curr, prev, n):
	'''
	Predicted arousals that are within 3s (minimum arousal length) of each other are combined.
	'''
	diff = timecol[curr[0]] - timecol[prev[1]]
	if diff < 3 * settings.SAMPLE_RATE:
		n += 1
		for j in range(prev[1], curr[0]):
			yhat[j] = 1
	return timecol, yhat, n

def summary_statistics(timecol, yhat, wake, nrem, rem, illegal):
	'''
	Computes summary statistics for the GUI.
	'''
	rec_dur_float = ((timecol[-1]-timecol[0])/settings.SAMPLE_RATE)/60
	rec_dur = str(int(rec_dur_float)) + ' min'
	#print(sum(wake), sum(nrem), sum(rem))
	ss_total = sum(wake) + sum(nrem) + sum(rem)
	p_wake = int((sum(wake)/ss_total)*100)
	pct_wake = str(p_wake) + '%'
	p_rem = int((sum(rem)/ss_total)*100)
	pct_rem = str(p_rem) + '%'
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