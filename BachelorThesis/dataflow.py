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
Nicklas Hansen,
Michael Kirkegaard
"""

def test_dataflow_LR(file='mesa-sleep-2451'):
	import os, matlab.engine, preprocessing

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

	X_,y = fs.load_csv(file)
	X,_,mask = make_features(X_, None, settings.SAMPLE_RATE, removal=True)

	yhat, timecol_hat = eng.LR_classify(fs.directory(), file+'.edf', float(settings.SAMPLE_RATE), nargout=2)
	timecol_hat = array([t[0] for t in timecol_hat])
	yhat = transform_yhat([1. if yh[0] else 0. for yh in yhat], timecol_hat, transpose(X)[0], transpose(X_)[0])

	X,_,mask = make_features(X_, None, settings.SAMPLE_RATE, removal=False)
	X = transpose(X)
	ss = X[6].copy()
	for i,_ in enumerate(ss):
		if X[7,i]:
			ss[i] = 2
		elif X[5,i]:
			ss[i] = 0

	plot_results(X[0]/settings.SAMPLE_RATE, [X[1], X[3], ss, yhat, y*(-1)], ['RR interval', 'PTT', 'Sleep stage', 'yhat', 'y'], region(X[5]), region(X[7]), None, None, int(X[0,-1]/settings.SAMPLE_RATE))

def count_epochs_removed():
	files = fs.load_splits()[1]
	for file in files:
		X,y = fs.load_csv(file)
		epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=True, removal=True)

def test_dataflow(file = 'mesa-sleep-1541'):
	X,y = fs.load_csv(file)
	epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=False, removal=True)
	epochs = gru(load_graph=True).predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	yhat,timecol = reconstruct(X, epochs)
	full = epochs_from_prep(X, None, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=False, removal=False)
	full.sort(key=lambda x: x.index_start, reverse=False)
	wake, nrem, rem, illegal = timeseries(full)
	summary = summary_statistics(timecol, yhat, wake, nrem, rem, illegal)
	print(summary)
	X,_,mask = make_features(X, None, settings.SAMPLE_RATE, removal=False)
	X = transpose(X)
	ss = X[6].copy()
	for i,_ in enumerate(ss):
		if X[7,i]:
			ss[i] = 2
		elif X[5,i]:
			ss[i] = 0
	plot_results(X[0]/settings.SAMPLE_RATE, [X[1], X[2], X[3], X[4], ss, yhat, y*(-1)], ['RR', 'RWA', 'PTT', 'PWA', 'Sleep stage', 'yhat', 'y'], region(X[5]), region(X[7]), None, None, int(X[0,-1]/settings.SAMPLE_RATE))

def dataflow(X, cmd_plot = False):
	epochs = epochs_from_prep(X, None, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE, filter=False, removal=True)
	epochs = gru(load_graph=True).predict(epochs)
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

def get_timeseries_prediction(X, model, y=None):
	epochs = epochs_from_prep(X, y, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, filter = False, removal=True)
	epochs = model.predict(epochs)
	epochs.sort(key=lambda x: x.index_start, reverse=False)
	y, yhat, wake, rem, illegal, timecol = timeseries(epochs, epochs, settings.EPOCH_LENGTH, settings.OVERLAP_FACTOR, settings.SAMPLE_RATE)
	if y is not None:
		return epochs, y, yhat, wake, rem, illegal, timecol
	return epochs, yhat, wake, rem, illegal, timecol

def postprocess(timecol, yhat, combine = False, remove = False):
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
	dur = timecol[curr[1]] - timecol[curr[0]]
	if dur < 1 * settings.SAMPLE_RATE:
		n += 1
		for j in range(curr[0], curr[1]+1):
			yhat[j] = 0
	return timecol, yhat, n
		

def conditional_combine(timecol, yhat, curr, prev, n):
	diff = timecol[curr[0]] - timecol[prev[1]]
	if diff < 3 * settings.SAMPLE_RATE:
		n += 1
		for j in range(prev[1], curr[0]):
			yhat[j] = 1
	return timecol, yhat, n

def region(array, count = False):
	regions, start, bin, n = [], 0, False, 0
	for i,val in enumerate(array):
		if val == 1:
			if not bin:
				start, bin = i, True
		elif bin:
			bin = False
			n += 1
			regions.append([start, i-1])
	if bin:
		regions.append([start, i])
		n += 1
	if count:
		return regions, n
	return regions

def summary_statistics(timecol, yhat, wake, nrem, rem, illegal):
	rec_dur_float = ((timecol[-1]-timecol[0])/settings.SAMPLE_RATE)/60
	rec_dur = str(int(rec_dur_float)) + ' min'
	#print(sum(wake), sum(nrem), sum(rem))
	ss_total = sum(wake) + sum(nrem) + sum(rem)
	p_wake = int((sum(wake)/ss_total)*100)
	pct_wake = str(p_wake) + '%'
	p_rem = int((sum(rem)/ss_total)*100)
	pct_rem = str(p_rem) + '%'
	#_, n_wake = region(wake, count = True)
	#p_wake = n_wake/len(wake)
	#pct_wake = str(int(p_wake*100)) + '%'
	#_, n_rem = region(rem, count = True)
	#pct_rem = str(int((n_rem/len(rem))*100)) + '%'
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