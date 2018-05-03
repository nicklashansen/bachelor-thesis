from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
WRITTEN BY:
Michael Kirkegaard,
Nicklas Hansen
"""

COLOR = ['b-', 'g-', 'r-']

def plot_results(signals, wake_states, illegals, arousals, duration = None):
	show_signals(signals, None, COLOR, duration)
	show_spans(wake_states, '0.5')
	show_spans(illegals, 'red')
	show_spans(arousals, 'green')

	plt.xlabel('Minutes')
	#plt.legend()
	plt.show()

def show_signals(array, labels = None, colors = COLOR, duration = None):
	if array is None:
		return
	if duration is None:
		duration = 0
		for i,signal in enumerate(array):
			if len(signal) > duration:
				duration = len(signal)
	x = [i/60 for i in range(duration)]
	for i,signal in enumerate(array):
		plt.plot(x, signal[:duration], colors[i], label='signal'+str(i+1))

def show_spans(array, color, alpha = 0.3):
	if array is None:
		return
	for _,obj in enumerate(array):
		plt.axvspan(obj[0]/60, obj[1]/60, color=color, alpha=alpha)

def plot_data(signals, peaksIndexs=None, labels=None, normalization=False):
	def normalize(X, scaler=MinMaxScaler()):
		return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))
	
	if normalization:
		signals = [normalize(sig) for sig in signals]

	color = ['b-', 'g-', 'r-']
	peakcolor = ['rx','bx','gx']
	#plt.figure(figsize=(6.5, 4))
	x = range(0,len(signals[0]))
	for i,signal in enumerate(signals):
		plt.plot(x, signal, color[i], label=(labels[i] if labels else 'signal'+str(i+1)))
	if peaksIndexs != None:
		for i,peaks in enumerate(peaksIndexs):
			plt.plot(peaks, [signals[i][j] for j in peaks], peakcolor[i], label=(labels[i]+' peaks' if labels else 'signal'+str(i+1)+' peaks'))
			plt.plot()
	plt.legend()
	plt.show()