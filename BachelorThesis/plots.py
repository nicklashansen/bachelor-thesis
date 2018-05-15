from numpy import array
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

COLOR = ['black', 'orange', 'cyan', 'green']

def plot_results(timecol, signals, labels, wake_states, rem, illegals, arousals, duration = None, figure = None):
	if figure:
		a = figure.add_subplot(111)
		a = show_signals(timecol, signals, labels, COLOR, duration, a=a)
		a = show_spans(wake_states, '0.5', a=a)
		a = show_spans(rem, 'purple', a=a)
		a = show_spans(illegals, 'red', a=a)
		a = show_spans(arousals, 'green', 0.9, a=a)
		a.set_xlim(0, duration/60)
		a.set_ylim(-1,1)
		a.set_xlabel('Minutes')
		a.set_ylabel('Normalised values')
		a.legend()
		return figure
	else:
		show_signals(timecol, signals, labels, COLOR, duration)
		show_spans(timecol, wake_states, '0.5')
		#show_spans(timecol, rem, 'purple', 0.15)
		show_spans(timecol, illegals, 'red')
		#show_spans(timecol, arousals, 'green', 0.9)
		plt.xlim(0, duration/60)
		plt.ylim(-1,1)
		plt.xlabel('Minutes')
		plt.ylabel('Normalised values')
		plt.legend()
		plt.show()

def show_signals(timecol, array, labels = None, colors = COLOR, duration = None, a = None):
	if array is None:
		return a
	if duration is None:
		duration = 0
		for i,signal in enumerate(array):
			if len(signal) > duration:
				duration = len(signal)
	x = timecol/60
	for i,signal in enumerate(array):
		linewidth = 1.8 if labels[i] == 'y' else 0.6
		if a:
			a.plot(x, signal/10, colors[i], label=labels[i], linewidth=linewidth)
		else:
			plt.plot(x, signal/10, colors[i], label=labels[i], linewidth=linewidth)
	return a

def show_spans(timecol, array, color, alpha = 0.3, a = None):
	if array is None:
		return a
	x = timecol/60
	for _,obj in enumerate(array):
		if a:
			a.axvspan(x[obj[0]], x[obj[1]], color=color, alpha=alpha)
		else:
			plt.axvspan(x[obj[0]], x[obj[1]], color=color, alpha=alpha)
	return a

def plot_data(signals, peaksIndexs=None, labels=None, normalization=False):
	def normalize(X, scaler=MinMaxScaler()):
		return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))
	
	if normalization:
		signals = [normalize(sig) for sig in signals]

	color = ['b-', 'g-', 'r-']
	peakcolor = ['rx','bx','gx']
	x = range(0,len(signals[0]))
	for i,signal in enumerate(signals):
		plt.plot(x, signal, color[i], label=(labels[i] if labels else 'signal'+str(i+1)))
	if peaksIndexs != None:
		for i,peaks in enumerate(peaksIndexs):
			plt.plot(peaks, [signals[i][j] for j in peaks], peakcolor[i], label=(labels[i]+' peaks' if labels else 'signal'+str(i+1)+' peaks'))
			plt.plot()
	plt.legend()
	plt.show()