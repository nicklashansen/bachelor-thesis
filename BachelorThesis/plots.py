from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
WRITTEN BY:
Michael Kirkegaard
"""

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