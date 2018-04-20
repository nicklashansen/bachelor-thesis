from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from filters import *

def plot_data(signals, labels=None, normalization = False):
	def normalize(X, scaler=MinMaxScaler()):
		return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))
	
	color = ['b-', 'g-', 'r-']
	plt.figure(figsize=(6.5, 4))
	x = range(0,len(signals[0]))
	for i,signal in enumerate(signals):
		signal = normalize(signal) if normalization else signal
		plt.plot(x, signal, color[i], label=(labels[i] if labels else 'signal'+str(i)))
	plt.legend()
	plt.show()