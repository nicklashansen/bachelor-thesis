from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from filters import *

def plot_data(signals, normalize_signals=False):
	def normalize(X, scaler=MinMaxScaler()):
		return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))
	
	if normalize_signals:
		for i,signal in enumerate(signals):
			signals[i] = normalize(signal)
	color = ['b-', 'g-', 'rx']
	plt.figure(figsize=(6.5, 4))
	x = range(0,len(signals[0]))
	for i,signal in enumerate(signals):
		if normalize_signals:
			signal = normalize(signal)
		plt.plot(x, signal, color[i], label='signal'+str(i))
	plt.legend()
	plt.show()