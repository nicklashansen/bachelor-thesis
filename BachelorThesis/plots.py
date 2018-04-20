from numpy import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from filters import *

def plot_data(signals):
	#def normalize(X, scaler=MinMaxScaler()):
	#	return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))
	
	color = ['b-', 'g-', 'rx']
	plt.figure(figsize=(6.5, 4))
	x = range(0,len(signals[0]))
	for i,signal in enumerate(signals):
			plt.plot(x, signal, color[i], label='signal'+str(i))
	plt.legend()
	plt.show()