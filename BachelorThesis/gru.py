from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional, GRU
import metrics
from stopwatch import *
import sys
from plots import *

"""
WRITTEN BY:
Nicklas Hansen
"""

class gru:
	def __init__(self, data, neurons = 10):
		self.data = data
		self.neurons = neurons
		self.graph = None
		self.build()

	def build(self):
		graph = Sequential()
		graph.add(Bidirectional(GRU(self.neurons, return_sequences=True), input_shape=(self.data.timesteps, self.data.features), merge_mode='concat'))
		#graph.add(GRU(self.neurons, return_sequences=True, input_shape=(self.data.timesteps, self.data.features)))
		graph.add(TimeDistributed(Dense(1, activation='sigmoid')))
		graph.compile(loss='binary_crossentropy', optimizer='adam')
		self.graph = graph

	def shape_epochs(self, epochs):
		X = empty((len(epochs), self.data.timesteps, self.data.features))
		y = empty((len(epochs), self.data.timesteps, 1))
		for i,epoch in enumerate(epochs):
			X[i] = epoch.X
			y[i] = reshape(epoch.y, (epoch.y.size, 1))
		return X,y

	def fit(self, epochs, iterations):
		loss = []
		X,y = self.shape_epochs(epochs)
		hist = self.graph.fit(X, y, epochs=iterations, batch_size=100, verbose=1)
		loss.append(hist.history['loss'])
		#plot_data([loss])

	# fix data pass format
	def cross_val(self, tuple: tuple, trainX=None, trainY=None, testX=None, testY=None, metric=metrics.TPR_FNR):
		print('Cross-validating...')
		if (tuple != None):
			trainX, trainY, testX, testY = tuple[0], tuple[1], tuple[2], tuple[3]
		timer, score = stopwatch(), []
		for fold in range(len(trainX)):
			self.build()
			X,y,_X,_y = trainX[fold], trainY[fold], testX[fold], testY[fold]
			#print('Fitting ', len(X), ' sequences...')
			self.fit(X, y, 1)
			#print('Evaluating ', len(_X), ' sequences...')
			score.append(self.evaluate(_X, _y, metric=metric))
			print(fold+1, '/', len(trainX), 'folds completed...')
		print('Duration: ', timer.stop(), 's')
		return score

	def shape_X(self, epoch):
		return reshape(epoch.X, (1, epoch.X.shape[0], epoch.X.shape[1]))

	def shape_y(self, epoch):
		return reshape(epoch.y, (1, epoch.y.size, 1))

	def evaluate(self, epochs, metric=metrics.TPR_TNR):
		TPR=FNR=0
		for epoch in epochs:
			#if (sum(epoch.y) == 0):
			#	continue
			yhat = self.graph.predict_classes(self.shape_X(epoch), verbose=0)
			p, n = metric(epoch.y, yhat)
			TPR += p
			FNR += n
		TPR /= len(epochs)
		FNR /= len(epochs)
		return TPR, FNR