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
		for iteration in range(iterations):
			X,y = self.shape_epochs(epochs)
			hist = self.graph.fit(X, y, epochs=1, batch_size=50, verbose=1)
			#for epoch in epochs:
			#	hist = self.graph.fit(self.shape_X(epoch), self.shape_y(epoch), epochs=1, batch_size=1, verbose=0)
			loss.append(hist.history['loss'])
			if (iterations > 1):
				print(iteration+1, '/', iterations, ' iterations completed')
		plot_data([loss])

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

	def evaluate(self, epochs, metric=metrics.TPR_FNR):
		TPR=FNR=0
		for epoch in epochs:
			yhat = self.graph.predict_classes(self.shape_X(epoch), verbose=0)
			p, n = metric(epoch.y, yhat)
			TPR += p
			FNR += n
		TPR /= len(epochs)
		FNR /= len(epochs)
		return TPR, FNR