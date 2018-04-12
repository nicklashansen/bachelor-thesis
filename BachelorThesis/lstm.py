from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, TimeDistributed, Bidirectional, GRU
import metrics
from stopwatch import *
from dataset import *
import sys

"""
WRITTEN BY:
Nicklas Hansen
"""

class lstm:
	data = dataset()
	model = Sequential()
	neurons = 10

	def __init__(self, data, neurons = 10):
		self.data = data
		self.neurons = neurons
		self.build()

	def build(self):
		model = Sequential()
		#model.add(Bidirectional(GRU(self.neurons, return_sequences=True), input_shape=(self.data.timesteps, self.data.features), merge_mode='sum'))
		model.add(GRU(self.neurons, return_sequences=True, input_shape=(self.data.timesteps, self.data.features)))
		model.add(TimeDistributed(Dense(1, activation='sigmoid')))
		model.compile(loss='binary_crossentropy', optimizer='rmsprop')
		self.model = model

	def fit(self, X, y, epochs):
		for epoch in range(epochs):
			for sample in range(len(X)):
				hist = self.model.fit(X[sample], y[sample], epochs=1, batch_size=1, verbose=0)
				#loss = hist.history['loss']	
				#print('loss:', loss[0])
			if (epochs > 1):
				print(epoch+1, '/', epochs, ' epochs completed')

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

	def evaluate(self, X, y, metric=metrics.TPR_FNR):
		TPR=FNR=0
		for epoch in range(len(X)):
			yhat = self.model.predict_classes(X[epoch], verbose=0)
			p, n = metric(y[epoch], yhat)
			TPR += p
			FNR += n
		TPR /= len(X)
		FNR /= len(X)
		return TPR, FNR