from numpy import *
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Embedding, TimeDistributed, Bidirectional, GRU
from keras.callbacks import EarlyStopping, TensorBoard, History
import metrics
from stopwatch import *
import sys
from plots import *
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
"""

MODEL = 'gru.h5'

class gru:
	def __init__(self, data = None, load_graph = False, path = MODEL, batch_size = 2 ** 11):
		self.batch_size = batch_size
		if load_graph:
			self.graph = load_model(path)
		elif data:
			self.data = data
			self.neurons = data.timesteps
			self.graph = None
			self.build()

	def build(self):
		graph = Sequential()
		#graph.add(Bidirectional(GRU(self.neurons, return_sequences=True), input_shape=(self.data.timesteps, self.data.features), merge_mode='concat'))
		graph.add(GRU(self.neurons, return_sequences=True, input_shape=(self.data.timesteps, self.data.features)))
		graph.add(TimeDistributed(Dense(1, activation='sigmoid')))
		graph.compile(loss='binary_crossentropy', optimizer='adam')
		self.graph = graph

	def save(self):
		self.graph.save(MODEL)

	def get_callbacks(self):
		early_stop = EarlyStopping(monitor='loss', patience=3, mode='auto', verbose=1)
		history = History()
		return [early_stop, history]

	def shape_epochs(self, epochs):
		X = empty((len(epochs), self.data.timesteps, self.data.features))
		y = empty((len(epochs), self.data.timesteps, 1))
		for i,epoch in enumerate(epochs):
			X[i] = epoch.X
			y[i] = reshape(epoch.y, (epoch.y.size, 1))
		return X,y

	def fit(self, epochs, iterations=256):
		X,y = self.shape_epochs(epochs)
		callbacks = self.get_callbacks()
		hist = self.graph.fit(X, y, epochs=iterations, batch_size=self.batch_size, verbose=1, callbacks=callbacks)
		savetxt('hist.csv', self.return_loss(callbacks[1]), delimiter=',')

	def return_loss(self, history):
		items = history.history.items()
		return [item[1] for item in items]

	def shape_X(self, epoch):
		return reshape(epoch.X, (1, epoch.X.shape[0], epoch.X.shape[1]))

	def shape_y(self, epoch):
		return reshape(epoch.y, (1, epoch.y.size, 1))

	def predict(self, epochs):
		predictions = []
		for epoch in epochs:
			yhat = self.graph.predict_classes(self.shape_X(epoch))
			epoch.yhat = squeeze(yhat)
		return epochs

	def evaluate(self, epochs):
		results = []
		self.predict(epochs)
		for epoch in epochs:
			scores = metrics.compute_score(epoch.y, epoch.yhat)
			results.append(scores)
		return results