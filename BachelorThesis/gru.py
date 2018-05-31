'''
WRITTEN BY:
Nicklas Hansen

Module is responsible for everything related to neural networks.
'''

from numpy import *
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Embedding, TimeDistributed, Bidirectional, GRU, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint
from keras.utils import plot_model
from stopwatch import *
from plots import *
import settings
import sys
import metrics
import filesystem as fs

MODEL = 'gru.h5'
HIST = 'hist.csv'
HISTVAL = 'histval.csv'
PLOT = 'gru.png'

class gru_config:
	'''
	Helper class used to format model architectures for easy configuration.
	'''
	def __init__(self, name = 'gru', rnn_layers = 1, dense_layers_before = 0, dense_layers_after = 1, bidirectional = False, bidirectional_mode = 'sum', dropout = 0, timesteps = settings.EPOCH_LENGTH, features = settings.FEATURES, verbose = 1):
		'''
		Initialises an instance of the object using specified parameters and default values for remaining values.
		'''
		self.name = name
		self.rnn_layers = rnn_layers
		self.dense_layers_before = dense_layers_before
		self.dense_layers_after = dense_layers_after
		self.bidirectional = bidirectional
		self.bidirectional_mode = bidirectional_mode
		self.dropout = dropout
		self.timesteps = timesteps
		self.features = features
		self.verbose = verbose

class gru:
	'''
	High-level class responsible for handling everything related to the usage of neural networks.
	'''
	def __init__(self, data = None, load_graph = False, path = MODEL, batch_size = 2 ** 8, config = None):
		'''
		Creates a new instance of a neural network.
		If data is provided, the network architecture uses that as input size.
		If a configuration is provided, the network creates a new model using that configuration.
		If load_graph is true, an already created model is loaded from file.
		'''
		self.batch_size = batch_size
		self.data = None
		self.timesteps = None
		self.graph = None
		self.config = None
		if load_graph:
			self.graph = load_model(path)
		elif data is not None:
			self.data = data
			self.timesteps = data.timesteps
			self.graph = None
			self.build()
		elif config is not None:
			self.config = config
			self.build(config)

	def build(self, config = None):
		'''
		Builds the graph of a new model.
		If a configuration is provided, its settings are used.
		Otherwise, a default configuration is used.
		'''
		graph = Sequential()
		if config is None:
			graph.add(GRU(1, return_sequences=True, input_shape=(self.data.timesteps, self.data.features)))
		else:
			for i in range(config.dense_layers_before):
				graph.add(TimeDistributed(Dense(1, activation='tanh'), input_shape=(config.timesteps, config.features)))
				if config.dropout > 0:
					graph.add(Dropout(config.dropout))
			for j in range(config.rnn_layers):
				if config.dense_layers_before == 0 and j == 0:
					if config.bidirectional:
						graph.add(Bidirectional(GRU(1, return_sequences=True), input_shape=(config.timesteps, config.features), merge_mode=config.bidirectional_mode))
					else:
						graph.add(GRU(1, return_sequences=True, input_shape=(config.timesteps, config.features)))
				else:
					if config.bidirectional:
						graph.add(Bidirectional(GRU(1, return_sequences=True), merge_mode=config.bidirectional_mode))
					else:
						graph.add(GRU(1, return_sequences=True))
					graph.add(GRU(1, return_sequences=True))
				if config.dropout > 0:
					graph.add(Dropout(config.dropout))
			if config.dense_layers_after > 1:
				for k in range(config.dense_layers_after - 1):
					graph.add(TimeDistributed(Dense(1, activation='tanh')))
					if config.dropout > 0:
						graph.add(Dropout(config.dropout))
		graph.add(TimeDistributed(Dense(1, activation='sigmoid')))
		graph.compile(loss='binary_crossentropy', optimizer='adam')
		self.graph = graph

	def save(self, path = MODEL, plot = False, plot_path = PLOT):
		'''
		Saves a model to a local file.
		Optionally saves a plot of the model as well.
		'''
		if self.config is not None:
			self.graph.save(self.config.name + '.h5')
		else:
			self.graph.save(path)
		if plot:
			self.plot(plot_path)

	def plot(self, path = PLOT):
		'''
		Saves a plot of the model to a local file.
		Plotting is only supported on Unix systems.
		'''
		if self.config is not None:
			plot_model(self.graph, self.config.name + '.png')
		else:
			plot_model(self.graph, to_file=path)

	def get_callbacks(self, validate = True):
		'''
		Returns the keras callback functions used during model training.
		'''
		metric = 'val_loss' if validate else 'loss'
		early_stop = EarlyStopping(monitor=metric, patience=3, mode='auto', verbose=1)
		checkpoint = ModelCheckpoint('best.h5', monitor=metric, save_best_only=True)
		history = History()
		return [early_stop, checkpoint, history]

	def shape_epochs(self, epochs):
		'''
		Reshapes epochs in order to fit keras input format.
		'''
		if self.config is not None:
			timesteps = self.config.timesteps
			features = self.config.features
		else:
			timesteps = self.data.timesteps
			features = self.data.features
		X = empty((len(epochs), timesteps, features))
		y = empty((len(epochs), timesteps, 1))
		for i,epoch in enumerate(epochs):
			X[i] = epoch.X
			y[i] = reshape(epoch.y, (epoch.y.size, 1))
		return X,y

	def fit(self, epochs, val = None, iterations=256):
		'''
		Fit the current model from provided epochs.
		This procdure will continue until early stopping is triggered or the maximum number of iterations is reached.
		'''
		verbose = 1
		validate = val is not None
		if self.config is not None:
			verbose = self.config.verbose
		callbacks = self.get_callbacks(validate=validate)
		X,y = self.shape_epochs(epochs)
		if validate:
			valX,valy = self.shape_epochs(val)
			hist = self.graph.fit(X, y, epochs=iterations, validation_data=(valX, valy), batch_size=self.batch_size, verbose=verbose, callbacks=callbacks)
		else:
			hist = self.graph.fit(X, y, epochs=iterations, batch_size=self.batch_size, verbose=verbose, callbacks=callbacks)
		if self.config is not None:
			savetxt(self.config.name + '.csv', self.return_loss(callbacks[2], 'loss'), delimiter=',')
			if validate:
				savetxt(self.config.name + '_val.csv', self.return_loss(callbacks[2], 'val_loss'), delimiter=',')
		else:
			savetxt(HIST, self.return_loss(callbacks[2], 'loss'), delimiter=',')
			if validate:
				savetxt(HISTVAL, self.return_loss(callbacks[2], 'val_loss'), delimiter=',')

	def return_loss(self, history, metric = 'loss'):
		'''
		Returns loss from the keras History callback object.
		'''
		items = history.history
		return [item for item in items[metric]]

	def shape_X(self, epoch):
		'''
		Reshapes feature matrix X to keras format.
		'''
		return reshape(epoch.X, (1, epoch.X.shape[0], epoch.X.shape[1]))

	def shape_y(self, epoch):
		'''
		Reshapes target y to keras format.
		'''
		return reshape(epoch.y, (1, epoch.y.size, 1))

	def predict(self, epochs, return_probabilities = False):
		'''
		Predicts for each of the passed epochs and adds the predictions directly to the epoch objects.
		'''
		predictions = []
		for epoch in epochs:
			yhat = self.graph.predict_classes(self.shape_X(epoch)) if not return_probabilities else self.graph.predict(self.shape_X(epoch))
			epoch.yhat = squeeze(yhat)
		return epochs