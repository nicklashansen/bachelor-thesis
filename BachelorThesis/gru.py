from numpy import *
from keras.models import Sequential
from keras.layers import Dense, Embedding, TimeDistributed, Bidirectional, GRU
from keras.callbacks import EarlyStopping, TensorBoard
import metrics
from stopwatch import *
import sys
from plots import *
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
"""

#def load_model():
#	model = ''

class gru:
	def __init__(self, data, batch_size = 100):
		self.data = data
		self.neurons = data.timesteps
		self.graph = None
		self.batch_size = batch_size
		self.build()

	def build(self):
		graph = Sequential()
		#graph.add(Bidirectional(GRU(self.neurons, return_sequences=True), input_shape=(self.data.timesteps, self.data.features), merge_mode='concat'))
		graph.add(GRU(self.neurons, return_sequences=True, input_shape=(self.data.timesteps, self.data.features)))
		graph.add(TimeDistributed(Dense(1, activation='sigmoid')))
		graph.compile(loss='binary_crossentropy', optimizer='adam')
		self.graph = graph

	def save(self):
		self.graph.save('gru.h5')

	def get_callbacks(self):
		early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto', verbose=1)
		#checkpoint = ModelCheckpoint(fs.Filepaths.Model + '', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
		#tensorboard = TensorBoard(log_dir=fs.Filepaths.Logs + 'TensorBoard', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		return [early_stop]

	def shape_epochs(self, epochs):
		X = empty((len(epochs), self.data.timesteps, self.data.features))
		y = empty((len(epochs), self.data.timesteps, 1))
		for i,epoch in enumerate(epochs):
			X[i] = epoch.X
			y[i] = reshape(epoch.y, (epoch.y.size, 1))
		return X,y

	def fit(self, epochs, iterations=1000):
		X,y = self.shape_epochs(epochs)
		hist = self.graph.fit(X, y, epochs=iterations, batch_size=self.batch_size, verbose=1, callbacks=self.get_callbacks())
		#plot_data([loss])

	# fix data pass format
	def cross_val(self, tuple: tuple, trainX=None, trainY=None, testX=None, testY=None, metric=metrics.TPR_FNR):
		if (tuple != None):
			trainX, trainY, testX, testY = tuple[0], tuple[1], tuple[2], tuple[3]
		timer, score = stopwatch(), []
		for fold in range(len(trainX)):
			self.build()
			X,y,_X,_y = trainX[fold], trainY[fold], testX[fold], testY[fold]
			self.fit(X, y, 1)
			score.append(self.evaluate(_X, _y, metric=metric))
			print(fold+1, '/', len(trainX), 'folds completed...')
		print('Duration: ', timer.stop(), 's')
		return score

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

	def evaluate(self, epochs, metric=metrics.TPR_TNR):
		TPR=TNR=0
		for epoch in epochs:
			yhat = self.graph.predict_classes(self.shape_X(epoch))
			p, n = metric(epoch.y, yhat)
			TPR += p
			TNR += n
		TPR /= len(epochs)
		TNR /= len(epochs)
		return TPR, TNR