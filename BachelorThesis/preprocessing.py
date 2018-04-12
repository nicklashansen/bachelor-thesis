from numpy import *
from dataset import *
from scipy.signal import medfilt
from qrs_detector import QRSDetectorOffline as QRS
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
"""

PATH = 'ecg'

def preprocess(path = PATH):
	print('Processing file...')
	qrs = QRS(path, True, True, False, False)
	index,amp = qrs.detected_peaks_indices, qrs.detected_peaks_values
	arousal = squeeze(fs.load_csv('arousal', type=int))
	y = array([arousal[i] for i in index])
	X = []
	X.append(heartrate(index))
	X.append(normalize(amp[1:len(amp)]))
	X = transpose(array(X))
	fs.write_csv(path + '_X', X)
	fs.write_csv(path + '_y', y[1:len(y)])

def heartrate(index):
	dur = medfilt([index[i]-index[i-1] for i in range(1,len(index))], 5)
	return normalize(dur)

def normalize(X, scaler=MinMaxScaler()):
	return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))

def cutout(X, dur):
	return X[dur:len(X)-dur]