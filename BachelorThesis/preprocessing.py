from numpy import *
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler
#from qrs_detector import QRSDetectorOffline as QRS
import filesystem as fs
import matlab.engine

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

def prepAll():
	# Status
	print('Extracting and preprocessing files..')

	# get subject names
	filenames,datasetCSV = fs.getAllSubjectFilenames()

	# extract all subjects
	for i, filename in enumerate(filenames):

		subject = fs.Subject(filename)
		# TODO: Cut data by datasetCsv rules
		
		X, y = preprocess(subject)
		fs.write_csv(filename, X, y)

		print('{0:.3f} %'.format((i+1) / len(filenames) * 100), end='\r')
	print('') # reset '\r'

def preprocess(subject):
	sig_ECG = subject.ECG_signal
	index, amp = QRS(subject)
	index = array(index).astype(int)
	anno_SleepStage = subject.SleepStage_anno
	anno_Arousal = subject.Arousal_anno
	x_DR, x_RPA = ECG(sig_ECG, index), array(amp)
	x_SS = SleepStageBin(anno_SleepStage, sig_ECG.sampleFrequency, index)
	y_AA = ArousalBin(anno_Arousal, sig_ECG.sampleFrequency, index)
	features = [x_DR, x_RPA, x_SS]
	X = empty((len(features), len(x_DR)))
	for i in range(len(features)):
		X[i] = features[i]
	X = transpose(X)
	y = array(y_AA)
	return X, y

def QRS(subject):
	eng = matlab.engine.start_matlab()
	eng.cd(fs.Filepaths.Matlab)
	index, amp = eng.peak_detect(fs.directory(), subject.filename + '.edf', 256.0, nargout=2)
	d = index[0]
	e = d[3]

	return [i for i in d], amp

def ECG(sig_ECG, index):
	DR = array(medfilt([0]+[index[i]-index[i-1] for i in range(1,len(index))], 5)).astype(float)
	return DR

def PPG(sig_PPG, index):
	# 1) Find all peak-indexs in PPG singal
	# 2) foreach R-index, find first PPG-index right after R-index.
	# 3) calculate ptt => PPG-index minus R-index
	
	PpgPeaks = array([idx+150 for idx in index]) # TODO: Not cheat ... PTT = 150/256 sec

	PPT = array([PpgPeaks[i]-idx for i,idx in enumerate(index)]).astype(float)
	PWA = array([sig_PPG.signal[jdx] for jdx in PpgPeaks])
	
	return PPT, PWA

def SleepStageBin(anno_SleepStage, frequency, index):
	# 0			= WAKE	=> -1.0
	# 1,2,3,4	= NREM	=> 0.0
	# 5			= REM	=> 1.0
	SS = [-1.0]*int(anno_SleepStage.duration*frequency)
	for start,dur,stage in anno_SleepStage.annolist:
		start = float(start)
		dur = float(dur)
		stage = int(stage)
		if stage > 0 and stage < 5:
			for i in range(int(start*frequency), int((start+dur)*frequency)):
				SS[i] = 0.0
		elif stage >= 5:
			for i in range(int(start*frequency), int((start+dur)*frequency)):
				SS[i] = 1.0

	return array([SS[idx] for idx in index])

def ArousalBin(anno_Arousal, frequency, index):
	# 0.0 = no arousal
	# 1.0 = arousal
	AA = [0.0]*int(anno_Arousal.duration*frequency)
	xlen = len(AA)
	for start,dur,_ in anno_Arousal.annolist:
		start = float(start)
		dur = float(dur)
		for i in range(int(start*frequency), int((start+dur)*frequency)):
			AA[i] = 1.0

	return array([AA[idx] for idx in index])

def normalize(X, scaler=MinMaxScaler()):
	return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))