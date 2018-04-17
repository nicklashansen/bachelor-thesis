from numpy import *
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler
from qrs_detector import QRSDetectorOffline as QRS
from PPGpeak_detector import PPG_Peaks
import filesystem as fs

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
	for i, filename in enumerate(filenames[1:]):

		subject = fs.Subject(filename)
		# TODO: Cut data by datasetCsv rules
		
		X, y = preprocess(subject)
		fs.write_csv(filename, X, y)

		print('{0:.3f} %'.format((i+1) / len(filenames) * 100), end='\r')
	print('') # reset '\r'
	
def preprocess(subject):
	# Load signals
	sig_ECG = subject.ECG_signal
	sig_PPG = subject.PPG_signal
	anno_SleepStage = subject.SleepStage_anno
	anno_Arousal = subject.Arousal_anno

	# Get Indexes / Time Series Stamps
	index,_ = _QRS(sig_ECG)

	# Process each signal
	x_DR, x_RPA = ECG(sig_ECG, index)
	x_PTT, x_PWA  = PPG(sig_PPG, index)
	x_SS = SleepStageBin(anno_SleepStage, subject.frequency, index)

	y_AA = ArousalBin(anno_Arousal, subject.frequency, index)
	
	X = transpose(array([x_DR, x_RPA, x_PTT, x_PWA, x_SS]))
	y = array(y_AA)

	return X, y 

def _QRS(sig_ECG):
	# find all peak-indexs in ECG signal
	qrs = QRS(sig_ECG.signal)# TODO: Better QRS-index detection
	index = qrs.detected_peaks_indices 
	amps = qrs.detected_peaks_values

	return index,amps

def ECG(sig_ECG, index):
	DR = [index[i]-index[i-1] for i in range(1,len(index))]
	DR = [average(DR)] + DR
	RPA = [sig_ECG.signal[idx] for idx in index]

	DR = medfilt(DR, 5).astype(float)
	RPA = array(RPA).astype(float)

	return DR,RPA

def PPG(sig_PPG, index):
	peaks, amps = PPG_Peaks(sig_PPG.signal, sig_PPG.sampleFrequency)
	
	def find_ptt(idx, idxplus, h):
		# Find h index of peak after R-peak
		while h < len(peaks) and peaks[h] < idx:
			h += 1
		# Errors
		if h >= len(peaks) or peaks[h] >= idxplus:
			return -1, h-1
		# peakid, index of ppg-peakid
		return peaks[h], h

	PTT = []
	PWA = []
	h = -1
	for i,idx in enumerate(index):
		if i < len(index)-1:	idxplus = index[i+1]
		else:					idxplus = sig_PPG.duration
		peak, h = find_ptt(idx,idxplus,h+1)
		if(peak != -1):
			PTT += [peak-idx]
			PWA += [amps[h]]
		else:
			PTT += [-1]
			PWA += [-1]

	PTT = array(PTT).astype(float)
	PWA = array(PWA).astype(float)

	return PTT, PWA

def SleepStageBin(anno_SleepStage, frequency, index):
	# 0			= WAKE	=> -1
	# 1,2,3,4	= NREM	=>  0
	# 5			= REM	=>  1
	SS = [-1]*int(anno_SleepStage.duration*frequency)
	for start,dur,stage in anno_SleepStage.annolist:
		start = float(start)
		dur = float(dur)
		stage = int(stage)
		range = range(int(start*frequency), int((start+dur)*frequency))
		if stage > 0 and stage < 5:
			for i in range:
				SS[i] = 0
		elif stage >= 5:
			for i in range:
				SS[i] = 1
	
	SS = array([SS[idx] for idx in index]).astype(float)
	return SS

def ArousalBin(anno_Arousal, frequency, index):
	# 0 = not arousal
	# 1 =     arousal
	AA = [0]*int(anno_Arousal.duration*frequency)
	xlen = len(AA)
	for start,dur,_ in anno_Arousal.annolist:
		start = float(start)
		dur = float(dur)
		for i in range(int(start*frequency), int((start+dur)*frequency)):
			AA[i] = 1

	AA = array([AA[idx] for idx in index]).astype(float)
	return AA

def normalize(X, scaler=MinMaxScaler()):
	return squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))