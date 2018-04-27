from numpy import *
from PPGpeak_detector import PPG_Peaks
import filesystem as fs
import matlab.engine
import os
from log import getLog
from stopwatch import stopwatch

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

def prepSingle(filename):
	#log, clock = getLog('Preprocessing', True), stopwatch()
	sub = fs.Subject(filename)
	X, y = preprocess(sub)
	fs.write_csv(filename, X, y)
	#log.print('Preprocessed {0} in {1}s'.format(filename, clock.round()))
	return X, y

def prepAll(force=False):
	log, clock = getLog('Preprocessing', echo=True), stopwatch()
	filenames = fs.getAllSubjectFilenames(preprocessed=False)

	# already completed files
	oldFiles = fs.getAllSubjectFilenames(preprocessed=True)
	if not force:
		filenames = [fn for fn in filenames if fn not in oldFiles]
		log.print('Files already completed:   {0}'.format(len(oldFiles)))
		log.print('Files remaining:           {0}'.format(len(filenames)))
		if(len(oldFiles) > 0):
			log.print('-'*35)
			for fn in oldFiles:
				log.print('{0} already completed'.format(fn))
	else:
		log.print('Files re-preprocessing:    {0}'.format(len(oldFiles)))
		log.print('Files remaining:           {0}'.format(len(filenames)))
	log.print('-'*35)

	# extract all subjects
	clock.round()
	for i, filename in enumerate(filenames):
		try:
			subject = fs.Subject(filename)
			X, y = preprocess(subject)
			fs.write_csv(filename, X, y)
			log.print('{0} preprocessed in {1}s'.format(filename, clock.round()))
		except Exception as e:
			log.print('{0} Exception: {1}'.format(filename, str(e)))
			clock.round()
	clock.stop()

def preprocess(subject):
	sig_ECG = subject.ECG_signal
	sig_PPG = subject.PPG_signal
	anno_SleepStage = subject.SleepStage_anno
	anno_Arousal = subject.Arousal_anno
	
	# Get Index
	index, amp = QRS(subject)
	
	# Preprocess Features
	x_DR, x_RPA = ECG(sig_ECG, index), array(amp).astype(float)
	x_PTT, x_PWA = PPG(sig_PPG, index)
	x_SS = SleepStageBin(anno_SleepStage, subject.frequency, index)
	y_AA = ArousalBin(anno_Arousal, subject.frequency, index)

	# Collect Matrix
	features = [index, x_DR, x_RPA, x_PTT, x_PWA, x_SS]
	X = empty((len(features), len(x_DR)-1))
	for i,feat in enumerate(features):
		X[i] = feat[1:len(feat)]
	X = transpose(X)
	y = array(y_AA[1:len(y_AA)])
	return X, y

def QRS(subject):
	eng = matlab.engine.start_matlab()
	os.makedirs(fs.Filepaths.Matlab, exist_ok=True)
	eng.cd(fs.Filepaths.Matlab)
	index, amp = eng.peak_detect(fs.directory(), subject.filename + '.edf', float(subject.frequency), nargout=2)
	index = [int(i) for i in index[0]]
	amp = [float(i) for i in amp[0]]
	return index, amp

def ECG(sig_ECG, index):
	DR = array([0]+[index[i]-index[i-1] for i in range(1,len(index))]).astype(float)
	return DR/sig_ECG.sampleFrequency

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
			PTT += [(peak-idx) / sig_PPG.sampleFrequency]
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
		ran = range(int(start*frequency), int((start+dur)*frequency))
		if stage > 0 and stage < 5:
			for i in ran:
				SS[i] = 0
		elif stage >= 5:
			for i in ran:
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