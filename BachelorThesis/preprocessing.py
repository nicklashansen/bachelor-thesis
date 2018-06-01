'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

Converting raw signals into measurable values. This file is responsible for
the main chunk of pre-processing, while sorting of reliable files, correcting
extreme outliers and epoch generations is handles in features.py and epoch.py 
'''
from numpy import *
from PPGpeak_detector import PPG_Peaks
import filesystem as fs
import matlab.engine
import os
from log import get_log
from stopwatch import stopwatch
import settings

def prep_X(edf, anno):
	'''
	converts two edf and anno filepaths into X
	'''
	sub = fs.Subject(edfPath = edf, annoPath = anno, ArousalAnno=False)
	return preprocess(sub, arousals=False)

def prepSingle(filename, save = True):
	'''
	preprocesses a single file by name - used for testing
	'''
	sub = fs.Subject(filename=filename)
	X, y = preprocess(sub)
	if save:
		fs.write_csv(filename, X, y)
	return X, y

def prepAll(force=False):
	'''
	preprocesses all files stored on properly on the drive. either mesa or shhs.
	the amount of time and errors are logged for testing purposes. optionally
	re-processing already completed files is possible.
	'''
	log, clock = get_log('Preprocessing', echo=True), stopwatch()
	filenames = fs.getAllSubjectFilenames(preprocessed=False)

	# determines already completed files
	oldFiles = fs.getAllSubjectFilenames(preprocessed=True)
	if not force:
		filenames = [fn for fn in filenames if fn not in oldFiles]
		log.print('Files already completed:   {0}'.format(len(oldFiles)))
		log.print('Files remaining:           {0}'.format(len(filenames)))
		if(len(oldFiles) > 0):
			log.printHL()
			for fn in oldFiles:
				log.print('{0} already completed'.format(fn))
	else:
		log.print('Files re-preprocessing:    {0}'.format(len(oldFiles)))
		log.print('Files remaining:           {0}'.format(len(filenames)))
	log.printHL()

	# processes each file with try/catch loop incase of errors in single files.
	clock.round()
	for i, filename in enumerate(filenames):
		try:
			subject = fs.Subject(filename=filename)
			X, y = preprocess(subject)
			fs.write_csv(filename, X, y)
			log.print('{0} preprocessed in {1}s'.format(filename, clock.round()))
		except Exception as e:
			log.print('{0} Exception: {1}'.format(filename, str(e)))
			clock.round()
	clock.stop()

def preprocess(subject, arousals = True):
	'''
	Preprocessing of a single subject. each feature and annotation is
	handled by submodules. This method creates X matrix and y vector.
	'''
	# Get data signals from container
	sig_ECG = subject.ECG_signal
	sig_PPG = subject.PPG_signal
	anno_SleepStage = subject.SleepStage_anno
	
	# ----- hardcode

	Xt = transpose(fs.load_csv(subject.filename)[0])
	rpeaks = Xt[0]
	ppeaks, _ = PPG_Peaks(sig_PPG.signal, sig_PPG.sampleFrequency)
	from plots import plot_data
	from peakutils import baseline

	#plot_data([sig_ECG.signal-baseline(sig_ECG.signal)], [None, array([p for i,p in enumerate(rpeaks) if Xt[3,i] == -1]).astype(int), None, rpeaks.astype(int)], ['ECG', 'PTT fail', None, 'R'], False, (0,int(sig_PPG.duration)))
	#plot_data([sig_ECG.signal-baseline(sig_ECG.signal)-1, sig_PPG.signal-baseline(sig_PPG.signal)+1], [rpeaks.astype(int), array(ppeaks).astype(int)], ['ECG', 'PPG'], False, (0,int(sig_PPG.duration-10)))
	#plot_data([sig_PPG.signal], [rpeaks.astype(int), array(ppeaks).astype(int)], ['R', 'P'], False, (0,int(sig_PPG.duration-10)))
	plot_data([sig_PPG.signal], [None, None, array(ppeaks).astype(int), rpeaks.astype(int)], ['PPG', 'PTT Failed', 'P', 'R'], False, (0,int(sig_PPG.duration)))

	# ----- hardcode

	# Gets R-peak indexes and amplitudes
	index, amp = QRS(subject)
	
	# Preprocess Features
	x_RR, x_RWA = RR(subject.frequency, index), array(amp).astype(float)
	x_PTT, x_PWA = PPG(sig_PPG, index)
	x_SS = SleepStageBin(anno_SleepStage, subject.frequency, index)

	# Collect Matrix
	features = [index, x_RR, x_RWA, x_PTT, x_PWA, x_SS]
	X = empty((len(features), len(x_RR)-1))
	for i,feat in enumerate(features):
		X[i] = feat[1:len(feat)]
	X = transpose(X)

	# Include Arousals
	if arousals:
		anno_Arousal = subject.Arousal_anno
		y_AA = ArousalBin(anno_Arousal, subject.frequency, index)
		y = array(y_AA[1:len(y_AA)])
		return X, y
	return X

def QRS(subject):
	'''
	Get R-peak indeses and amplitudes from Mads Olsens QRS algorithm using matlab engine
	'''
	# Start Matlab Engine
	eng = matlab.engine.start_matlab()
	os.makedirs(fs.Filepaths.Matlab, exist_ok=True)
	eng.cd(fs.Filepaths.Matlab)
	if subject.filename:
		edf = subject.filename + '.edf'
	else:
		edf = subject.edfPath
	if not settings.SHHS:
		index, amp = eng.peak_detect(fs.directory(), edf, float(subject.frequency), nargout=2)
	else:
		index, amp = eng.peak_detect_shhs(fs.directory(), edf, float(subject.frequency), nargout=2)
	index = [int(i) for i in index[0]]
	amp = [float(i) for i in amp[0]]
	return index, amp

def RR(freq, index):
	'''
	Calculates RR distance for each peak after the first
	'''
	DR = array([0]+[index[i]-index[i-1] for i in range(1,len(index))]).astype(float)
	return DR/freq # durations in seconds

def PPG(sig_PPG, index):
	'''
	Calculates PTT for each R-index
	'''
	# peaks and amplitudes from PPGpeak_detector.py
	peaks, amps = PPG_Peaks(sig_PPG.signal, sig_PPG.sampleFrequency)
	
	# finds P-peak between two r-peaks
	def find_RpR(idx, idxplus, h):
		# Find h index of peak after R-peak
		while h < len(peaks) and peaks[h] < idx:
			h += 1
		# Errors
		if h >= len(peaks) or peaks[h] >= idxplus:
			return -1, h-1
		# peakid, index of ppg-peakid
		return peaks[h], h

	# attempts to find a peak between all RR-intervals
	PTT = []
	PWA = []
	h = -1
	for i,idx in enumerate(index):
		idxplus = index[i+1] if i < len(index)-1 else sig_PPG.duration
		peak, h = find_RpR(idx,idxplus,h+1)
		# if peak is found, store both ptt and pwa, else mark as error '-1'
		if(peak != -1):
			PTT += [(peak-idx) / sig_PPG.sampleFrequency]
			PWA += [amps[h]]
		else:
			PTT += [-1]
			PWA += [-1]
	
	# store as floats
	PTT = array(PTT).astype(float)
	PWA = array(PWA).astype(float)

	return PTT, PWA

def SleepStageBin(anno_SleepStage, frequency, index):
	'''
	converts sleep stage containers into a value sleep stage
	scorings into series of {-1,0,1} for each R-index.
	'''
	#	0		= WAKE	=> -1
	#	1,2,3,4	= NREM	=>  0
	#	5		= REM	=>  1
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

	# Extracts sleep stage for each R-index
	SS = array([SS[idx] for idx in index]).astype(float)
	return SS


def ArousalBin(anno_Arousal, frequency, index):
	'''
	converts arousal containers into a value for each R-index
	'''
	# 0 = not arousal
	# 1 =     arousal
	AA = [0]*int(anno_Arousal.duration*frequency)
	xlen = len(AA)
	for start,dur,_ in anno_Arousal.annolist:
		start = float(start)
		dur = float(dur)
		for i in range(int(start*frequency), int((start+dur)*frequency)):
			AA[i] = 1

	# Extracts arousal score for each R-peak
	AA = array([AA[int(idx)] for idx in index]).astype(float)
	return AA