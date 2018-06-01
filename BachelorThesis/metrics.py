'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

Compute performance values for validation and evaluations using
TP, FP, TN, FN confusion matrix. This module also extracts those
values given a predicted time series yhat and a score series y.
'''

from numpy import *
import settings

def compute_score(y, yhat, timecol, secOverlap=settings.OVERLAP_SCORE, sampleRate=settings.SAMPLE_RATE):
	'''
	Coputes a dictionary over all implemented scoring criteria, given y and yhat.
	Both confusion matrix with and without overlap is computed.
	'''
	scores = {}
	for cm in [cm_standard, cm_overlap]:
		TP,FP,TN,FN = cm(y, yhat, timecol, secOverlap, sampleRate)
		scores[cm.__name__] = compute_cm_score(TP,FP,TN,FN)['score']
	return scores

def compute_cm_score(TP, FP, TN, FN):
	'''
	Coputes a dictionary over all implemented scoring criteria, given TP, FP, TN, FN.
	'''
	scores = {}
	d = scores['score'] = {}
	d['TP_FP_TN_FN'] = TP,FP,TN,FN
	for metric in [accuracy, sensitivity, specificity, precision, f1_score, mcc]:
		d[metric.__name__] = metric(TP,FP,TN,FN)
	return scores

def accuracy(TP,FP,TN,FN):
	return safe_divide(TP+TN, TP+FP+TN+FN)

def sensitivity(TP,FP,TN,FN):
	return safe_divide(TP, TP+FN)

def specificity(TP,FP,TN,FN):
	return safe_divide(TN, TN+FP)

def precision(TP,FP,TN,FN):
	return safe_divide(TP, TP+FP)

def f1_score(TP,FP,TN,FN):
	return safe_divide(2*TP, 2*TP + FP + FN)

def mcc(TP,FP,TN,FN):
	return safe_divide((TP*TN) - (FP*FN), ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2))

def safe_divide(a, b):
	return b if b==0 else a/b

def cm_standard(y, yhat, timecol=None, secOverlap=None, sampleRate=None):
	'''
	Computes confusion matrix timestep by timestep
	'''
	assert(len(y) == len(yhat))
	n = len(y)

	TP=FP=TN=FN = 0
	for i in range(n):
		a = bool(yhat[i])
		b = bool(y[i])
		TP += int(a & b)
		TN += int((not a) & ( not b))
		FP += int(a & (not b))
		FN += int(b & (not a))
	return TP,FP,TN,FN

def cm_overlap(y, yhat, timecol, secOverlap, sampleRate):
	'''
	Computes confusion matrix with overlap and by combining series of 1's into single units.
	'''
	assert(len(y) == len(yhat) == len(timecol))
	n = len(y)
	
	class Arousal:
		'''
		Container class for checking overlap between arousals
		'''
		# Initilise
		def __init__(self, start, end):
			self.start = start
			self.end = end
			self.min = self.__getVal(start, -1) ; self.min = 0 if self.min < 0 else self.min
			self.max = self.__getVal(end, 1) ; self.max = n if self.max > n else self.max

		# the min/max distance is calculated and stored
		def __getVal(self, init, inc):
			dist = int((secOverlap*sampleRate)/2) # /2 because both arousals have overlap
			i = 0
			while(0 <= init+i < n and abs(timecol[init]-timecol[init+i]) <= dist):
				i += inc
			return init+i+(inc*-1)
		
		def compareTo(self,other):
			'''
			Comparsison method for arousal, self, and arousal, other.
			0        : Self == Other
			negative : Self < Other
			positive : Other < Self
			'''
			if self.min <= other.min <= self.max or self.min <= other.max <= self.max:
				return 0
			return self.min - other.max

	# Extracts all arousals series from time series
	def a_transform(y):
		ax = []
		a = None
		for i,val in enumerate(y):
			if val == 1 and not a:
				a = i
			elif val == 0 and a:
				ax += [Arousal(a,i)]
				a = None
		return ax

	y = a_transform(y)
	yhat = a_transform(yhat)
	
	# Iteration over both y and yhat's arousal simultaniously
	i=j = 0				# index values for y and yhat
	tpfptnfn = [None]*n # timestep labels
	TP=FP=TN=FN = 0		# confusion matrix valus
	while (i<len(y) or j<len(yhat)):
		# Previous checked Scored Arousal overlaps
		if(i > 0 and j < len(yhat) and y[i-1].compareTo(yhat[j])==0):
			#
			for z in range(yhat[j].start if yhat[j].start >= 0 else 0, yhat[j].end if yhat[j].end < n else n):
				tpfptnfn[z] = 'TP' # True Positive
			#
			TP += 1
			j += 1
		# Previous checked predicted arousal
		elif(j > 0 and i < len(y) and y[i].compareTo(yhat[j-1])==0):
			#
			for z in range(y[i].start if y[i].start >= 0 else 0, y[i].end if y[i].end < n else n):
				tpfptnfn[z] = 'TP' # True Positive
			#
			TP += 1
			i += 1
		# All scored arousals checked => rest of predicted = FP
		elif i == len(y):
			#
			for z in range(yhat[j].start if yhat[j].start >= 0 else 0, yhat[j].end if yhat[j].end < n else n):
				tpfptnfn[z] = 'FP' # False Positive
			#
			FP += 1
			j += 1
		# All predicted arousals checked => rest of scored = FN
		elif j == len(yhat):
			#
			for z in range(y[i].start if y[i].start >= 0 else 0, y[i].end if y[i].end < n else n):
				tpfptnfn[z] = 'FN' # False Negative
			#
			FN += 1
			i += 1
		else:
			co = y[i].compareTo(yhat[j])
			# y_i and yhat_i overlaps => TP++, increase i,j
			if co == 0:
				#
				for z in range(yhat[j].start if yhat[j].start >= 0 else 0, yhat[j].end if yhat[j].end < n else n):
					tpfptnfn[z] = 'TP' # True Positive
				for z in range(y[i].start if y[i].start >= 0 else 0, y[i].end if y[i].end < n else n):
					tpfptnfn[z] = 'TP' # True Positive
				#
				TP += 1
				i += 1
				j += 1
			# y_i comes before yhat_j => FN++, increase i
			elif co < 0:
				#
				for z in range(y[i].start if y[i].start >= 0 else 0, y[i].end if y[i].end < n else n):
					tpfptnfn[z] = 'FN' # False Negative
				#
				FN += 1
				i += 1
			# yhat_j comes before y_i => FP++, increase j
			else: #co > 0
				#
				for z in range(yhat[j].start if yhat[j].start >= 0 else 0, yhat[j].end if yhat[j].end < n else n):
					tpfptnfn[z] = 'FP' # False Positive
				#
				FP += 1
				j += 1
	
	# sum of all timesteps not labeled TP or FP
	neg = [l for l in tpfptnfn if l not in ['TP', 'FP']]
	fn = neg.count('FN')		# sum of all timesteps labeled FN
	tn = neg.count(None)		# sum of all timesteps supposed to be TN
	TN = int(FN * (tn / fn))	# TN estimated

	return TP,FP,TN,FN