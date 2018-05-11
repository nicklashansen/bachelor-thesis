from numpy import *

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

def compute_score(y, yhat, secOverlap=3.0, sampleRate=256):
	scores = {}
	for cm in [cm_standard, cm_overlap]:
		d = scores[cm.__name__] = {}
		TP,FP,TN,FN = cm(y, yhat, secOverlap, sampleRate)
		d['TP_FP_TN_FN'] = TP,FP,TN,FN
		for metric in [accuracy, sensitivity, specificity, precision, f1_score, mcc]:
			d[metric.__name__] = metric(TP,FP,TN,FN)
	return scores

def compute_cm_score(TP, FP, TN, FN):
	scores = {}
	d = scores['score'] = {}
	d['TP_FP_TN_FN'] = TP,FP,TN,FN
	for metric in [accuracy, sensitivity, specificity, precision, f1_score, mcc]:
		d[metric.__name__] = metric(TP,FP,TN,FN)
	return scores

def accuracy(TP,FP,TN,FN):
	return divide(TP+TN, TP+FP+TN+FN)

def sensitivity(TP,FP,TN,FN):
	return divide(TP, TP+FN)

def specificity(TP,FP,TN,FN):
	return divide(TN, TN+FP)

def precision(TP,FP,TN,FN):
	return divide(TP, TP+FP)

def f1_score(TP,FP,TN,FN):
	return divide(2*TP, 2*TP + FP + FN)

def mcc(TP,FP,TN,FN):
	return divide((TP*TN) - (FP*FN), sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

def divide(a, b):
	return b if b==0 else a/b

def cm_standard(y, yhat, secOverlap=None, sampleRate=None):
	assert(len(y) == len(yhat))
	n = len(y)
	TP=FP=TN=FN=0
	for i in range(n):
		a = bool(yhat[i])
		b = bool(y[i])
		TP += int(a & b)
		TN += int((not a) & ( not b))
		FP += int(a & (not b))
		FN += int(b & (not a))
	return TP,FP,TN,FN

def cm_overlap(y, yhat, timecol, secOverlap, sampleRate):
	assert(len(y) == len(yhat) == len(timecol))
	n = len(y)
	
	class Arousal:
		def __init__(self, start, end):
			self.min = self.__getVal(start, -1) ; self.min = 0 if self.min < 0 else self.min
			self.max = self.__getVal(end, 1) ; self.max = n if self.max > n else self.max

		def __getVal(self, init, inc):
			dist = int((secOverlap*sampleRate)/2) # both arousals have overlap
			i = 0
			while(0 <= init+i < n and abs(timecol[init]-timecol[init+i]) <= dist):
				i += inc
			return init+i+(inc*-1)
			
		def compareTo(self,other):
			if self.min <= other.min <= self.max or self.min <= other.max <= self.max:
				return 0
			return self.min - other.max

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

	TP=FP=TN=FN = 0
	i=j = 0
	while (i<len(y) or j<len(yhat)):
		# Previous checked Scored Arousal overlaps
		if(i > 0 and j < len(yhat) and y[i-1].compareTo(yhat[j])==0):
			TP += 1
			j += 1
		# Previous checked predicted arousal
		elif(j > 0 and i < len(y) and y[i].compareTo(yhat[j-1])==0):
			TP += 1
			i += 1
		# All scored arousals checked => rest of predicted = FP
		elif i == len(y):
			FP += len(yhat)-j
			j = len(yhat)
		# All predicted arousals checked => rest of scored = FN
		elif j == len(yhat):
			FN += len(y)-i
			i = len(y)
		else:
			co = y[i].compareTo(yhat[j])
			# y_i and yhat_i overlaps => TP++, increase i,j
			if co == 0:
				TP += 1
				i += 1
				j += 1
			# y_i comes before yhat_j => FN++, increase i
			elif co < 0:
				FN += 1
				i += 1
			# yhat_j comes before y_i => FP++, increase j
			else: #co > 0
				FP += 1
				j += 1

	return TP,FP,TN,FN