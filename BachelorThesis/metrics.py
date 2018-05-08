from numpy import *

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

def compute_scores(y, yhat, secOverlap=3.0, sampleRate=256):
	scores = {}
	for cm in [cm_standard, cm_overlap]:
		d = scores[cm.__name__] = {}
		TP,FP,TN,FN = cm(y, yhat, secOverlap, sampleRate)
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

def cm_overlap(y, yhat, secOverlap, sampleRate):
	assert(len(y) == len(yhat))
	n = len(y)
	
	class Arousal:
		def __init__(self, n, start, end):
			self.n = n
			self.start = start
			self.end = end
			self.combined = end-start

		def compareTo(self,other):
			min = self.start-(secOverlap*sampleRate) ; min = 0 if min < 0 else min
			max = self.end+(secOverlap*sampleRate) ; max = n if max > n else max
			if min <= other.start <= max or min <= other.end <= max:
				return 0
			return 1 if min > other.end else -1

	def transform(y, dontCombine):
		ax = []
		a = None
		for i,val in enumerate(y):
			if val == 1 and not a:
				a = i
			elif val == 0 and a:
				if dontCombine or not ax or Arousal(n,a,i).compareTo(ax[-1]) > 0 or i == len(y):
					ax += [Arousal(n,a,i)]
				else:
					ax[-1].end = i
					ax[-1].combined += i-a
				a = None
		return ax

	y = transform(y, True)
	yhat = transform(yhat, False)

	TP=FP=TN=FN = 0
	i=j = 0
	while (i<len(y) or j<len(yhat)):
		# All scored arousals checked => rest of predicted = FP
		if i == len(y):
			FP += len(yhat)-j
			j = len(yhat)
		# last predicted arousal is combined but overlaps => split yy[i], TP++, increase j
		elif j > 0 and yhat[j-1].combined > 0 and yhat[j-1].compareTo(y[i]) == 0:
			yhat[j-1].combined -= y[i].combined
			TP += 1
			i += 1
		# All predicted arousals checked => rest of scored = FN
		elif j == len(yhat):
			FN += len(y)-i
			i = len(y)
		else:
			co = y[i].compareTo(yhat[j])
			# yy_i and yyhat_i overlaps => TP++, increase i,j
			if co == 0:
				yhat[j].combined -= y[i].combined
				TP += 1
				i += 1
				j += 1
			# yy_i comes before yyhat_j => FN++, increase i
			elif co < 0:
				FN += 1
				i += 1
			# yy_i comes after yyhat_j => FP++, increase j
			else: #co > 0
				FP += 1
				j += 1

	return TP,FP,TN,FN

'''
y =    [0,0,0,1,1,1,0,0,1,1,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0]
yhat = [1,1,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1]

std = 2,7,12,8
ovl = 3,1,0 ,1
scores = compute_scores(y,yhat,3,1)
breakpoint = 0
'''