from numpy import *

"""
WRITTEN BY:
Nicklas Hansen
"""

def compute_score(score, metric):
	result,dict,metrics = [],{},[]
	if (metric==TPR_FNR):
		metrics.append('tpr')
		metrics.append('fnr')
	for j in range(len(score[0])):
		x = []
		for i in range(len(score)):
			y = score[i]
			x.append(y[j])
		result.append(mean(array(x)))
	for k in range(len(result)):
		dict[metrics[k]] = result[k]
	return dict

def F1(y, yhat):
	TP,FP,TN,FN = cm(y, yhat)
	return 2*TP / (2*TP + FP + FN)

def TPR_TNR(y, yhat):
	TP,FP,TN,FN = cm(y, yhat)
	TPR = TP / (TP + FP)
	TNR = TN / (TN + FP)
	return TPR, TNR

def TPR_FNR(y, yhat):
	TP,FP,TN,FN = cm(y, yhat)
	TPR = TP / (TP + FP)
	FNR = FN / (TP + FN)
	return TPR, FNR

def cm(y, yhat):
	yhat = squeeze(yhat)
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