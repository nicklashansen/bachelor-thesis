from stopwatch import stopwatch
from lstm import gru
from dataset import dataset
from preprocessing import prepAll
import filesystem as fs
import metrics

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

def eval():
	neurons = 20
	folds = 5
	print('Loading data...')
	data = dataset()
	data.load_mesa(summary=True)	# Mesa dataset
	#data.load_physionet()			# PhysioNet dataset
	#data.load_example(1000, 60)	# Example dataset
	model = gru(data, neurons)
	score = model.cross_val(data.kfold(folds))	# K-fold
	#score = model.cross_val(data.holdout())	# Holdout
	print(metrics.compute_score(score, metrics.TPR_FNR).items())

if __name__ == '__main__':
	prepAll()
	eval()