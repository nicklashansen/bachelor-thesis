from stopwatch import stopwatch
from lstm import lstm
from dataset import dataset
from preprocessing import *
import metrics

"""
WRITTEN BY:
Nicklas Hansen
"""

def eval():
	neurons = 20
	folds = 5
	print('Loading data...')
	data = dataset()
	#data.load_physionet()	# PhysioNet dataset
	data.load_example(1000, 60)	# Example dataset
	model = lstm(data, neurons)
	score = model.cross_val(data.kfold(folds))
	print(metrics.compute_score(score, metrics.TPR_FNR).items())

if __name__ == '__main__':
	eval()
	#preprocess()