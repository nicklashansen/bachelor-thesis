from stopwatch import *
from preprocessing import *
from features import *
#	from dataflow import *
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

'''
def eval(neurons = 20, folds = 5):
	data = dataset()
	data.load_example(1000, 60)
	model = gru(data, neurons)
	score = model.cross_val(data.kfold(folds))
	print(metrics.compute_score(score, metrics.TPR_FNR).items())
#'''

if __name__ == '__main__':
	#flow_all()
	#filename = 'mesa-sleep-0002'
	#X, y = prepSingle(filename)
	#X_, y_ = fs.load_csv(filename)
	#epochs = make_features(X_, y_)
	#data = dataset()
	prepAll(force=False)
	#eval()
	breakpoint = 0