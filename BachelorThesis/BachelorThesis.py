#from stopwatch import *
#from gru import *
#from dataset import *
from preprocessing import *
#import metrics
#import matlab.engine

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
'''

if __name__ == '__main__':
	prepAll()
	#eval()
	#eng = matlab.engine.start_matlab()
	#print(eng.isprime(37))