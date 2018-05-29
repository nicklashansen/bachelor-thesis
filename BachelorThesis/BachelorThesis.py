from features import process_epochs, make_splits#, hours_of_sleep_files
#from dataflow import test_dataflow, test_dataflow_LR
from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate, evaluate_LR, predict_file
from preprocessing import prepAll, prepSingle
#from metrics import compute_score
#import h5py
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

#def test():
#	import numpy as np
#	import matplotlib.pyplot as plt
#	import features
#	import scipy.stats as stats
#	import settings

#	X,y = fs.load_csv(fs.load_splits()[2][0])
#	Xt = np.transpose(X)

#	_X,_,_ = features.make_features(X,y,settings.SAMPLE_RATE)
#	_Xt = np.transpose(_X)

#	for i in range(1,5):
#		sig = sorted([s for s in Xt[i] if s != -1])
#		fit = stats.norm.pdf(sig, np.mean(sig), np.std(sig))
#		plt.plot(sig,fit,'-')
#		plt.hist(sig,bins=100, normed=True)
#		plt.show()

#		_sig = sorted(_Xt[i])
#		_fit = stats.norm.pdf(_sig, np.mean(_sig), np.std(_sig))
#		plt.plot(_sig,_fit,'-')
#		plt.hist(_sig,bins=100, normed=True)
#		plt.show()

if __name__ == '__main__':
	#parameter_tuning(evaluate_model=False)
	#test_bidirectional()
	#fit_validate_test(only_arousal = True)
	#evaluate(validation=False)
	#test_dataflow(file='mesa-sleep-4618')
	#count_epochs_removed()
	#test_dataflow_LR()
	#evaluate_LR()
	#hours_of_sleep_files()
	#make_splits()
	#process_epochs()
	prepSingle(fs.load_splits()[2][0], False)
	#test()