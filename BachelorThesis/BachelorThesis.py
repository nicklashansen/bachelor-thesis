from features import process_epochs, make_splits, hours_of_sleep_files
from dataflow import test_dataflow, test_dataflow_LR
#from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate, evaluate_LR, predict_file
from model_selection import evaluate
from preprocessing import prepAll, prepSingle
#from metrics import compute_score
#import h5py

"""
AUTHOR(S):
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	#parameter_tuning(evaluate_model=False)
	#train_validate_test()
	#fit_validate_test(only_arousal = True)
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rr.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rwa.h5')
	evaluate(validation=False, path='FEATURE_SELECTION\\best_ecg.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_ppg.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rr_ppg.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rwa_ppg.h5')
	#test_dataflow(file='mesa-sleep-1541')
	#count_epochs_removed()
	#test_dataflow_LR()
	#evaluate_LR()
	#t = hours_of_sleep_files()
	#make_splits()
	#process_epochs()