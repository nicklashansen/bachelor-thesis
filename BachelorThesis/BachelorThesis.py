from features import process_epochs, make_splits, hours_of_sleep_files
#from dataflow import test_dataflow, test_dataflow_LR
#from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate, evaluate_LR, predict_file
from preprocessing import prepAll, prepSingle
#from metrics import compute_score
#import h5py

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	#parameter_tuning(evaluate_model=False)
	#test_bidirectional()
	#fit_validate_test(only_arousal = True)
	#evaluate(validation=False)
	#test_dataflow(file='mesa-sleep-4618')
	count_epochs_removed()
	#test_dataflow_LR()
	#evaluate_LR()
	t = hours_of_sleep_files()
	#make_splits()
	#process_epochs()