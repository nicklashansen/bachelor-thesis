from features import process_epochs, make_splits
from dataflow import test_dataflow
from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate
import h5py

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
	test_dataflow()
	#make_splits()
	#process_epochs()