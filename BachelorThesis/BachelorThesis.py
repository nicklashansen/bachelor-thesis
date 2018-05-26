from features import process_epochs, make_splits
from dataflow import test_dataflow, test_dataflow_LR
from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate, evaluate_LR
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
	#test_dataflow()
	#test_dataflow_LR()
	#evaluate_LR()
	make_splits()
	#process_epochs()