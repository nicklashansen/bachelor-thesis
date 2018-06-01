'''
AUTHOR(S):
Nicklas Hansen
Michael Kirkegaard
'''

from features import process_epochs, make_splits, hours_of_sleep_files
#from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate, evaluate_LR, predict_file
from model_selection import evaluate, evaluate_LR
from preprocessing import prepAll, prepSingle
from dataflow import dataflow
import filesystem as fs

if __name__ == '__main__':
	#train_validate_test()
	#evaluate(validation=False, path='best_rwa.h5')
	X, _ = fs.load_csv('mesa-sleep-5810')
	dataflow(X, cmd_plot=True)
	#make_splits()
	#process_epochs()