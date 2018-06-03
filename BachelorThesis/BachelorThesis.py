'''
AUTHOR(S):
Nicklas Hansen
Michael Kirkegaard
'''

from features import process_epochs
from model_selection import evaluate
from preprocessing import prepAll
from dataflow import dataflow
import filesystem as fs

if __name__ == '__main__':
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rr_ppg.h5')
	X, y = fs.load_csv('mesa-sleep-6422')
	y = y * (-1)
	dataflow(X, y, cmd_plot=True)
	#process_epochs()