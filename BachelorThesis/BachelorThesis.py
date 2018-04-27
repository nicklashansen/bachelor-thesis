from stopwatch import *
from preprocessing import *
from features import *
from dataflow import flow_all
import filesystem as fs

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	flow_all()
	#filename = 'mesa-sleep-0002'
	#X, y = prepSingle(filename)
	#X_, y_ = fs.load_csv(filename)
	#epochs = make_features(X_, y_)
	#data = dataset()
	#prepAll(force=False)
	breakpoint = 0