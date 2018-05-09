# Import GUI
#import sys
#sys.path.insert(0, './GUI')
#from BachelorGUI import BachelorGUI
#from preprocessing import prepAll
from dataflow import dataflow, test, fit_eval, process_epochs
import h5py

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	test()
	#dataflow()
	#BachelorGUI()
	#fit_eval(gpu=False)
	#process_epochs()
	#prepAll(force=False)