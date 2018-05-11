# Import GUI
#import sys
#sys.path.insert(0, './GUI')
#from BachelorGUI import BachelorGUI
#from preprocessing import prepAll
from dataflow import dataflow, fit_validate, evaluate, process_epochs
import h5py

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	#fit_validate(gpu=True, balance = True)
	#evaluate()
	#dataflow()
	#BachelorGUI()
	process_epochs()
	#prepAll(force=False)