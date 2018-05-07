# Import GUI
#import sys
#sys.path.insert(0, './GUI')
#from BachelorGUI import BachelorGUI
import h5py

from preprocessing import prepAll
from dataflow import dataflow, fit_eval, process_epochs

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	#dataflow()
	#BachelorGUI()
	#fit_eval(gpu=False)
	process_epochs()
	#prepAll(force=False)