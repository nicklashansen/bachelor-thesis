# Import GUI
#import sys
#sys.path.insert(0, './GUI')
#from BachelorGUI import BachelorGUI
#from preprocessing import prepAll
from dataflow import process_epochs
import h5py

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