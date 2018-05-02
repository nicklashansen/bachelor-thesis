# Import GUI
import sys
sys.path.insert(0, './GUI')
from BachelorGUI import BachelorGUI

from preprocessing import prepAll
from dataflow import fit_eval, process_epochs

"""
WRITTEN BY:
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	BachelorGUI()
	#fit_eval()
	#process_epochs()
	#prepAll(force=False)