from time import gmtime, strftime
import filesystem as fs
import os

'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

Module is a framework for logging activities from other modules.
'''

TABLE = dict()

def get_log(directory = 'Evaluation', echo = False):
	'''
	Returns the desired log file if it exists, otherwise a new one is created for this instance of the program.
	'''
	if directory not in TABLE:
		TABLE[directory] = Log(directory, echo)
	else:
		TABLE[directory].printHL()
	TABLE[directory].echo = echo
	return TABLE[directory]

class Log:
	'''
	Class responsible for logging activities.'
	'''
	def __init__(self, directory = 'Evaluation', echo = False):
		'''
		Creates a new file at the desired destination.
		If echo is true, all prints are echoed to the console as well.
		'''
		self.filename = strftime("%Y-%m-%d_%H-%M", gmtime()) + '_log.txt'
		self.directory = fs.Filepaths.Logs + directory + '\\'
		self.echo = echo
		os.makedirs(self.directory, exist_ok=True)
		open(self.directory + self.filename, 'w+', encoding='utf8').close()

	def print(self, line):
		'''
		Prints a line to a given log.
		'''
		fs.append_log(self.directory, self.filename, line)
		if self.echo:
			try:
				print(line)
			except Exception as e:
				None

	def printHL(self):
		'''
		Prints a horizontal line for styling purposes.
		'''
		self.print('-'*35)