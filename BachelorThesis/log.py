from time import gmtime, strftime
import filesystem as fs
import os

"""
WRITTEN BY
Nicklas Hansen
"""

TABLE = dict()

def get_log(directory = 'Evaluation', echo = False):
	if directory not in TABLE:
		TABLE[directory] = Log(directory, echo)
	else:
		TABLE[directory].printHL()
	TABLE[directory].echo = echo
	return TABLE[directory]

class Log:
	def __init__(self, directory = 'Evaluation', echo = False):
		self.filename = strftime("%Y-%m-%d_%H-%M", gmtime()) + '_log.txt'
		self.directory = fs.Filepaths.Logs + directory + '\\'
		self.echo = echo
		os.makedirs(self.directory, exist_ok=True)
		open(self.directory + self.filename, 'w+', encoding='utf8').close()

	def print(self, line):
		fs.write(self.directory, self.filename, line)
		if self.echo:
			try:
				print(line)
			except Exception as e:
				None

	def printHL(self):
		self.print('-'*35)