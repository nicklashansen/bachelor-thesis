from time import gmtime, strftime
import filesystem as fs
import os

"""
WRITTEN BY
Nicklas Hansen
"""

class log:
	def __init__(self, directory = 'Evaluation'):
		self.filename = strftime("%Y-%m-%d_%H-%M", gmtime()) + '_log.txt'
		self.directory = fs.Filepaths.Logs + directory + '\\'
		os.makedirs(self.directory, exist_ok=True)
		open(self.directory + self.filename, 'w+', encoding='utf8').close()

	def print(self, line):
		fs.write(self.directory, self.filename, line)