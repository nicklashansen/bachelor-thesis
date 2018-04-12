import os
from numpy import array

"""
WRITTEN BY:
Nicklas Hansen
"""

folder = 'BachelorThesis\\BachelorThesis\\Files\\'

def load_csv(s, type=float):
	X = []
	with open(get(s), "r", encoding='utf8') as f:
		for line in f:
			cols = array(line.replace('\n','').split(','))
			cols = cols.astype(type)
			X.append(cols)
	return array(X)

def write_csv(s, data, wa="w"):
	with open(get(s), wa, encoding='utf8') as f:
		for line in data:
			s = ''
			if (data.ndim > 1):
				for feature in line:
					s = s + str(feature) + ','
				s = s[:len(s)-1]
			else:
				s = str(line)
			f.write(s + '\n')

def get(s, endtag='.csv'):
	path = directory() + s + endtag
	return path

def directory(s=folder):
	path = os.path.dirname(os.path.abspath(__file__))
	if type(path) == str:
		i,j = len(path),0
		while (j!=2):
			i = i-1
			if path[i] == '\\':
				j = j + 1
		return path[0:i+1] + s
	return None	