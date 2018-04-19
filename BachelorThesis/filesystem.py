import os
from numpy import *
import pyedflib
from os import listdir, makedirs
from os.path import isfile, join
import xml.etree.ElementTree as xmlTree

"""
WRITTEN BY:
Nicklas Hansen,
Michael Kirkegaard
"""

def directory():
	path = os.path.dirname(os.path.abspath(__file__))
	i,j = len(path),0
	while (j!=1):
		i = i-1
		if path[i] == '\\':
			j = j + 1
	return path[0:i+1]

class Filepaths(object):
	# Folder 
	Folder = directory() + 'Files\\'
	Matlab = directory() + 'Matlab\\'
	Logs = directory() + 'Logs\\'

	# Save paths
	SaveSubject = Folder + 'Subjects\\'

	# Load paths
	LoadCsv = Folder + 'Data\\mesa\\datasets\\mesa-sleep-dataset-0.3.0.csv'
	LoadPsg = Folder + 'Data\\mesa\\polysomnography\\edfs\\'
	LoadAnno = Folder + 'Data\\mesa\\polysomnography\\annotations-events-nsrr\\'

class Annotation(object):
	def __init__(self, label, annolist, dur):
		self.label = label
		self.annolist = annolist
		self.duration = dur

class Signal(object):
	def __init__(self, label, sig, freq, dur):
		self.label = label
		self.signal = sig
		self.sampleFrequency = freq
		self.duration = dur

class Subject(object):
	def __init__(self, filename):
		self.id = int(filename[-4:])
		self.filename = filename

		self.ECG_signal = self.get_signal('EKG')
		self.PPG_signal = self.get_signal('Pleth')
		self.SleepStage_anno = self.get_anno('Stages')
		self.Arousal_anno = self.get_anno('Arousal')

		assert(self.ECG_signal.duration == self.PPG_signal.duration)
		assert(self.ECG_signal.duration <= self.SleepStage_anno.duration)
		assert(self.ECG_signal.duration <= self.Arousal_anno.duration)
		self.duration = self.ECG_signal.duration

		assert(self.ECG_signal.sampleFrequency == self.PPG_signal.sampleFrequency)
		self.frequency = self.ECG_signal.sampleFrequency

	def get_signal(self, label):
		filepath = Filepaths.LoadPsg + self.filename + ".edf"
		with pyedflib.EdfReader(filepath) as file:

			id = file.getSignalLabels().index(label)
			sig = file.readSignal(id)
			freq = file.getSampleFrequency(id)
			dur = len(sig) / freq

		return Signal(label, sig, freq, dur)

	def get_anno(self, label):	
		filepath = Filepaths.LoadAnno + self.filename + "-nsrr.xml"
		xml = xmlTree.parse(filepath).getroot()

		dict = self.make_dict_from_tree(xml)

		eventlist = dict['PSGAnnotation']['ScoredEvents']['ScoredEvent']
		dur = sorted([float(d['Start']) + float(d['Duration']) for d in eventlist])[-1]
		
		ax = [[d['Start'], d['Duration'], d['EventConcept'].split('|')[1]] for d in eventlist if d['EventType'] and label.lower() in d['EventType'].lower()]
		ax = array(sorted(ax, key = lambda x: x[0]))
		
		return Annotation(label, ax, dur)

	''' https://ericscrivner.me/2015/07/python-tip-convert-xml-tree-to-a-dictionary/ '''
	def make_dict_from_tree(self, element_tree):
		def internal_iter(tree, accum):
			if tree is None:
				return accum	
			if tree.getchildren():
				accum[tree.tag] = {}
				for each in tree.getchildren():
					result = internal_iter(each, {})
					if each.tag in accum[tree.tag]:
						if not isinstance(accum[tree.tag][each.tag], list):
							accum[tree.tag][each.tag] = [accum[tree.tag][each.tag]]
						accum[tree.tag][each.tag].append(result[each.tag])
					else:
						accum[tree.tag].update(result)
			else:
				accum[tree.tag] = tree.text
			return accum
		return internal_iter(element_tree, {})

def getAllSubjectFilenames(preprocessed=False):
	if(preprocessed):
		path = Filepaths.SaveSubject
	else:
		path = Filepaths.LoadPsg
	datasetCSV = ''
	filenames = [f[:-4] for f in listdir(path) if isfile(join(path, f))]
	return filenames, datasetCSV

def load_csv(filename):
	path = Filepaths.SaveSubject + filename + '.csv'
	data = []
	with open(path, "r", encoding='utf8') as f:
		for line in f:
			cols = line.replace('\n','').split(',')
			cols = array(cols).astype(float)
			data.append(cols)
	data = array(data)
	X = data[:,:-1]
	y = array(data[:,-1])
	return X, y

def write_csv(filename, X, y):
	os.makedirs(Filepaths.SaveSubject, exist_ok=True)
	path = Filepaths.SaveSubject + filename + '.csv'

	Xy = insert(X, X.shape[1], y, axis=1)
	with open(path, 'w+', encoding='utf8') as f:
		for row in Xy:
			s = ''
			for val in row:
				s = s + str(val) + ','
			s = s[:-1]
			f.write(s + '\n')

def write(directory, filename, line=None, wra='a'):
	os.makedirs(directory, exist_ok=True)
	with open(directory + filename, wra, encoding='utf8') as f:
		f.write(line + '\n')