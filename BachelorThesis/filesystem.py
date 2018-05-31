'''
AUTHOR(S):
Nicklas Hansen,
Michael Kirkegaard

File management for the entire solution. Handles directories, reading- and writing of files.
csv, xml, edf, aplot, epoch-pickle and log use these functions.
'''

import os
from os.path import isfile, join
from numpy import *
import pyedflib
import pandas as pd
import pickle as pck
import xml.etree.ElementTree as xmlTree
import settings

class Filepaths:
	'''
	Container class for filespaths
	'''
	# Directory
	def directory():
		path = os.path.dirname(os.path.abspath(__file__))
		i,j = len(path),0
		while (j!=1):
			i = i-1
			if path[i] == '\\':
				j = j + 1
		return path[0:i+1]

	directory = directory()
	#directory = 'D:\\BachelorThesis\\' # Michael Path (not enough harddrive space)

	# Folder 
	Files = directory + 'Files\\'
	Matlab = directory + 'Matlab\\'
	Model = directory + 'Model\\'
	Logs = directory + 'Logs\\'

	# Save paths
	SaveSubject = Files + 'Subjects\\'
	SaveSplits  = Files + 'Splits\\'
	SaveEpochs = Files + 'Epochs\\'

	# Load paths
	LoadDatabaseCsv = Files + 'Data\\mesa\\datasets\\{0}.csv'.format('mesa-sleep-dataset-0.3.0')
	LoadPsg = Files + 'Data\\mesa\\polysomnography\\edfs\\'
	LoadAnno = Files + 'Data\\mesa\\polysomnography\\annotations-events-nsrr\\'

	# Specific changes for the SHHS dataset
	if settings.SHHS:

		# Save paths
		SaveSubject = SaveSubject[-2] + '_shhs\\'
		SaveSplits  = SaveSplits[-2] + '_shhs\\'
		SaveEpochs = SaveEpochs[-2] + '_shhs\\'

		# Load paths
		LoadDatabaseCsv = Files + 'Data\\mesa\\datasets\\{0}.csv'.format('missing') # Not required
		LoadPsg = LoadPsg.replace('mesa','shhs2')
		loadAnno = LoadAnno.replace('mesa','shhs2')

	# GUI
	TempAplotFile = Files + 'temp.aplot' 

class Annotation:
	'''
	Container class for annotations
	'''
	def __init__(self, label, annolist, dur):
		self.label = label
		self.annolist = annolist
		self.duration = dur

class Signal:
	'''
	Containeer class for signals
	'''
	def __init__(self, label, sig, freq, dur):
		self.label = label
		self.signal = sig
		self.sampleFrequency = freq
		self.duration = dur

class Subject:
	'''
	Container class for subjects. Holds ECG+PPG signals, arousal+sleepsstage annotations, duration and frequncy.
	For mesa and shhs files, it also holds id and filename, but subjects can be instanised only from filepaths.
	'''
	# Initilise from either filename or edf- and annopath
	def __init__(self, filename=None, edfPath=None, annoPath=None, ArousalAnno=True, ArousalOnly=False):
		# if mesa or shhs file
		if filename:
			self.id = int(filename[-4:])
			self.filename = filename
		else:
			self.id = None
			self.filename = None

		# if not only arousals are to be extracted
		if not ArousalOnly:
			self.edfPath = edfPath
			self.annoPath = annoPath

			# shhs does not support PPG, to work with the dataflow, a series of 0's with two peaks are inserterd.
			if settings.SHHS:
				shhs_ppg = Signal('Pleth', zeros(int(self.ECG_signal.duration)), self.ECG_signal.sampleFrequency, self.ECG_signal.duration)
				shhs_ppg[10] = 1 ; shhs_ppg[-10] = 1

			# save signals and annotations
			self.ECG_signal = self.get_signal('EKG|ECG')
			self.PPG_signal = self.get_signal('Pleth') if not settings.SHHS else shhs_ppg
			self.SleepStage_anno = self.get_anno('Stages')
			self.Arousal_anno = self.get_anno('Arousal') if ArousalAnno else None

			# assert files match in duration
			assert(self.ECG_signal.duration == self.PPG_signal.duration)
			assert(self.ECG_signal.duration <= self.SleepStage_anno.duration)
			if ArousalAnno:
				assert(self.ECG_signal.duration <= self.Arousal_anno.duration)
			self.duration = self.ECG_signal.duration

			# assert ecg and ppg was recorded with same frequency
			assert(self.ECG_signal.sampleFrequency == self.PPG_signal.sampleFrequency)
			self.frequency = self.ECG_signal.sampleFrequency

		# only extracts arousal annotations
		else:
			self.annoPath = annoPath
			self.Arousal_anno = self.get_anno('Arousal')
	
	def get_signal(self, label):
		'''
		Reads signal of label from edf-file
		'''
		filepath = Filepaths.LoadPsg + self.filename + ".edf" if not self.edfPath else self.edfPath

		# split label-options and find index of first instance 
		labels = label.split('|')
		with pyedflib.EdfReader(filepath) as file:
			for i,l in enumerate(labels):
				try:
					id = file.getSignalLabels().index(l)
					break
				except:
					if i == len(labels)-1:
						raise

			# read signal and properties
			sig = file.readSignal(id)
			freq = file.getSampleFrequency(id)
			dur = len(sig) / freq # duration in seconds

		# return container
		return Signal(label, sig, freq, dur)

	def get_anno(self, label):
		'''
		Reads annotation of label from xml-file
		'''
		filepath = Filepaths.LoadAnno + self.filename + "-nsrr.xml" if not self.annoPath else self.annoPath

		# parse tree into dictionary
		xml = xmlTree.parse(filepath).getroot()
		dict = make_dict_from_tree(xml)

		# get duration - this is either the 'duration' scoredevent or the last scoredevent's endpoint
		eventlist = dict['PSGAnnotation']['ScoredEvents']['ScoredEvent']
		dur = sorted([float(d['Start']) + float(d['Duration']) for d in eventlist])[-1]
		
		# extracts all annotations for label as onset, duration and concept
		ax = [[d['Start'], d['Duration'], d['EventConcept'].split('|')[1]] for d in eventlist if d['EventType'] and label.lower() in d['EventType'].lower()]
		ax = array(sorted(ax, key = lambda x: x[0]))
		
		# returns container
		return Annotation(label, ax, dur)

def validate_fileformat(edfpath, annopath):
	'''
	Validates fileformat of edf and xml file
	'''
	return validate_edf(edfpath) and validate_anno(annopath)


def validate_edf(edfpath):
	'''
	validates that edf file contains both ecg and pleth signal
	'''
	try:
		with pyedflib.EdfReader(edfpath) as file:
			signals = file.getSignalLabels()
		# true if edf contains any instance of each labeloption
		return all(any(l in signals for l in label.split('|')) for label in ['EKG|ECG', 'Pleth'])
	except Exception as e:
		# Exception if no file exists or is corrupt
		return False


def validate_anno(annopath):
	'''
	validates that xml file is properly formatted and contains sleep stage annotations
	'''
	try:
		xml = xmlTree.parse(annopath).getroot()
		dict = make_dict_from_tree(xml)
		eventlist = dict['PSGAnnotation']['ScoredEvents']['ScoredEvent']
		# true if any event contains sleep stages 
		return any([d['EventType'] and 'stages' in d['EventType'].lower() for d in eventlist])
	except Exception as e:
		# Exception if no file exists or is corrupt
		return False

def getAllSubjectFilenames(preprocessed=False):
	'''
	returns all subject filenames, optinally only preprocessed files
	'''
	if(preprocessed):
		path = Filepaths.SaveSubject
	else:
		path = Filepaths.LoadPsg
	os.makedirs(path, exist_ok=True)
	filenames = [f[:-4] for f in os.listdir(path) if isfile(join(path, f))]
	return filenames

def getDataset_csv():
	'''
	returns the dataset variable container
	'''
	return pd.read_csv(Filepaths.LoadDatabaseCsv)

def load_csv(filename):
	'''
	loads a preprocessed subject - Deserialing into X,y
	'''
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
	'''
	saves a preprocessed subject - Serialising to .csv
	'''
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

def load_splits(name='splits'):
	'''
	loads the training, validation and testing file sets
	'''
	if settings.SHHS:
		return [],[],[]
	file = Filepaths.SaveSplits + name + '.txt'
	with open(file, 'r') as f:
		tte = list(f.readlines())
	train = tte[0].replace('\n','').split(',')
	vali = tte[1].replace('\n','').split(',')
	test = tte[2].replace('\n','').split(',')
	return train,vali,test

def write_splits(train, test, eval, name='splits'):
	'''
	saves the training, validation and testing file sets
	'''
	os.makedirs(Filepaths.SaveSplits, exist_ok=True)
	file = Filepaths.SaveSplits + name + '.txt'
	with open(file, 'w') as f:
		f.write(','.join(train)+'\n')
		f.write(','.join(test)+'\n')
		f.write(','.join(eval))

def load_epochs(name='epochs'):
	'''
	loads generated epochs for model training
	'''
	file = Filepaths.SaveEpochs + name +'.pickle'
	with open(file, 'rb') as f:
		epochs = pck.load(f)
	return epochs

def write_epochs(epochs, name='epochs'):
	'''
	saves generated epochs for model training
	'''
	os.makedirs(Filepaths.SaveEpochs, exist_ok=True)
	file = Filepaths.SaveEpochs + name + '.pickle'
	with open(file, 'wb') as handle:
		pck.dump(epochs, handle, protocol=pck.HIGHEST_PROTOCOL)

def load_aplot(filepath):
	'''
	loads deserialised plot and property data using pickle
	'''
	with open(filepath, 'rb') as f:
		plotdata, properties = pck.load(f)
	return plotdata, properties

def write_aplot(filepath, plotdata, properties):
	'''
	saves the plot and property data by serilising object with pickle
	'''
	with open(filepath, 'wb') as f:
		pck.dump([plotdata,properties], f, protocol=pck.HIGHEST_PROTOCOL)

def append_log(directory, filename, line=None, wra='a'):
	'''
	Creates file if none exists and appends a line to the file.
	'''
	os.makedirs(directory, exist_ok=True)
	with open(directory + filename, wra, encoding='utf8') as f:
		f.write(line + '\n')

def make_dict_from_tree(element_tree):
	'''
	Converts xml trees into a python dictionary. Has modified to fit this project.
	Original Author: Eri Scrivner
	Original Source: https://ericscrivner.me/2015/07/python-tip-convert-xml-tree-to-a-dictionary/
	'''
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