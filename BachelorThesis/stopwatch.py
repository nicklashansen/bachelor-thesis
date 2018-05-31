'''
AUTHOR(S):
Nicklas Hansen

Module is a framework for evaluating time durations for use in other modules.
'''

import time

class stopwatch:
	'''
	Class responsible for keeping track of time.
	'''
	def __init__(self):
		'''
		Creates a new instace of the stopwatch object and automatically starts the counter.
		'''
		self.a = time.time()
		self.b = self.a
		self.c = 0

	def round(self):
		'''
		Calculates time between now and last round.
		If this is the first round, time between now and start is calculated instead.
		'''
		t = time.time()
		r = t - self.b 
		self.b = t
		return int(r)

	def stop(self):
		'''
		Stops the stopwatch and returns total duration.
		'''
		self.c = time.time() - self.a
		return int(self.c)