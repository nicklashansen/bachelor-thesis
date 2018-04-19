import time

"""
WRITTEN BY:
Nicklas Hansen
"""

class stopwatch:

	def __init__(self):
		self.a = time.time()
		self.b = self.a
		self.c = 0

	def round(self):
		t = time.time()
		r = t - self.b 
		self.b = t
		return int(r)

	def stop(self):
		self.c = time.time() - self.a
		return int(self.c)