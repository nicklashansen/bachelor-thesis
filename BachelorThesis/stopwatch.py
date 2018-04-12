import time

"""
WRITTEN BY:
Nicklas Hansen
"""

class stopwatch:
	a=b=c=0

	def __init__(self):
		self.a = int(time.time())
		self.b = self.a

	def round(self):
		self.b = int(time.time() - self.b)
		return self.b

	def stop(self):
		self.c = int(time.time() - self.a)
		return self.c