from scipy.signal import butter, filtfilt
from peakutils.peak import indexes
import numpy as np
from plots import plot_data

"""
WRITTEN BY:
Michael Kirkegaard
"""

def lowpass_butter_filter(data, Norder=5, lowcut=0.03):
	B, A = butter(Norder, Wn=lowcut, btype='lowpass', output='ba')
	return filtfilt(B,A, data)

def cubing_filter(data):
	return data**3 #power of three to keep negative values negative

def PPG_Peaks(data, freq, plot=False):
	_data = data
	_data = lowpass_butter_filter(_data)
	_data = cubing_filter(_data)

	slice = 1/2
	peaks = indexes(_data, min_dist=freq*slice) # Heartbeat should not be faster than 120 BPM (2 Beats per second)
	peaks, amps = zip(*[getMax(data, i, freq) for i in peaks])

	if plot:
		plot_data([data, _data], peaksIndexs=[peaks], labels=['Signal','Filtered'], normalization=True)

	return peaks, amps

# Max value within a 1/4 (2*1/8) frequency span (min-dist)
def getMax(data, i, freq):
	slice = 1/8
	h,j = int(max(i-(slice*freq),0)), int(min(i+(slice*freq), len(data)))
	k = h + np.argmax(data[h:j])
	amp = data[k]
	return k, amp
