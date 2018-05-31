'''
AUTHOR(S):
Michal Kirkegaard

Handle the PPG peak detection using lowpass filter, cubing filter, adaptive thresholds and minimum distance.
'''
from scipy.signal import butter, filtfilt
from peakutils.peak import indexes
from peakutils import baseline
import numpy as np
from plots import plot_data


def lowpass_butter_filter(data, Norder=5, lowcut=0.03):
	'''
	lowpass filter
	'''
	B, A = butter(Norder, Wn=lowcut, btype='lowpass', output='ba')
	return filtfilt(B,A, data)


def cubing_filter(data):
	'''
	Cubing filter (instead of squaring, to keep ngative values negativ)
	'''
	return data**3

def PPG_Peaks(data, freq, plot=False):
	'''
	Performs the peak detection in steps. filtering (lowpass and cubic), peak detections (adaptive treshold
	and minimum distance) and lastly find the amplitudes for each peak, from the baseline removed signal.
	'''

	# filters
	_data = data
	_data = lowpass_butter_filter(_data)
	_data = cubing_filter(_data)

	# peak detection provided by peakutils package, it uses adaptive treshold and minimum distance
	slice = 1/3
	_peaks = indexes(_data, min_dist=freq*slice)
	peaks = [softTemplate(data, i, freq) for i in _peaks]

	# peak amps from baseline removed data on corrected peaks
	b_data = data-baseline(data)
	amps = [b_data[i] for i in peaks]

	if plot:
		plot_data([data, b_data], labels=['PPG', 'Baseline PPG'], normalization=True, indice=(0,10000)) # Baselined
		plot_data([data, _data], peaksIndexs=None, labels=['PPG','PPG Filtered'], normalization=True, indice = (0,10000)) # filtered
		plot_data([None, _data], peaksIndexs=[None, _peaks], labels=['PPG','PPG Filtered'], normalization=True, indice = (0,10000)) # non-corrected peaks
		plot_data([data], peaksIndexs=[peaks], labels=['PPG','PPG Filtered'], normalization=True, indice = (0,10000)) # Final
		plot_data([data, _data], peaksIndexs=[peaks, _peaks], labels=['PPG','PPG Filtered'], normalization=True, indice = (0,10000)) # Everything

	return peaks, amps

def softTemplate(data, i, freq):
	'''
	Gets maximum value within interval to determine absolute peak value.
	'''
	slice = 1/8
	h,j = int(max(i-(slice*freq),0)), int(min(i+(slice*freq), len(data)))
	k = h + np.argmax(data[h:j])
	return k
