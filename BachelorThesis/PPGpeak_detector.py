'''
AUTHOR(S):
Michal Kirkegaard

Handle the PPG peak detection using lowpass filter, cubing filter, adaptive thresholds and minimum distance.
'''
from scipy.signal import butter, filtfilt, medfilt
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

def extreme_removal(data):
	'''
	Removes extreme values
	'''
	mean = np.mean(data)
	return np.array([d if abs(d) < abs(mean*10) else mean for d in data])

def PPG_Peaks(data, freq, plot=False, remove_extreme=False):
	'''
	Performs the peak detection in steps. filtering (lowpass and cubic), peak detections (adaptive treshold
	and minimum distance) and lastly find the amplitudes for each peak, from the baseline removed signal.
	'''

	# filters
	_data = data
	_data = lowpass_butter_filter(_data)
	_data = extreme_removal(_data) if remove_extreme else _data
	_data = cubing_filter(_data)

	# peak detection provided by peakutils package, it uses adaptive treshold and minimum distance
	slice = 1/3
	_peaks = indexes(_data, min_dist=freq*slice)
	peaks = [softTemplate(data, i, freq) for i in _peaks]

	# peak amps from filtered data
	amps = [_data[i] for i in peaks]

	if plot:
		b_data = data-baseline(data, 2)
		plot_data([data+10, b_data], labels=['PPG', 'PPG Baselined'], normalization=True, indice=(0,len(data)))
		#plot_data([None, b_data], peaksIndexs=[None,peaks], labels=[None,'PPG Baselined'], normalization=False, indice = (0,len(data)))
		#plot_data([None, None, _data], peaksIndexs=[None, None, _peaks], labels=[None,'PPG Baselined', 'PPG Filtered'], normalization=False, indice = (0,len(data))) 
		#plot_data([data, None, _data], peaksIndexs=[peaks, None, _peaks], labels=['PPG', 'PPG Baselined','PPG Filtered'], normalization=False, indice = (0,len(data))) 

	return peaks, amps

def softTemplate(data, i, freq):
	'''
	Gets maximum value within interval to determine absolute peak value.
	'''
	slice = 1/8
	h,j = int(max(i-(slice*freq),0)), int(min(i+(slice*freq), len(data)))
	k = h + np.argmax(data[h:j])
	return k
