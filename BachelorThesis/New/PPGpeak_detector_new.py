'''
AUTHOR(S):
Michal Kirkegaard

Handle the PPG peak detection using lowpass filter, cubing filter, adaptive thresholds and minimum distance.
'''
from scipy.interpolate import CubicSpline
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

def upsample(data, freq, freq_up):
	'''
	upsampling filter
	'''
	N = len(data)
	cs = CubicSpline(range(N),data)
	_timecol = np.linspace(0,N,N*(int(freq_up/freq)))
	return cs(_timecol), _timecol

def extreme_removal(data):
	'''
	Removes extreme values
	'''
	mean = np.mean(data)
	return np.array([d if abs(d) < abs(mean*20) else mean*20*(d/abs(d)) for i,d in enumerate(data)])

def PPG_Peaks(data, freq, plot=False, remove_extreme=True, freq_up=256):
	'''
	Performs the peak detection in steps. filtering (lowpass and cubic), peak detections (adaptive treshold
	and minimum distance) and lastly find the amplitudes for each peak, from the baseline removed signal.
	'''
	
	timecol = np.linspace(0,len(data),len(data))

	# filters
	_data, _timecol = data, timecol
	#_data, _timecol = upsample(data, freq, freq_up)
	_data = lowpass_butter_filter(_data)
	_data = extreme_removal(_data) if remove_extreme else _data
	_data = cubing_filter(_data)

	# peak detection provided by peakutils package, it uses adaptive treshold and minimum distance
	peaks = indexes(_data, min_dist=(1/2.5)*freq_up)
	peaks_sec = [_timecol[i]/freq for i in peaks]

	# peak amps from filtered data
	amps = np.array([abs(_data[i])**(1/3) for i in peaks])

	if plot:
		import matplotlib.pyplot as plt
		plt.plot(timecol/freq, data, 'blue', label='raw')
		plt.plot(_timecol/freq, _data, 'green', label='filt')

		b = [_data[int(j*freq)] for j in peaks_sec]
		b = np.array(b)
		plt.plot(peaks_sec, b, 'rx', label='filt peaks')

		plt.legend()
		plt.show()
		#b_data = data-baseline(data, 2)
		#plot_data([data+10, b_data], labels=['PPG', 'PPG Baselined'], normalization=True, indice=(0,len(data)))
		#plot_data([None, b_data], peaksIndexs=[None,peaks], labels=[None,'PPG Baselined'], normalization=False, indice = (0,len(data)))
		#plot_data([None, None, _data], peaksIndexs=[None, None, _peaks], labels=[None,'PPG Baselined', 'PPG Filtered'], normalization=False, indice = (0,len(data))) 
		#plot_data([data, None, _data], peaksIndexs=[(peaks_sec*freq_up).astype(int), None, peaks], labels=['PPG', 'PPG Baselined','PPG Filtered'], normalization=True, indice = (0,15000)) 

	return peaks_sec, amps

def softTemplate(data, i, freq):
	'''
	Gets maximum value within interval to determine absolute peak value.
	'''
	slice = 1/8
	h,j = int(max(i-(slice*freq),0)), int(min(i+(slice*freq), len(data)))
	k = h + np.argmax(data[h:j])
	return k