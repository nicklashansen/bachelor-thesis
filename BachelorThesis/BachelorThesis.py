#from features import process_epochs, make_splits, hours_of_sleep_files
#from model_selection import parameter_tuning, test_bidirectional, fit_validate_test, evaluate, evaluate_LR, predict_file
#from model_selection import evaluate, evaluate_LR
#from preprocessing import prepAll, prepSingle
import filesystem as fs
import numpy as np
#from dataflow import dataflow
import log
import metrics
#import h5py

"""
AUTHOR(S):
Nicklas Hansen
Michael Kirkegaard
"""

if __name__ == '__main__':
	#parameter_tuning(evaluate_model=False)
	#train_validate_test()
	#fit_validate_test(only_arousal = True)
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rr.h5')
	#evaluate(validation=False, path='best_rwa.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_ecg.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_ppg.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rr_ppg.h5')
	#evaluate(validation=False, path='FEATURE_SELECTION\\best_rwa_ppg.h5')
	#test_dataflow(file='mesa-sleep-1541')
	#count_epochs_removed()
	#test_dataflow_LR()
	#evaluate_LR()
	#t = hours_of_sleep_files()

	# ---- remove

	#lg = log.get_log('arousal_index',True)
	#lg.print('scored,predicted')
	#for file in fs.getAllSubjectFilenames(preprocessed=True):
	#	id = int(file[-6:])
	#	filter = ['ai_all']
	#	datasetCsv = fs.getDataset_csv()
	#	ai_all = datasetCsv[datasetCsv['nsrrid'] == id][filter].iloc[0][0]

	#	X,y = fs.load_csv(file)
	#	_,summary = dataflow(X)
	#	ai_all_hat = dict(summary)['arousals_hr']
	#	lg.print('{0},{1}'.format(ai_all, ai_all_hat))

	#filename = lg.directory + lg.filename
	filename = 'D:\\BachelorThesis\\Logs\\arousal_index\\2018-06-03_03-45_log.txt'
	with open(filename) as f:
		a,b = zip(*[line.replace('\n','').split(',') for line in f])
	a = np.array(a[1:]).astype(float)
	b = np.array(b[1:]).astype(float)
	from scipy.stats import linregress
	slope, intercept, r_value, p_value, std_err = linregress(a, b)
	
	from matplotlib import pyplot as plt
	plt.scatter(a,b, color=(150/255, 0, 0))
	abline_values = [slope * i + intercept for i in a]
	plt.plot(a, abline_values, 'b')
	plt.legend(['(R = {0:.3f}, p < {1:.11f})'.format(r_value, p_value*10),'Scored AAI vs. Predicted AAI'])
	plt.show()

	#lg.printHL()
	print(np.corrcoef(a,b)[0,1])
	print(r_value**2)
	print(r_value)
	print(p_value)

	#lg.printHL()

	a = [1 if ai >= 20 else 0 for ai in a]
	b = [1 if ai >= 20 else 0 for ai in b]
	tp,fp,tn,fn = metrics.cm_standard(a,b)
	score = metrics.compute_cm_score(tp,fp,tn,fn)
	for k,d in score.items():
		print(str(k))
		for key,val in d.items():
			print(str(key)+':'+str(val))

	# ---- remove

	#prepSingle('mesa-sleep-0085', False) # illustrating errors
	#prepSingle('mesa-sleep-2915', False) #new found
	#prepSingle('mesa-sleep-3150', False) #big errors
	
	#X,y = fs.load_csv('mesa-sleep-0001')
	#dataflow(X, y,True)