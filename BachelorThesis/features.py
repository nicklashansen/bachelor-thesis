import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
WRITTEN BY
Micheal Kirkegaard
"""

def make_features(X, y):
	# Transpose X
	Xt = np.transpose(X)

	# Make Masks
	m_DR = maskDR(Xt[0])
	m_RPA = maskRPA(Xt[1])
	m_PTT = maskPTT(Xt[2])
	m_PWA = maskPTT(Xt[3])
	m_SS = maskSS(Xt[4])
	m_AA = maskAA(y)

	mask = mergeMasks([m_DR, m_RPA, m_PTT, m_PWA, m_SS, m_AA])

	# Make Epoch Slices
	e = epoch_Slices(X, y)

	# Cut Epochs with XX percent uncertainty in mask
	e = epoch_Cut(e, mask)

	# Create and Normalize each epoch
	E = epoch_Create(e, X, y)

	return E

# Input [ [0, 1, ...], ...]
def mergeMasks(masks):
	def AnyAboveZero(tub):
		for i in tub:
			if i > 0:
				return True
		return False
	# Returns [0, 1, ...]
	return [(1 if AnyAboveZero(tub) else 0) for tub in zip(*masks)]

# Makes Feature Masks

# DR
def maskDR(x_DR):
	mask = [0]*len(x_DR)
	for index in mask:
		if (x_DR[index] > 2.0 or x_DR[index] < 0.4):
			mask[index] = 1
	return mask

# RPA
def maskRPA(x_RPA):
	mask = [0]*len(x_RPA)
	return mask

# PTT
def maskPTT(x_PTT):
	mask = [0]*len(x_PTT)
	# TODO
	return mask

# PWA
def maskPWA(x_PWA):
	mask = [0]*len(x_PWA)
	# TODO
	return mask

# SS
def maskSS(x_SS):
	mask = [0]*len(x_SS)
	# TODO
	return mask

# AA
def maskAA(y_AA):
	mask = [0]*len(y_AA)
	# TODO
	return mask

# Make Epochs indexes
# TODO: improve
def epoch_Slices(X, y):
	l = len(X)
	li = np.arange(0, l, l/180)
	lis = [[int(li[i]),int(li[i+1])] for i,_ in enumerate(li) if i+1 < len(li)]
	
	return lis # 180 Epochs for full set

# Cut Epochs with XX percent uncertainty in mask
def epoch_Cut(e, mask, percent = 0.45):
	def cut(epoch):
		i,j = epoch[0], epoch[1]
		maskslice = mask[i:j]
		masksum = sum(maskslice)
		if (1-masksum/len(maskslice)) <= percent:
			return True
		return False

	return [epoch for epoch in e if not cut(epoch)]

# Create and Normalize each epoch
# TODO: improve
def epoch_Create(e, X, y):

	def create(epoch):
		i,j = epoch[0], epoch[1]
		
		# Normalize each feature as row
		X_ = np.transpose([normalize(x) for x in np.transpose(X[i:j])])
		y_ = y[i:j]

		return [X_, y_]

	return [create(epoch) for epoch in e]
	"""
	Returns:
	[ [ X_0, y_0 ], [ X_1, y_1 ], ... ]

	[
		[ [ [x0_DR[0], ...], [x0_DR[1], ...], ... ], [y0_AA[0], y0_AA[1], ...] ],
	    [ [ [x1_DR[0], ...], [x1_DR[1], ...], ... ], [y1_AA[0], y0_AA[1], ...] ],
		...
	    ]
	
	[
		# epoch_0, i_0 : j_0
		[
			[
				[DR_0, RDA_0, PPT_0, PWA_0, SS_0],
				[DR_1, RDA_1, PPT_1, PWA_1, SS_1], ...
			],
			[
				AA_0,
				AA_1, ...
			],
		],
		# epoch_1, i_1 : j_1
		[
			[
				[DR_0, RDA_0, PPT_0, PWA_0, SS_0],
				[DR_1, RDA_1, PPT_1, PWA_1, SS_1], ...
			],
			[
				AA_0,
				AA_1, ...
			],
		], ...
	]
		
	"""

def normalize(X, scaler=MinMaxScaler()):
	return np.squeeze(scaler.fit_transform(X.reshape(X.shape[0], 1)))
