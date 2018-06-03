'''
AUTHOR(S):
Nicklas Hansen,
Micael Kirkegaard

Setting used across the solution
'''

SHHS = False
BEST_MODEL = 'best_rwa.h5'

SAMPLE_RATE = 250 if SHHS else 256
FEATURES = 6
EPOCH_LENGTH = 120
OVERLAP_FACTOR = 3
OVERLAP_SCORE = 10
MASK_THRESHOLD = 0.125

PREDICT_THRESHOLD = 0.5