'''
AUTHOR(S):
Nicklas Hansen,
Micael Kirkegaard

Setting used across the solution
'''

SHHS = False

SAMPLE_RATE = 250 if SHHS else 256
FEATURES = 7
EPOCH_LENGTH = 120
OVERLAP_FACTOR = 3
OVERLAP_SCORE = 10
MASK_THRESHOLD = 0.125

PREDICT_THRESHOLD = 0.5