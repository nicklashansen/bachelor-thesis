from time import time
from os import listdir
from os.path import isfile, join
from subject import Subject
import pickle
import pandas as pd
#from MESA.library.subject import Subject
#from MESA.library.annotation_reader import AnnotationReader as ar


def pre_processing(filename):

    mesaid = int(filename[-4:])
    filter = ['mesaid', 'apnhyppr5', 'respevpr5']
    # respevor5 - Scoring respiratory events (RDI) unreliable
    # apnhyppr5 - Scoring apnea/hypapnea unreliable

    # Check if it is too uncertain
    csv_file = pd.read_csv("C:\\nssr\data\mesa\datasets\mesa-sleep-dataset-0.3.0.csv")
    dataset_info = csv_file.loc[:, filter]
    idx = [i for i, id in enumerate(dataset_info.loc[:, 'mesaid']) if id == mesaid][0]
    if any(dataset_info.loc[idx, filter[1:]]):
        print('not a reliable recoring and is not saved')
        return None, None, None, None, None, None

    
    events_classes = ['Central apnea|Central Apnea',
                      'Hypopnea|Hypopnea',
                      'Obstructive apnea|Obstructive Apnea',
                      'SpO2 desaturation|SpO2 desaturation',
                      'Mixed apnea|Mixed Apnea'
                      ]

    discard_events = [['Unsure|Unsure'],
                      ['SpO2 artifact|SpO2 artifact'],
                      ['Respiratory artifact|Respiratory artifact'],
                      ['Narrow complex tachycardia|Narrow Complex Tachycardia'],
                      ['Limb movement - left|Limb Movement (Left)'],
                      ['Periodic leg movement - left|PLM (Left)'],
                      ['Arousal|Arousal ()'],
                      ['Spontaneous arousal|Arousal (ARO SPONT)'],
                      ['ASDA arousal|Arousal (ASDA)']
					  ]

    event_penalties = [[5], [5], [5], [5], [5], [5], [5], [5], [5]]

    channels_to_include = ['Pleth', 'EKG', 'OxStatus', 'HR', 'Pos', 'SpO2']

    discard_signals = [['OxStatus']]
    signal_discard_values = [[3, 4]]
    signal_discard_penalies = [[5, 5]]

    # apply edf pre-processing rules
    sub = Subject(filename)
    sub.add_channel(channels_to_include)
    for i, signal in enumerate(discard_signals):
        sub.add_mask(signal, signal_discard_values[i], signal_discard_penalies[i])

    # Apply annotation pre-processing rules
    anno = ar(filename)
    for i, types in enumerate(discard_events):
        x_mask = anno.add_mask(xmlEventConcept=types, penalties=event_penalties[i])
        sub.set_mask(x_mask)

    # Remove wake
    wake = anno.get_specific_stage(['Wake|0'], section_length=1, penalties=[0])
    sub.set_mask(wake)

    # set minimum valid duration for mask:
    sub.set_valid_mask(min_valid_duration=60)

    if sub.valid_recordDur < 3600*2:
        print('Too little valid duration: ', sub.valid_recordDur)
        return None, None, None, None, None, None

    # Extract sections for classification:
    x_pleth, y_pleth = sub.get_valid_segments(label=['Pleth'], segment_duration=30, overlap=0.5,
                                              annotations=events_classes)
    x_HR, y_HR = sub.get_valid_segments(label=['HR'], segment_duration=30, overlap=0.5,
                                        annotations=events_classes)
    x_SpO2, y_SpO2 = sub.get_valid_segments(label=['SpO2'], segment_duration=30, overlap=0.5,
                                            annotations=events_classes)
    x_Pos, y_Pos = sub.get_valid_segments(label=['Pos'], segment_duration=30, overlap=0.5,
                                          annotations=events_classes)
    x_EKG, y_EKG = sub.get_valid_segments(label=['EKG'], segment_duration=30, overlap=0.5,
                                          annotations=events_classes)

    x_Pos = x_Pos[:, ::32]
    return x_pleth, x_EKG, x_HR, x_SpO2, x_Pos, y_pleth

# save files:
start = time()

# load files
save_path = "C:\\nssr\data\mesa\polysomnography\dataset"
mypath = "C:\\nssr\data\mesa\polysomnography\edfs"
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i, filename in enumerate(filenames):
    x_pleth, x_EKG, x_HR, x_SpO2, x_Pos, y = pre_processing(filename[:-4])

    if x_pleth is None:
        print(filename, ' is not saved')
    else:
        pickleFile = open(save_path + '\\' + filename[:-4] + '.txt', 'wb')
        pickle.dump([x_pleth, x_EKG, x_HR, x_SpO2, x_Pos, y], pickleFile)
        pickleFile.close()
    if i % 100 == 0 and i > 0:
        print(i)
        print('elapse ', time()-start)

print('final elapes: ', time()-start)

# Excluding criteria
# TODO - consider discarding technical errors from annotation files
# TODO - Only include patients with > 2 hours of sleep

# Model
# TODO - build machine learning model. - Use andrew NG's paper.
# TODO - setup: training, validation and testing using this approach
# TODO - build batch_generator.
# TODO - setup accuracy measurements.

# stanford articles
# TODO - Hyatt made a good LM detector, see what restrictions, etc. he included.
# TODO - haryat made a nice apnea detection algorithm. See what she did with SpO2 vs non-SpO2 desaturation apnea events.

