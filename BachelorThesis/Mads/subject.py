import pyedflib
import numpy as np
#from MESA.library.annotation_reader import AnnotationReader as ar
#from MESA.library.signal_processing import window_idx

class Subject(object):

    def __init__(self, filename):
        self.filename = filename
        self.path = 'C:\\nsrr\data\mesa\polysomnography\edfs'

        self.signal_info = None
        self.mask = None
        self.recording_duration = self.get_record_duration()
        self.valid_recordDur = 0

        # annotation reader:
        # self.anno = ar(self.filename)
        # anno_duration = self.anno.RecordDur

        # make sure annotation and edf file have same duration
        # assert(self.recording_duration == anno_duration)

        # TODO - make sure annotation and edf are started same date. - But they are.

    def get_edf_signals(self, signal_labels=None):
        """ Loads edf files from the added channels
        :return: signal
        """
        with pyedflib.EdfReader(self.path + '\\' + self.filename + ".edf") as file:
            signals = []
            for signal_label in signal_labels:
                signal_idx = self.get_signal_idx(signal_label)
                signal = file.readSignal(signal_idx)
                fs = self.get_signal_fs(signal_label)
                signals += [signal[:fs*self.recording_duration]]
                #print('signal from ' + signal_label + ' has been loaded.')
            return signals

    def set_valid_mask(self, min_valid_duration=300):
        """ section duration is the trianing size. Valid """
        mask = self.mask

        # valid sections:
        counter = 0
        new_mask = np.ones(mask.shape, dtype=bool)
        for i, value in enumerate(mask):
            if value:
                counter = 0
            else:
                counter += 1
            if counter >= min_valid_duration:
                new_mask[i-counter:i+1] = False
        self.set_mask(new_mask) # set new mask
        self.valid_recordDur = self.recording_duration - sum(new_mask)

    def get_adjusted_mask(self, label):

        fs = self.get_signal_fs(label[0])
        x = self.get_edf_signals(label)
        x = x[0]
        mask = self.mask

        # Re-size mask to signal:
        adjusted_mask = np.zeros((len(x), 1), dtype=bool)
        for i, value in enumerate(mask):
            adjusted_mask[i * fs: (i + 1) * fs] = value
        return adjusted_mask

    def get_valid_segments(self, label=None, segment_duration=10, overlap=0, annotations=None):

        fs = self.get_signal_fs(label[0])
        x = self.get_edf_signals(label)
        x = x[0]
        ori_mask = self.mask

        x_train = []
        y_train = []

        # Events for classification
        events_for_class = self.anno.get_event_vector(type=None, xmlEventConcept=annotations, penalties=None)
        sig_len, N_classes = events_for_class.shape
        adjusted_mask = self.get_adjusted_mask(label)
        anno_idx = window_idx(ori_mask, window_length=segment_duration, overlap=overlap)
        signal_idx = window_idx(x, window_length=segment_duration*fs, overlap=overlap)

        # Apply mask
        N_windows, segment_duration = signal_idx.shape
        for r in range(N_windows):
            if not any(adjusted_mask[signal_idx[r, :]]):
                y_train_temp = []
                keep = True
                for n in range(N_classes):
                    if sum(events_for_class[anno_idx[r, :], n]) >= 10:
                        y_train_temp.append(True)
                    elif sum(events_for_class[anno_idx[r, :], n]) >= 2:
                        keep = False
                        y_train_temp.append(False)
                    else:
                        y_train_temp.append(False)
                if keep or any(y_train_temp):
                    if not any(y_train_temp):
                        # TODO - remember it is only normal if not any other event.
                        y_train_temp.append(True) #
                    else:
                        y_train_temp.append(False)
                    y_train += [y_train_temp]
                    x_train += [x[signal_idx[r, :]]]
        return np.array(x_train), np.array(y_train)

    def add_mask(self, label=None, mask_values=None, penalties=None):

        x = self.get_edf_signals(label)
        x_mask = np.zeros((self.recording_duration, 1), dtype=bool)
        for value_idx, mask_value in enumerate(mask_values):
            mask_idx = [i for i, value in enumerate(x[0]) if value == mask_value]
            # penalty
            if penalties is not None:
                penalty = penalties[value_idx]
                for n in mask_idx:
                    x_mask[max(n-penalty, 0):min(n+penalty, self.recording_duration) + 1] = 1
            else:
                x_mask[mask_idx] = 1
        self.set_mask(x_mask)

    def set_mask(self, mask):
        if self.mask is None:
            #print('---adding masks---')
            self.mask = mask
        else:
            #print('---adding masks---')
            self.mask = ((self.mask + mask) > 0) * 1
            self.mask = self.mask > 0


    def add_channel(self, signal_labels=None):
        """ Adding a channel, with label name, sampling frequency, and channel number."""

        with pyedflib.EdfReader(self.path + '\\' + self.filename + ".edf") as file:
            edf_labels = file.getSignalLabels()

            signal_info_list = []
            for signal_label in signal_labels:
                signal_info = dict()
                signal_info['signal_label'] = signal_label
                signal_info['signal_idx'] = [i for i, edf_label in enumerate(edf_labels) if edf_label == signal_label][0]
                signal_info['signal_fs'] = file.getSampleFrequency(signal_info['signal_idx'])
                signal_info_list.append(signal_info)

                # print('Signal info for: ' + signal_label + ' is added...')
            self.signal_info = signal_info_list

    def get_record_duration(self):
        with pyedflib.EdfReader(self.path + '\\' + self.filename + ".edf") as file:
            recordDuration = file.getFileDuration()

        # Adjust for 30 second epochs
        recordDuration -= (recordDuration % 30)
        return recordDuration

    def get_all_signal_labels(self):
        if self.signal_info:
            labels = []
            for label in self.signal_info:
                labels.append(label['signal_label'])
            return labels

    def get_all_signal_fs(self):
        if self.signal_info:
            fs = []
            for label in self.signal_info:
                fs.append(label['signal_fs'])
            return fs

    def get_all_signal_idx(self):
        if self.signal_info:
            idx = []
            for label in self.signal_info:
                idx.append(label['signal_idx'])
            return idx

    def get_signal_label_position(self, label):
        if self.signal_info:
            signal_labels = self.get_all_signal_labels()
            label_pos = [i for i, signal_label in enumerate(signal_labels) if signal_label == label][0]
            return label_pos

    def get_signal_fs(self, label):
        if self.signal_info:
            label_pos = self.get_signal_label_position(label)
            all_fs = self.get_all_signal_fs()
            fs = all_fs[label_pos]
            return fs

    def get_signal_idx(self, label):
        if self.signal_info:
            label_pos = self.get_signal_label_position(label)
            all_idx = self.get_all_signal_idx()
            idx = all_idx[label_pos]
            return idx