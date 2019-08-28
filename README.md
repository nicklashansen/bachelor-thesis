# Automatic Multi-Modal Detection of Autonomic Arousals in Sleep

Repository for bachelor thesis on Automatic Multi-Modal Detection of Autonomic Arousals in Sleep. The thesis itself (236 pages) and all related data is confidential and thus not publicly available, but **access to the thesis can be granted upon request**. We have however made our presentation slides available for review here.

### Abstract

Manual scoring of arousals is a tiring process prone to high inter-scorerdisagreement. Current arousal detection is based on electroencephalog-raphy (EEG), which is not suitable for sleep studies at home. Basner et al. (2008) proposes a semi-automatic detection algorithm based on electrocardiography (ECG) and acknowledges that it cannot replace EEG, but a strong correlation between cortical and autonomic arousals is evident. This study explores the possibility of better autonomic arousal detection algorithms in a multi-modal setting by inclusion of features from photoplethysmography (PPG).

A state-of-the-art algorithm provided by M. Olsen (2018) is applied for reliable RR-tachogram extraction in ECG signals, and a pulse peak detection algorithm is developed for extraction of pulse transit time and corresponding pulse wave amplitude in PPG signals. Classification is performed by a recurrent neural network (RNN) utilising gated recurrent unit (GRU) cells with an architecture derived from model- and feature selection using 2-layered cross-validation on the Multi-Ethnic Study of Atherosclerosis (MESA) data-set.

It was found that the best performing model was a 2-layered bi-directional RNN using GRU cells, and cross-validation shows slight performance increase when introducing PTT and PWA to a RR-interval based model. The best performing modality in this study was an RR-interval calculated at 800Hz upsampling.
A generalisation estimation is performed on a sub-set of an additional data-set, Sleep Heart Health Study (SHHS), in order to provide unbiased results for comparison to other studies. A sensitivity of 58.53%, precision of 71.68%, F1-score of 0.6444 and a correlation between scored and predicted arousal indices of (R=0.762, p<0.01) was found.
These results compare well to state-of-the-art research but with a use of far less features. It was found that the model performed poorly on subjects with abnormal heart rates, suggesting ECG-based algorithms cannot reliably be employed unless deployed multi-modally.

### Results

A few figures from the thesis illustrating our work and results is shown below.

![mesa](https://i.imgur.com/76ohaxa.png)

![shhs](https://i.imgur.com/Ye8Hbyz.png)

![view](https://i.imgur.com/PEcRhl1.png)

### Questions?

You are more than welcome to contact me directly at hello@nicklashansen.com if you have any questions regarding our work.
