import pickle
import os
import numpy as np
import python_speech_features

def wav2melspec(wave, sr=8000, max_len=499):
    melspec = python_speech_features.base.logfbank(wave, samplerate=sr, nfft=1024, nfilt=120)
    melspec = melspec.T
    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec = melspec[:, :max_len]
    return melspec

pk_list = os.listdir('data_extend/') 

#Bark

with open('data_extend/'+pk_list[0], 'rb') as fp:
    dataset = pickle.load(fp)
bark_extend = [wav2melspec(data) for data in dataset]
bark_org = np.load('train_npy/Barking.npy')

for i in bark_extend: