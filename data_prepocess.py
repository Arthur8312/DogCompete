# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:34:57 2021

@author: arthur
"""
import csv
import python_speech_features
import librosa
import numpy as np
import os
import nlpaug.augmenter.audio as ag
from sklearn.model_selection import train_test_split

#讀取CSV
with open('meta_train.csv') as csvfile:
    rows = csv.reader(csvfile)
    label = list(rows)
    
label.remove(label[0])
mel_list = []
other = []

X_train, X_validation = train_test_split(label, test_size = 0.1, random_state = 42)
X_train.sort(key= lambda s:s[1])
X_validation.sort(key= lambda s:s[1])

#資料轉melspec 並切齊
def wav2melspec(wave, sr, max_len=499):

    melspec = python_speech_features.base.logfbank(wave, samplerate=sr, nfft=1024, nfilt=120)
    melspec = melspec.T
    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec = melspec[:, :max_len]
    return melspec

category = ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']
# #label data 之後改掉
# for data in category:
#   os.makedirs('data/'+data, exist_ok=True)
index = 0
mel_list = []
temp = category[index]
valid_path = 'val_npy/'

for data in X_validation:
    wave, sr = librosa.load('train/'+data[0]+'.wav', sr=None)
    mel = wav2melspec(wave, sr)
    if index == int(data[1]):
        mel_list.append(mel)
    else:
        # print(len(mel_list))
        np.save(valid_path+temp+'.npy', mel_list)
        #Update to new category
        index = index + 1
        temp = category[index]
        mel_list = []
        mel_list.append(mel)
np.save(valid_path+temp+'.npy', mel_list)

total_aug = 10
mel_list = []
index = 0
temp = category[index]
train_path = 'train_npy/'

for data in X_train:
    wave, sr = librosa.load('train/'+data[0]+'.wav', sr=None)
    for i in range(total_aug):
        aug = ag.VtlpAug(sr)
        mel = wav2melspec(wave, sr)
        if index == int(data[1]):
            mel_list.append(mel)
        else:
            # print(len(mel_list))
            np.save(train_path+temp+'.npy', mel_list)
            #Update to new category
            index = index + 1
            temp = category[index]
            mel_list = []
            mel_list.append(mel)
np.save(train_path+temp+'.npy', mel_list)

# np.save('npy/'+temp+'.npy', mel_list)

