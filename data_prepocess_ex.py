# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:09:34 2021

@author: arthurchien
"""

import csv
import python_speech_features
import librosa
import numpy as np
import os
import nlpaug.augmenter.audio as ag
import nlpaug.flow as augf
from sklearn.model_selection import train_test_split
import pickle

with open('meta_train.csv') as csvfile:
    rows = csv.reader(csvfile)
    label = list(rows)

def wav2melspec(wave, sr, max_len=499):
    melspec = python_speech_features.base.logfbank(wave, samplerate=sr, nfft=1024, nfilt=120)
    melspec = melspec.T
    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec = melspec[:, :max_len]
    return melspec

pk_list = os.listdir('data_extend/') 
dataset = []
for i, pk in enumerate(pk_list):
    with open('data_extend/'+pk, 'rb') as fp:
        dataset.append(pickle.load(fp))

def spilt(label):
    X_train, X_validation = train_test_split(label ,test_size = 0.1, random_state=42)
    X_train.sort(key= lambda s:s[1])
    X_validation.sort(key= lambda s:s[1])
    return X_train, X_validation

aug = augf.Sequential([ag.VtlpAug(8000, zone=(0,1), coverage=1)])
def aug_data(data_list, n):
    aug_data_list = []
    for wav in data_list:
        aug_datas = aug.augment(wav, n)
        aug_data_list.extend(aug_datas)
    return aug_data_list

data_list = []
for data in label:
    if data[1] == '4':
        wave, sr = librosa.load('train/'+data[0]+'.wav', sr=None)
        data_list.append(wave)
        
train_data = data_list
train, valid = spilt(train_data)
train_aug = aug_data(train, 9)
valid_aug = aug_data(valid, 9)
train.extend(train_aug)
valid.extend(valid_aug)
train = [wav2melspec(wav, 8000) for wav in train]
valid = [wav2melspec(wav, 8000) for wav in valid]
np.save('train_npy/Glassbreaking.npy',train)
np.save('val_npy/Glassbreaking.npy',valid)
