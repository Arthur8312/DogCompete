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
with open('meta_train.csv') as csvfile:
    rows = csv.reader(csvfile)
    label = list(rows)
    
label.remove(label[0])
label.sort(key= lambda s:s[1])
mel_list = []
other = []
for data in label:
    if data[1] == '5':
        other.append(data)
        audio, sr = librosa.load('train/'+other[0][0]+'.wav', sr=16000)
        mel = python_speech_features.logfbank(audio, samplerate=16000, nfft=2048, nfilt=120)
        mel_list.append(mel.T)
np.save('npy/'+'Other'+'.npy', mel_list)
category = ['Barking', 'Howling', 'Crying', 'CO_Smoke', 'GlassBreaking', 'Other']
for data in category:
  os.makedirs('data/'+data, exist_ok=True)
mel_list = []
index = 0
temp = category[index]
for data in label:
    audio, sr = librosa.load('train/'+data[0]+'.wav', sr=16000)
    mel = python_speech_features.logfbank(audio, samplerate=16000, nfft=2048, nfilt=120)
    if index == int(data[1]):
        mel_list.append(mel.T)
    else:
        np.save('npy/'+temp+'.npy', mel_list)
        #Update to new category
        index = index + 1
        temp = category[index]
        mel_list = []
        mel_list.append(mel.T)

X = np.load('npy/'+'Barking'+'.npy', allow_pickle=True)
X1 = np.load('npy/'+'Other'+'.npy', allow_pickle=True)
