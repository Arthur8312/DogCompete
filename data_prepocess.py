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
category = ['Barking', 'Howling', 'Crying', 'CO_Smoke', 'GlassBreaking', 'Other']
for data in category:
  os.makedirs('data/'+data, exist_ok=True)
mel_list = []
index = 0
temp = category[index]
for data in label:
    audio, sr = librosa.load('train/'+data[0]+'.wav', sr=16000)
    mel = python_speech_features.logfbank(audio, sr)
    if index == int(data[1]):
        mel_list.append(mel)
    else:
        np.save(temp+'.npy', mel_list)
        #Update to new category
        index = index + 1
        tempe = category[index]
        mel_list = []
        mel_list.append(mel)
