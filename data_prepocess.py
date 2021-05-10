# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:34:57 2021

@author: arthur
"""
import csv
import python_speech_features
import librosa
import numpy as np
with open('train/meta_train.csv') as csvfile:
    rows = csv.reader(csvfile)
    label = list(rows)
    
label.remove(label[0])

mel_list = []
index = 0
temp = label[index][2]
for data in label:
    audio, sr = librosa.load('train/train/'+data[0]+'.wav', sr=16000)
    mel = python_speech_features.logfbank(audio, sr)
    if index == int(data[1]):
        mel_list.append(mel)
    else:
        np.save(temp+'.npy', mel_list)
        temp = label[index][2]
        #Update to new category
        index = int(data[1])
        mel_list = []
        mel_list.append(mel)
