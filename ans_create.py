# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:23:48 2021

@author: arthurchien
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:43:45 2021

@author: arthurchien
"""

import os
import librosa
import python_speech_features
import tensorflow.keras as keras
import csv
from scipy import signal
import numpy as np
test_path = 'public_test/'
test_list = os.listdir(test_path)
weight_path = 'model_log/best.h5'
model = keras.models.load_model(weight_path, compile = False)
category = ['Filename','Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']
ans_list = []
ans_list.append(category)
def speech_roi(y, n=5, sample_r=8000, sec=1.5): 
    index = 0
    y_t = abs(y)
    index_temp = 0
    t0 = y_t[0:int(sec*sample_r)]
    max_temp = sum(t0)
    temp = sum(t0)
    while index+sample_r+n < len(y):
        a1 = sum(y_t[index: index+n]) #最前面n筆
        a2 = sum(y_t[index+sample_r: index+sample_r+n]) #最後面n筆
        temp = temp - a1 +a2 #更新1s
        if temp>max_temp:
            index_temp = index
            max_temp = temp
        index =index+n
    return y[index_temp:index_temp+sample_r]

#資料轉melspec 並切齊
def wav2melspec(wave, sr, max_len=499):

    melspec = python_speech_features.base.logfbank(wave, samplerate=sr, nfft=1024, nfilt=120)
    melspec = melspec.T
    # frq, time, melspec = signal.spectrogram(wave, fs=sr, window='hann', scaling='spectrum', nperseg=256)

    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec = melspec[:, :max_len]
    return melspec

for data in test_list:
    data_path = test_path+data
    wave, sr = librosa.load(data_path, sr=None)
    # wave =speech_roi(audio)
    mel = wav2melspec(wave, sr, max_len=499)
    mel = mel.reshape(1, 120, 499, 1)
    ans = model.predict(mel)
    ans = ans.tolist()
    data_l = [data.split('.')[0]] + ans[0]
    ans_list.append(data_l)
    
# with open('test.csv','w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(ans_list)
with open('sample_submission.csv') as csvfile:
    rows = csv.reader(csvfile)
    new = list(rows)

new[1:10001] = ans_list[1:10001]
with open('submission.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(new)