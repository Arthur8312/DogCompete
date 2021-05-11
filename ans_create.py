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
test_path = 'public_test/'
test_list = os.listdir(test_path)
weight_path = 'SpeechModel_0217.hdf5'
model = keras.models.load_model(weight_path, compile = False)
category = ['Filename','Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']
ans_list = []
ans_list.append(category)
for data in test_list:
    data_path = test_path+data
    audio, sr = librosa.load(data_path, sr=8000)
    mel = python_speech_features.logfbank(audio, samplerate=8000, nfft=2048, nfilt=120)
    mel = mel.T
    mel = mel.reshape(1, 120, 499, 1)
    ans = model.predict(mel)
    ans = ans.tolist()
    data_l = [data.split('.')[0]] + ans[0]
    ans_list.append(data_l)
    
with open('submission.csv','w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(ans_list)
