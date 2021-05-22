# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:43:21 2021

@author: arthur
"""
import os
import librosa
import python_speech_features
import tensorflow.keras as keras
import csv
import numpy as np
test_path = 'public_test/'
test_list = os.listdir(test_path)
weight_path = 'step1.hdf5'
weight_path_s2 = 'step2.hdf5'
model = keras.models.load_model(weight_path, compile = False)
model_step2 = keras.models.load_model(weight_path_s2, compile = False)
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
    # print(np.argmax(ans))
    if np.argmax(ans) == 0:
        ans = ans.tolist()[0]
        ans2 = model_step2.predict(mel)
        ans2 = ans2.tolist()[0]
        ans2 = [ans[0]*ans2[0],ans[0]*ans2[1],ans[0]*ans2[1]]
    else:
        ans = ans.tolist()[0]
        ans2 = [ans[0]/3, ans[0]/3, ans[0]/3]
       

    ans = ans2 + ans[1:3]
    ans.append(1-sum(ans))
    print(sum(ans))
    data_l = [data.split('.')[0]] + ans
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