# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:38:13 2021

@author: arthurchien
"""

import os
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
LABEL = ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']

np_path = 'val_npy/'
wave_list = []
index = 0
weight_path = 'model_log/best.h5'
model = load_model(weight_path, compile=False)
confusion = np.zeros([len(LABEL), len(LABEL)])
for i, np_file in enumerate(LABEL):
    np_temp = np.load(np_path+np_file+'.npy')
    wave_list.append(np_temp)
    
for i, mel_list in enumerate(wave_list):
    for mel in mel_list:
        idx_true = i
        mel = mel.reshape(1, 120, 499, 1)
        pred = model.predict(mel)
        value = np.max(pred)
        idx_pred = np.argmax(pred)
    

        confusion[idx_true][idx_pred] = confusion[idx_true][idx_pred] + 1
        
df_cm = pd.DataFrame(confusion, index=[i for i in LABEL], columns=[i for i in LABEL])
plt.figure(figsize = (10, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.ylabel('truth')
plt.xlabel('prediction')