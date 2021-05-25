# -*- coding: utf-8 -*-

import os

import librosa
import numpy as np
import pandas as pd
import seaborn as sn
import python_speech_features
from keras.models import load_model
import matplotlib.pyplot as plt

LABEL = ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']

train_wav_path = 'D:/dev/task/dog_voice/train'
train_label_path = 'D:/dev/task/dog_voice/meta_train.csv'
train_label = pd.read_csv(train_label_path)  

weight_path = 'D:/dev/task/dog_voice/DogCompete/weight_acc_97.hdf5'
model = load_model(weight_path, compile=False)

table = []
confusion = np.zeros([len(LABEL), len(LABEL)])
for idx in range(len(train_label)):
    row = train_label.iloc[idx]
    file_name = row.Filename
    idx_true = row.Label

    wav_path = f'{train_wav_path}/{file_name}.wav'
    audio, sr = librosa.load(wav_path, sr=8000)
    if len(audio) != 40000:
        audio = np.pad(audio, (0, 40000-len(audio)), 'constant')
    mel = python_speech_features.logfbank(audio, samplerate=8000, nfft=2048, nfilt=120)
    mel = mel.T
    mel = mel.reshape(1, 120, 499, 1)
    pred = model.predict(mel)
    value = np.max(pred)
    idx_pred = np.argmax(pred)
    
    if value < 0.9:
        confusion[idx_true][idx_pred] = confusion[idx_true][idx_pred] + 1
        print(file_name, idx_true, idx_pred, value)
    

df_cm = pd.DataFrame(confusion, index=[i for i in LABEL], columns=[i for i in LABEL])
plt.figure(figsize = (10, 7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.ylabel('truth')
plt.xlabel('prediction')




