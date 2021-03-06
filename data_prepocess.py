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
import nlpaug.flow as augf
from sklearn.model_selection import train_test_split
import pickle
from scipy import signal

    
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
    # melspec = python_speech_features.base.logfbank(wave, samplerate=sr, nfft=1024, nfilt=120, winlen=0.032, winstep=0.016)
    melspec = melspec.T
    # frq, time, melspec = signal.spectrogram(wave, fs=sr, window='hann', scaling='spectrum', nperseg=256)

    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec = melspec[:, :max_len]
    return melspec


# fw = open(train_path+temp+'.pkl','wb')
# pickle.dump(mel_list, fw)


if __name__ == '__main__':
    #讀取CSV
    with open('meta_train.csv') as csvfile:
        rows = csv.reader(csvfile)
        label = list(rows)
        
    with open('data_extend/Other.pkl', 'rb') as fp:
        noise = pickle.load(fp)
    label.remove(label[0])
    mel_list = []
    other = []
    
    y = [num[1] for num in label]
    X_train, X_validation = train_test_split(label, stratify = y ,test_size = 0.1, random_state=38)
    
    X_train.sort(key= lambda s:s[1])
    X_validation.sort(key= lambda s:s[1])
    with open('val_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(X_validation)

    category = ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']
    #label data 之後改掉
    for data in category:
      os.makedirs('data/'+data, exist_ok=True)
    index = 0
    mel_list = []
    temp = category[index]
    valid_path = 'val_npy/'
    aug = augf.Sometimes([ag.VtlpAug(8000, zone=(0,1)),
                          ag.ShiftAug(8000, 1.5),
                          ag.SpeedAug(factor=(0.8, 1.5), zone=(0, 1)),
                          ag.LoudnessAug(zone=(0,1), factor=(0.95, 2)),
                          ag.NoiseAug(noises=noise),
                          ag.NoiseAug()])    
    for data in X_validation:
        wave, sr = librosa.load('train/'+data[0]+'.wav', sr=None)
        # wave =speech_roi(wave)
        aug_datas = aug.augment(wave, 14)
        aug_datas.append(wave)
        for aug_data in aug_datas:
            mel = wav2melspec(aug_data, sr)
            
            if index == int(data[1]):
                mel_list.append(mel)
            else:

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
        # wave =speech_roi(wave)
        aug_datas = aug.augment(wave, 14)
        aug_datas.append(wave)
        for aug_data in aug_datas:
            mel = wav2melspec(aug_data, sr)
            
            if index == int(data[1]):
                mel_list.append(mel)
            else:
                # print(len(mel_list))            
                # fw = open(train_path+temp+'.pkl','wb')
                # pickle.dump(mel_list, fw)
                np.save(train_path+temp+'.npy', mel_list)
                #Update to new category
                index = index + 1
                temp = category[index]
                mel_list = []
                mel_list.append(mel)
    np.save(train_path+temp+'.npy', mel_list)