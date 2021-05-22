# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:39:35 2021

@author: arthur
"""
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import keras.layers as layers
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from classification_models.keras import Classifiers
feature_dim_2 = 499
feature_dim_1 = 120
channel = 1
epochs = 20
batch_size = 80
verbose = 1


def get_data(labels, npy_path):

    X = np.load(npy_path + labels[0] + '.npy', allow_pickle=True )
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(npy_path + label + '.npy', allow_pickle=True )
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return X,y
category = ['Barking', 'Howling', 'Crying', 'COSmoke', 'GlassBreaking', 'Other']
X_train, y_train = get_data(category, 'train_npy/')
X_test, y_test = get_data(category, 'val_npy/')

X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_step1 = y_train.copy()
y_train_step1[y_train_step1<3] = 0
y_train_step1[y_train_step1==3] = 1
y_train_step1[y_train_step1==4] = 2
y_train_step1[y_train_step1==5] = 3
y_train_step1_hot = to_categorical(y_train_step1)

y_test_step1 = y_test.copy()
y_test_step1[y_test_step1<3] = 0
y_test_step1[y_test_step1==3] = 1
y_test_step1[y_test_step1==4] = 2
y_test_step1[y_test_step1==5] = 3
y_test_step1_hot = to_categorical(y_test_step1)

y_train_step2 = y_train[y_train<3].copy()
y_train_step2_hot = to_categorical(y_train_step2)
y_test_step2 = y_test[y_test<3].copy()
y_test_step2_hot = to_categorical(y_test_step2)
X_train_step2 = X_train[0:len(y_train_step2),:,:,:]
X_test_step2 = X_test[0:len(y_test_step2),:,:,:]

def audio_model(num_classes):
  model = Sequential()
  model.add(Conv2D(64, kernel_size=(3, 35), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel), padding='same'))
  model.add(MaxPooling2D(pool_size=(3, 1)))
  model.add(Conv2D(128, kernel_size=(7, 1), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(4, 1)))
  model.add(Conv2D(256, kernel_size=(10, 1), activation='relu', padding='valid'))
  model.add(Conv2D(512, kernel_size=(1, 35), activation='relu', padding='same'))
  model.add(keras.layers.GlobalMaxPooling2D())
  model.add(Dropout(0.25))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
  return model

model_step1 = audio_model(4)
weight_dir = "model_step1"
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
checkpoint1 = keras.callbacks.ModelCheckpoint(filepath=weight_dir+'/checkpoint-{epoch:02d}.hdf5')
model_step1.summary()
model_step1.fit(X_train, y_train_step1_hot, batch_size=batch_size, epochs=20, verbose=verbose, validation_data=(X_test, y_test_step1_hot), callbacks=[checkpoint1])
model_step1.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=1e-6),
            metrics=['accuracy'])
model_step1.fit(X_train, y_train_step1_hot, batch_size=batch_size, epochs=30, verbose=verbose, validation_data=(X_test, y_test_step1_hot), callbacks=[checkpoint1])
del model_step1
model_step2 = audio_model(3)
weight_dir = "model_step2"
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
checkpoint2 = keras.callbacks.ModelCheckpoint(filepath=weight_dir+'/checkpoint-{epoch:02d}.hdf5')
model_step2.summary()
model_step2.fit(X_train_step2, y_train_step2_hot, batch_size=batch_size, epochs=20, verbose=verbose, validation_data=(X_test_step2, y_test_step2_hot), callbacks=[checkpoint2])
model_step2.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=1e-6),
            metrics=['accuracy'])
model_step2.fit(X_train_step2, y_train_step2_hot, batch_size=batch_size, epochs=30, verbose=verbose, validation_data=(X_test_step2, y_test_step2_hot), callbacks=[checkpoint2])
