# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:06:36 2021

@author: arthurchien
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
import transformer
feature_dim_2 = 499
feature_dim_1 = 120
channel = 1
epochs = 10
batch_size = 80
verbose = 1
num_classes = 6



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

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def audio_model():
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


weight_dir = "model_log"
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=weight_dir+'/checkpoint-{epoch:02d}.hdf5')

# # model = audio_model()
# model = keras.models.load_model('model_log/checkpoint-150.hdf5')
NUM_LAYERS = 2
D_MODEL = X_train.shape[2]
NUM_HEADS = 4
UNITS = 1024
DROPOUT = 0.1
TIME_STEPS= X_train.shape[1]
OUTPUT_SIZE=80
EPOCHS = 10
EXPERIMENTS=10

model = transformer.transformer(time_steps=TIME_STEPS,
  num_layers=NUM_LAYERS,
  units=UNITS,
  d_model=D_MODEL,
  num_heads=NUM_HEADS,
  dropout=DROPOUT,
  output_size=OUTPUT_SIZE,  
  projection='linear')


model.summary()
# model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=[checkpoint])
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=1e-6),
            metrics=['accuracy'])

seed = 7
np.random.seed(seed)
kf = KFold(n_splits=10,shuffle=False, random_state=seed)
for train_index , test_index in kf.split(X_test):
    model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test[test_index], y_test_hot[test_index]), callbacks=[checkpoint])
model.save('Dog_0512_2.hdf5')

