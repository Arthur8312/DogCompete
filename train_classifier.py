# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:06:36 2021

@author: arthurchien
"""

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import preprocess
import numpy as np
import os
import keras.layers as layers
import tensorflow_addons as tfa
feature_dim_2 = 499
feature_dim_1 = 120
channel = 1
epochs = 250
batch_size = 80
verbose = 1
num_classes = 6


X_train, X_test, y_train, y_test = preprocess.get_train_test()



X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

def model_2nd():
  model = Sequential()
  model.add(Conv2D(64, kernel_size=(3, 7), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel), padding='same'))
  model.add(MaxPooling2D(pool_size=(3, 1)))
  model.add(Conv2D(128, kernel_size=(7, 1), activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(4, 1)))
  model.add(Conv2D(256, kernel_size=(10, 1), activation='relu', padding='valid'))
  model.add(Conv2D(512, kernel_size=(1, 7), activation='relu', padding='same'))
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
checkpoint = keras.callbacks.ModelCheckpoint(filepath=weight_dir+'/checkpoint-{epoch:02d}.hdf5', period = 50)

# model = Res15()
model = model_2nd()

# model.compile(optimizer=tfa.optimizers.NovoGrad(learning_rate = 0.001,
#                                                     beta_1=0.98,
#                                                     beta_2=0.5,
#                                                     weight_decay=0.001),
#                   loss=keras.losses.categorical_crossentropy,
#                   metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=[checkpoint])
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=1e-6),
            metrics=['accuracy'])
model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=[checkpoint])
model.save('SpeechModel_0217.hdf5')

