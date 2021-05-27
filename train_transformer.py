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
# import tensorflow.keras.callbacks.ModelCheckpoint as ModelCheckpoint
import matplotlib.pyplot as plt

feature_dim_2 = 499
feature_dim_1 = 120
channel = 1
epochs = 100
batch_size = 2
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
X_train = X_train.transpose(0, 2, 1, 3) #transpose Freq and Time dominate
X_test = X_test.transpose(0, 2, 1, 3)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)




weight_dir = "model_log/"
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)

callback = [
    keras.callbacks.ModelCheckpoint(weight_dir+'-{epoch:02d}-{val_loss:.3f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
]
# # model = audio_model()
# model = keras.models.load_model('model_log/checkpoint-150.hdf5')
NUM_LAYERS = 10
D_MODEL = X_train.shape[2]
NUM_HEADS = 2
UNITS = 1024
DROPOUT = 0.25
TIME_STEPS= X_train.shape[1]
OUTPUT_SIZE=6


model = transformer.transformer(time_steps=TIME_STEPS,
  num_layers=NUM_LAYERS,
  units=UNITS,
  d_model=D_MODEL,
  num_heads=NUM_HEADS,
  dropout=DROPOUT,
  output_size=OUTPUT_SIZE,  
  projection='linear')


model.summary()
# model = keras.models.load_model('model_log/checkpoint-01.hdf5')
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=1e-4),
            metrics=['accuracy'])
history = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=callback)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('result_log/acc1.png')
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('result_log/loss1.png')
plt.close()

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=1e-5),
            metrics=['accuracy'])


model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=callback)
# model.save('Dog_0512_2.hdf5')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('result_log/acc2.png')
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('result_log/loss2.png')
plt.close()
