# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:18:26 2021

@author: arthurchien
"""

import tensorflow.keras as keras
from transformer import MultiHeadAttention
model = keras.models.load_model('model_log/checkpoint-01.hdf5',  custom_objects={'MultiHeadAttention': MultiHeadAttention})
model.summary()