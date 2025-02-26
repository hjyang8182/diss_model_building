import tensorflow as tf
import train
import numpy as np
import os
import pandas as pd
import sys
script_dir = os.getcwd()
sys.path.append(script_dir)
import utils as utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Flatten, Conv1D, MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense
from tensorflow.keras.regularizers import l2

def lstm_delta_model_dense(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    model.add(LSTM(32))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def gru_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20, )))
    model.add(Reshape((20, 1)))

    model.add(GRU(2048, return_sequences=True))
    model.add(GRU(1024, return_sequences=True))
    model.add(GRU(512, return_sequences=True))
    model.add(GRU(256))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def bi_gru_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20, )))
    model.add(Reshape((20, 1)))

    base_units = 16
    num_layers = 5
    for i in range(num_layers, 0, -1): 
        model.add(Bidirectional(GRU(base_units * (2 ** i), return_sequences = True)))
    model.add(Bidirectional(GRU(base_units)))

    num_dense_layers = 1
    base_dense_units = 64
    for i in range(num_dense_layers - 1, -1, -1): 
        model.add(Dense(base_dense_units * ( 2 ** i ), activation = 'relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

train.train_model_full_data(lstm_delta_model_dense, 'spo2_biclass_lstm_model',train.biclass_spo2_data)
train.train_model_full_data(gru_model, 'spo2_biclass_gru_model',train.biclass_spo2_data)
train.train_model_full_data(bi_gru_model, 'spo2_biclass_bigru_model',train.biclass_spo2_data)