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


def el_moaqet_lstm_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(40))

    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def dense_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def gru_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20, )))
    model.add(Reshape((20, 1)))

    model.add(GRU(16, return_sequences=True))
    model.add(GRU(16, return_sequences=True))
    model.add(GRU(16, return_sequences=True))
    model.add(GRU(16, return_sequences=True))
    model.add(GRU(16))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))

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
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def lstm_model(): 
    # tuned lstm units complete
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    model.add(LSTM(128, return_sequences= True))
    model.add(LSTM(64, return_sequences= True))
    model.add(LSTM(16))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def lstm_delta_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    model.add(LSTM(512, return_sequences= True))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(128))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def bilstm_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    model.add(Bidirectional(LSTM(1024, return_sequences = True)))
    model.add(Bidirectional(LSTM(512, return_sequences = True)))
    model.add(Bidirectional(LSTM(128)))

    model.add(Dense(128, activation='softmax'))
    model.add(Dense(64, activation='softmax'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()

    return model

def cnn_model(): 
    cnn_model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    cnn_model.add(tf.keras.layers.InputLayer(input_shape = (40,), dtype = 'float32'))
    cnn_model.add(tf.keras.layers.Reshape((1, 40, 1)))

    cnn_model.add(tf.keras.layers.Conv2D(32, (1,2), activation = 'relu', kernel_initializer = 'he_normal'))

    cnn_model.add(tf.keras.layers.Conv2D(32, (1,4), activation = 'relu', kernel_initializer = 'he_normal'))

    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(256, activation = 'relu'))
    cnn_model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    cnn_model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    cnn_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return cnn_model


def cnn_1d_model(): 
    # Tuned kernel complete
    model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    model.add(Reshape((-1, 1)))

    model.add(Conv1D(32, 4, activation = 'relu', kernel_initializer = 'he_normal'))
    model.add(MaxPool1D(2))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return model


train.train_model_full_data(cnn_1d_model, 'spo2_cnn_model_full_data', train.spo2_data_full)
train.train_model_full_data(gru_model, 'spo2_gru_model_full_data', train.spo2_data_full)
train.train_model_full_data(lstm_model, 'spo2_lstm_model_full_data', train.spo2_data_full)
train.train_model_full_data(el_moaqet_lstm_model, 'spo2_el_moaqet_lstm_full_data', train.spo2_data_full)

