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

def gru_raw_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (40, )))
    model.add(Reshape((40, 1)))

    model.add(GRU(512, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(128))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def delta_gru_model(): 
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
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
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

def lstm_norm_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (40,)))
    model.add(Reshape((40, 1)))

    # model.add(LSTM(4096, return_sequences = True))
    # model.add(LSTM(2048, return_sequences = True))
    # model.add(LSTM(1024, return_sequences = True))
    model.add(LSTM(512, return_sequences = True))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(128))

    model.add(Dense(64, activation='relu'))
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
    cnn_model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    cnn_model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    cnn_model.add(Reshape((-1, 1)))

    num_layers = 1 
    filter_num = 16
    filter_size = 3
    base_dense_units = 64
    num_dense_layers = 3
    for i in range(num_layers-1, -1, -1):
        cnn_model.add(Conv1D(filter_num * (2 ** i), filter_size, activation = 'relu', kernel_initializer = 'he_normal'))
        cnn_model.add(MaxPool1D(2))

    cnn_model.add(Flatten())
    for i in range(num_dense_layers, 0, -1):
        cnn_model.add(Dense(base_dense_units  * (2 ** (i-1)), activation='relu'))
    cnn_model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    cnn_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return cnn_model

train.load_and_train(cnn_1d_model, 'spo2_cnn_model_full_data', train.spo2_delta_data, initial_epoch = 2)
train.train_model_full_data(cnn_1d_model, 'spo2_cnn_model_full_data', train.spo2_delta_data)
train.load_and_train(delta_gru_model, 'bi_gru_delta_full_data', train.spo2_delta_data, initial_epoch = 1)

train.train_model_full_data(delta_gru_model, 'bi_gru_delta_full_data', train.spo2_delta_data)
# train.train_model(lstm_delta_model_dense, 'lstm_delta_more_dense_data_subset', train.delta_model_data)
# train.load_and_train(lstm_delta_model_dense, 'lstm_delta_more_dense_layers_model', train.spo2_delta_data, initial_epoch = 8)
# train.train_model(cnn_1d_model, 'cnn_delta_model', train.delta_model_data)
# train.train_model(lstm_delta_model_dense, 'lstm_delta_more_dense_with_dropout', train.delta_model_data)
