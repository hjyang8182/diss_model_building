import tensorflow as tf
import train
import numpy as np
import os
import pandas as pd
import sys
script_dir = os.getcwd()
sys.path.append(script_dir)
from data import DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, GRU, Dropout, Flatten, Conv1D, MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense
from tensorflow.keras.regularizers import l2


def cnn_1d_model(): 
    # Tuned kernel complete
    model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    model.add(Reshape((-1, 1)))

    model.add(Conv1D(128, 2, activation = 'relu', kernel_initializer = 'he_normal'))
    model.add(MaxPool1D(2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())

    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return model


def dense_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (20,)))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def gru_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (8, 5)))

    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(256))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def windowed_bilstm_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (8,5)))

    model.add(Bidirectional(LSTM(256, return_sequences= True)))
    model.add(Bidirectional(LSTM(256, return_sequences = True)))
    model.add(Bidirectional(LSTM(256, return_sequences = True)))
    model.add(Bidirectional(LSTM(256, return_sequences = True)))
    model.add(Bidirectional(LSTM(256)))

    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model


def lstm_model(): 
    # tuned lstm units complete
    model = Sequential()
    model.add(InputLayer(input_shape = (8,5)))

    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(256))

    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

####
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

data_loader = DataLoader(train_batch_size=128, valid_test_batch_size=32)
windowed_spo2_data_full = data_loader.load_full_data('spo2', window_spo2= True)
spo2_data_full = data_loader.load_full_data('spo2')

train.train_model_full_data(cnn_1d_model, spo2_data_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
train.train_model_full_data(dense_model, spo2_data_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
train.train_model_full_data(gru_model, windowed_spo2_data_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
train.train_model_full_data(windowed_bilstm_model, windowed_spo2_data_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
train.train_model_full_data(lstm_model, windowed_spo2_data_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
