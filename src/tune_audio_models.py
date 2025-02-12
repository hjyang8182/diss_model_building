import tensorflow as tf
from tensorflow.keras.layers import LSTM, InputLayer, MelSpectrogram, Dropout, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Normalization
import numpy as np
import os
import pandas as pd
import sys
import keras_tuner as kt
import matplotlib.pyplot as plt
from tune_models import tune, visualize_hp

script_dir = os.getcwd()
sys.path.append(script_dir)
import utils

def mfcc_cnn_model(): 
    num_mel_bins = 64
    sampling_rate = 8000
    sequence_stride = 160
    fft_length = 400
    n_samples = 320000
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (320000,), dtype = 'float32'))
    # Convert audio to mel spectrogram -> outputs (num_mel_bins = 80, 2501)
    model.add(MelSpectrogram(num_mel_bins = num_mel_bins, sampling_rate = sampling_rate, sequence_stride = sequence_stride, fft_length = fft_length, min_freq = 70, max_freq = 7500, power_to_db=True))
    model.add(tf.keras.layers.Lambda(lambda x: utils.mel_to_mfcc(x, 30)))
    model.add(Reshape((num_mel_bins, -1, 1)))

    model.add(Conv2D(32, (2,2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (2, 2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mfcc_model_lstm(hp): 
    num_mel_bins = 64
    sampling_rate = 8000
    sequence_stride = 160
    fft_length = 400
    n_samples = 320000

    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (320000,), dtype = 'float32'))
    # Convert audio to mel spectrogram -> outputs (num_mel_bins = 80, 2501)
    model.add(MelSpectrogram(num_mel_bins = num_mel_bins, sampling_rate = sampling_rate, sequence_stride = sequence_stride, fft_length = fft_length, min_freq = 70, max_freq = 7500, power_to_db=True))
    model.add(tf.keras.layers.Lambda(lambda x: utils.mel_to_mfcc(x, 30)))

    num_lstm_base_units = hp.Choice('num_lstm_base_units', values = [16, 32, 64, 128])
    num_lstm_layers = hp.Int('num_lstm_layers', min_value = 0, max_value = 5, step = 1)
    for i in range(num_lstm_layers, 0, -1): 
        model.add(tf.keras.layers.LSTM(num_lstm_base_units * (2 ** i), return_sequences = True))
    model.add(tf.keras.layers.LSTM(num_lstm_base_units))

    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 

def romero_cnn(hp): 
    num_mel_bins = 64
    sampling_rate = 8000
    sequence_stride = 160
    fft_length = 400
    n_samples = 320000

    model = tf.keras.models.Sequential()
    model.add(InputLayer(input_shape = (320000,), dtype = 'float32'))
    model.add(MelSpectrogram(num_mel_bins = num_mel_bins, sampling_rate = sampling_rate, sequence_stride = sequence_stride, fft_length = fft_length, min_freq = 70, max_freq = 7500))
    model.add(Normalization(axis = [1,2]))
    model.add(Reshape((num_mel_bins, -1, 1)))

    num_kernels = hp.Choice('num_kernels_base', values = [16, 32, 64, 128])
    kernel_size = hp.Int('num_layers', min_value = 2, max_value = 4, step = 1)

    for i in range(3): 
        model.add(Conv2D(num_kernels * (2 ** i), (kernel_size, kernel_size), activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(MaxPool2D((3,2)))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

tune(mfcc_model_lstm, 'mfcc_model_lstm', 'audio')