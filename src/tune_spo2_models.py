import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense, Conv1D, Flatten, GRU

script_dir = os.getcwd()
sys.path.append(script_dir)
import utils
from tune_models import tune, visualize_hp


def gru_delta_model(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('gru_base_units', values = [16, 32, 64, 128, 256])
    num_layers = hp.Int('gru_num_layers', min_value = 0, max_value = 5, step = 1)

    # base_units = 16
    # num_layers = 5
    # base_dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 1)
    base_dense_units = 64
    num_dense_layers = 1

    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    for i in range(num_layers, 0, -1): 
        model.add(GRU(base_units * (2 ** i), return_sequences = True))
    model.add(GRU(base_units))

    for i in range(num_dense_layers - 1, -1, -1): 
        model.add(Dense(base_dense_units * ( 2 ** i ), activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def bi_gru_delta_model(hp): 
    model = tf.keras.models.Sequential()

    # base_units = hp.Choice('gru_base_units', values = [16, 32, 64, 128, 256])
    # num_layers = hp.Int('gru_num_layers', min_value = 0, max_value = 5, step = 1)

    base_units = 16
    num_layers = 5
    base_dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 1)

    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    for i in range(num_layers, 0, -1): 
        model.add(Bidirectional(GRU(base_units * (2 ** i), return_sequences = True)))
    model.add(Bidirectional(GRU(base_units)))

    for i in range(num_dense_layers - 1, -1, -1): 
        model.add(Dense(base_dense_units * ( 2 ** i ), activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def lstm_delta_model(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('base_units', values = [16, 32, 64, 128])
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 5, step = 1)
    base_dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 1)

    model.add(tf.keras.layers.InputLayer(input_shape = (20,)))
    model.add(tf.keras.layers.Reshape((20, 1)))
    for i in range(num_layers, 0, -1): 
        model.add(tf.keras.layers.LSTM(base_units * (2 ** i), return_sequences = True))
    model.add(tf.keras.layers.LSTM(base_units))

    for i in range(num_dense_layers, 0, -1):
        model.add(tf.keras.layers.Dense(base_dense_units  * (2 ** (i-1)), activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def lstm_norm_model(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('base_units', values = [16, 32, 64, 128])
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 5, step = 1)
    base_dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    num_dense_layers = hp.Int('num_layers', min_value = 1, max_value = 5, step = 1)

    model.add(tf.keras.layers.InputLayer(input_shape = (40,)))
    model.add(tf.keras.layers.Reshape((40, 1)))
    for i in range(num_layers, 0, -1): 
        model.add(tf.keras.layers.LSTM(base_units * (2 ** i), return_sequences = True))
    model.add(tf.keras.layers.LSTM(base_units))

    for i in range(num_dense_layers, 0, -1):
        model.add(tf.keras.layers.Dense(base_dense_units  * (2 ** (i-1)), activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model
    
def cnn_1d_model(hp): 
    cnn_model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    cnn_model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    cnn_model.add(Reshape((-1, 1)))

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)
    filter_num = hp.Choice('num_filters', values = [16, 64, 128, 256])
    filter_size = hp.Int('filter_size', min_value = 2, max_value = 4, step = 1)

    # num_layers = 1 
    # filter_num = 16
    # filter_size = 3
    # base_dense_units = hp.Choice('dense_units', values = [16, 64, 128])
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 2)
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

def bilstm_delta_model(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('base_units', values = [16, 32, 64])
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 5, step = 2)
    base_dense_units = hp.Choice('dense_units', values = [16, 64, 128])
    num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 2)
    
    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))

    for i in range(num_layers, 0, -1): 
        model.add(Bidirectional(LSTM(base_units * (2 ** i), return_sequences = True)))
    model.add(Bidirectional(LSTM(base_units)))

    for i in range(num_dense_layers, 0, -1):
        model.add(Dense(base_dense_units  * (2 ** (i-1)), activation='relu'))

    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

tune(bilstm_delta_model, '/home/hy381/rds/hpc-work/model_tuning', 'tune_bilstm_delta_model_all')
# tune(cnn_1d_model, 'tune_cnn_1d_model_filters')