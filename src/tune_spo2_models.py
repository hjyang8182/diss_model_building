import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense, Conv1D, Flatten, GRU, BatchNormalization

script_dir = os.getcwd()
sys.path.append(script_dir)
import utils
from tune_models import tune, visualize_hp


def gru_delta_model(hp): 
    model = tf.keras.models.Sequential()

    # base_units = hp.Choice('gru_base_units', values = [16, 32, 64, 128, 256])
    num_layers = hp.Int('gru_num_layers', min_value = 0, max_value = 5, step = 1)

    # base_units = 16
    # num_layers = 5
    # base_dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 1)
    base_dense_units = 64
    num_dense_layers = 1

    model.add(InputLayer(input_shape = (20,)))
    model.add(Reshape((20, 1)))
    for i in range(num_layers): 
        model.add(GRU(hp.Choice(f'gru_unit_{i}', [16, 32, 64, 128]), return_sequences=True))
    model.add(GRU(hp.Choice(f'gru_unit_{num_layers}', [16, 32, 64, 128])))

    # for i in range(num_dense_layers - 1, -1, -1): 
    #     model.add(Dense(base_dense_units * ( 2 ** i ), activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.summary()
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

    num_layers = hp.Int('num_layers', min_value = 0, max_value = 4, step = 1)

    model.add(tf.keras.layers.InputLayer(input_shape = (20,)))
    model.add(tf.keras.layers.Reshape((20, 1)))
    for i in range(num_layers): 
        model.add(LSTM(hp.Choice(f'lstm_unit_{(i+1)}', [16, 32, 64, 128]), return_sequences=True))
    model.add(LSTM(hp.Choice(f'lstm_unit_{num_layers}', [16, 32, 64, 128])))

    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def cnn_1d_model(hp): 
    model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    model.add(Reshape((-1, 1)))

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)

    for i in range(num_layers): 
        model.add(Conv1D(hp.Choice(f'num_filters_{(i+1)}', [16, 32, 64, 128]), hp.Choice(f'filter_size_{(i+1)}', [2, 3, 4]), activation = 'relu'))
        model.add(MaxPool1D(2))
        model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return model

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

# tune(lstm_delta_model, 'tune_lstm_delta_model_lstm_units', 'spo2')
# tune(lstm_delta_model, 'tune_lstm_delta_model_lstm_units', 'spo2')
tune(gru_delta_model, 'tune_gru_delta_model_gru_units', 'spo2')
tune(cnn_1d_model, 'tune_cnn_model_kernels', 'spo2')
# tune(lstm_delta_model, 'tune_lstm_delta_model_dense_units', 'spo2')
# visualize_hp(lstm_delta_model, 'tune_lstm_delta_model_dense_units')
# tune(bilstm_delta_model, '/home/hy381/rds/hpc-work/model_tuning', 'tune_bilstm_delta_model_all')
# tune(cnn_1d_model, 'tune_cnn_1d_model_filters')