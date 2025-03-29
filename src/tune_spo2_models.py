import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import ReLU, Dropout, GlobalAveragePooling1D, MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense, Conv1D, Flatten, GRU, BatchNormalization
from tensorflow.keras.regularizers import l1 
script_dir = os.getcwd()
sys.path.append(script_dir)
from data import DataLoader
from tune_models import tune, visualize_hp


# Tune complete 

def lstm_delta_model(hp): 
    model = tf.keras.models.Sequential()

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 4, step = 1)
    base_units = hp.Choice('lstm_units', values = [16, 32, 64, 128, 256])

    model.add(tf.keras.layers.InputLayer(input_shape = (8,5)))
    # tuned
    # base_units = 256
    # num_layers = 2
    for i in range(num_layers): 
        model.add(LSTM(base_units, return_sequences=True))

    model.add(GlobalAveragePooling1D())

    # dense_units = hp.Choice('dense_units', values = [16, 64, 128, 256])
    # num_dense_layers = hp.Int('num_layers', min_value = 0, max_value = 4, step = 1)
    # for i in range(num_dense_layers): 
    #     model.add(tf.keras.layers.Dense(dense_units, activation='softmax'))
    model.add(Dense(256, activation = 'relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def cnn_1d_model(hp): 
    model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    model.add(Reshape((-1, 1)))

    # num_layers = hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)
    # filter_size = hp.Int('filter_size', min_value = 2, max_value = 4, step = 1)
    # num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128])
    # tuned
    num_layers = 1 
    filter_size = 2
    num_filters = 128

    num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128])

    for i in range(num_layers): 
        model.add(Conv1D(num_filters, filter_size, activation = 'relu'))
        model.add(MaxPool1D(2))
        model.add(BatchNormalization())
    
    model.add(GlobalAveragePooling1D())

    for i in range(num_dense_layers): 
        model.add(Dense(dense_units, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate= 3e-4, clipnorm=1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return model


def gru_delta_model(hp): 
    model = tf.keras.models.Sequential()

    # base_units = hp.Choice('gru_base_units', values = [16, 32, 64, 128, 256])
    # num_layers = hp.Int('gru_num_layers', min_value = 0, max_value = 5, step = 1)
    #Tuned
    base_units = 256
    num_layers = 5
    ####
    model.add(InputLayer(input_shape = (8,5)))
    for i in range(num_layers): 
        model.add(GRU(base_units, return_sequences=True))
    model.add(GRU(base_units))


    dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    num_dense_layers = hp.Int('dense_num_layers', min_value = 0, max_value = 5, step = 1)
    
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def bilstm_windowed_delta_model(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('lstm_units', values = [16, 64, 128, 256])
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 5, step = 1)

    # base_dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 5, step = 1)
    
    model.add(InputLayer(input_shape = (8,5)))

    for i in range(num_layers): 
        model.add(Bidirectional(LSTM(base_units, return_sequences = True)))
    model.add(Bidirectional(LSTM(base_units)))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model


def delta_dense_model(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('dense_units', values = [16, 64, 128, 256])
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 5, step = 1)
    # base_dense_units = hp.Choice('dense_units', values = [16, 64, 128])
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 2)
    
    model.add(InputLayer(input_shape = (20,)))

    for i in range(num_layers): 
        model.add(Dense(base_units))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model
# all tuned

def bilstm_changing_units(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('lstm_units', values = [16, 64, 128, 256])
    num_layers = hp.Int('num_layers', min_value = 1, max_value = 5, step = 1)

    model.add(InputLayer(input_shape = (8,5)))

    for i in range(num_layers-1): 
        model.add(Bidirectional(LSTM(base_units * int(2 ** i), return_sequences = True)))
    model.add(Bidirectional(LSTM(base_units * int(2 ** (num_layers - 1)))))

    # for i in range(num_dense_layers):
    #     model.add(Dense(base_dense_units * int(2**i), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def gru_changing_units(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('gru_base_units', values = [16, 32, 64, 128])
    num_layers = hp.Int('gru_num_layers', min_value = 1, max_value = 5, step = 1)
    #Tuned
    # base_units = 256
    # num_layers = 5
    ####
    model.add(InputLayer(input_shape = (8,5)))
    for i in range(num_layers-1): 
        model.add(GRU(base_units * int(2 ** i), return_sequences=True))
    model.add(GRU(base_units * int(2 ** (num_layers-1))))

    # dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    # num_dense_layers = hp.Int('dense_num_layers', min_value = 0, max_value = 5, step = 1)
    
    # for i in range(num_dense_layers):
    #     model.add(Dense(dense_units * int(2 ** i), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def cnn_1d_model_changing_param(hp): 
    model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20,), dtype = 'float32'))
    model.add(Reshape((-1, 1)))

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)
    filter_size = hp.Int('filter_size', min_value = 2, max_value = 4, step = 1)
    num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128])
    # tuned

    # num_same_layers = hp.Int('num_same_layers', min_value = 1, max_value = 4, step = 1)
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    # dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128])

    for i in range(num_layers): 
    # for i in range(num_same_layers): 
        model.add(Conv1D(num_filters * (2 ** i), filter_size, activation = 'relu', padding = 'same'))
        model.add(BatchNormalization())
        model.add(MaxPool1D(2))
        model.add(Dropout(0.2))
    
    model.add(Flatten())

    # for i in range(num_dense_layers): 
    #     model.add(Dense(dense_units * (2 ** i), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate= 3e-4, clipnorm=1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return model


def delta_dense_mode_changing_units(hp): 
    model = tf.keras.models.Sequential()

    base_units = hp.Choice('dense_units', values = [16, 64, 128, 256])
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 5, step = 1)
    reg_alpha = hp.Choice('reg_alpha', values = [0, 0.0001, 0.001, 0.01, 0.1])
    # base_dense_units = hp.Choice('dense_units', values = [16, 64, 128])
    # num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 2)
    
    model.add(InputLayer(input_shape = (20,)))

    for i in range(num_layers): 
        model.add(Dense(base_units * (2 ** i), kernel_regularizer = l1(reg_alpha)))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model


def lstm_changing_units(hp): 
    model = tf.keras.models.Sequential()

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 4, step = 1)
    base_units = hp.Choice('lstm_units', values = [16, 32, 64, 128])

    model.add(tf.keras.layers.InputLayer(input_shape = (8,5)))
    # tuned
    # base_units = 256
    # num_layers = 2
    for i in range(num_layers-1): 
        model.add(LSTM(base_units * (2 ** i), return_sequences=True))
        model.add(Dropout(0.1))
    
    model.add(LSTM(base_units * int(2 ** (num_layers - 1))))

    model.add(Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

data_loader = DataLoader()
spo2_windowed_data = data_loader.load_full_data('spo2', window_spo2 = True)
spo2_data = data_loader.load_full_data('spo2')

# tune(cnn_1d_model_changing_param, spo2_data, 'spo2_cnn_changing_filter_num', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(gru_changing_units, spo2_windowed_data, 'spo2_gru_changing_params_gru_units', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(lstm_changing_units, spo2_windowed_data, 'spo2_lstm_changing_units', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(bilstm_windowed_delta_model, spo2_windowed_data, 'spo2_bilstm_changing_param_bilstm_units', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)

# visualize_hp(lstm_changing_units, 'spo2_lstm_changing_units')
# visualize_hp(bilstm_windowed_delta_model, 'spo2_bilstm_changing_param_bilstm_units')



# tune(delta_dense_model, spo2_data, 'dense_spo2_data')
