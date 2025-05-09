import tensorflow as tf
from tensorflow.keras.layers import Lambda, GlobalAveragePooling1D, GlobalAveragePooling2D, ReLU, Conv1D, MaxPool1D, InputLayer, MelSpectrogram, Dropout, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
import numpy as np
import os
import pandas as pd
import sys
import keras_tuner as kt
import matplotlib.pyplot as plt
from data import ApneaDataLoader
from tune_models import tune, visualize_hp

script_dir = os.getcwd()
sys.path.append(script_dir)
import data

tf.keras.backend.clear_session()

# Tune complete
def mfcc_model_bilstm(hp):
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    # Convert audio to mel spectrogram -> outputs (num_mel_bins = 80, 2501)
    model.add(Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1])))

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 4, step = 1)
    base_units = hp.Choice(f'lstm_unit', [16, 32, 64, 128, 256, 512, 1024])

    # num_layers = 0 
    # base_units = 128
    for i in range(num_layers-1): 
        model.add(Bidirectional(LSTM(base_units, return_sequences=True)))
        model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(base_units)))
    model.add(Dropout(0.3))
    num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256, 512, 1024])
    # num_dense_layers = 2
    # dense_units = 32
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu'))
        model.add(Dropout(0.3))
    # mode,l.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 


def mfcc_model_lstm(hp):
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1])))

    # Tuned lstm
    num_layers = hp.Int('num_layers', min_value = 1, max_value = 4, step = 1)
    lstm_units = hp.Choice(f'lstm_units', [16, 32, 64, 128, 256])

    for i in range(num_layers-1, 0, -1): 
        model.add(LSTM(lstm_units * (2 ** i), return_sequences=True))
        model.add(Dropout(0.3))
    model.add(LSTM(lstm_units))
    model.add(Dropout(0.3))

    # num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    # dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256])
    # for i in range(num_dense_layers):
    #     model.add(Dense(dense_units, activation = 'relu'))
    num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256])
    # num_dense_layers = 2
    # dense_units = 32
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu'))
        model.add(Dropout(0.3))
    # model.add(Dense(64, activation = 'relu') )
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 

###########

def romero_cnn(hp): 

    model = tf.keras.models.Sequential()
    model.add(InputLayer(input_shape = (128, 1251), dtype = 'float32'))
    model.add(Reshape((128, -1, 1)))

    # num_layers = hp.Int('num_layers', min_value = 1, max_value = 4, step = 1)
    # num_kernels = hp.Choice('num_kernels_base', values = [16, 32, 64, 128])
    # filter_size = hp.Int('filter_size', min_value = 2, max_value = 4, step = 1)
    num_layers = 4
    num_kernels = 32 
    filter_size = 2

    for i in range(num_layers): 
        model.add(Conv2D(num_kernels, (filter_size, filter_size), activation = 'relu'))
        model.add(MaxPool2D((3,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    model.add(GlobalAveragePooling2D())
    num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    dense_units = hp.Choice('dense_units', values = [16, 32, 64, 128, 256])
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model


def mel_spec_lstm(hp): 
    num_mel_bins = 64
    sampling_rate = 8000
    sequence_stride = 50
    fft_length = 100
    n_samples = 320000
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (320000,), dtype = 'float32'))
    # Convert audio to mel spectrogram -> outputs (num_mel_bins = 80, 2501)
    model.add(MelSpectrogram(num_mel_bins = num_mel_bins, sampling_rate = sampling_rate, sequence_stride = sequence_stride, fft_length = fft_length, min_freq = 70, max_freq = 7500, power_to_db=True))

    num_lstm_base_units = hp.Choice('num_lstm_base_units', values = [16, 64, 128, 512])
    num_lstm_layers = hp.Int('num_lstm_layers', min_value = 0, max_value = 5, step = 1)
    
    for i in range(num_lstm_layers, 0, -1): 
        model.add(Bidirectional(LSTM(num_lstm_base_units, return_sequences = True)))
        model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(num_lstm_base_units)))

    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mel_spec_png_cnn(hp): 

    num_filters = hp.Choice('num_filters', values = [16, 64, 128, 512])
    filter_size = hp.Int('num_cnn_layers', min_value = 2, max_value = 4, step = 1)
    num_layers = hp.Int('num_cnn_layers', min_value = 0, max_value = 5, step = 1)
    
    model = Sequential()
    model.add(InputLayer(input_shape = (224, 224, 3), dtype = 'float32'))

    for i in range(num_layers): 
        model.add(Conv2D(num_filters, (filter_size, filter_size), kernel_regularizer=l1(0.005)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model


def mfcc_model_lstm_changing_units(hp):
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))

    # Tuned lstm
    num_layers = hp.Int('num_layers', min_value = 0, max_value = 4, step = 1)
    lstm_units = hp.Choice(f'lstm_units', [16, 32, 64, 128, 256])

    # num_layers = 3
    # lstm_units = 128
    for i in range(num_layers-1): 
        model.add(LSTM(lstm_units * (2 ** i), return_sequences=True))
        model.add(Dropout(0.1))
    model.add(LSTM(base_units * int(2 ** (num_layers-1))))
    model.add(Dropout(0.1))

    num_dense_layers = hp.Int('num_dense_layers', min_value = 0, max_value = 4, step = 1)
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256]) 
    
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 

def mfcc_cnn_model(hp): 
    num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128])
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256, 512])
    num_dense_layers = hp.Int('dense_layers', min_value = 1, max_value = 4, step = 1)

    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(num_filters, (2, 7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (2, 8), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (3, 9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (3, 10), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 5)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu', kernel_regularizer = l1(0.0001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model
#
def mel_spec_cnn(hp): 
    num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128])
    # kernel_size= hp.Int(f'kernel_size', min_value = 2, max_value = 4)
    # num_layers = hp.Int('num_layers', min_value = 1, max_value = 4, step = 1)
    # num_same_layers = hp.Int('num_same_layers', min_value = 1, max_value = 4, step = 1)
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256, 512])
    num_dense_layers = hp.Int('dense_layers', min_value = 1, max_value = 4, step = 1)

    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (128, 1251), dtype = 'float32'))
    model.add(Reshape((128, -1, 1)))

    model.add(Conv2D(num_filters, (5,7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (6,8), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (7,9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (8,10), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,4))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (9,11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,5)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu', kernel_regularizer=l1(0.0001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model



def mfcc_cnn_bilstm_model(hp): 
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256, 512])
    num_dense_layers = hp.Int('dense_layers', min_value = 1, max_value = 4, step = 1)

    lstm_units = hp.Choice(f'lstm_units', [16, 32, 64, 128, 256, 512])
    num_lstm_layers = hp.Int('lstm_layers', min_value = 1, max_value = 4, step = 1)

    # num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128, 256, 512])
    num_filters = 512
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(num_filters, (2, 5), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((1, 4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (2, 7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((1, 4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (3, 9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 5)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (3, 11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 5)))
    model.add(Dropout(0.3))
    
    model.add(Reshape((15, num_filters)))
  
    for i in range(num_lstm_layers):
        model.add(Bidirectional(LSTM(lstm_units, return_sequences = True)))
        model.add(Dropout(0.3))
    model.add(GlobalAveragePooling1D())

    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu', kernel_regularizer = l1(0.0001)))
        model.add(Dropout(0.3))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mfcc_cnn_lstm_model(hp): 
    # dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256, 512])
    # num_dense_layers = hp.Int('dense_layers', min_value = 1, max_value = 4, step = 1)

    # lstm_units = hp.Choice(f'lstm_units', [16, 32, 64, 128, 256, 512])
    # num_lstm_layers = hp.Int('lstm_layers', min_value = 1, max_value = 4, step = 1)

    num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128, 256, 512])
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(num_filters, (2, 5), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((1, 4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (2, 7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((1, 4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (3, 9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 5)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (3, 11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 5)))
    model.add(Dropout(0.3))
    
    model.add(Reshape((15, num_filters)))

    model.add(LSTM(512, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(32, activation = 'relu', kernel_regularizer = l1(0.0001)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation = 'softmax'))
    # for i in range(num_lstm_layers):
    #     model.add(LSTM(lstm_units, return_sequences = True))
    #     model.add(Dropout(0.3))
    # model.add(GlobalAveragePooling1D())

    # for i in range(num_dense_layers):
    #     model.add(Dense(dense_units, activation = 'relu', kernel_regularizer = l1(0.0001)))
    #     model.add(Dropout(0.3))
    # model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model



def ms_cnn_lstm_model(hp):
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256, 512])
    num_dense_layers = hp.Int('dense_layers', min_value = 1, max_value = 4, step = 1)

    lstm_units = hp.Choice(f'lstm_units', [16, 32, 64, 128, 256, 512])
    num_lstm_layers = hp.Int('lstm_layers', min_value = 1, max_value = 4, step = 1)

    num_filters = 64
    # num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128, 256, 512])
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (128, 1251), dtype = 'float32'))
    model.add(Reshape((128, -1, 1)))

    model.add(Conv2D(num_filters, (4,8), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (5,9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (7,10), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,5)))
    model.add(Dropout(0.3))

    model.add(Conv2D(num_filters, (9,11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,5)))
    model.add(Dropout(0.3))

    # model.add(Reshape((-1, num_filters)))
    # model.add(LSTM(512, return_sequences = True))
    # model.add(Dropout(0.3))
    # model.add(GlobalAveragePooling1D())

    model.add(Reshape((-1, num_filters)))
    for i in range(num_lstm_layers):
        model.add(LSTM(lstm_units, return_sequences = True))
        model.add(Dropout(0.3))

    model.add(GlobalAveragePooling1D())

    for i in range(num_dense_layers):
        model.add(Dense(dense_units, activation = 'relu', kernel_regularizer = l1(0.0001)))
        model.add(Dropout(0.3))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model
data_loader = ApneaDataLoader()
audio_mel_spec = data_loader.load_full_data(['audio_mel_spec'], parse_type = 'audio_features')
audio_mfcc = data_loader.load_full_data(['audio_mfcc'], parse_type = 'audio_features')


# tune(mfcc_cnn_bilstm_model, audio_mfcc, 'mfcc_cnn_bilstm_lstm_dense', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)

tune(ms_cnn_lstm_model, audio_mel_spec, 'ms_cnn_lstm', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# visualize_hp(ms_cnn_lstm_model, 'ms_cnn_lstm')

# visualize_hp(mfcc_cnn_lstm_model, 'mfcc_cnn_lstm_num_filters')

# visualize_hp(mfcc_cnn_lstm_model, 'mfcc_cnn_lstm_lstm_dense')

# tune(mfcc_cnn_lstm_model, audio_mfcc, 'mfcc_cnn_lstm_lstm_dense', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)

# tune(mfcc_cnn_model, audio_mfcc, 'mfcc_cnn_kernel_num_dense', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(mfcc_model_bilstm, audio_mfcc, 'mfcc_bilstm_transposed', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(mfcc_model_lstm, audio_mel_spec, 'mfcc_lstm_transposed', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)


# tune(mfcc_model_lstm, audio_mfcc, 'audio_mfcc_lstm_changing_lstm_units', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(mfcc_model_bilstm, audio_mfcc, 'audio_mfcc_bilstm_changing_lstm_units', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(mel_spec_cnn, audio_mel_spec, 'audio_mel_spec_cnn_filter_num', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(mfcc_cnn_model_changing_param, audio_mfcc, 'audio_mfcc_cnn_filter_num', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)

# tune(mfcc_cnn_model_changing_param, audio_mfcc, 'audio_mfcc_model_changing_param_equalized_data', data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# tune(romero_cnn_changing_params, subset_mel_spec_data, 'audio_romero_changing_param_changing_num_same_layer', data_loader.TRAIN_STEPS_PER_EPOCH, data_loader.VALID_STEPS_PER_EPOCH)
# visualize_hp(mfcc_model_lstm, 'audio_mfcc_lstm_changing_lstm_units')
# visualize_hp(mfcc_model_bilstm, 'audio_mfcc_bilstm_changing_lstm_units')
# visualize_hp(mel_spec_cnn, 'audio_mel_spec_cnn_filter_num')
# visualize_hp(mel_spec_cnn, 'mel_spec_cnn_kernel_num_dense')