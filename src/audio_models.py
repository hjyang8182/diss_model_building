import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, InputLayer, MelSpectrogram, Dropout, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
import os 
import sys 
script_dir = os.getcwd()
sys.path.append(script_dir)
import utils as utils
import train

def mel_spec_png_cnn(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (224, 224, 3), dtype = 'float32'))

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(256, (3,3), activation = 'relu'))
    model.add(Conv2D(256, (3,3), activation = 'relu'))
    model.add(Conv2D(256, (3,3), activation = 'relu'))
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(512, (3,3), activation = 'relu'))
    model.add(Conv2D(512, (3,3), activation = 'relu'))
    model.add(Conv2D(512, (3,3), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mel_to_mfcc_std_wrapper(x): 
    mfcc = utils.mel_to_mfcc(x, 50)
    std_mfcc = utils.standardize(mfcc)
    return std_mfcc

def mfcc_lstm_model(): 
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
    model.add(tf.keras.layers.Lambda(lambda x: utils.mel_to_mfcc(x)))
    # model.add(Reshape((num_mel_bins, -1, 1)))
    model.add(LSTM(64))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

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
    model.add(tf.keras.layers.Lambda(mel_to_mfcc_std_wrapper))
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


def mfcc_cnn_lstm_model(): 
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
    model.add(tf.keras.layers.Lambda(lambda x: utils.mel_to_mfcc(x)))
    model.add(Reshape((num_mel_bins, -1, 1)))

    model.add(Conv2D(32, (2,2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPool2D((3,3)))
    model.add(BatchNormalization())

    model.add(Reshape((-1, 1)))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(32))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model



def cnn_lstm_model(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (320000,)))
    model.add(Reshape((-1, 1)))

    model.add(Conv1D(3, 64))
    model.add(MaxPool1D(3))
    model.add(Conv1D(4, 128))
    model.add(MaxPool1D(4))
    model.add(Conv1D(5, 256))
    model.add(MaxPool1D(5))
    model.add(Conv1D(5, 256))
    model.add(MaxPool1D(5))

    # model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(32))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def cnn_lstm_mel_spec(): 
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
    model.add(Reshape((num_mel_bins, -1, 1)))
    
    model.add(Conv2D(32, (2,2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPool2D((3,3)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPool2D((3,3)))
    model.add(BatchNormalization())

    model.add(Reshape((-1, 1)))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(32))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def romero_cnn(): 
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
    model.add(Reshape((num_mel_bins, -1, 1)))

    model.add(Conv2D(64, (2,2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (2,2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (2,2), activation = 'relu'))
    model.add(MaxPool2D((2,2)))
    model.add(BatchNormalization())

    # model.add(Conv2D(128, (2,2), activation = 'relu'))
    # model.add(MaxPool2D((2,2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

train.test_model(mel_spec_png_cnn, train.mel_spec_data)
# custom_objects = {'mel_to_mfcc_std_wrapper': mel_to_mfcc_std_wrapper}
# train.test_model(romero_cnn, train.audio_data)
# train.test_model(mfcc_cnn_model, train.audio_data)
# train.train_model_full_data(mfcc_cnn_model, 'mfcc_cnn_full', train.audio_data_full)
# train.train_model_full_data(romero_cnn, 'romero_full',train.audio_data_full)
# train.test_model(mfcc_cnn_lstm_model, train.audio_data)
# train.test_model(romero_cnn, train.audio_data)
# train.test_model(mfcc_cnn_model, train.audio_data)
