import tensorflow as tf
from tensorflow.keras.layers import ReLU, Conv1D, MaxPool1D, InputLayer, MelSpectrogram, Dropout, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
import os 
import sys 
script_dir = os.getcwd()
sys.path.append(script_dir)
import utils as utils
import train


def mfcc_bilstm_model():
    # tuned bilstm units
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


    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))

    model.add(Dense(256, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 

def mel_spec_png_cnn(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (224, 224, 3), dtype = 'float32'))

    model.add(Conv2D(32, (2,2), kernel_regularizer=l1(0.005)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, clipnorm = 1.0)
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
    # tuned kernels 
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

    model.add(Conv2D(128, (3,3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model


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

    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(32, return_sequences = True))
    model.add(LSTM(32, return_sequences = True))
    model.add(LSTM(32))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mel_spec_lstm(): 
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

    # model.add(Bidirectional(LSTM(512, return_sequences = True)))
    # model.add(Bidirectional(LSTM(256, return_sequences = True)))
    # model.add(Bidirectional(LSTM(128, return_sequences = True)))
    model.add(LSTM(512))

    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

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

    model.add(Conv2D(64, (2,2), activation = 'relu', strides = (1, 2), kernel_regularizer=l1(0.005)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))

    # model.add(Conv2D(128, (2,2), activation = 'relu', kernel_regularizer=l1(0.005)))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D((2,2)))

    model.add(Flatten())
    # model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

train.test_model(romero_cnn, train.audio_data)
train.test_model(mfcc_cnn_model, train.audio_data)
train.test_model(mel_spec_png_cnn, train.mel_spec_data)
train.train_model_full_data(mfcc_bilstm_model, 'audio_mfcc_bilstm_model', train.audio_data_full)
train.train_model_full_data(mfcc_cnn_model, 'audio_mfcc_cnn_model', train.audio_data_full)
