import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, ReLU, Conv1D, MaxPool1D, InputLayer, MelSpectrogram, Dropout, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
import os 
import sys 
script_dir = os.getcwd()
sys.path.append(script_dir)
# import only what you need from utils
from data import DataLoader
import train

def mfcc_bilstm_model():
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))

    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.1))

    model.add(Dense(256, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 

def mfcc_lstm_model(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))

    model.add(LSTM(64))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mfcc_cnn_model(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(16, (2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(16, (2,2)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model


#####

def mel_spec_png_cnn(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (224, 224, 3), dtype = 'float32'))

    model.add(Conv2D(16, (2,2), kernel_regularizer=l1(0.001)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (2,2), kernel_regularizer=l1(0.001)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))

    # model.add(Conv2D(64, (2,2), kernel_regularizer=l1(0.001)))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    # model.add(MaxPool2D((2,2)))
    # model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
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
    sequence_stride = 256
    fft_length = 2048
    n_samples = 320000
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (64,), dtype = 'float32'))
    # Convert audio to mel spectrogram -> outputs (num_mel_bins = 80, 2501)
    # model.add(MelSpectrogram(num_mel_bins = num_mel_bins, sampling_rate = sampling_rate, sequence_stride = sequence_stride, fft_length = fft_length, min_freq = 70, max_freq = 7500, power_to_db=True))
    model.add(Reshape((64, -1, 1)))

    model.add(Conv2D(32, (2,2), kernel_regularizer=l1(0.005)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(64, (2,2), kernel_regularizer=l1(0.005)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(128, (2,2), kernel_regularizer=l1(0.005)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

# train.test_model(mfcc_dense_model, train.audio_data)

data_loader = DataLoader(subset_train_count=15000, subset_test_valid_count=3000, train_batch_size=128, valid_test_batch_size=32)
mel_spec_data = data_loader.load_mel_spec_images(subset=True)
train.test_model(mel_spec_png_cnn, mel_spec_data, data_loader.TRAIN_STEPS_PER_EPOCH, data_loader.VALID_STEPS_PER_EPOCH)

# audio_mfcc_subset = data_loader.load_subset_data('audio_mfcc', dataset = 'audio_feature')
# audio_mfcc_full = data_loader.load_full_data('audio_mfcc', dataset = 'audio_feature')

# train.test_model(mfcc_lstm_model, audio_mfcc_subset, data_loader.TRAIN_STEPS_PER_EPOCH, data_loader.VALID_STEPS_PER_EPOCH)
# train.train_model_full_data(mfcc_bilstm_model, 'mfcc_bilstm_full', audio_mfcc_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# train.train_model_full_data(mfcc_cnn_model, 'mfcc_cnn_full', audio_mfcc_full, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
