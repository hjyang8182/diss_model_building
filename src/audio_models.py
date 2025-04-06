import tensorflow as tf
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D, ReLU, Conv1D, MaxPool1D, InputLayer, Dropout, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os 
import sys 
from data import ApneaDataLoader
import train



def mfcc_bilstm_model():
    model = tf.keras.models.Sequential()
    
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))

    model.add(Bidirectional(LSTM(512, return_sequences = True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(256, return_sequences = True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(128, return_sequences = True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model 

def mfcc_lstm_model(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))

    model.add(LSTM(512, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mfcc_cnn_model(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(64, (2, 5), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 4)))
    model.add(Dropout(0.3))

    # model.add(Conv2D(64, (2, 7), activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D((1, 3)))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(64, (3, 9), activation = 'relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D((2, 4)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3, 5)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu', kernel_regularizer = l1(0.0001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model
#

def mel_spec_cnn(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (128, 1251), dtype = 'float32'))
    model.add(Reshape((128, -1, 1)))

    model.add(Conv2D(32, (5,7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    # model.add(Conv2D(32, (6,8), activation = 'relu', padding = 'same'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D((2,4)))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(32, (7,9), activation = 'relu', padding = 'same'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D((2,4)))
    # model.add(Dropout(0.3))

    # model.add(Conv2D(32, (8,10), activation = 'relu', padding = 'same'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(2,4))
    # model.add(Dropout(0.3))

    model.add(Conv2D(32, (9,11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,5)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(16, activation = 'relu', kernel_regularizer=l1(0.0001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def opera_feature_dense(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (768,)))

    model.add(Dense(512, activation='relu', kernel_regularizer = l2(0.001)))
    
    model.add(Dense(256, activation='relu', kernel_regularizer = l2(0.001)))

    model.add(Dense(128, activation='relu', kernel_regularizer = l2(0.001)))

    model.add(Dense(64, activation='relu', kernel_regularizer = l2(0.001)))

    model.add(Dense(3, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

data_loader = ApneaDataLoader(train_batch_size=64)
# audio = data_loader.load_full_data(['audio'], parse_type = 'default')
audio_ms = data_loader.load_full_data(['audio_mel_spec'], parse_type = 'audio_feature')
audio_mfcc_denoised = data_loader.load_full_data(['audio_mfcc'], parse_type = 'audio_feature')
# full_mel_spec_png_data =  data_loader.load_mel_spec_images()
# opera_features = data_loader.load_opera_data/()
# train.train_model_full_data(mel_spec_cnn, 'audio_ms_new', audio_ms, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
train.test_model(mfcc_cnn_model, audio_mfcc_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)

# train.train_model_full_data(mfcc_cnn_model_denoised_new, 'audio_mfcc_new_denoise', audio_mfcc_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mel_spec_cnn_denoised, 'audio_mel_spec_new_denoise', audio_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)

# full_mel_spec_data = data_loader.load_full_data('audio_mel_spec', parse_type = 'audio_feature')
# full_mfcc_data = data_loader.load_full_data('audio_mfcc', parse_type = 'audio_feature')

# train.test_model(mfcc_cnn_model, full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# train.train_model_full_data(mel_spec_cnn, 'audio_mel_spec_cnn', full_mel_spec_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_bilstm_model, 'audio_mfcc_bilstm', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_lstm_model, 'audio_mfcc_lstm', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_cnn_model, 'audio_mfcc_cnn', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)