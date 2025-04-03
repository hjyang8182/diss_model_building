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

    model.add(Dense(64, activation='softmax'))
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


#####

def vgg_16_mel_spec_png(): 
    vgg_model = VGG16(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
    # for layer in vgg_model.layers[:-4]:
    #     layer.trainable = False
    vgg_model.trainable = False
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', kernel_regularizer = l2(0.01)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model



def resnet_model(): 
    resnet_model = ResNet50(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
    resnet_model.trainable = False
    model = Sequential()
    model.add(resnet_model)

    model.add(MaxPool2D((7,7)))
    model.add(Flatten())

    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mel_spec_png_cnn(): 
    model = Sequential()
    model.add(InputLayer(input_shape = (224, 224, 3), dtype = 'float32'))

    model.add(Conv2D(64, (5,5), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (7,7), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (9,9), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
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
    model.add(InputLayer(input_shape = (128, 1251), dtype = 'float32'))
    # Convert audio to mel spectrogram -> outputs (num_mel_bins = 80, 2501)
    # model.add(MelSpectrogram(num_mel_bins = num_mel_bins, sampling_rate = sampling_rate, sequence_stride = sequence_stride, fft_length = fft_length, min_freq = 70, max_freq = 7500, power_to_db=True))
    model.add(Reshape((128, -1, 1)))

    model.add(Conv2D(16, (3,3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((4,3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3,3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((4,3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool2D((4,3)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def mfcc_cnn_model(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(16, (2, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(16, (2, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 3)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (2, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(32, (2, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 3)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (2, 5), activation = 'relu', padding = 'same'))
    model.add(Conv2D(64, (2, 5), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 5)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 7), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128, (2, 7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 7)))
    model.add(Dropout(0.2))

    model.add(Flatten())
#
    model.add(Dense(64, activation = 'relu', kernel_regularizer = l1(0.005)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    model.summary()
    return model

def mel_spec_cnn(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (128, 321), dtype = 'float32'))
    model.add(Reshape((128, -1, 1)))

    model.add(Conv2D(64, (6,8), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (7,9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (8,10), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3,4))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (9,11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,4)))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(64, activation = 'relu', kernel_regularizer=l1(0.001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

def mfcc_cnn_model_denoised_new(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (20, 1251), dtype = 'float32'))
    model.add(Reshape((20, -1, 1)))

    model.add(Conv2D(64, (2, 5), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((1, 3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (2, 7), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((1, 3)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 9), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 11), activation = 'relu'))
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

def mel_spec_cnn_denoised(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (128, 1251), dtype = 'float32'))
    model.add(Reshape((128, -1, 1)))

    model.add(Conv2D(64, (5,7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (6,8), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (7,9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,4)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (8,10), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,4))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (9,11), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3,5)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu', kernel_regularizer=l1(0.0001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model

data_loader = ApneaDataLoader(train_batch_size=64)
train_files, valid_files, test_files = data_loader.split_train_valid_test()
audio = data_loader.load_full_data(['audio'], parse_type = 'default')
audio_denoised = data_loader.load_full_data(['audio_mel_spec'], parse_type = 'audio_feature')
audio_mfcc_denoised = data_loader.load_full_data(['audio_mfcc'], parse_type = 'audio_feature')
full_mel_spec_png_data =  data_loader.load_mel_spec_images()

train.test_model(mel_spec_cnn_denoised, audio_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.test_model(mfcc_cnn_model_denoised_new, audio_mfcc_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)

# train.train_model_full_data(mfcc_cnn_model_denoised_new, 'audio_mfcc_new_denoise', audio_mfcc_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mel_spec_cnn_denoised, 'audio_mel_spec_new_denoise', audio_denoised, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)

# full_mel_spec_data = data_loader.load_full_data('audio_mel_spec', parse_type = 'audio_feature')
# full_mfcc_data = data_loader.load_full_data('audio_mfcc', parse_type = 'audio_feature')

# train.test_model(mfcc_cnn_model, full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# train.train_model_full_data(mel_spec_cnn, 'audio_mel_spec_cnn', full_mel_spec_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_bilstm_model, 'audio_mfcc_bilstm', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_lstm_model, 'audio_mfcc_lstm', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_cnn_model, 'audio_mfcc_cnn', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)