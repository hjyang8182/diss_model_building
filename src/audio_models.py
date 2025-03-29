import tensorflow as tf
from tensorflow.keras.layers import Activation, GlobalAveragePooling2D, ReLU, Conv1D, MaxPool1D, InputLayer, MelSpectrogram, Dropout, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, AveragePooling2D, Bidirectional
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os 
import sys 
script_dir = os.getcwd()
sys.path.append(script_dir)
# import only what you need from utils
from data import DataLoader
import train


print(tf.__version__)



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

    model.add(Conv2D(32, (2,2), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(Conv2D(32, (2,2), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(BatchNormalization())
    # model.add(ReLU())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (2,2), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(Conv2D(64, (2,2), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    model.add(BatchNormalization())
    # model.add(ReLU())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.3))

    # model.add(Conv2D(64, (2,2), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001)))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    # model.add(MaxPool2D((2,2)))
    # model.add(Dropout(0.3))


    model.add(Flatten())
    model.add(Dense(512, activation = 'relu', kernel_regularizer = l2(0.01)))
    model.add(Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)))
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


def wav2fec_feat_cnn_model(): 
    model = tf.keras.models.Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (60, 1251), dtype = 'float32'))
    model.add(Reshape((60, -1, 1)))

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

    model.add(Conv2D(32, (5,5), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (7,7), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3,3))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (9,9), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((4,4)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(64, activation = 'relu', kernel_regularizer=l1(0.001)))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    model.summary()
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    return model


data_loader = DataLoader(train_batch_size=64)


audio_feature = data_loader.load_full_data(['audio'], parse_type = 'default')

# full_mel_spec_png_data =  data_loader.load_mel_spec_images()
train.train_model_full_data(mel_spec_cnn, 'audio_mel_spec_new_no_denoise', audio_feature, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 80)
# train.train_model_full_data(resnet_model, 'audio_mel_spec_resnet', full_mel_spec_png_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 200)

# full_mel_spec_data = data_loader.load_full_data('audio_mel_spec', parse_type = 'audio_feature')
# full_mfcc_data = data_loader.load_full_data('audio_mfcc', parse_type = 'audio_feature')

# train.test_model(mfcc_cnn_model, full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH)
# train.train_model_full_data(mel_spec_cnn, 'audio_mel_spec_cnn', full_mel_spec_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_bilstm_model, 'audio_mfcc_bilstm', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_lstm_model, 'audio_mfcc_lstm', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
# train.train_model_full_data(mfcc_cnn_model, 'audio_mfcc_cnn', full_mfcc_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)