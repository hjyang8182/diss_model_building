import tensorflow as tf
import train
import numpy as np
import os
import pandas as pd
import sys
script_dir = os.getcwd()
sys.path.append(script_dir)
from data import DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, GRU, Dropout, Flatten, Conv1D, MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense
from tensorflow.keras.regularizers import l2, l1 


def cnn_1d_model(): 
    model = Sequential()
    # Input Layer - all inputs are dimensions (320,000, ) -> outputs (320,000, ) 
    model.add(InputLayer(input_shape = (140,), dtype = 'float32'))
    model.add(Reshape((-1, 1)))

    model.add(Conv1D(32, 2, activation = 'relu', padding = 'same'))
    model.add(MaxPool1D(4))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D(32, 4, activation = 'relu', padding = 'same'))
    model.add(MaxPool1D(4))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D(32, 8, activation = 'relu', padding = 'same'))
    model.add(MaxPool1D(4))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall']) 
    return model

data_loader = DataLoader()
concat_data = data_loader.load_full_data('concat_features', apply_delta = False)
train.test_model(cnn_1d_model, concat_data,  data_loader.FULL_TRAIN_STEPS_PER_EPOCH, data_loader.FULL_VALID_STEPS_PER_EPOCH, epochs = 100)
