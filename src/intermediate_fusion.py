import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer, MaxPool2D, GlobalAveragePooling1D, Input, BatchNormalization, Conv1D, Conv2D, Flatten, Dense, Concatenate, Dropout, Reshape, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential

import os
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from data import ApneaDataLoader
import numpy as np
from tune_models import tune, visualize_hp
import visualkeras

def spo2_mfcc(): 
    num_filters = 512
    spo2_model = Sequential([
        InputLayer(input_shape = (20,), dtype = 'float32'), 
        Reshape((-1, 1)), 
        Conv1D(num_filters, 6, activation = 'relu', padding = 'same'),
        MaxPool1D(2),
        BatchNormalization(),
        Dropout(0.2),
        GlobalAveragePooling1D()
    ])
    
    audio_model = Sequential([
        InputLayer(input_shape=(20, 1251), dtype='float32'),
        Reshape((20, -1, 1)),

        Conv2D(num_filters, (2, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((1, 3)),
        Dropout(0.3),

        Conv2D(num_filters, (2, 7), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((1, 3)),
        Dropout(0.3),

        Conv2D(num_filters, (3, 9), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 4)),
        Dropout(0.3),

        Conv2D(num_filters, (3, 11), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 4)),
        Dropout(0.3), 
        
        Reshape((-1, num_filters)),
        GlobalAveragePooling1D()
    ])


    spo2_input = Input(shape=(20,), name="spo2_input")
    audio_input = Input(shape=(20, 1251), name="audio_input")

    spo2_feature = spo2_model(spo2_input)
    audio_feature = audio_model(audio_input)

    fused = Concatenate()([spo2_feature, audio_feature])
    fused = BatchNormalization()(fused)

    x = Dense(512, activation = 'relu', kernel_regularizer = l1(0.001))(fused)
    x = Dropout(0.3)(x)

    output = Dense(3, activation = 'softmax')(x)
    fusion_model = Model(inputs=[spo2_input, audio_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    fusion_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    fusion_model.summary()
    return fusion_model



def spo2_ms(): 
    spo2_model = Sequential([
        InputLayer(input_shape = (20,), dtype = 'float32'), 
        Reshape((-1, 1)), 
        Conv1D(64, 6, activation = 'relu', padding = 'same'),
        MaxPool1D(2),
        BatchNormalization(),
        Dropout(0.2),
        GlobalAveragePooling1D()
    ])
    
    audio_model = Sequential([
        InputLayer(input_shape = (128, 1251), dtype = 'float32'), 
        Reshape((128,-1, 1)), 

        Conv2D(64, (3, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 3)),
        Dropout(0.3),
        
        Conv2D(64, (3, 7), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 3)),
        Dropout(0.3),
        
        Conv2D(64, (5, 9), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((4, 5)),
        Dropout(0.3), 

        Conv2D(64, (5, 11), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((4, 5)),
        Dropout(0.3), 

        Reshape((-1, 64)),
        GlobalAveragePooling1D()
    ])


    spo2_input = Input(shape=(20,), name="spo2_input")
    audio_input = Input(shape=(128, 1251), name="audio_input")

    spo2_feature = spo2_model(spo2_input)
    audio_feature = audio_model(audio_input)
    print(spo2_feature.shape)
    print(audio_feature.shape)

    fused = Concatenate()([spo2_feature, audio_feature])
    fused = BatchNormalization()(fused)

    x = Dense(64, activation = 'relu', kernel_regularizer = l1(0.001))(fused)
    x = Dropout(0.3)(x)

    output = Dense(3, activation = 'softmax')(x)
    fusion_model = Model(inputs=[spo2_input, audio_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    fusion_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    fusion_model.summary()
    return fusion_model

def train_interm_model(interm_model, spo2_data, audio_data, model_name): 
    fusion_model = interm_model()
    data_loader = ApneaDataLoader(train_batch_size=64)
  
    models_dir = '/home/hy381/rds/hpc-work/models'
    model_dir = os.path.join(models_dir, model_name)
    try: 
        os.mkdir(model_dir)
    except FileExistsError: 
            print(f"Directory {model_dir} already exists")
    except PermissionError: 
        print(f"Permission Denied to create directory {model_dir}")
    except Exception as e: 
        print(f"Error: {e} occurred making directory {model_dir}")
        
    model_file = os.path.join('/home/hy381/rds/hpc-work/models', model_name, f"{model_name}.keras")
    output_file = os.path.join(model_dir, f"{model_name}.txt")
    model_file = os.path.join(model_dir, f"{model_name}.keras")
    log_file = os.path.join(model_dir, f"{model_name}.log")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor = 'val_accuracy', save_best_only=True, mode = "max", verbose =1)
    csv_logger = CSVLogger(log_file)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, epsilon=1e-4, mode='min')

    train_dataset, valid_dataset, test_dataset = prepare_interm_data(spo2_data, audio_data)
    history = fusion_model.fit(train_dataset, steps_per_epoch = data_loader.FULL_TRAIN_STEPS_PER_EPOCH, epochs = 100, verbose = 1, validation_data = valid_dataset, callbacks = [checkpoint, csv_logger, reduce_lr_loss], validation_steps = data_loader.FULL_VALID_STEPS_PER_EPOCH)
    
    return history

def test_interm_model(spo2_data, audio_data, model_name): 
    model_file = os.path.join('/home/hy381/rds/hpc-work/models', model_name, f"{model_name}.keras")
    model = load_model(model_file)

    test_dataset = prepare_interm_data(spo2_data, audio_data)[2]

    preds = model.predict(test_dataset, steps = 3484)
    model_dir = os.path.join('/home/hy381/rds/hpc-work/models', model_name)
    predictions_output_file = os.path.join(model_dir, 'predictions.txt')
    np.savetxt(predictions_output_file, preds, fmt='%f')


def merge_fn(spo2_data, audio_data):
    spo2_input, spo2_label, _ = spo2_data
    audio_input, audio_label , _= audio_data
    return (spo2_input, audio_input), spo2_label

def prepare_interm_data(spo2_data, audio_data): 
    train_spo2_data, valid_spo2_data, test_spo2_data = spo2_data
    train_audio_data, valid_audio_data, test_audio_data = audio_data
    
    train_dataset = tf.data.Dataset.zip((train_spo2_data, train_audio_data))
    train_dataset = train_dataset.map(merge_fn)
    train_dataset = train_dataset.shuffle(buffer_size=10000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(64).repeat()

    valid_dataset = tf.data.Dataset.zip((valid_spo2_data, valid_audio_data))
    valid_dataset = valid_dataset.map(merge_fn)
    valid_dataset = valid_dataset.shuffle(buffer_size=10000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(64).repeat()

    test_dataset = tf.data.Dataset.zip((test_spo2_data, test_audio_data))
    test_dataset = test_dataset.map(merge_fn)
    test_dataset = test_dataset.batch(1)

    return train_dataset, valid_dataset, test_dataset

def tune_spo2_mfcc(hp): 
    num_filters = hp.Choice(f'num_filters', [16, 32, 64, 128, 256, 512])
    num_dense_layers = hp.Int('num_dense_layers', min_value = 1, max_value = 5, step = 1)
    dense_units = hp.Choice(f'dense_units', [16, 32, 64, 128, 256])
    
    spo2_model = Sequential([
        InputLayer(input_shape = (20,), dtype = 'float32'), 
        Reshape((-1, 1)), 
        Conv1D(num_filters, 6, activation = 'relu', padding = 'same'),
        MaxPool1D(2),
        BatchNormalization(),
        Dropout(0.2),
        GlobalAveragePooling1D()
    ])
    
    audio_model = Sequential([
        InputLayer(input_shape=(20, 1251), dtype='float32'),
        Reshape((20, -1, 1)),

        Conv2D(num_filters, (2, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((1, 3)),
        Dropout(0.3),

        Conv2D(num_filters, (2, 7), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((1, 3)),
        Dropout(0.3),

        Conv2D(num_filters, (3, 9), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 4)),
        Dropout(0.3),

        Conv2D(num_filters, (3, 11), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 4)),
        Dropout(0.3), 

        Reshape((-1, num_filters)),
        GlobalAveragePooling1D()
    ])

    spo2_input = Input(shape=(20,), name="spo2_input")
    audio_input = Input(shape=(20, 1251), name="audio_input")

    spo2_feature = spo2_model(spo2_input)
    audio_feature = audio_model(audio_input)

    fused = Concatenate()([spo2_feature, audio_feature])
    x = BatchNormalization()(fused)

    for i in range(num_dense_layers): 
        x = Dense(dense_units, activation = 'relu', kernel_regularizer = l1(0.001))(x)
        x = Dropout(0.3)(x)

    output = Dense(3, activation = 'softmax')(x)
    fusion_model = Model(inputs=[spo2_input, audio_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    fusion_model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy', 'precision', 'recall'])
    fusion_model.summary()
    return fusion_model

data_loader = ApneaDataLoader()
spo2_data = data_loader.load_full_data(['spo2'], apply_delta = True, batch = False)
mfcc_data = data_loader.prepare_data(['audio_mfcc'], parse_type = 'audio_feature', batch = False)
ms_data = data_loader.prepare_data(['audio_mel_spec'], parse_type = 'audio_feature', batch = False)

interm_spo2_mfcc_data = prepare_interm_data(spo2_data, mfcc_data)
interm_spo2_ms_data = prepare_interm_data(spo2_data, ms_data)
test_interm_model(spo2_data, ms_data, 'if_spo2_ms')
# train_interm_model(spo2_ms, spo2_data, ms_data, 'if_spo2_ms')


# def spo2_mfcc_inline():
    
#     spo2_input = Input(shape=(64,), name="spo2_input")
#     audio_input = Input(shape=(64, ), name="audio_input")

#     fused = Concatenate()([spo2_input, audio_input])
#     fused = BatchNormalization()(fused)

#     x = Dense(512, activation = 'relu', kernel_regularizer = l1(0.001))(fused)
#     x = Dropout(0.3)(x)

#     output = Dense(3, activation = 'softmax')(x)
#     fusion_model = Model(inputs=[spo2_input, audio_input], outputs=output)
#     return fusion_model

# model = spo2_mfcc_inline()
# visualkeras.layered_view(model, legend=True, to_file='intermediate_fusion_fused_part.png', max_xy=400)
