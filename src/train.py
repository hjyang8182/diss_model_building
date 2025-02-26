import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
import utils as utils
# import spo2_models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
# import audio_models

os.chdir('/home/hy381/rds/hpc-work/segmented_data_new')

train_files, valid_files, test_files = utils.split_train_valid_test_subject()
full_train_count = len(train_files) 
full_valid_count = len(valid_files) 
full_test_count = len(test_files)
FULL_TRAIN_STEPS_PER_EPOCH = full_train_count//128
FULL_VALID_STEPS_PER_EPOCH = full_valid_count//32 -1

spo2_data_full = utils.prepare_data_subject(feature = 'spo2', p_train = 0.7, p_valid = 0.15, use_delta = True)
audio_data_full = utils.prepare_data_subject(feature = 'audio')
biclass_spo2_data = utils.prepare_biclass_data(feature = 'spo2', use_delta = True)
biclass_audio_data = utils.prepare_biclass_data(feature = 'audio', use_delta = True)
mel_spec_data = utils.load_mel_spec_images()

spo2_data_subset = utils.prepare_data_subset(feature = 'spo2', train_length= 8000, valid_length = 800, test_length = 800, train_batch_size= 128, valid_test_batch_size= 32, use_delta = True)
audio_data = utils.prepare_data_subset(feature = 'audio', train_length= 8000, valid_length = 800, test_length = 800, train_batch_size= 128, valid_test_batch_size= 32)

train_count = 8000 
valid_count = 800 
test_count = 800 
TRAIN_STEPS_PER_EPOCH = train_count//128
VALID_STEPS_PER_EPOCH = valid_count//32 -1


def test_model(model_fn, model_data): 

    train_data, valid_data, test_data = model_data
    
    model = model_fn()
    # Make directory to store all output files related to the model
   
    # Plot the shape of the model 
    
    # Callbacks for the model 
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights = True)
    
    # Fit the model 
    csv_logger = CSVLogger(f'/home/hy381/model_training/src/{model_fn.__name__}.log')
    history = model.fit(train_data, steps_per_epoch = TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_data, callbacks = [csv_logger], validation_steps = VALID_STEPS_PER_EPOCH)

    return history

def train_model(model_fn, model_name, model_data): 
    models_dir = '/home/hy381/rds/hpc-work/models'

    train_data, valid_data, test_data = model_data
    
    model = model_fn()
    # Make directory to store all output files related to the model
    model_dir = os.path.join(models_dir, model_name)
    try: 
        os.mkdir(model_dir)
    except FileExistsError: 
            print(f"Directory {model_dir} already exists")
    except PermissionError: 
        print(f"Permission Denied to create directory {model_dir}")
    except Exception as e: 
        print(f"Error: {e} occurred making directory {model_dir}")

    model_plot_file = os.path.join(model_dir, f"{model_name}.png")
    output_file = os.path.join(model_dir, f"{model_name}.txt")
    model_file = os.path.join(model_dir, f"{model_name}.keras")

    # Plot the shape of the model 
    tf.keras.utils.plot_model(model, show_shapes = True, to_file = model_plot_file)
    
    # Callbacks for the model 
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, save_best_only=True)
    
    # Fit the model 
    history = model.fit(train_data, steps_per_epoch = TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_data, callbacks = [checkpoint], validation_steps = VALID_STEPS_PER_EPOCH)

    with open(output_file, 'w') as file: 
        json.dump(history.history, file)
    
    return history


def train_model_full_data(model_fn, model_name, model_data): 
    models_dir = '/home/hy381/rds/hpc-work/models'

    train_data, valid_data, test_data = model_data
    
    model = model_fn()
    # Make directory to store all output files related to the model
    model_dir = os.path.join(models_dir, model_name)
    try: 
        os.mkdir(model_dir)
    except FileExistsError: 
            print(f"Directory {model_dir} already exists")
    except PermissionError: 
        print(f"Permission Denied to create directory {model_dir}")
    except Exception as e: 
        print(f"Error: {e} occurred making directory {model_dir}")

    model_plot_file = os.path.join(model_dir, f"{model_name}.png")
    output_file = os.path.join(model_dir, f"{model_name}.txt")
    model_file = os.path.join(model_dir, f"{model_name}.keras")
    log_file = os.path.join(model_dir, f"{model_name}.log")
    # Plot the shape of the model 
    tf.keras.utils.plot_model(model, show_shapes = True, to_file = model_plot_file)
    
    # Callbacks for the model 
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_file)
    csv_logger = CSVLogger(log_file)
    # Fit the model 
    history = model.fit(train_data, steps_per_epoch = FULL_TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_data, callbacks = [checkpoint, csv_logger], validation_steps = FULL_VALID_STEPS_PER_EPOCH)

    with open(output_file, 'w') as file: 
        json.dump(history.history, file)
    
    return history

def load_and_train(model_fn, model_name, model_data, initial_epoch, custom_objects = {}): 
    model_file = os.path.join('/home/hy381/rds/hpc-work/models', model_name, f"{model_name}.keras")
    model = model_fn()
    model = load_model(model_file, custom_objects = custom_objects, safe_mode = False)

    train_data, valid_data, test_data = model_data

    output_file = os.path.join('/home/hy381/rds/hpc-work/models', f"{model_name}.txt")
    print(output_file)

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights = True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file)
    
    # Fit the model 
    history = model.fit(train_data, initial_epoch = initial_epoch, steps_per_epoch = FULL_TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_data, callbacks = [checkpoint], validation_steps = FULL_VALID_STEPS_PER_EPOCH)

    with open(output_file, 'w') as file: 
        json.dump(history.history, file)# train_model('audio_romero', 'audio_romero_modified', audio_data)

    return history