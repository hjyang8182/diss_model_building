import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
from data import DataLoader
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

def test_model(model_fn, model_data, TRAIN_STEPS_PER_EPOCH, VALID_STEPS_PER_EPOCH): 

    train_data, valid_data, test_data = model_data
    
    model = model_fn()
    # Make directory to store all output files related to the model
   
    # Plot the shape of the model 
    
    # Callbacks for the model 
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights = True)
    
    # Fit the model 
    csv_logger = CSVLogger(f'/home/hy381/model_training/src/{model_fn.__name__}.log')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    history = model.fit(train_data, steps_per_epoch = TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_data, callbacks = [csv_logger, reduce_lr_loss], validation_steps = VALID_STEPS_PER_EPOCH)

    return history

def train_model(model_fn, model_name, model_data, TRAIN_STEPS_PER_EPOCH, VALID_STEPS_PER_EPOCH): 
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


def train_model_full_data(model_fn, model_name, model_data, FULL_TRAIN_STEPS_PER_EPOCH, FULL_VALID_STEPS_PER_EPOCH): 
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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, save_best_only=True)
    csv_logger = CSVLogger(log_file)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    # Fit the model 
    history = model.fit(train_data, steps_per_epoch = FULL_TRAIN_STEPS_PER_EPOCH, epochs = 50, verbose = 1, validation_data = valid_data, callbacks = [checkpoint, csv_logger, reduce_lr_loss], validation_steps = FULL_VALID_STEPS_PER_EPOCH)

    with open(output_file, 'w') as file: 
        json.dump(history.history, file)
    
    return history

def load_and_train(model_fn, model_name, model_data, initial_epoch, custom_objects, FULL_TRAIN_STEPS_PER_EPOCH, FULL_VALID_STEPS_PER_EPOCH): 
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