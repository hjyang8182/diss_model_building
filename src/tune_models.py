import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
import keras_tuner as kt
from train import spo2_data_subset, audio_data, TRAIN_STEPS_PER_EPOCH, VALID_STEPS_PER_EPOCH
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense, Conv1D, Flatten, GRU

script_dir = os.getcwd()
sys.path.append(script_dir)
import utils


def tune(model_fn, project_name, feature): 
    if feature == 'audio': 
        train_data, valid_data, test_data = audio_data
    else: 
        train_data, valid_data, test_data = spo2_data_subset
    tuner = kt.Hyperband(model_fn, objective = 'val_accuracy', max_epochs = 50, factor = 3, directory = '/home/hy381/rds/hpc-work/model_tuning', project_name = project_name, overwrite = True)
    tuner.search(train_data, epochs = 50, validation_data = valid_data, steps_per_epoch = TRAIN_STEPS_PER_EPOCH, validation_steps = VALID_STEPS_PER_EPOCH)

def visualize_hp(model_fn, project_name): 
    directory = '/home/hy381/rds/hpc-work/model_tuning'
    tuner = kt.Hyperband(model_fn, objective = 'val_accuracy', max_epochs = 50, factor = 3, directory = directory, project_name = project_name)
    
    hp_values = []
    scores = []

    for trial in tuner.oracle.trials.values(): 
        hp_values.append(trial.hyperparameters.values)
        scores.append(trial.score)
    
    trials = tuner.oracle.get_best_trials(5)
    sorted_trials = sorted(trials, key=lambda t: t.score, reverse=True)  # Reverse if accuracy-based
    sorted_hp_values = [trial.hyperparameters.values for trial in sorted_trials]
    sorted_scores = [trial.score for trial in sorted_trials]

    print(sorted_hp_values)
    print(sorted_scores)