import tensorflow as tf
import numpy as np
import os
import pandas as pd
import sys
import keras_tuner as kt
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPool1D, InputLayer, Reshape, LSTM, Bidirectional, Dense, Conv1D, Flatten, GRU

script_dir = os.getcwd()
sys.path.append(script_dir)
import data



def tune(model_fn, data, project_name, TRAIN_STEPS_PER_EPOCH, VALID_STEPS_PER_EPOCH): 
    train_data, valid_data, test_data = data
    tuner = kt.Hyperband(model_fn, objective = 'val_loss', max_epochs = 50, factor = 2, directory = '/home/hy381/rds/hpc-work/final_model_tuning', project_name = project_name)
    tuner.search(train_data, epochs = 50, validation_data = valid_data, steps_per_epoch = TRAIN_STEPS_PER_EPOCH, validation_steps = VALID_STEPS_PER_EPOCH)

def visualize_hp(model_fn, project_name): 
    directory = '/home/hy381/rds/hpc-work/final_model_tuning'
    tuner = kt.Hyperband(model_fn, objective = 'val_loss', max_epochs = 50, factor = 2, directory = directory, project_name = project_name)
    
    hp_values = []
    scores = []
    trials = []
    
    for trial in tuner.oracle.trials.values(): 
        if trial.score is not None: 
            # hp_values.append(trial.hyperparameters.values)
            # scores.append(trial.score)
            trials.append(trial)
    # trials = tuner.oracle.get_best_trials(5)
    sorted_trials = sorted(trials, key=lambda t: t.score, reverse=True)  # Reverse if accuracy-based
    sorted_hp_values = [trial.hyperparameters.values for trial in sorted_trials]
    sorted_scores = [trial.score for trial in sorted_trials]

    print(sorted_hp_values)
    print(sorted_scores)