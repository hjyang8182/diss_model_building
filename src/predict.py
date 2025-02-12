import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
import src.utils as utils
import models

from train import delta_model_data
from tensorflow.keras.models import load_model

_, _, test_data = delta_model_data

def write_predictions_truths(model_fname):
    model = load_model(model_fname)
    predictions = model.predict(test_data)

    model_dir = os.path.dirname(model_fname)
    predictions_output_file = os.path.join(model_dir, 'predictions.txt')
    np.savetxt(predictions_output_file, predictions, fmt='%f')

    ground_truth_labels = []
    for _, label in test_data:
        ground_truth_labels.append(label)

    ground_truth_labels = np.array(ground_truth_labels)
    ground_truth_labels = ground_truth_labels.reshape(-1, 3)

    ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
    np.savetxt(ground_truth_output_file, ground_truth_labels, fmt = '%f')

    return ground_truth_labels, predictions_output_file

def confusion_matrix(model_fname): 
    model_dir = os.path.dirname(model_fname)
    predictions_output_file = os.path.join(model_dir, 'predictions.txt')
    ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
    
    preds = np.loadtxt(predictions_output_file)
    gt = np.loadtxt(ground_truth_output_file)

    gt_labels = np.argmax(gt, axis = 1)
    pred_labels = np.argmax(preds, axis = 1)
    cm = confusion_matrix(gt_labels, pred_labels)
    return cm
