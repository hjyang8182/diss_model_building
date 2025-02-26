import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
import utils as utils
import matplotlib.pyplot as plt

from train import delta_model_data
from tensorflow.keras.models import load_model

_, _, test_data = delta_model_data

def write_predictions_truths(model_name):
    model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
    model_fname = os.path.join(model_dir, model_name)
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

def confusion_matrix(model_name): 
    model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
    write_predictions_truths(model_name)
    model_fname = os.path.join(model_dir, model_name)
    predictions_output_file = os.path.join(model_dir, 'predictions.txt')
    ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
    
    preds = np.loadtxt(predictions_output_file)
    gt = np.loadtxt(ground_truth_output_file)

    gt_labels = np.argmax(gt, axis = 1)
    pred_labels = np.argmax(preds, axis = 1)
    cm = confusion_matrix(gt_labels, pred_labels)
    return cm

def plot_history(model_name): 
    results_file = f'/home/hy381/rds/hpc-work/models/{model_name}/{model_name}.txt'
    with open(results_file, 'r') as file: 
        results = json.load(file)
    train_acc = results['accuracy']
    train_precision = results['precision']
    train_recall = results['recall']
    val_acc = results['val_accuracy']
    val_precision = results['val_precision']
    val_recall = results['val_recall']
    fig, ax = plt.subplots(2, 3, figsize = (12, 8))
    ax = ax.flatten()

    ax[0].plot(val_acc)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Validation Accuracy")

    ax[1].plot(val_precision)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Validation Precision")

    ax[2].plot(val_recall)
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Validation Recall")

    ax[3].plot(train_acc)
    ax[3].set_xlabel("Epochs")
    ax[3].set_ylabel("Train Accuracy")

    ax[4].plot(train_precision)
    ax[4].set_xlabel("Epochs")
    ax[4].set_ylabel("Train Precision")

    ax[5].plot(train_recall)
    ax[5].set_xlabel("Epochs")
    ax[5].set_ylabel("Train Recall")
    plt.tight_layout()
    plt.show()    
