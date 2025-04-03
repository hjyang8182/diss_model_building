import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import glob

class Evaluator():
    def __init__(self): 
        gt_fname = '/home/hy381/rds/hpc-work/models/ground_truth.txt'
        gt = np.loadtxt(gt_fname)
        self.gt_labels = np.argmax(gt, axis = 1)
        self.test_count = 3484

    def save_cm_png(self, cm, cm_file): 
        labels = ['Normal', 'Hypopnea', 'Apnea']
        plt.figure(figsize=(8,8))
        ax = sns.heatmap(cm , annot = True, fmt = 'd', cmap = 'Blues', xticklabels = labels, yticklabels = labels, square = True)
        ax.set_xlabel('Predicted Label', fontsize = 12, fontweight = 'bold', labelpad = 18)
        ax.set_ylabel('True Label', fontsize = 12, fontweight = 'bold', labelpad = 18)
        ax.set_aspect("equal")
        plt.savefig(cm_file, bbox_inches='tight')
        plt.close()

    def save_report(self, report, report_file): 
        with open(report_file, 'w') as f: 
            f.write(report)

    def evaluate_saved_model(self, model_name, test_data): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        model = load_model(model_fname)
        test_results = model.evaluate(test_data)
        return test_results

    def write_predictions_truths(self, model_name, test_data):
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        model = load_model(model_fname)
        predictions = model.predict(test_data, steps = self.test_count)
        
        ground_truth_labels = []
        for _, label, _ in test_data:
            ground_truth_labels.append(label)
    
        ground_truth_labels = np.array(ground_truth_labels)
        ground_truth_labels = ground_truth_labels.reshape(-1, 3)
    
        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
        np.savetxt(ground_truth_output_file, ground_truth_labels, fmt = '%f')
    
        model_dir = os.path.dirname(model_fname)
        predictions_output_file = os.path.join(model_dir, 'predictions.txt')
        np.savetxt(predictions_output_file, predictions, fmt='%f')

        return predictions

    def find_error_files(model_name):
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        predictions_output_file = os.path.join(model_dir, 'predictions.txt')
        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')

        preds = np.loadtxt(predictions_output_file)
        gt = np.loadtxt(ground_truth_output_file)

        pred_labels = np.argmax(preds, axis = 1)
        gt_labels = np.argmax(gt, axis = 1)
        diff_indices = np.where(pred_labels != gt_labels)[0]
    

    def save_confusion_matrix(self, model_name, model_dir = None, test_data = None): 
        if model_dir is None: 
            model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'

        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        predictions_output_file = os.path.join(model_dir, 'predictions.txt')
        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
    
        if not (os.path.exists(predictions_output_file)): 
            self.write_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        gt = np.loadtxt(ground_truth_output_file)
        
        pred_labels = np.argmax(preds, axis = 1)
        gt_labels = np.argmax(gt, axis = 1)
        cm = confusion_matrix(gt_labels, pred_labels)

        labels = ['Normal', 'Hypopnea', 'Apnea']
        plt.figure(figsize=(8,8))
        ax = sns.heatmap(cm , annot = True, fmt = 'd', cmap = 'Blues', xticklabels = labels, yticklabels = labels, square = True)
        ax.set_xlabel('Predicted Label', fontsize = 12, fontweight = 'bold', labelpad = 18)
        ax.set_ylabel('True Label', fontsize = 12, fontweight = 'bold', labelpad = 18)
        ax.set_aspect("equal")
        plt.savefig(f'/home/hy381/model_training/img/cm/{model_name}_cm.png', bbox_inches='tight')
        print(f"Saved confusion matrix for {model_name}")
        return 

    def write_model_classification_report(self, model_name, test_data = None): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')

        predictions_output_file = os.path.join(model_dir, 'predictions.txt')

        if not (os.path.exists(predictions_output_file)): 
            self.write_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        pred_labels = np.argmax(preds, axis = 1)

        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
        gt = np.loadtxt(ground_truth_output_file)
        gt_labels = np.argmax(gt, axis = 1)
        
        report = classification_report(gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypopnea', 'Apnea'])
        report_file = f'/home/hy381/model_training/model_results/reports/{model_name}.txt'
        with open(report_file, 'w') as f: 
            f.write(report)
        return report

    def write_model_binary_report(self, model_name, test_data = None): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')

        predictions_output_file = os.path.join(model_dir, 'predictions.txt')

        if not (os.path.exists(predictions_output_file)): 
            self.write_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        pred_labels = np.argmax(preds, axis = 1)

        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
        gt = np.loadtxt(ground_truth_output_file)
        gt_labels = np.argmax(gt, axis = 1)
        
        pred_labels_binary = np.where(pred_labels == 0, 0, 1)
        gt_labels_binary = np.where(self.gt_labels == 0, 0, 1)           

        report = classification_report(gt_labels_binary, pred_labels_binary, labels = [0,1], target_names = ['Normal', 'Abnormal'])
        report_file = f'/home/hy381/model_training/model_results/individual_models/binary_reports/{model_name}.txt'
        with open(report_file, 'w') as f: 
            f.write(report)
        return report
    
    def plot_model_history(self, model_name): 
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
        
        # ax.set_xticklabels(ax.get_xticks(), fontsize=10)
        plt.tight_layout()
        plt.savefig(f'/home/hy381/model_training/img/history/{model_name}_history.png')    

    def save_biclass_report(self, model_name, test_data = None): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'

        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        predictions_output_file = os.path.join(model_dir, 'predictions.txt')
        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
    
        if not (os.path.exists(predictions_output_file)): 
            self.write_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        gt = np.loadtxt(ground_truth_output_file)
        
        pred_labels = np.argmax(preds, axis = 1)
        gt_labels = np.argmax(gt, axis = 1)

        pred_labels_binary = np.where(pred_labels == 0, 0, 1)
        gt_labels_binary = np.where(gt_labels == 0, 0, 1)           

        report = classification_report(gt_labels_binary, pred_labels_binary, labels = [0,1], target_names = ['Normal', 'Abnormal'])
        report_file = f'/home/hy381/model_training/model_results/individual_models/binary_reports/{model_name}.txt'
        with open(report_file, 'w') as f: 
            f.write(report)
        return report
    
    def find_best_fused_model(self, models, metric): 
        eval_metrics = []
        for model in models: 
            preds = np.loadtxt(f'/home/hy381/rds/hpc-work/models/{model}/predictions.txt')
            pred_labels = np.argmax(preds, axis = 1)
            report = classification_report(self.gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypoponea', 'Apnea'], output_dict=True)
            if metric != 'accuracy': 
                eval_metrics.append(report['weighted avg'][metric])
            else: 
                eval_metrics.append(report[metric])
        best_indexes = np.argsort(eval_metrics)[::-1]
        print(eval_metrics)
        print(best_indexes)
        models_np = np.array(models)
        return models_np[best_indexes]
    

evaluator = Evaluator()
fused_models = glob.glob('/home/hy381/rds/hpc-work/models/lf_*')
fused_model_names = [os.path.basename(f) for f in fused_models]
best_accs = evaluator.find_best_fused_model(fused_model_names, 'accuracy')
print(best_accs)