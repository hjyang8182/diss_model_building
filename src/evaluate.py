import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
import seaborn as sns
from data import ApneaDataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from tensorflow.keras.models import load_model
import glob
import json
from scipy.stats import ttest_ind

def load_predictions(model_name): 
    predictions_output_file = f'/home/hy381/rds/hpc-work/models/{model_name}/predictions.txt'
    preds = np.loadtxt(predictions_output_file) 
    pred_labels = np.argmax(preds, axis = 1)
    return pred_labels

def load_gts(): 
    gt_fname = '/home/hy381/rds/hpc-work/models/ground_truth.txt'
    gt = np.loadtxt(gt_fname)
    gt_labels = np.argmax(gt, axis = 1)
    return gt_labels

class Evaluator():
    def __init__(self): 
        gt_fname = '/home/hy381/rds/hpc-work/models/ground_truth.txt'
        gt = np.loadtxt(gt_fname)
        self.gt_labels = np.argmax(gt, axis = 1)
        self.test_count = 3484
        
    def evaluate_saved_model(self, model_name, test_data): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        model = load_model(model_fname)
        test_results = model.evaluate(test_data)
        return test_results


    def write_test_predictions_truths(self, model_name, test_data):
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

    def save_confusion_matrix(self, model_name, model_dir = None, test_data = None): 
        if model_dir is None: 
            model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'

        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        predictions_output_file = os.path.join(model_dir, 'predictions.txt')
        ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')
    
        if not (os.path.exists(predictions_output_file)): 
            self.write_test_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        
        pred_labels = np.argmax(preds, axis = 1)
        cm = confusion_matrix(self.gt_labels, pred_labels)

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
            self.write_test_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        pred_labels = np.argmax(preds, axis = 1)

        report = classification_report(self.gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypopnea', 'Apnea'], digits = 4)
        report_file = f'/home/hy381/model_training/model_results/individual_models/reports/{model_name}.txt'
        with open(report_file, 'w') as f: 
            f.write(report)
        return report

    def write_model_binary_report(self, model_name, test_data = None): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')

        predictions_output_file = os.path.join(model_dir, 'predictions.txt')

        if not (os.path.exists(predictions_output_file)): 
            self.write_test_predictions_truths(model_name, test_data)

        preds = np.loadtxt(predictions_output_file)
        pred_labels = np.argmax(preds, axis = 1)

        pred_labels_binary = np.where(pred_labels == 0, 0, 1)
        gt_labels_binary = np.where(self.gt_labels == 0, 0, 1)           

        report = classification_report(gt_labels_binary, pred_labels_binary, labels = [0,1], target_names = ['Normal', 'Abnormal'], digits = 4)
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

    
    def find_best_model(self, models, metric): 
        eval_metrics = []
        for model in models: 
            preds = np.loadtxt(f'/home/hy381/rds/hpc-work/models/{model}/predictions.txt')
            pred_labels = np.argmax(preds, axis = 1)
            report = classification_report(self.gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypoponea', 'Apnea'], output_dict=True, digits = 4)
            if metric != 'accuracy': 
                eval_metrics.append(report['macro avg'][metric])
            else: 
                eval_metrics.append(report[metric])
        best_indexes = np.argsort(eval_metrics)[::-1]
        models_np = np.array(models)
        return models_np[best_indexes]
    
    def find_best_fused_model(self, metric): 
        spo2_models =['spo2_cnn_1d_model', 'spo2_dense_model', 'spo2_gru_model', 'spo2_bilstm_model', 'spo2_lstm_model']
        audio_models = ['audio_mfcc_lstm', 'audio_mfcc_cnn', 'audio_ms_cnn', 'audio_mfcc_bilstm', 'audio_mfcc_cnn_lstm']
        methods = ['avg', 'weighted', 'prod']
        
        fused_models = []
        for method in methods: 
            for spo2_model in spo2_models: 
                for audio_model in audio_models: 
                    spo2_model_type = spo2_model.split('_')[1]
                    audio_model_type = audio_model.removeprefix('audio_')
                    fused_model_name = f'lf_{method}_{spo2_model_type}+{audio_model_type}'
                    fused_models.append(fused_model_name)

        best_model = self.find_best_model(fused_models, metric)
        return best_model

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
    
    def return_classification_report(self, pred_labels): 
        report = classification_report(self.gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypopnea', 'Apnea'], digits = 4)
        return report

    def return_biclass_report(self, pred_labels): 
        pred_labels_binary = np.where(pred_labels == 0, 0, 1)
        gt_labels_binary = np.where(self.gt_labels == 0, 0, 1)          
        report = classification_report(gt_labels_binary, pred_labels_binary, labels = [0,1], target_names = ['Normal', 'Abnormal'], digits = 4)
        return report

    def write_train_cm(self, model_name, train_data, steps): 
        model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
        model_fname = os.path.join(model_dir, f'{model_name}.keras')
        model = load_model(model_fname)
        
        predictions = model.predict(train_data, steps = steps)
        pred_labels = np.argmax(predictions, axis = 1)

        predictions_output_file = f'/home/hy381/model_training/model_results/individual_models/{model_name}.txt'
        preds = np.loadtxt(predictions_output_file)
        pred_labels = np.argmax(preds, axis = 1)

        gt_labels = np.load(f"/home/hy381/model_training/model_results/individual_models/train_labels.npy")
        print(len(gt_labels))
        report = classification_report(gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypopnea', 'Apnea'], output_dict=True, digits = 4)
        with open(f'/home/hy381/model_training/model_results/individual_models/cm_json/{model_name}.json', 'w') as json_file:
            json.dump(report, json_file)
    
    def bootstrap_performance(self, model_name, n_bootstraps = 1000,): 
        pred_labels = load_predictions(model_name)
        bootstrap_accs = []
        bootstrap_f1s = []
        rng = np.random.default_rng(seed = 42)

        for i in range(n_bootstraps): 
            indices = rng.choice(len(self.gt_labels), len(self.gt_labels), replace = True)
            acc = accuracy_score(self.gt_labels[indices], pred_labels[indices])
            f1 = f1_score(self.gt_labels[indices], pred_labels[indices], average = 'macro')
            bootstrap_accs.append(acc)
            bootstrap_f1s.append(f1)
        bootstrap_accs = np.array(bootstrap_accs)
        bootstrap_f1s = np.array(bootstrap_f1s)

        np.save(f'/home/hy381/rds/hpc-work/models/{model_name}/bootstrap_accs.npy', bootstrap_accs)
        np.save(f'/home/hy381/rds/hpc-work/models/{model_name}/bootstrap_f1s.npy', bootstrap_f1s)
        return bootstrap_accs, bootstrap_f1s

    def find_perf_significance(self, model1, model2): 
        bootstrap_acc1 = np.load(f'/home/hy381/rds/hpc-work/models/{model1}/bootstrap_accs.npy')
        bootstrap_acc2 = np.load(f'/home/hy381/rds/hpc-work/models/{model2}/bootstrap_accs.npy')
        print(np.mean(bootstrap_acc1))
        print(np.std(bootstrap_acc1))

        _, p_val_acc = ttest_ind(bootstrap_acc1, bootstrap_acc2)

        bootstrap_f1_1 = np.load(f'/home/hy381/rds/hpc-work/models/{model1}/bootstrap_f1s.npy')
        bootstrap_f1_2 = np.load(f'/home/hy381/rds/hpc-work/models/{model2}/bootstrap_f1s.npy')
        print(np.mean(bootstrap_f1_1))
        print(np.std(bootstrap_f1_1))
        _, p_val_f1 = ttest_ind(bootstrap_f1_1, bootstrap_f1_2)


        return p_val_acc, p_val_f1 
    
def find_error_files(model_name):
    model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
    predictions_output_file = os.path.join(model_dir, 'predictions.txt')
    ground_truth_output_file =  os.path.join(model_dir, 'ground_truth.txt')

    preds = np.loadtxt(predictions_output_file)
    gt = np.loadtxt(ground_truth_output_file)

    pred_labels = np.argmax(preds, axis = 1)
    gt_labels = np.argmax(gt, axis = 1)
    diff_indices = np.where(pred_labels != gt_labels)[0]
    
def main(): 
    data_loader = ApneaDataLoader()
    evaluator = Evaluator()
    _, _, test_data = data_loader.load_full_data(['spo2'], parse_type = 'default', apply_delta = True, window_spo2= True)

    evaluator.write_model_binary_report('if_spo2_ms')
    evaluator.write_model_classification_report('if_spo2_ms')
    evaluator.save_confusion_matrix('if_spo2_ms')
   
    # base_dir_default = '/home/hy381/rds/hpc-work/segmented_data_new/'
    # base_dir_audio = '/home/hy381/rds/hpc-work/audio_feature_data/'
    # train_files = data_loader.split_train_valid_test[0]
    # default_train_files = np.char.add(base_dir_default, train_files)
    # audio_train_files = np.char.add(base_dir_audio, train_files)

    # spo2_train_data = data_loader.load_tfrecord_dataset(default_train_files, ['spo2'], parse_type = 'default', dataset_type= 'train', apply_delta = True)
    # spo2_windowed_data = data_loader.load_tfrecord_dataset(default_train_files, ['spo2'], parse_type = 'default', dataset_type= 'train', apply_delta = True, window_spo2=True)
    # audio_mfcc = data_loader.load_tfrecord_dataset(audio_train_files, ['audio_mfcc'], parse_type = 'audio_feature', dataset_type= 'train', apply_delta = True, window_spo2=True)
    # audio_mel_spec = data_loader.load_tfrecord_dataset(audio_train_files, ['audio_mel_spec'], parse_type = 'audio_feature', dataset_type= 'train', apply_delta = True, window_spo2=True)

    # evaluator.write_train_cm('spo2_cnn_1d_model', spo2_train_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('spo2_lstm_model', spo2_windowed_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('spo2_bilstm_model', spo2_windowed_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('spo2_gru_model', spo2_windowed_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('spo2_dense_model', spo2_train_data, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)

    # audio_mel_spec, _, test_data = data_loader.load_full_data(['audio_mel_spec'], parse_type = 'audio_feature')
    # audio_mfcc, _, test_data = data_loader.load_full_data(['audio_mfcc'], parse_type = 'audio_feature')
    # evaluator.write_train_cm('audio_ms_new', audio_mel_spec, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('audio_mfcc_cnn', audio_mfcc, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('audio_mfcc_lstm', audio_mfcc, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)
    # evaluator.write_train_cm('audio_mfcc_lstm', audio_mfcc, data_loader.FULL_TRAIN_STEPS_PER_EPOCH)

    
if __name__ == '__main__':
    main()