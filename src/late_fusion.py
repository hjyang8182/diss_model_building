import tensorflow as tf
import train
import numpy as np
import os
import pandas as pd
import sys
from evaluate import Evaluator
from data import ApneaDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix

class ModelFuser(): 
    def __init__(self, model, model_type, fusion_method):
        fusion_methods = {'avg': self.avg_voting, 'weighted': self.weighted_fusion, 'prod': self.product_rule}
        
        self.model = model
        self.method_name = fusion_method
        self.fusion_method = fusion_methods[fusion_method] 

        gt_fname = '/home/hy381/rds/hpc-work/models/ground_truth.txt'
        gt = np.loadtxt(gt_fname)
        self.gt_labels = np.argmax(gt, axis = 1)

        self.fusion_dir = '/home/hy381/model_training/model_results/late_fusion/' + fusion_method
        try: 
            os.mkdir(self.fusion_dir)
            os.mkdir(os.path.join(self.fusion_dir, 'cm'))
            os.mkdir(os.path.join(self.fusion_dir, 'reports'))
            os.mkdir(os.path.join(self.fusion_dir, 'binary_reports'))

        except FileExistsError: 
                print(f"Directory {self.fusion_dir} already exists")
        except PermissionError: 
            print(f"Permission Denied to create directory {self.fusion_dir}")
        except Exception as e: 
            print(f"Error: {e} occurred making directory {self.fusion_dir}")

        if model_type == 'audio': 
            self.other_models = ['spo2_cnn_1d_model', 'spo2_dense_model', 'spo2_gru_model', 'spo2_bilstm_model', 'spo2_lstm_model']
        else: 
            self.other_models = ['audio_ms_cnn', 'audio_mfcc_lstm', 'audio_mfcc_cnn', 'audio_ms_new']
        # gt_fname = '/home/hy381/rds/hpc-work/models/ground_truth.txt'
        # gt = np.loadtxt(gt_fname)
        # gt_labels = np.argmax(gt, axis = 1)

    def avg_voting(self, models): 
        all_predictions = []
        for model_name in models:
            model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
            model_fname = os.path.join(model_dir, f'{model_name}.keras')
            predictions_output_file = os.path.join(model_dir, 'predictions.txt')
            preds = np.loadtxt(predictions_output_file)
            all_predictions.append(preds)
        all_predictions = np.array(all_predictions)
        # print(all_predictions)
        mean_preds = np.mean(all_predictions, axis = 0)
        return mean_preds

    def weighted_fusion(self, models):
        all_predictions = []
        for model_name in models:
            model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
            model_fname = os.path.join(model_dir, f'{model_name}.keras')

            predictions_output_file = os.path.join(model_dir, 'predictions.txt')
            preds = np.loadtxt(predictions_output_file)
            pred_labels = np.argmax(preds, axis = 1)

            gt_file = os.path.join(model_dir, 'ground_truth.txt')
            gts = np.loadtxt(gt_file)
            gt_labels = np.argmax(gts, axis = 1)

            mcm = multilabel_confusion_matrix(gt_labels, pred_labels, labels = [0, 1, 2])
            weights = []
            for class_idx, cm in enumerate(mcm):
                tn, fp, fn, tp = cm.ravel()
                dr = tp/(tn + fp + fn + tp)
                weight = 1 - dr
                weights.append(weight)
            weights = np.array(weights)
            weighted_preds = preds * weights
            all_predictions.append(weighted_preds)
        mean_preds = np.mean(all_predictions, axis = 0)
        return mean_preds

    def naive_bayes(self, models):
        all_predictions = []
        for model_name in models:
            model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
            model_fname = os.path.join(model_dir, f'{model_name}.keras')

            predictions_output_file = os.path.join(model_dir, 'predictions.txt')
            preds = np.loadtxt(predictions_output_file)
            pred_labels = np.argmax(preds, axis = 1)

            gt_file = os.path.join(model_dir, 'ground_truth.txt')
            gts = np.loadtxt(gt_file)
            gt_labels = np.argmax(gts, axis = 1)

            mcm = confusion_matrix(gt_labels, pred_labels, labels = [0, 1, 2])
            weights = []
            
        return mean_preds

    def product_rule(self, models):
        all_predictions = []
        for model_name in models:
            model_dir = f'/home/hy381/rds/hpc-work/models/{model_name}'
            model_fname = os.path.join(model_dir, f'{model_name}.keras')
            predictions_output_file = os.path.join(model_dir, 'predictions.txt')
            preds = np.loadtxt(predictions_output_file)
            all_predictions.append(np.log(preds))
        all_predictions = np.array(all_predictions)
        log_product = np.sum(all_predictions, axis = 0)
        return log_product
    
    def fuse_models(self): 
        for other_model in self.other_models: 
            spo2_model_type = self.model.split('_')[1]
            audio_model_type = other_model.removeprefix('audio_')
            fused_model_name = f'lf_{self.method_name}_{spo2_model_type}+{audio_model_type}'
            model_dir = f'/home/hy381/rds/hpc-work/models/{fused_model_name}'
            try: 
                os.mkdir(model_dir)
            except FileExistsError: 
                print(f"Directory {model_dir} already exists")
            except PermissionError: 
                print(f"Permission Denied to create directory {model_dir}")
            except Exception as e: 
                print(f"Error: {e} occurred making directory {model_dir}")

            fused_preds = self.fusion_method([self.model, other_model])
            # pred_labels = np.argmax(fused_preds, axis = 1)
            pred_fname = os.path.join(model_dir, 'predictions.txt')
            np.savetxt(pred_fname, fused_preds, fmt = '%f')

    def visualize_fusion(self):
        # visualize the model on its own 
        models = []
        models.append(self.model)
        f1_scores = []
        precisions = []
        recalls = []
        accs = []
        model_dir = f'/home/hy381/rds/hpc-work/models/{self.model}'
        
        predictions_output_file = os.path.join(model_dir, 'predictions.txt')
        
        preds = np.loadtxt(predictions_output_file)
        pred_labels = np.argmax(preds, axis = 1)
        
        report = classification_report(self.gt_labels, pred_labels, output_dict = True)
        
        acc = report['accuracy']
        accs.append(acc)

        f1_score = report['weighted avg']['f1-score']
        f1_scores.append(f1_score)

        precision = report['weighted avg']['precision']
        precisions.append(precision)

        recall = report['weighted avg']['recall']
        recalls.append(recall)

        for other_model in self.other_models: 
            fused_preds = self.fusion_method([self.model, other_model])
            pred_labels = np.argmax(fused_preds, axis = 1)

            display_model_name = f'{self.model} \n + \n {other_model}'
            models.append(display_model_name)

            model_name = f'{self.model}+{other_model}'

            cm_file = os.path.join(self.fusion_dir, 'cm', f"{model_name}_cm.png")
            cm = confusion_matrix(self.gt_labels, pred_labels)
            evaluator.save_cm_png(cm, cm_file)

            report_file = os.path.join(self.fusion_dir, 'reports', f"{model_name}_report.txt")
            report = classification_report(self.gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypopnea', 'Apnea'])
            evaluator.save_report(report, report_file)

            bi_report_file = os.path.join(self.fusion_dir, 'binary_reports', f"{model_name}_report.txt")
            binary_report = evaluator.return_biclass_report(pred_labels)
            evaluator.save_report(binary_report, bi_report_file)
            
            report_dict = classification_report(self.gt_labels, pred_labels, labels = [0,1,2], target_names = ['Normal', 'Hypopnea', 'Apnea'], output_dict=True)

            
            acc = report_dict['accuracy']
            accs.append(acc)

            f1_score = report_dict['weighted avg']['f1-score']
            f1_scores.append(f1_score)

            precision = report_dict['weighted avg']['precision']
            precisions.append(precision)

            recall = report_dict['weighted avg']['recall']
            recalls.append(recall)

        model_result_dir = os.path.join(self.fusion_dir, self.model)
        try: 
            os.mkdir(model_result_dir)
        except FileExistsError: 
                print(f"Directory {model_result_dir} already exists")
        except PermissionError: 
            print(f"Permission Denied to create directory {model_result_dir}")
        except Exception as e: 
            print(f"Error: {e} occurred making directory {model_result_dir}")

        plt.figure(figsize = (10, 5))
        plt.grid(True, zorder=0)
        sns.barplot(x = models, y = accs)
        plt.xlabel("Model Combination")
        plt.ylabel("Accuracy")
        plt.xticks(multialignment="center", fontsize = 8, rotation = 45)
        
        plt.savefig(f'{model_result_dir}/{self.model}_acc', bbox_inches = 'tight')
        plt.close()

        plt.figure(figsize = (10, 5))
        plt.grid(True, zorder=0)
        sns.barplot(x = models, y = f1_scores)
        plt.xlabel("Model Combination")
        plt.ylabel("Weighted F1 Score")
        plt.xticks(multialignment="center", fontsize = 8, rotation = 45)
        
        plt.savefig(f'{model_result_dir}/{self.model}_f1', bbox_inches = 'tight')
        plt.close()

        plt.figure(figsize = (10, 5))
        plt.grid(True, zorder=0)
        sns.barplot(x = models, y = precisions)
        plt.xlabel("Model Combination")
        plt.ylabel("Weighted Precision Score")
        plt.xticks(multialignment="center", fontsize = 8, rotation = 45)
        plt.savefig(f'{model_result_dir}/{self.model}_precision', bbox_inches = 'tight')
        plt.close()

        plt.figure(figsize = (10, 5))
        plt.grid(True, zorder=0)
        sns.barplot(x = models, y = recalls)
        plt.xlabel("Model Combination")
        plt.ylabel("Weighted Recall Score")
        plt.xticks(multialignment="center", fontsize = 8, rotation = 45)
        plt.savefig(f'{model_result_dir}/{self.model}_recall', bbox_inches = 'tight')
        plt.close()
     
def main(): 
    pass

if __name__ == '__main__':
    main()
evaluator = Evaluator()

spo2_models = ['spo2_cnn_1d_model', 'spo2_dense_model', 'spo2_gru_model', 'spo2_bilstm_model', 'spo2_lstm_model']

methods = ['avg', 'weighted', 'prod']

# fused_model_names = {}
# for method in methods: 
#     for i in range(len(spo2_models)):
#         spo2_model = spo2_models[i]
#         model_fuser = ModelFuser(spo2_model, 'spo2', method)  
#         model_fuser.visualize_fusion()


