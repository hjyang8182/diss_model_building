import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import pandas as pd
import os
import sys
sys.path.append('/home/hy381/model_training/src')
import seaborn as sns
from data import DataLoader

os.chdir('/home/hy381/rds/hpc-work/segmented_data_new')
data_labels = pd.read_csv('data_labels.csv')
data_loader = DataLoader()
all_files = np.concatenate(list(data_loader.split_train_valid_test()))

from scipy.signal import butter, sosfiltfilt

@tf.function
def mel_spec(audio):
    def mel_spec_wrapper(audio_np): 
        sr = 8000
        n_fft = int(0.25 * 8000)
        hop_length = int(0.125 * 8000)
        audio_np = _butter_bandpass_filter(audio_np, 100, 3400, 8000)
        mel_spec = librosa.feature.melspectrogram(y = audio_np, sr = sr, n_fft = n_fft, hop_length = hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
        spec = tf.convert_to_tensor(log_mel_spec, dtype = tf.float32)
        return spec
    
    spec = tf.numpy_function(mel_spec_wrapper, [audio], tf.float32)
    return spec

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low,high], btype='band', analog=False, output='sos')
    y = sosfiltfilt(sos, data)
    return y

def visualize_spo2_audio_stacked(files): 
    dataset = tf.data.Dataset.from_tensor_slices(files)

    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, buffer_size=1000000),
        cycle_length=4,  # Number of files read in parallel
        num_parallel_calls=tf.data.experimental.AUTOTUNE  # Optimize for performance
    )

    parsed_data = data_loader.parse_raw_tf_record_dataset(dataset, ['spo2', 'audio'], 'train')

    for i, data in enumerate(parsed_data): 
        fname = os.path.basename(files[i])
        spo2, spectrogram, label, _ = data

        label_names = ['Normal', 'Hypopnea', 'Apnea']
        raw_label = np.argmax(label.numpy())
        
        fig, ax = plt.subplots(2, 1, figsize = (8, 6), sharex = True)
        ax = ax.flatten()

        sr = 8000
        n_fft = int(0.25 * sr)
        # 0.125 seconds -> 125 ms, hop_length = 1000 
        long_hop_length = int(0.125 * sr)

        ax[0].plot(spo2)
        ax[0].set_ylabel('SpO2 Signal')
        ax[0].set_ylim(80, 100)
        ax[1].set_ylim(0, 1500)
        fig.suptitle(f"Visualisation for {fname} - {label_names[raw_label]}")
        mel_spec_display = librosa.display.specshow(spectrogram.numpy(), sr=8000, x_axis="time", y_axis="linear", ax = ax[1],  hop_length=long_hop_length)
        plt.savefig(f'/home/hy381/model_training/img/data_visualisation/{fname}.png', bbox_inches='tight')
        plt.close()

def visualize_audio_subject(subject): 
    sr = 8000
    n_fft = int(0.75 * sr)
    # 0.125 seconds -> 125 ms, hop_length = 1000 
    long_hop_length = int(0.125 * sr)

    hypopnea_files = data_labels.loc[(data_labels['label'] == 1) & (data_labels['subject'] == subject) & (data_labels['file'].isin(all_files)), 'file'].values
    apnea_files = data_labels.loc[(data_labels['label'] == 2) & (data_labels['subject'] == subject)  & (data_labels['file'].isin(all_files)), 'file'].values
    normal_files = data_labels.loc[(data_labels['label'] == 0) & (data_labels['subject'] == subject) & (data_labels['file'].isin(all_files)), 'file'].values

    hypopnea_files_random = np.random.choice(hypopnea_files, 1, replace = False)
    apnea_files_random = np.random.choice(apnea_files, 1, replace = False)
    normal_files_random = np.random.choice(normal_files, 1, replace = False)

    visualize_files = np.concatenate([hypopnea_files_random, apnea_files_random, normal_files_random])
    visualize_data = tf.data.TFRecordDataset(visualize_files)
    parsed_data = data_loader.parse_raw_tf_record_dataset(visualize_data, ['audio'], 'train')
    fig, ax = plt.subplots(3, 1, figsize = (8, 8), sharex = True)
    ax = ax.flatten()
    labels = ['Normal', 'Hypopnea', 'Apnea']

    for i, data in (enumerate(parsed_data)):
        spectrogram, label, _ = data
        pred_label = np.argmax(label)
        print(spectrogram.numpy().shape)
        mel_spec_display = librosa.display.specshow(spectrogram.numpy(), sr=8000, x_axis="time", y_axis="linear", ax = ax[i],  hop_length=long_hop_length)
        plt.colorbar(mel_spec_display, ax = ax[i], label='Power (dB)')
        ax[i].set_title(f"{labels[pred_label]} Event")
        ax[i].set_ylim(0, 750)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].grid(alpha = 0.5)

    fig.suptitle(f"Spectrogram Visualisation for {subject}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/home/hy381/model_training/img/data_visualisation/audio/{subject}.png', bbox_inches='tight')

    # plt.subplots_adjust(top=0.93)
    # plt.tight_layout()
    plt.show()
    plt.close()

def visualize_spo2_subject(subject): 
    hypopnea_files = data_labels.loc[(data_labels['label'] == 1) & (data_labels['subject'] == subject) & (data_labels['file'].isin(all_files)), 'file'].values
    apnea_files = data_labels.loc[(data_labels['label'] == 2) & (data_labels['subject'] == subject)  & (data_labels['file'].isin(all_files)), 'file'].values
    normal_files = data_labels.loc[(data_labels['label'] == 0) & (data_labels['subject'] == subject) & (data_labels['file'].isin(all_files)), 'file'].values

    hypopnea_files_random = np.random.choice(hypopnea_files, 1, replace = False)
    apnea_files_random = np.random.choice(apnea_files, 1, replace = False)
    normal_files_random = np.random.choice(normal_files, 1, replace = False)

    visualize_files = np.concatenate([hypopnea_files_random, apnea_files_random, normal_files_random])
    visualize_data = tf.data.TFRecordDataset(visualize_files)
    parsed_data = data_loader.parse_raw_tf_record_dataset(visualize_data, ['spo2'], 'train')
    fig, ax = plt.subplots(3, 1, figsize = (6, 6), sharex = True)
    ax = ax.flatten()
    labels = ['Normal', 'Hypopnea', 'Apnea']
    for i, data in (enumerate(parsed_data)):
        spo2, label, _ = data
        pred_label = np.argmax(label)
        ax[i].set_title(f"{labels[pred_label]} Event")
        ax[i].set_ylim(80, 100)
        ax[i].legend()
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].grid(alpha = 0.5)
        ax[i].plot(spo2)

    fig.suptitle(f"SpO2 Event Visualisation for {subject}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/home/hy381/model_training/img/data_visualisation/audio/{subject}.png', bbox_inches='tight')

    # plt.subplots_adjust(top=0.93)
    # plt.tight_layout()
    plt.show()
    plt.close()


# data_loader = DataLoader()
# non_apnea_files = data_labels[data_labels['label'] == 0]['file'].values.astype(str)
# hypopnea_files = data_labels[data_labels['label'] == 1]['file'].values.astype(str)
# apnea_files = data_labels[data_labels['label'] == 2]['file'].values.astype(str)

# random_non_apnea_files = np.random.choice(non_apnea_files, 10, replace = False)
# random_hypopnea_files = np.random.choice(hypopnea_files, 10, replace = False)
# random_apnea_files = np.random.choice(apnea_files, 10, replace = False)

# visualize_spo2_audio_stacked(random_non_apnea_files)
# visualize_spo2_audio_stacked(random_apnea_files)
# visualize_spo2_audio_stacked(random_hypopnea_files)

subjects = data_labels['subject'].values
random_subjects = np.random.choice(subjects, 30, replace = False)
# for subject in random_subjects: 
#     visualize_spo2_subject(subject)

for subject in random_subjects:
    visualize_audio_subject(subject)
    print(subject)