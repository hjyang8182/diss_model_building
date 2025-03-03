import tensorflow as tf
import numpy as np
import os
import sys
import librosa
import matplotlib.pyplot as plt 
import pandas as pd
import glob
from scipy.signal import butter,sosfiltfilt
from noisereduce import reduce_noise

script_dir = os.getcwd()
sys.path.append(script_dir)
from data import DataLoader


data_labels_path = '/home/hy381/rds/hpc-work/segmented_data_new/data_labels.csv'
data_labels = pd.read_csv(data_labels_path)

def bandpass_filter(data, low_cutoff, high_cutoff, fs, order):
    nyq = fs * 0.5
    normal_cutoff_low = low_cutoff / nyq
    normal_cutoff_high = high_cutoff / nyq
    sos = butter(order, [normal_cutoff_low,normal_cutoff_high], btype='band', analog=False, output='sos')
    y = sosfiltfilt(sos, data)
    return y

def compute_mel_spec(audio, num_bins = 64, sr = 8000, n_fft = 2048, hop_length = 256): 
    audio_np = audio.numpy()
    denoised_audio = reduce_noise(audio_np, sr = 8000)
    mel_spec = librosa.feature.melspectrogram(y = denoised_audio, sr = sr, n_fft = n_fft, hop_length = hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
    return log_mel_spec

def save_mel_spec(spec_db, filename): 
    spec_db = spec_db[:, 3:-3]
    plt.figure(figsize = (6, 6))
    librosa.display.specshow(spec_db, sr=8000)  # Adjust sample rate if needed
    plt.axis('off')  # Remove axes for a clean image
    plt.tight_layout(pad = 0)
    plt.savefig(filename, bbox_inches = 'tight', pad_inches=0)
    plt.close()

def make_fname(filename): 
    png_filename = filename.replace('.png', '.tfrecord')
    png_filename = "/".join(png_filename.split("/")[-2:])
    return png_filename

os.chdir('/home/hy381/rds/hpc-work/audio_feature_data/')
data_loader = DataLoader()
data_files = data_loader.split_train_valid_test()[2]
dataset = tf.data.TFRecordDataset(data_files)
parsed_dataset = data_loader.parse_audio_feature_tf_record_dataset(dataset, ['audio_mel_spec'], 'train')
for i, record in enumerate(parsed_dataset): 
    data_file = data_files[i]
    subject = os.path.dirname(data_file)
    segment_name = os.path.splitext(os.path.basename(data_file))[0]
    mel_spec, _, _ = record
    mel_spec_file = os.path.join('/home/hy381/rds/hpc-work/segmented_data_new', subject, f"{segment_name}.png")
    save_mel_spec(mel_spec.numpy(), mel_spec_file)
    print(f"Mel spectrogram saved for {data_file} to {mel_spec_file}")
#     data_labels.loc[data_labels['file'] == data_file, 'mel_spec'] = True
# data_labels.to_csv(data_labels_path, index = False)