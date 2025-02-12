import tensorflow as tf
import numpy as np
import os
import sys
import librosa
import matplotlib.pyplot as plt 
import pandas as pd

script_dir = os.getcwd()
sys.path.append(script_dir)
import utils

os.chdir('/home/hy381/rds/hpc-work/segmented_data_new')

data_labels_path = '/home/hy381/rds/hpc-work/segmented_data_new/segmented_data_combined_apnea.csv'
data_labels = pd.read_csv(data_labels_path)

def compute_mel_spec(audio, num_bins, sample_rate, stride, fft_length, min_freq = 70, max_freq = 4000): 
    audio_np = audio.numpy()
    spec = librosa.feature.melspectrogram(y = audio_np, sr = sample_rate, n_mels = num_bins, hop_length = stride, n_fft = fft_length, fmin = min_freq, fmax = max_freq)    
    spec_db = librosa.power_to_db(spec, ref = np.max)
    return spec_db

def save_mel_spec(spec_db, filename): 
    plt.figure(figsize = (6, 6))
    librosa.display.specshow(spec_db, cmap='viridis', sr=22050)  # Adjust sample rate if needed
    plt.axis('off')  # Remove axes for a clean image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

data_files = np.unique(data_labels.loc[data_labels['label'] != 3, 'file'].values)
dataset = tf.data.TFRecordDataset(data_files)
num_mel_bins = 64
sampling_rate = 8000
sequence_stride = 160
fft_length = 400
n_samples = 320000
no_mel_spec_files = data_labels.loc[(data_labels['mel_spec'] == False) & (data_labels['label'] != 3), 'file']
for i, record in enumerate(dataset): 
    data_file = data_files[i]
    subject = os.path.dirname(data_file)
    segment_name = os.path.splitext(os.path.basename(data_file))[0]
    audio, _, _ = utils._parse_data(record, ['audio'])
    spec_db = compute_mel_spec(audio, num_bins=num_mel_bins, sample_rate=sampling_rate, stride = sequence_stride, fft_length = fft_length)
    mel_spec_file = os.path.join('/home/hy381/rds/hpc-work/segmented_data_new', subject, f"{segment_name}.png")
    save_mel_spec(spec_db, mel_spec_file)
    print(f"Mel spectrogram saved for {data_file}")

    data_labels.loc[data_labels['file'] == data_file, 'mel_spec'] = True
    data_labels.to_csv(data_labels_path, index = False)