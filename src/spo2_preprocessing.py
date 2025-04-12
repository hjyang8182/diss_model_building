import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import math
import tensorflow as tf
# import data as data
from data import ApneaDataLoader
from noisereduce import reduce_noise

data_labels_path = '/home/hy381/rds/hpc-work/segmented_data_new/data_labels.csv'
data_labels = pd.read_csv(data_labels_path)
# os.chdir('/home/hy381/rds/hpc-work/segmented_data_new')

def _float_feature(value): 
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

data_loader = ApneaDataLoader()

class Preprocessor: 
    def apply_delta(self, spo2_np): 
        # spo2_np = spo2_tensor.numpy()
        threshold = 10
        delta_spo2 = []
        # t ranging from 1 ~ 10
        t_vals = np.arange(1, 11)

        for n in range(10, 30): 
            t_samples_after = n + t_vals
            t_samples_before = n - t_vals
            pre_spo2 = spo2_np[t_samples_before]
            post_spo2 = spo2_np[t_samples_after]
            delta_val = np.sum(t_vals * (post_spo2 - pre_spo2))/(2 * np.sum(t_vals ** 2))
            delta_spo2.append(delta_val) 

        return np.array(delta_spo2)
    
    def apply_denoise(self, audio_np):
        denoised_audio = reduce_noise(audio_np, sr = 8000)
        return denoised_audio
    
    def compute_mel_spec(self, audio_np, num_bins = 64, sr = 8000, n_fft = 2048, hop_length = 256): 
        denoised_audio = reduce_noise(audio_np, sr = 8000)
        mel_spec = librosa.feature.melspectrogram(y = denoised_audio, sr = sr, n_fft = n_fft, hop_length = hop_length)
        log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
        return log_mel_spec

    def mel_to_mfcc(self, mel_spectrogram):
        # Convert the Mel Spectrogram to MFCC using TensorFlow
        mfccs = librosa.feature.mfcc(S = mel_spectrogram)  # [batch, time, num_mfcc]
        
        # Keep only the desired number of MFCC coefficients (typically 13 or 20)
        return mfccs
    
    def _data_example(self, audio_denoised, audio_denoised_mel_spec, audio_denoised_mfcc, audio_mel_spec_shape, audio_mfcc_shape, label): 
        feature = {
            'label': _int64_feature(label.tolist()), 
            'audio_denoised': _float_feature(audio_denoised.tolist()), 
            'audio_mel_spec': _float_feature(audio_denoised_mel_spec.flatten().tolist()), 
            'audio_mfcc': _float_feature(audio_denoised_mfcc.flatten().tolist()), 
            'audio_mel_spec_shape': _int64_feature(audio_mel_spec_shape), 
            'audio_mfcc_shape': _int64_feature(audio_mfcc_shape)
        }
        return tf.train.Example(features = tf.train.Features(feature = feature)).SerializeToString()

    def save_data_tfrecord(self, files): 
        dataset = tf.data.TFRecordDataset(files)
        dataset_parsed = data_loader.parse_raw_tf_record_dataset(dataset, ['audio'], dataset_type='train')
        
        save_dir = '/home/hy381/rds/hpc-work/audio_feature_data'
        for i, record in enumerate(dataset_parsed): 
            audio, label, _ = record
            file = files[i]
            segment_fpath = os.path.basename(file)
            subject = os.path.dirname(file)
            subject_dir = os.path.join(save_dir, subject)
            print(save_dir)
            try: 
                os.makedirs(subject_dir)
            except FileExistsError: 
                print(f"Directory {subject_dir} already exists")
            except PermissionError: 
                print(f"Permission Denied to create directory {subject_dir}")
            except Exception as e: 
                print(f"Error: {e} occurred making directory {subject_dir}")

            record_fpath = os.path.join(subject_dir, segment_fpath)
            if os.path.exists(record_fpath):
                print(f"{record_fpath} exists - skipping")
                continue
            
            label = label.numpy().astype(int)
            # Compute delta metric for spo2
        
            # Compute denoising for audio 
            denoised_audio = self.apply_denoise(audio.numpy())
            print(denoised_audio.shape)
            # Compute mel spectrogram, mfcc for denoised audio 
            mel_spec_audio = self.compute_mel_spec(denoised_audio)
            mfcc_audio = self.mel_to_mfcc(mel_spec_audio)

            # keep track of dimensions of mel spec, mfcc audio 
            mel_spec_shape = mel_spec_audio.shape
            mfcc_shape = mfcc_audio.shape

            data_example = self._data_example(denoised_audio, mel_spec_audio, mfcc_audio, mel_spec_shape, mfcc_shape, label)

            with tf.io.TFRecordWriter(record_fpath) as writer:
                writer.write(data_example)
            print(f"Feature TFRecord saved for {record_fpath}")
    
    def save_data_tfrecord(self, files): 
        dataset = tf.data.TFRecordDataset(files)
        dataset_parsed = data_loader.parse_audio_feature_tf_record_dataset(dataset, ['audio_denoised'], dataset_type='train')
        
        save_dir = '/home/hy381/rds/hpc-work/mfcc_ms_data'
        for i, record in enumerate(dataset_parsed): 
            denoised_audio, label, _ = record
            file = files[i]
            segment_fpath = os.path.basename(file)
            subject = os.path.dirname(file)
            subject_dir = os.path.join(save_dir, subject)
            print(save_dir)
            try: 
                os.makedirs(subject_dir)
            except FileExistsError: 
                print(f"Directory {subject_dir} already exists")
            except PermissionError: 
                print(f"Permission Denied to create directory {subject_dir}")
            except Exception as e: 
                print(f"Error: {e} occurred making directory {subject_dir}")

            record_fpath = os.path.join(subject_dir, segment_fpath)

            mel_spec_audio = self.compute_mel_spec(denoised_audio)
            mfcc_audio = self.mel_to_mfcc(mel_spec_audio)
            
            
            label = label.numpy().astype(int)
            # Compute delta metric for spo2
        
            # Compute denoising for audio 
            denoised_audio = self.apply_denoise(audio.numpy())
            print(denoised_audio.shape)
            # Compute mel spectrogram, mfcc for denoised audio 
            mel_spec_audio = self.compute_mel_spec(denoised_audio)
            mfcc_audio = self.mel_to_mfcc(mel_spec_audio)

            # keep track of dimensions of mel spec, mfcc audio 
            mel_spec_shape = mel_spec_audio.shape
            mfcc_shape = mfcc_audio.shape

            data_example = self._data_example(denoised_audio, mel_spec_audio, mfcc_audio, mel_spec_shape, mfcc_shape, label)

            with tf.io.TFRecordWriter(record_fpath) as writer:
                writer.write(data_example)
            print(f"Feature TFRecord saved for {record_fpath}")
train_files, valid_files, test_files = data_loader.split_train_valid_test()
base_dir = '/home/hy381/rds/hpc-work/audio_feature_data/'
train_files = np.char.add(base_dir, train_files)
valid_files = np.char.add(base_dir, valid_files)
test_files = np.char.add(base_dir, test_files)
all_files = set(np.concatenate([train_files, valid_files, test_files]))
all_avail_files = set(glob.glob('/home/hy381/rds/hpc-work/audio_feature_data/*/*.tfrecord'))
diff = np.array(list(all_files - all_avail_files))
print(len(all_files))
print(len(all_avail_files))
print(len(diff))
# '/home/hy381/rds/hpc-work/audio_feature_data/1463/1463_segment_187.tfrecord' in diff
def make_fname(fname): 
    basename = os.path.basename(fname)
    subject = basename.split('_')[0]
    return f"{subject}/{basename}"
diff = np.vectorize(make_fname)(diff)
diff = np.char.add('/home/hy381/rds/hpc-work/segmented_data_new/', diff)
# print(diff)
# processor = Preprocessor()
# processor.save_data_tfrecord(diff)

dataset = tf.data.TFRecordDataset(diff)
dataset_parsed = data_loader.parse_raw_tf_record_dataset(dataset, ['audio'], dataset_type='train')
for data in  dataset_parsed:
    print(data)