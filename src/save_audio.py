from data import ApneaDataLoader
import tensorflow as tf
import numpy as np
import sys
import librosa
from soundfile import SoundFile
import os
sys.path.insert(0, '/home/hy381/rds/hpc-work/OPERA')

from src.benchmark.model_util import extract_opera_feature

def save_og_audio(): 
    data_loader = ApneaDataLoader()
    train_files, valid_files, test_files = data_loader.split_train_valid_test()
    base_dir = '/home/hy381/rds/hpc-work/segmented_data_new/'
    train_files = np.char.add(base_dir, train_files)
    valid_files = np.char.add(base_dir, valid_files)
    test_files = np.char.add(base_dir, test_files)

    all_files = [train_files, valid_files, test_files]
    dataset_type = ['train', 'valid', 'test']
    for i, files in enumerate(all_files): 
        audio_list = []
        label_list = []
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.interleave(
                    lambda filename: tf.data.TFRecordDataset(filename, buffer_size=1000000),
                    cycle_length=4,  # Number of files read in parallel
                    num_parallel_calls=tf.data.experimental.AUTOTUNE  # Optimize for performance
                )
        dataset = data_loader.parse_raw_tf_record_dataset(dataset, ['audio'], dataset_type='train')
        save_dir = '/home/hy381/rds/hpc-work/audio_wav'
        for j, data in enumerate(dataset):
            audio, label, _ = data
            fname = os.path.basename(files[j])
            fname = os.path.splitext(fname)[0]
            audio_np = audio.numpy()
            label_np = label.numpy()
            label = np.argmax(label_np)
            audio_list.append(audio_np)
            label_list.append(label)
            output_path = os.path.join(save_dir, f"{fname}.wav")
            with SoundFile(output_path, 'w', 8000, 1, format = 'wav') as sf: 
                sf.write(audio_np)
            print(f"Audio saved for {output_path}")
        audio_array = np.array(audio_list)
        label_array = np.array(label_list)
        # fname_array = np.array(fnames_list)
        print(label_array.shape)
        # np.save(f"{save_dir}/{dataset_type[i]}_audio_data.npy", audio_array)
        np.save(f"{save_dir}/{dataset_type[i]}_labels.npy", label_array)
        # np.save(f"{save_dir}/{dataset_type[i]}_files.npy", fname_array)

def save_denoised_audio(): 
    data_loader = ApneaDataLoader()
    train_files, valid_files, test_files = data_loader.split_train_valid_test()
    base_dir = '/home/hy381/rds/hpc-work/audio_feature_data/'
    test_files = np.char.add(base_dir, test_files)
    dataset = tf.data.Dataset.from_tensor_slices(test_files)
    dataset = dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename, buffer_size=1000000),
                cycle_length=4,  # Number of files read in parallel
                num_parallel_calls=tf.data.experimental.AUTOTUNE  # Optimize for performance
            )
    dataset = data_loader.parse_audio_feature_tf_record_dataset(dataset, features = ['audio_denoised'], dataset_type = 'test')
    save_dir = '/home/hy381/rds/hpc-work/denoised_audio_wav'
    for j, data in enumerate(dataset):
        audio, label, _ = data
        fname = os.path.basename(test_files[j])
        fname = os.path.splitext(fname)[0]
        audio_np = audio.numpy()
        output_path = os.path.join(save_dir, f"{fname}.wav")
        with SoundFile(output_path, 'w', 8000, 1, format = 'wav') as sf: 
            sf.write(audio_np)
        print(f"Audio saved for {output_path}")


def save_opera_features(): 
    os.chdir('/home/hy381/rds/hpc-work/audio_wav')
    data_loader = ApneaDataLoader()
    train_files, valid_files, test_files = data_loader.split_train_valid_test()
    train_files, valid_files, test_files = np.vectorize(os.path.basename)(train_files), np.vectorize(os.path.basename)(valid_files), np.vectorize(os.path.basename)(test_files)
    train_files, valid_files, test_files = np.char.replace(train_files, '.tfrecord', '.wav'), np.char.replace(valid_files, '.tfrecord', '.wav'), np.char.replace(test_files, '.tfrecord', '.wav')
    all_files = [train_files, valid_files, test_files]
    data_type = ['train', 'valid', 'test']
    for i in range(3): 
        files = all_files[i]
        print(f"Saving features for {data_type[i]:} {len(files)}")
        feature_dir = '/home/hy381/rds/hpc-work/opera_features/'
        opera_features = extract_opera_feature(files,  pretrain="operaCT", input_sec=40, dim=768)
        np.save(feature_dir + f"{data_type[i]}_operaCT_feature.npy", np.array(opera_features))
#         from src.util import get_split_signal_librosa
#         x_data = []
#         for audio_file in files:
#             print(audio_file)
#             data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=40)[0]
#             # print(data.shape)
#             x_data.append(data)
#         x_data = np.array(x_data)
#         np.save(feature_dir + f"{data_type[i]}_spectrogram.npy", x_data)
# save_opera_features()
# data_loader = ApneaDataLoader()
# train_files, valid_files, test_files = data_loader.split_train_valid_test()

# save_og_audio()
save_opera_features()
