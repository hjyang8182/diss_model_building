from data import ApneaDataLoader
import tensorflow as tf
import numpy as np
import librosa
from soundfile import SoundFile
import os




# dataset = tf.data.Dataset.from_tensor_slices(train_files[:10])
# dataset = dataset.interleave(
#             lambda filename: tf.data.TFRecordDataset(filename, buffer_size=1000000),
#             cycle_length=4,  # Number of files read in parallel
#             num_parallel_calls=tf.data.experimental.AUTOTUNE  # Optimize for performance
#         )
# dataset = data_loader.parse_raw_tf_record_dataset(dataset, ['audio'], dataset_type='train')
# save_dir = '/home/hy381/rds/hpc-work/audio/'
# audio_list = []
# label_list = []
# fnames_list = []
# for i, data in enumerate(dataset):
#     audio, label, _ = data
#     fname = valid_files[i]
#     audio_np = audio.numpy()
#     label_np = label.numpy()
#     audio_list.append(audio_np)
#     label_list.append(label_np)
#     fnames_list.append(fname)
#     print(f"Audio Files saved fr {fname}")
# audio_array = np.array(audio_list)
# label_array = np.array(label_list)
# fname_array = np.array(fnames_list)
# assert(audio_array.shape[0] == len(valid_files))

# # Save the NumPy arrays
# np.save(f"{save_dir}/valid_audio_data.npy", audio_array)
# np.save(f"{save_dir}/valid_labels.npy", label_array)
# np.save(f"{save_dir}/valid_files.npy", fname_array)

def save_og_audio(): 
    data_loader = ApneaDataLoader()
    train_files, valid_files, test_files = data_loader.split_train_valid_test_subset(train_length = 8000, valid_length = 1000, test_length = 1000)
    base_dir = '/home/hy381/rds/hpc-work/segmented_data_new/'
    base_dir = '/home/hy381/rds/hpc-work/audio_feature_data/'
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
        save_dir = '/home/hy381/rds/hpc-work/opera_features'
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
        np.save(f"{save_dir}/{dataset_type[i]}__labels.npy", label_array)
        # np.save(f"{save_dir}/{dataset_type[i]}__files.npy", fname_array)

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

save_denoised_audio()