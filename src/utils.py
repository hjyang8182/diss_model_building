import tensorflow as tf
import numpy as np
from scipy.fft import fft
from scipy.signal import savgol_filter
from sklearn.utils.class_weight import compute_class_weight
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from noisereduce import reduce_noise

data_labels_path = '/home/hy381/rds/hpc-work/segmented_data_new/data_labels.csv'
data_labels = pd.read_csv(data_labels_path)

def load_mel_spec_images(train_batch_size = 128, valid_test_batch_size = 32):
    train_files, valid_files, test_files = split_train_valid_test_subset() 
    
    img_dir = '/home/hy381/rds/hpc-work/segmented_data_new/'
    train_files_mel_spec = np.char.replace(train_files, '.tfrecord', '.png')
    train_files_mel_spec = np.char.add(img_dir, train_files_mel_spec)
    train_files_mel_spec = tf.convert_to_tensor(train_files_mel_spec, dtype=tf.string)
    
    valid_files_mel_spec = np.char.replace(valid_files, '.tfrecord', '.png')
    valid_files_mel_spec = np.char.add(img_dir, valid_files_mel_spec)
    valid_files_mel_spec = tf.convert_to_tensor(valid_files_mel_spec, dtype=tf.string)
    
    test_files_mel_spec = np.char.replace(test_files, '.tfrecord', '.png')
    test_files_mel_spec = np.char.add(img_dir, test_files_mel_spec)
    test_files_mel_spec = tf.convert_to_tensor(test_files_mel_spec, dtype=tf.string)

    train_labels = data_labels.loc[data_labels['file'].isin(train_files), 'label']
    train_labels = tf.convert_to_tensor(train_labels.values, dtype = tf.int32)
    train_labels = tf.one_hot(train_labels, depth=3, dtype=tf.int32)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files_mel_spec, train_labels))
    train_dataset = train_dataset.map(load_image).batch(train_batch_size, drop_remainder = True).repeat()

    valid_labels = data_labels.loc[data_labels['file'].isin(valid_files), 'label']
    valid_labels = tf.convert_to_tensor(valid_labels.values, dtype = tf.int32)
    valid_labels = tf.one_hot(valid_labels, depth=3, dtype=tf.int32)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_files_mel_spec, valid_labels))
    valid_dataset = valid_dataset.map(load_image).batch(valid_test_batch_size, drop_remainder = True).repeat()

    test_labels = data_labels.loc[data_labels['file'].isin(test_files), 'label']
    test_labels = tf.convert_to_tensor(test_labels.values, dtype = tf.int32)
    test_labels = tf.one_hot(test_labels, depth=3, dtype=tf.int32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files_mel_spec, test_labels))
    test_dataset = test_dataset.map(load_image).batch(valid_test_batch_size, drop_remainder = True)

    return train_dataset, valid_dataset, test_dataset

def load_image(image_path, label):
    img = tf.io.read_file(image_path)  # Read file
    img = tf.image.decode_png(img, channels=3)  # Decode PNG
    img = tf.image.resize(img, [224, 224])  # Resize
    img = img / 255.0  # Normalize
    return img, label

def apply_delta(spo2_tensor): 
    # spo2_np = spo2_tensor.numpy()
    threshold = 10
    delta_spo2 = []
    # t ranging from 1 ~ 10
    t_vals = np.arange(1, 11)

    for n in range(10, 30): 
        t_samples_after = n + t_vals
        t_samples_before = n - t_vals
        pre_spo2 = spo2_tensor[t_samples_before]
        post_spo2 = spo2_tensor[t_samples_after]
        delta_val = np.sum(t_vals * (post_spo2 - pre_spo2))/(2 * np.sum(t_vals ** 2))
        delta_spo2.append(delta_val) 

    return tf.convert_to_tensor(delta_spo2, dtype=tf.float32)

def apply_norm(spo2): 
    norm_spo2 = spo2/100
    return tf.convert_to_tensor(norm_spo2, dtype = tf.float32)

def apply_standardization(spo2): 
    standard_spo2 = (spo2 - np.mean(spo2))/np.std(spo2)
    return tf.convert_to_tensor(standard_spo2, dtype = tf.float32)

def apply_denoise(audio_np):
    denoised_audio = reduce_noise(audio_np, sr = 8000)
    return tf.convert_to_tensor(denoised_audio, dtype = tf.float32)

def _parse_data(proto, features, dataset_type, use_delta = False, normalize = False):
    audio_sample_rate = 8000
    keys_to_features = {
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'audio': tf.io.FixedLenFeature([audio_sample_rate * 40], tf.float32), 
        'spo2': tf.io.FixedLenFeature([40], tf.float32) 
    }

    try: 
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        
        label = parsed_features['label']

        class_weights = return_class_weights()[dataset_type]
        
        label = tf.squeeze(label)
        label = tf.where(label == tf.constant(3, dtype=tf.int64), tf.constant(2, dtype=tf.int64), label)
        # label = tf.where(label == tf.constant(4, dtype=tf.int64), tf.constant(3, dtype=tf.int64), label)

        weight = tf.gather(class_weights, label)
        label_enc = tf.one_hot(label, 3)
        label_enc.set_shape([3])
        if use_delta: 
            spo2 = parsed_features['spo2']
            delta_spo2 = tf.numpy_function(apply_delta, [spo2], tf.float32)
            # delta_spo2.set_shape([20])
            parsed_features['spo2'] = delta_spo2
            parsed_features['spo2'].set_shape([20])
        if normalize: 
            spo2 = parsed_features['spo2']
            norm_spo2 = tf.numpy_function(apply_standardization, [spo2], tf.float32)
            # delta_spo2.set_shape([20])
            parsed_features['spo2'] = norm_spo2
            parsed_features['spo2'].set_shape([40])

        audio = parsed_features['audio']
        denoised_audio = tf.numpy_function(apply_denoise, [audio], tf.float32)
        parsed_features['audio'] = denoised_audio
        parsed_features['audio'].set_shape([320000])

        return_vals = [parsed_features[feature] for feature in features]
        return_vals.append(label_enc)
        return_vals.append(weight)
        return tuple(return_vals)
    except tf.errors.InvalidArgumentError:
        return None
    except Exception as e:
        print(e) 
        return None
    

def _parse_data_biclass(proto, features, use_delta = False, normalize = False):
    audio_sample_rate = 8000
    keys_to_features = {
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'audio': tf.io.FixedLenFeature([audio_sample_rate * 40], tf.float32), 
        'spo2': tf.io.FixedLenFeature([40], tf.float32) 
    }

    try: 
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        
        label = parsed_features['label']

        class_weights = return_biclass_weights()
        
        label = tf.squeeze(label)
        # Change all hypopnea/central apnea/obstructive apnea labels to 'apnea'
        label = tf.where(label == tf.constant(2, dtype=tf.int64), tf.constant(1, dtype=tf.int64), label)
        label = tf.where(label == tf.constant(3, dtype=tf.int64), tf.constant(1, dtype=tf.int64), label)
        
        # label = tf.where(label == tf.constant(4, dtype=tf.int64), tf.constant(3, dtype=tf.int64), label)

        weight = tf.gather(class_weights, label)
        label_enc = tf.one_hot(label, 2)
        label_enc.set_shape([2])
        if use_delta: 
            spo2 = parsed_features['spo2']
            delta_spo2 = tf.numpy_function(apply_delta, [spo2], tf.float32)
            # delta_spo2.set_shape([20])
            parsed_features['spo2'] = delta_spo2
            parsed_features['spo2'].set_shape([20])
        if normalize: 
            spo2 = parsed_features['spo2']
            norm_spo2 = tf.numpy_function(apply_standardization, [spo2], tf.float32)
            # delta_spo2.set_shape([20])
            parsed_features['spo2'] = norm_spo2
            parsed_features['spo2'].set_shape([40])

        return_vals = [parsed_features[feature] for feature in features]
        return_vals.append(label_enc)
        return_vals.append(weight)
        return tuple(return_vals)
    except tf.errors.InvalidArgumentError:
        return None
    except Exception as e:
        print(e) 
        return None

def parse_biclass_tf_record_dataset(dataset, features): 
    parsed_data = dataset.map(lambda x : _parse_data_biclass(x, features=features, use_delta=True))
    return parsed_data

def parse_tf_record_dataset(dataset, features, dataset_type, use_delta = False, normalize = False):
    parsed_data = dataset.map(lambda x : _parse_data(x, features=features, dataset_type= dataset_type, use_delta=use_delta, normalize=normalize))
    # parsed_data = parsed_data.filter(lambda *x: x[0] is not None)
    return parsed_data

def return_class_weights(): 
    train_files, valid_files, test_files = split_train_valid_test_subject()

    train_labels = data_labels.loc[(data_labels['file'].isin(train_files)), 'label'].values
    valid_labels = data_labels.loc[(data_labels['file'].isin(valid_files)), 'label'].values
    test_labels = data_labels.loc[(data_labels['file'].isin(test_files)), 'label'].values

    classes = np.array([0, 1, 2])

    train_class_weights  = compute_class_weight(class_weight = 'balanced', classes = classes, y= train_labels)
    valid_class_weights  = compute_class_weight(class_weight = 'balanced', classes = classes, y= valid_labels)
    test_class_weights  = compute_class_weight(class_weight = 'balanced', classes = classes, y= test_labels)
    
    train_class_weights_dict = dict(zip(classes, train_class_weights))
    valid_class_weights_dict = dict(zip(classes, valid_class_weights))
    test_class_weights_dict = dict(zip(classes, test_class_weights))

    train_weights = tf.convert_to_tensor(np.array([train_class_weights_dict[label] for label in classes]), tf.float32)
    valid_weights = tf.convert_to_tensor(np.array([valid_class_weights_dict[label] for label in classes]), tf.float32)
    test_weights = tf.convert_to_tensor(np.array([test_class_weights_dict[label] for label in classes]), tf.float32)

    class_weights = {'train': train_weights, 'valid': valid_weights, 'test': test_weights}
    return class_weights


def return_biclass_weights(): 
    non_apnea_files = data_labels[data_labels['label'] == 0]['file'].values.astype(str)
    
    hypopnea_files = data_labels[data_labels['label'] == 1]['file'].values.astype(str)
    
    apnea_files = data_labels[data_labels['label'] == 2]['file'].values.astype(str)
    
    num_non_apnea = len(non_apnea_files)
    num_apnea = len(apnea_files) + len(hypopnea_files)

    classes = np.array([0, 1])
    samples_per_class = [num_non_apnea, num_apnea]

    class_weights = compute_class_weight(class_weight = 'balanced', classes = classes, y=np.repeat(classes, samples_per_class))
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    sample_weights = np.array([class_weight_dict[label] for label in classes])
    sample_weights = tf.convert_to_tensor(sample_weights, tf.float32)
    sample_weights.set_shape([2])
    return sample_weights

def split_train_valid_test(p_train = 0.7, p_valid = 0.15):
    # find filenames that match the labels 
    non_apnea_files = data_labels[data_labels['label'] == 0]['file'].values.astype(str)
    
    hypopnea_files = data_labels[data_labels['label'] == 1]['file'].values.astype(str)
    
    apnea_files = data_labels[data_labels['label'] == 2]['file'].values.astype(str)
    
    # mixed_apnea_files = data_labels[data_labels['label'] == 3]['file'].values.astype(str)

    # find length of filenames 
    num_non_apnea = len(non_apnea_files)
    num_hypopnea = len(hypopnea_files)
    num_apnea = len(apnea_files)

    # num_mixed = len(mixed_apnea_files)
    
    # find length of train, valid, test for each label
    num_non_apnea_train = int(p_train * num_non_apnea)
    num_hypopnea_train = int(p_train * num_hypopnea)
    num_apnea_train = int(p_train * num_apnea)
    # num_mixed_train = int(p_train * num_mixed)
    
    num_non_apnea_valid = int(p_valid * num_non_apnea)
    num_hypopnea_valid = int(p_valid * num_hypopnea)
    num_apnea_valid = int(p_valid * num_apnea)
    # num_mixed_valid = int(p_valid * num_mixed)
        
    # shuffle the filenames randomly
    np.random.seed(42)
    non_apnea_files = np.random.permutation(non_apnea_files)
    
    np.random.seed(42)
    hypopnea_files = np.random.permutation(hypopnea_files)
    
    np.random.seed(42)
    apnea_files = np.random.permutation(apnea_files)
    # mixed_apnea_files = np.random.permutation(mixed_apnea_files)
    
    # find training test filenames
    training_files = np.concatenate([
                        non_apnea_files[:num_non_apnea_train], 
                        hypopnea_files[:num_hypopnea_train],
                        apnea_files[:num_apnea_train], 
                        # mixed_apnea_files[:num_mixed_train]
                     ])
    
    valid_files = np.concatenate([
                    non_apnea_files[num_non_apnea_train : num_non_apnea_train + num_non_apnea_valid], 
                    hypopnea_files[num_hypopnea_train : num_hypopnea_train + num_hypopnea_valid],
                    apnea_files[num_apnea_train : num_apnea_train + num_apnea_valid], 
                    # mixed_apnea_files[num_mixed_train : num_mixed_train + num_mixed_valid]
                  ])

    test_files = np.concatenate([
                    non_apnea_files[num_non_apnea_train + num_non_apnea_valid:], 
                    hypopnea_files[num_hypopnea_train + num_hypopnea_valid:],
                    apnea_files[num_apnea_train + num_apnea_valid:], 
                    # mixed_apnea_files[num_mixed_train + num_mixed_valid:]
                 ])
    
    np.random.seed(42)
    training_files, valid_files, test_files = np.random.permutation(training_files), np.random.permutation(valid_files), np.random.permutation(test_files)
    return training_files, valid_files, test_files


def split_train_valid_test_subject(p_train = 0.7, p_valid = 0.15):
    # find filenames that match the labels 
    all_subjects = np.unique(data_labels.loc[data_labels['label'] != 3, 'subject'].values)
    
    p_test = 1 - (p_train + p_valid)
    train_subjects, test_valid_subjects = train_test_split(all_subjects, test_size = (p_valid + p_test), random_state=42)
    valid_subjects, test_subjects = train_test_split(test_valid_subjects, test_size = p_test/(p_valid + p_test), random_state=42)

    np.random.seed(42)
    train_subjects = np.random.choice(train_subjects, len(train_subjects) // 2)

    np.random.seed(42)
    valid_subjects = np.random.choice(valid_subjects, len(valid_subjects) // 2)

    np.random.seed(42)
    test_subjects = np.random.choice(train_subjects, len(test_subjects) // 2)

    train_files = data_labels.loc[(data_labels['subject'].isin(train_subjects)) & (data_labels['label'] != 3), 'file'].values.astype('str')
    valid_files = data_labels.loc[(data_labels['subject'].isin(valid_subjects)) & (data_labels['label'] != 3), 'file'].values.astype('str')
    test_files = data_labels.loc[(data_labels['subject'].isin(test_subjects)) & (data_labels['label'] != 3), 'file'].values.astype('str')

    return train_files, valid_files, test_files

def split_train_valid_test_subset(train_length = 600, valid_length = 60, test_length = 60):
    # find filenames that match the labels 
    non_apnea_files = data_labels[data_labels['label'] == 0]['file'].values.astype(str)
    
    hypopnea_files = data_labels[data_labels['label'] == 1]['file'].values.astype(str)
    
    apnea_files = data_labels[data_labels['label'] == 2]['file'].values.astype(str)
    
    # mixed_apnea_files = data_labels[data_labels['label'] == 3]['file'].values.astype(str)

    # find length of filenames 
    # shuffle the filenames randomly
    np.random.seed(42)
    non_apnea_files = np.random.permutation(non_apnea_files)

    np.random.seed(42)
    hypopnea_files = np.random.permutation(hypopnea_files)

    np.random.seed(42)
    apnea_files = np.random.permutation(apnea_files)
    # mixed_apnea_files = np.random.permutation(mixed_apnea_files)
    
    train_split = train_length//4
    valid_split = valid_length//4
    test_split =  test_length //4 
    # find training test filenames
    training_files = np.concatenate([
                        non_apnea_files[:train_split], 
                        hypopnea_files[:train_split],
                        apnea_files[:train_split], 
                        # mixed_apnea_files[:train_split]
                     ])
    
    valid_files = np.concatenate([
                    non_apnea_files[train_split : train_split + valid_split], 
                    hypopnea_files[train_split : train_split + valid_split],
                    apnea_files[train_split : train_split + valid_split], 
                    # mixed_apnea_files[train_split : train_split + valid_split]
                  ])

    test_files = np.concatenate([
                    non_apnea_files[train_split + valid_split:], 
                    hypopnea_files[train_split + valid_split:],
                    apnea_files[train_split + valid_split:], 
                    # mixed_apnea_files[train_split + valid_split: ]
                 ])
    np.random.seed(42)
    training_files, valid_files, test_files = np.random.permutation(training_files), np.random.permutation(valid_files), np.random.permutation(test_files)
    return training_files, valid_files, test_files

def prepare_data_subset(feature, train_length = 600, valid_length = 60, test_length = 60, train_batch_size = 32, valid_test_batch_size = 16, use_delta = False, normalize = False): 
    train_files, valid_files, test_files = split_train_valid_test_subject() 

    train_files_subset, valid_files_subset, test_files_subset = np.random.choice(train_files, train_length, replace = False), np.random.choice(valid_files, valid_length, replace = False), np.random.choice(test_files, test_length, replace = False)
    train_data_tf = tf.data.TFRecordDataset(train_files_subset)
    train_data = parse_tf_record_dataset(train_data_tf, [feature],  dataset_type= 'train', use_delta= use_delta, normalize=normalize)
    train_data_batched = train_data.batch(train_batch_size, drop_remainder=True).repeat()

    valid_data_tf = tf.data.TFRecordDataset(valid_files_subset)
    valid_data = parse_tf_record_dataset(valid_data_tf, [feature], dataset_type= 'valid', use_delta= use_delta, normalize=normalize)
    valid_data_batched = valid_data.batch(valid_test_batch_size, drop_remainder = True).repeat()

    test_data_tf = tf.data.TFRecordDataset(test_files_subset)
    test_data = parse_tf_record_dataset(test_data_tf, [feature], dataset_type= 'test', use_delta= use_delta, normalize=normalize)
    test_data_batched = test_data.batch(valid_test_batch_size, drop_remainder = True)

    return train_data_batched, valid_data_batched, test_data_batched

def prepare_data(feature, p_train = 0.7 , p_valid = 0.15, train_batch_size = 128, valid_test_batch_size = 32, use_delta = False):
    """
    Loads and batches the training, validation, test data and returns the result.

    Parameters: 
    data_labels (pd.DataFrame): DataFrame containing the TFRecord filenames and their labels
    feature: feature to be loaded from the dataset - either 'spo2' or 'audio'
    p_train: percentage of total dataset to be used as train data
    p_valid: percentage of total dataset to be used as valid data 
    p_test: percentage of total dataset to be used as train data
    batch_size: number of records in each batch 

    Returns: 
    Batched train data, valid data, and test data as TFRecordDatasets
    """
    train_files, valid_files, test_files = split_train_valid_test_subject() 
   
    train_data_tf = tf.data.TFRecordDataset(train_files)
    train_data = parse_tf_record_dataset(train_data_tf, [feature], dataset_type= 'train', use_delta= use_delta)
    train_data_batched = train_data.batch(train_batch_size, drop_remainder=True).repeat()

    valid_data_tf = tf.data.TFRecordDataset(valid_files)
    valid_data = parse_tf_record_dataset(valid_data_tf, [feature], dataset_type= 'valid',  use_delta= use_delta)
    valid_data_batched = valid_data.batch(valid_test_batch_size, drop_remainder = True).repeat()

    test_data_tf = tf.data.TFRecordDataset(test_files)
    test_data = parse_tf_record_dataset(test_data_tf, [feature],  dataset_type= 'test', use_delta= use_delta)
    test_data_batched = test_data.batch(valid_test_batch_size, drop_remainder = True)

    return train_data_batched, valid_data_batched, test_data_batched


def prepare_data_subject(feature, p_train = 0.7 , p_valid = 0.15, train_batch_size = 128, valid_test_batch_size = 32, use_delta = False):
    """
    Loads and batches the training, validation, test data and returns the result.

    Parameters: 
    data_labels (pd.DataFrame): DataFrame containing the TFRecord filenames and their labels
    feature: feature to be loaded from the dataset - either 'spo2' or 'audio'
    p_train: percentage of total dataset to be used as train data
    p_valid: percentage of total dataset to be used as valid data 
    p_test: percentage of total dataset to be used as train data
    batch_size: number of records in each batch 

    Returns: 
    Batched train data, valid data, and test data as TFRecordDatasets
    """
    train_files, valid_files, test_files = split_train_valid_test_subject() 
   
    train_data_tf = tf.data.TFRecordDataset(train_files)
    train_data = parse_tf_record_dataset(train_data_tf, [feature], dataset_type= 'train', use_delta= use_delta)
    train_data_batched = train_data.batch(train_batch_size, drop_remainder=True).repeat()

    valid_data_tf = tf.data.TFRecordDataset(valid_files)
    valid_data = parse_tf_record_dataset(valid_data_tf, [feature], dataset_type= 'valid', use_delta= use_delta)
    valid_data_batched = valid_data.batch(valid_test_batch_size, drop_remainder = True).repeat()

    test_data_tf = tf.data.TFRecordDataset(test_files)
    test_data = parse_tf_record_dataset(test_data_tf, [feature], dataset_type= 'test', use_delta= use_delta)
    test_data_batched = test_data.batch(valid_test_batch_size, drop_remainder = True)

    return train_data_batched, valid_data_batched, test_data_batched

def prepare_biclass_data(feature, p_train = 0.7 , p_valid = 0.15, train_batch_size = 128, valid_test_batch_size = 32, use_delta = False):
    """
    Loads and batches the training, validation, test data and returns the result.

    Parameters: 
    data_labels (pd.DataFrame): DataFrame containing the TFRecord filenames and their labels
    feature: feature to be loaded from the dataset - either 'spo2' or 'audio'
    p_train: percentage of total dataset to be used as train data
    p_valid: percentage of total dataset to be used as valid data 
    p_test: percentage of total dataset to be used as train data
    batch_size: number of records in each batch 

    Returns: 
    Batched train data, valid data, and test data as TFRecordDatasets
    """
    train_files, valid_files, test_files = split_train_valid_test() 
   
    train_data_tf = tf.data.TFRecordDataset(train_files)
    train_data = parse_biclass_tf_record_dataset(train_data_tf, [feature])
    train_data_batched = train_data.batch(train_batch_size, drop_remainder=True).repeat()

    valid_data_tf = tf.data.TFRecordDataset(valid_files)
    valid_data = parse_biclass_tf_record_dataset(valid_data_tf, [feature])
    valid_data_batched = valid_data.batch(valid_test_batch_size, drop_remainder = True).repeat()

    test_data_tf = tf.data.TFRecordDataset(test_files)
    test_data = parse_biclass_tf_record_dataset(test_data_tf, [feature])
    test_data_batched = test_data.batch(valid_test_batch_size, drop_remainder = True)

    return train_data_batched, valid_data_batched, test_data_batched

def mel_to_mfcc(mel_spectrogram, num_mfcc=13):
    # Convert the Mel Spectrogram to MFCC using TensorFlow
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrogram)  # [batch, time, num_mfcc]
    
    # Keep only the desired number of MFCC coefficients (typically 13 or 20)
    mfccs = mfccs[:, :, :num_mfcc]

    return mfccs

def standardize(x): 
    # x_tensor = tf.convert_to_tensor(x, dtype = tf.float32)
    mean = tf.reduce_mean(x)
    std = tf.math.reduce_std(x)
    return (x - mean)/std