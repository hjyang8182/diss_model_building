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
AUTOTUNE = tf.data.experimental.AUTOTUNE


@tf.function
def apply_delta(spo2_tensor): 
    # spo2_np = spo2_tensor.numpy()
    threshold = 10
    delta_spo2 = []
    # t ranging from 1 ~ 10
    t_vals = tf.range(1, 11, dtype=tf.float32)

    for n in range(10, 30): 
        t_samples_after = n + t_vals
        t_samples_before = n - t_vals

        t_samples_before = tf.cast(t_samples_before, tf.int32)
        t_samples_after = tf.cast(t_samples_after, tf.int32)
        
        pre_spo2 = tf.gather(spo2_tensor, t_samples_before)
        post_spo2 = tf.gather(spo2_tensor,t_samples_after)
        delta_val = tf.reduce_sum(t_vals * (post_spo2 - pre_spo2)) / (2 * tf.reduce_sum(t_vals ** 2))
        delta_spo2.append(delta_val)      
    delta_spo2 = tf.convert_to_tensor(delta_spo2,  dtype=tf.float32)
    return delta_spo2

@tf.function
def apply_window(delta_spo2): 
    window_size = 5
    stride = 2
    delta_frames = tf.signal.frame(delta_spo2, window_size, stride)
    return delta_frames

@tf.function
def apply_denoise(audio):
    def denoise_wrapper(audio_np): 
        denoised_audio = reduce_noise(audio_np, sr = 8000)
        denoised_audio_tensor = tf.convert_to_tensor(denoised_audio, dtype = tf.float32)
        return denoised_audio_tensor
    
    denoised_audio = tf.numpy_function(denoise_wrapper, [audio], tf.float32)
    return denoised_audio


class DataLoader(): 
    def __init__(self, subset_train_count = 8000, subset_test_valid_count = 800, train_batch_size = 128, valid_test_batch_size = 32):
        train_files, valid_files, test_files = self.split_train_valid_test()

        self.train_batch_size = train_batch_size
        self.valid_test_batch_size = valid_test_batch_size 

        self.full_train_count = len(train_files) 
        self.full_valid_count = len(valid_files) 
        self.full_test_count = len(test_files)

        self.FULL_TRAIN_STEPS_PER_EPOCH = self.full_train_count//128
        self.FULL_VALID_STEPS_PER_EPOCH = self.full_valid_count//32 -1

        self.subset_train_count = subset_train_count 
        self.subset_valid_count = subset_test_valid_count 
        self.subset_test_count = subset_test_valid_count 
        self.TRAIN_STEPS_PER_EPOCH = self.subset_train_count//128
        self.VALID_STEPS_PER_EPOCH = self.subset_valid_count//32 -1

    def load_full_data(self, feature, window_spo2 = False, dataset = 'default'): 
        full_data = self.prepare_data(feature = feature, p_train = 0.7, p_valid = 0.15, window_spo2=window_spo2, dataset = dataset)
        return full_data

    def load_subset_data(self, feature, window_spo2 = False, dataset = 'default'): 
        data_subset = self.prepare_data(feature = feature, subset = True, train_batch_size= self.train_batch_size, valid_test_batch_size= self.valid_test_batch_size, window_spo2 = window_spo2, dataset = dataset)
        return data_subset

    def load_mel_spec_png(self): 
        return self.load_mel_spec_images()
    
    def _parse_data_default(self, proto, features, dataset_type, window_spo2 = False, biclass = False):
        audio_sample_rate = 8000
        keys_to_features = {
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'audio': tf.io.FixedLenFeature([audio_sample_rate * 40], tf.float32), 
            'spo2': tf.io.FixedLenFeature([40], tf.float32) 
        }

        try: 
            parsed_features = tf.io.parse_single_example(proto, keys_to_features)
            
            label = parsed_features['label']

            class_weights = self.return_class_weights()[dataset_type]
            
            label = tf.squeeze(label)

            if biclass: 
                label = tf.where(label == tf.constant(2, dtype=tf.int64), tf.constant(1, dtype=tf.int64), label)
                label = tf.where(label == tf.constant(3, dtype=tf.int64), tf.constant(1, dtype=tf.int64), label)
            
            else: 
                label = tf.where(label == tf.constant(3, dtype=tf.int64), tf.constant(2, dtype=tf.int64), label)

            weight = tf.gather(class_weights, label)
            label_enc = tf.one_hot(label, 3)
            label_enc.set_shape([3])
            if 'spo2' in features: 
                spo2 = parsed_features['spo2']
                delta_spo2 = apply_delta(spo2)
                if window_spo2: 
                    windowed_spo2 = apply_window(delta_spo2)
                    parsed_features['spo2'] = windowed_spo2
                    parsed_features['spo2'].set_shape([8, 5])
                else: 
                    parsed_features['spo2'] = delta_spo2
                    parsed_features['spo2'].set_shape([20])

            return_vals = [parsed_features[feature] for feature in features]
            return_vals.append(label_enc)
            return_vals.append(weight)
            return tuple(return_vals)
        except tf.errors.InvalidArgumentError:
            return None
        except Exception as e:
            print(e) 
            return None
    
    def _parse_audio_features(self, proto, features, dataset_type): 
        audio_sample_rate = 8000
        keys_to_features = {
            'label': tf.io.FixedLenFeature([3], tf.int64),
            'audio_denoised': tf.io.FixedLenFeature([audio_sample_rate * 40], tf.float32), 
            'audio_mel_spec': tf.io.FixedLenFeature([128 * 1251], tf.float32), 
            'audio_mfcc': tf.io.FixedLenFeature([20 * 1251], tf.float32), 
            'audio_mel_spec_shape': tf.io.FixedLenFeature([2], tf.int64), 
            'audio_mfcc_shape': tf.io.FixedLenFeature([2], tf.int64), 
        }

        try: 
            parsed_features = tf.io.parse_single_example(proto, keys_to_features)
            class_weights = self.return_class_weights()[dataset_type]
            
            one_hot_label = parsed_features['label']
            int_label = tf.argmax(one_hot_label)

            weight = tf.gather(class_weights, int_label)

            parsed_features['audio_mel_spec'] = tf.reshape(parsed_features['audio_mel_spec'], (128, 1251))
            parsed_features['audio_mfcc'] = tf.reshape(parsed_features['audio_mfcc'], (20, 1251))

            return_vals = [parsed_features[feature] for feature in features]
            return_vals.append(one_hot_label)
            return_vals.append(weight)
            return tuple(return_vals)
        except tf.errors.InvalidArgumentError:
            return None
        except Exception as e:
            print(e) 
            return None
        
    def parse_raw_tf_record_dataset(self, dataset, features, dataset_type, window_spo2 = False, biclass = False):
        parsed_data = dataset.map(lambda x : self._parse_data_default(x, features=features, dataset_type= dataset_type, window_spo2=window_spo2))
        # parsed_data = parsed_data.filter(lambda *x: x[0] is not None)
        return parsed_data
    
    def parse_audio_feature_tf_record_dataset(self, dataset, features, dataset_type, window_spo2 = False, biclass = False):
        parsed_data = dataset.map(lambda x : self._parse_audio_features(x, features=features, dataset_type= dataset_type))
        return parsed_data

    def split_train_valid_test(self, p_train = 0.7, p_valid = 0.15):
        data_labels_no_mixed = data_labels.loc[data_labels['label'] != 3]
        all_subjects = np.unique(data_labels_no_mixed['subject'].values)
        p_train = 0.7
        p_valid = 0.15

        p_test = 1 - (p_train + p_valid)
        train_subjects, test_valid_subjects = train_test_split(all_subjects, test_size = (p_valid + p_test), random_state=42)
        valid_subjects, test_subjects = train_test_split(test_valid_subjects, test_size = p_test/(p_valid + p_test), random_state=42)

        np.random.seed(42)
        train_subjects_df = data_labels_no_mixed.loc[data_labels_no_mixed['subject'].isin(train_subjects)]
        sampled_train_df = train_subjects_df.groupby('subject', group_keys=False).apply(lambda x : x.sample(frac = 0.5))
        sampled_train_df = sampled_train_df.reset_index(drop = True)

        np.random.seed(42)
        valid_subjects_df = data_labels_no_mixed.loc[data_labels_no_mixed['subject'].isin(valid_subjects)]
        sampled_valid_df = valid_subjects_df.groupby('subject', group_keys=False).apply(lambda x : x.sample(frac = 0.5))
        sampled_valid_df = sampled_valid_df.reset_index(drop = True)

        np.random.seed(42)
        test_subjects_df = data_labels_no_mixed.loc[data_labels_no_mixed['subject'].isin(test_subjects)]
        sampled_test_df = test_subjects_df.groupby('subject', group_keys=False).apply(lambda x : x.sample(frac = 0.5))
        sampled_test_df = sampled_test_df.reset_index(drop = True)

        train_files = sampled_train_df['file'].values.astype(str)
        valid_files = sampled_valid_df['file'].values.astype(str)
        test_files = sampled_test_df['file'].values.astype(str)
        
        # train_files = data_labels.loc[data_labels['subject'].isin(train_subjects), 'file'].values
        # valid_files = data_labels.loc[data_labels['subject'].isin(valid_subjects), 'file'].values
        # test_files = data_labels.loc[data_labels['subject'].isin(test_subjects), 'file'].values

        return train_files, valid_files, test_files
    

    def split_train_valid_test_subset(self, train_length = 600, valid_length = 60, test_length = 60):

        train_files, valid_files, test_files = self.split_train_valid_test()
        
        np.random.seed(42)
        train_files_subset = np.random.choice(train_files, train_length, replace = False) 

        np.random.seed(42)
        valid_files_subset = np.random.choice(valid_files, valid_length, replace = False) 
        
        np.random.seed(42)
        test_files_subset = np.random.choice(test_files, test_length, replace = False)

        return train_files_subset, valid_files_subset, test_files_subset

    def prepare_data(self, feature, subset = False, p_train = 0.7 , p_valid = 0.15,train_batch_size = 128, valid_test_batch_size = 32, window_spo2 = False, dataset = 'default'):
        """
        Loads and batches the training, validation, test data and returns the result.

        Parameters: 
        data_labels (pd.DataFrame): DataFrame containing the TFRecord filenames and their labels
        dataset: either the default raw tfrecord dataset or the audio features dataset
        feature: feature to be loaded from the dataset - either 'spo2' or 'audio'
        p_train: percentage of total dataset to be used as train data
        p_valid: percentage of total dataset to be used as valid data 
        p_test: percentage of total dataset to be used as train data
        batch_size: number of records in each batch 

        Returns: 
        Batched train data, valid data, and test data as TFRecordDatasets
        """
        if subset: 
            train_files, valid_files, test_files = self.split_train_valid_test_subset(8000, 800, 800)
        else: 
            train_files, valid_files, test_files = self.split_train_valid_test() 
        
        if dataset == 'default': 
            base_dir = '/home/hy381/rds/hpc-work/segmented_data_new/'
            train_files = np.char.add(base_dir, train_files)
            valid_files = np.char.add(base_dir, valid_files)
            test_files = np.char.add(base_dir, test_files)

            train_data_tf = tf.data.TFRecordDataset(train_files)
            train_data = self.parse_raw_tf_record_dataset(train_data_tf, [feature], dataset_type= 'train', window_spo2=window_spo2)
            train_data_batched = train_data.batch(train_batch_size, drop_remainder=True).repeat().prefetch(AUTOTUNE)

            valid_data_tf = tf.data.TFRecordDataset(valid_files)
            valid_data = self.parse_raw_tf_record_dataset(valid_data_tf,  [feature], dataset_type= 'valid', window_spo2=window_spo2)
            valid_data_batched = valid_data.batch(valid_test_batch_size, drop_remainder = True).repeat().prefetch(AUTOTUNE)

            test_data_tf = tf.data.TFRecordDataset(test_files)
            test_data = self.parse_raw_tf_record_dataset(test_data_tf, [feature],  dataset_type= 'test', window_spo2=window_spo2)
            test_data_batched = test_data.batch(valid_test_batch_size, drop_remainder = True).prefetch(AUTOTUNE)

        else: 
            base_dir = '/home/hy381/rds/hpc-work/audio_feature_data/'

            train_files = np.char.add(base_dir, train_files)
            valid_files = np.char.add(base_dir, valid_files)
            test_files = np.char.add(base_dir, test_files)

            train_data_tf = tf.data.TFRecordDataset(train_files)
            train_data = self.parse_audio_feature_tf_record_dataset(train_data_tf, [feature], dataset_type= 'train', window_spo2=window_spo2)
            train_data_batched = train_data.batch(train_batch_size, drop_remainder=True).repeat().prefetch(AUTOTUNE)

            valid_data_tf = tf.data.TFRecordDataset(valid_files)
            valid_data = self.parse_audio_feature_tf_record_dataset(valid_data_tf,  [feature], dataset_type= 'valid', window_spo2=window_spo2)
            valid_data_batched = valid_data.batch(valid_test_batch_size, drop_remainder = True).repeat().prefetch(AUTOTUNE)

            test_data_tf = tf.data.TFRecordDataset(test_files)
            test_data = self.parse_audio_feature_tf_record_dataset(test_data_tf, [feature],  dataset_type= 'test', window_spo2=window_spo2)
            test_data_batched = test_data.batch(valid_test_batch_size, drop_remainder = True).prefetch(AUTOTUNE)
            
        return train_data_batched, valid_data_batched, test_data_batched

    def load_mel_spec_images(self, subset = False):
        if subset: 
            train_files, valid_files, test_files = self.split_train_valid_test_subset(self.subset_train_count, self.subset_valid_count, self.subset_test_count)
        else:  
            train_files, valid_files, test_files = self.split_train_valid_test() 
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
        train_dataset = train_dataset.map(lambda img, label: self.load_image(img, label, 'train')).batch(self.train_batch_size, drop_remainder = True).repeat()

        valid_labels = data_labels.loc[data_labels['file'].isin(valid_files), 'label']
        valid_labels = tf.convert_to_tensor(valid_labels.values, dtype = tf.int32)
        valid_labels = tf.one_hot(valid_labels, depth=3, dtype=tf.int32)
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_files_mel_spec, valid_labels))
        valid_dataset = valid_dataset.map(lambda img, label: self.load_image(img, label, 'valid')).batch(self.valid_test_batch_size, drop_remainder = True).repeat()

        test_labels = data_labels.loc[data_labels['file'].isin(test_files), 'label']
        test_labels = tf.convert_to_tensor(test_labels.values, dtype = tf.int32)
        test_labels = tf.one_hot(test_labels, depth=3, dtype=tf.int32)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_files_mel_spec, test_labels))
        test_dataset = test_dataset.map(lambda img, label: self.load_image(img, label, 'test')).batch(self.valid_test_batch_size, drop_remainder = True)

        return train_dataset, valid_dataset, test_dataset

    def load_image(self, image_path, label, dataset_type):
        class_weights = self.return_class_weights()[dataset_type]
        int_label = tf.argmax(label)
        weight = tf.gather(class_weights, int_label)
        img = tf.io.read_file(image_path)  # Read file
        img = tf.image.decode_png(img, channels=3)  # Decode PNG
        img = tf.image.resize(img, [224, 224])  # Resize
        img = img / 255.0  # Normalize
        return img, label, weight
    
    def return_class_weights(self): 
        train_files, valid_files, test_files = self.split_train_valid_test()

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


# def mel_to_mfcc(mel_spectrogram, num_mfcc=13):
#     # Convert the Mel Spectrogram to MFCC using TensorFlow
#     mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrogram)  # [batch, time, num_mfcc]
    
#     # Keep only the desired number of MFCC coefficients (typically 13 or 20)
#     mfccs = mfccs[:, :, :num_mfcc]

#     return mfccs

# def standardize(x): 
#     # x_tensor = tf.convert_to_tensor(x, dtype = tf.float32)
#     mean = tf.reduce_mean(x)
#     std = tf.math.reduce_std(x)
#     return (x - mean)/std

        