import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json
import data as data

os.chdir('/home/hy381/rds/hpc-work/segmented_data_new')

train_files, valid_files, test_files = data.split_train_valid_test()
full_train_count = len(train_files) 
full_valid_count = len(valid_files) 
full_test_count = len(test_files)
print(full_train_count)
FULL_TRAIN_STEPS_PER_EPOCH = full_train_count//128
FULL_VALID_STEPS_PER_EPOCH = full_valid_count//32 -1

subset_train_count = 8000 
subset_valid_count = 800 
subset_test_count = 800 
TRAIN_STEPS_PER_EPOCH = subset_train_count//128
VALID_STEPS_PER_EPOCH = subset_valid_count//32 -1

# spo2_data_full = utils.prepare_data_subject(feature = 'spo2', p_train = 0.7, p_valid = 0.15, use_delta = False)
# audio_data_full = utils.prepare_data_subject(feature = 'audio')
# biclass_spo2_data = utils.prepare_biclass_data(feature = 'spo2', use_delta = True)
# biclass_audio_data = utils.prepare_biclass_data(feature = 'audio', use_delta = True)
# mel_spec_data = utils.load_mel_spec_images()

class DataLoader(): 
    def __init__(self, train_batch_size = 128, valid_test_batch_size = 32):
        train_files, valid_files, test_files = data.split_train_valid_test()

        self.train_batch_size = train_batch_size
        self.valid_test_batch_size = valid_test_batch_size 

        self.full_train_count = len(train_files) 
        self.full_valid_count = len(valid_files) 
        self.full_test_count = len(test_files)

        self.FULL_TRAIN_STEPS_PER_EPOCH = full_train_count//128
        self.FULL_VALID_STEPS_PER_EPOCH = full_valid_count//32 -1

        self.subset_train_count = 8000 
        self.subset_valid_count = 800 
        self.subset_test_count = 800 
        self.TRAIN_STEPS_PER_EPOCH = subset_train_count//128
        self.VALID_STEPS_PER_EPOCH = subset_valid_count//32 -1

    def load_full_data(self, feature, window_spo2 = False): 
        full_data = data.prepare_data(feature = feature, p_train = 0.7, p_valid = 0.15, window_spo2=window_spo2)
        return full_data

    def load_subset_data(self, feature, train_length = subset_train_count, valid_length = subset_valid_count, test_length = subset_test_count, window_spo2 = False): 
        data_subset = data.prepare_data_subset(feature = feature, train_length= train_length, valid_length = valid_length, test_length = test_length, train_batch_size= self.train_batch_size, valid_test_batch_size= self.valid_test_batch_size, window_spo2 = window_spo2)
        return data_subset

    def load_mel_spec_png(self): 
        return data.load_mel_spec_images()
    
    # todo : load biclass data