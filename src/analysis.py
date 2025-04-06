import pandas as pd
from data import ApneaDataLoader 
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import tensorflow as tf
def fused_results_to_csv(spo2_model, audio_model, fuse_method):
    csv_dict = {}

    # Load test file names
    data_loader = ApneaDataLoader()
    _, _, test_files = data_loader.split_train_valid_test()
    csv_dict['filename'] = test_files 
    
    # Load ground truth values for the test files
    gt_fname = '/home/hy381/rds/hpc-work/models/ground_truth.txt'
    gt = np.loadtxt(gt_fname)
    gt_labels = np.argmax(gt, axis = 1)
    csv_dict['true_label'] = gt_labels

    spo2_preds = np.loadtxt(f'/home/hy381/rds/hpc-work/models/{spo2_model}/predictions.txt')
    spo2_pred_labels = np.argmax(spo2_preds, axis = 1)
    csv_dict['spo2_model_correct'] = (spo2_pred_labels == gt_labels).astype(int)

    audio_preds = np.loadtxt(f'/home/hy381/rds/hpc-work/models/{audio_model}/predictions.txt')
    audio_pred_labels = np.argmax(audio_preds, axis = 1)
    csv_dict['audio_model_correct'] = (audio_pred_labels == gt_labels).astype(int)

    spo2_model_type = spo2_model.split('_')[1]
    audio_model_type = audio_model.removeprefix('audio_')
    fused_model_name = f'lf_{fuse_method}_{spo2_model_type}+{audio_model_type}'
    
    fused_pred = np.loadtxt(f'/home/hy381/rds/hpc-work/models/{fused_model_name}/predictions.txt')
    fused_pred_labels = np.argmax(fused_pred, axis = 1)
    csv_dict['fused_model_correct'] = (fused_pred_labels == gt_labels).astype(int)

    df = pd.DataFrame(csv_dict)
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    df.to_csv(csv_path, index = False)
    print(f"CSV saved for {fused_model_name}")
    return

def save_audio_snr(): 
    data_loader = ApneaDataLoader()
    csv_dict = {}
    _, _, test_files = data_loader.split_train_valid_test()
    denoised_audio_files = [os.path.basename(f).replace('tfrecord', 'wav') for f in test_files]
    csv_dict['filename'] = test_files
    snr_vals = []
    rms_vals = []
    zcr_vals = []
    denoised_audio_dir = '/home/hy381/rds/hpc-work/denoised_audio_wav'
    og_audio_dir = '/home/hy381/rds/hpc-work/audio_wav'
    for file in denoised_audio_files: 
        denoised_audio_fname = os.path.join(denoised_audio_dir, file)
        og_audio_fname = os.path.join(og_audio_dir, file)
        snr, rms, zcr = compute_audio_feats(og_audio_fname, denoised_audio_fname)
        snr_vals.append(snr)
        rms_vals.append(rms)
        zcr_vals.append(zcr)
    csv_dict['snr'] = snr_vals
    csv_dict['rms'] = rms_vals
    csv_dict['zcr'] = zcr_vals
    df = pd.DataFrame(csv_dict)
    csv_path = f'/home/hy381/model_training/eval/audio_snr.csv'
    df.to_csv(csv_path, index = False)

def compute_audio_feats(og_audio_file, clean_audio_file): 
    og_audio, _ = librosa.load(og_audio_file, sr = 8000)
    clean_audio, _ = librosa.load(clean_audio_file, sr = 8000)
    assert(np.max(clean_audio) <= 1 and np.min(clean_audio) >= -1)

    noise = clean_audio - og_audio
    signal_power = np.mean(clean_audio**2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    rms = librosa.feature.rms(y = clean_audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y = clean_audio)[0]
    zcr_mean = np.mean(zcr)
    rms_mean = np.mean(rms)

    return snr, rms_mean, zcr_mean

def analyse_snr_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    snr = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')

    audio_only_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    spo2_only_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    audio_only_snr = snr.loc[snr['filename'].isin(audio_only_correct), 'snr'].values
    spo2_only_snr = snr.loc[snr['filename'].isin(spo2_only_correct), 'snr'].values
    print(audio_only_snr)
    labels = ['Audio Only Correct', 'SpO2 Only Correct']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_only_snr)), audio_only_snr, label = 'Audio Only Correct', color = 'red')
    plt.scatter(np.repeat(1, len(spo2_only_snr)), spo2_only_snr, label = 'SpO2 Only Correct', color = 'blue')
    plt.legend()
    plt.savefig(f'/home/hy381/model_training/eval/snr_plot_{label}.png')


def analyse_rms_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    audio_feat_csv = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')

    audio_only_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    spo2_only_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    
    audio_only_rms = audio_feat_csv.loc[audio_feat_csv['filename'].isin(audio_only_correct), 'rms'].values
    spo2_only_rms = audio_feat_csv.loc[audio_feat_csv['filename'].isin(spo2_only_correct), 'rms'].values
    labels = ['Audio Only Correct', 'SpO2 Only Correct']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_only_rms)), audio_only_rms, label = 'Audio Only Correct', color = 'red')
    plt.scatter(np.repeat(1, len(spo2_only_rms)), spo2_only_rms, label = 'SpO2 Only Correct', color = 'blue')
    plt.legend()
    plt.savefig(f'/home/hy381/model_training/eval/rms_plot_{label}.png')
    plt.close()

def analyse_zcr_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    snr = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')

    audio_only_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    spo2_only_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    
    audio_only_zcr = snr.loc[snr['filename'].isin(audio_only_correct), 'zcr'].values
    spo2_only_zcr = snr.loc[snr['filename'].isin(spo2_only_correct), 'zcr'].values
    labels = ['Audio Only Correct', 'SpO2 Only Correct']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_only_zcr)), audio_only_zcr, label = 'Audio Only Correct', color = 'red')
    plt.scatter(np.repeat(1, len(spo2_only_zcr)), spo2_only_zcr, label = 'SpO2 Only Correct', color = 'blue')
    plt.legend()
    plt.savefig(f'/home/hy381/model_training/eval/zcr_plot_{label}.png')
    plt.close()

def visualize_audio_only(files): 
    data_loader = ApneaDataLoader()
    spo2_base_dir = '/home/hy381/rds/hpc-work/segmented_data_new/'
    audio_base_dir = '/home/hy381/rds/hpc-work/audio_feature_data/'

    spo2_files = np.char.add(spo2_base_dir, files)
    audio_files = np.char.add(audio_base_dir, files)
    
    spo2_dataset = tf.data.Dataset.from_tensor_slices(spo2_files)
    spo2_dataset = spo2_dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, buffer_size=1000000).shuffle(10000),
        cycle_length=4,  # Number of files read in parallel
        num_parallel_calls=tf.data.experimental.AUTOTUNE   # Optimize for performance
    )
    parsed_spo2_data = data_loader.parse_raw_tf_record_dataset(spo2_dataset, ['spo2'], 'train')

    audio_dataset = tf.data.Dataset.from_tensor_slices(audio_files)
    audio_dataset = audio_dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, buffer_size=1000000).shuffle(10000),
        cycle_length=4,  # Number of files read in parallel
        num_parallel_calls=tf.data.experimental.AUTOTUNE   # Optimize for performance
    )
    parsed_audio_data = data_loader.parse_audio_feature_tf_record_dataset(audio_dataset, ['audio_mel_spec'], 'train')

    dataset = tf.data.Dataset.zip((parsed_spo2_data, parsed_audio_data))
    for i, data in enumerate(dataset.take(20)): 
        fname = os.path.basename(files[i])
        spo2_all, audio_all = data
        spo2 = spo2_all[0]
        spectrogram = audio_all[0]
        fig, ax = plt.subplots(2, 1, figsize = (8, 6))
        ax = ax.flatten()

        sr = 8000
        n_fft = int(0.25 * sr)
        # 0.125 seconds -> 125 ms, hop_length = 1000 
        long_hop_length = int(0.125 * sr)

        ax[0].plot(spo2)
        ax[0].set_ylabel('SpO2 Signal')
        ax[0].set_ylim(60, 100)
        ax[1].set_ylim(0, 1500)
        # fig.suptitle(f"Visualisation for {fname} - {label_names[raw_label]}")
        mel_spec_display = librosa.display.specshow(spectrogram.numpy(), sr=8000, x_axis="time", y_axis="linear", ax = ax[1],  hop_length=long_hop_length)
        save_fname = f'/home/hy381/model_training/eval/{fname}.png'
        plt.savefig(save_fname, bbox_inches='tight')
        plt.close()

# def fused_error_files(): 
def main(): 
    # spo2_model = 'spo2_lstm_model'
    # audio_model = 'audio_ms_cnn'
    # fused_results_to_csv(spo2_model, audio_model, 'prod')
    # save_audio_snr()
    fused_model_name = 'lf_prod_lstm+ms_cnn'
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    # audio_only_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == 2), 'filename'].values
    # visualize_audio_only(audio_only_correct)

    spo2_only_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1)  & (model_results['true_label'] == 0), 'filename'].values
    # visualize_audio_only(spo2_only_correct)
    
    # np.savetxt(f'/home/hy381/model_training/eval/audio_only.txt', audio_only_correct, fmt='%s')
    # np.savetxt(f'/home/hy381/model_training/eval/spo2_only.txt', spo2_only_correct, fmt='%s')

    # analyse_snr_files('lf_prod_lstm+ms_cnn')
    analyse_zcr_files('lf_prod_lstm+ms_cnn', 0)
    analyse_zcr_files('lf_prod_lstm+ms_cnn', 1)
    analyse_zcr_files('lf_prod_lstm+ms_cnn', 2)

    analyse_rms_files('lf_prod_lstm+ms_cnn', 0)
    analyse_rms_files('lf_prod_lstm+ms_cnn', 1)
    analyse_rms_files('lf_prod_lstm+ms_cnn', 2)

    analyse_snr_files('lf_prod_lstm+ms_cnn', 0)
    analyse_snr_files('lf_prod_lstm+ms_cnn', 1)
    analyse_snr_files('lf_prod_lstm+ms_cnn', 2)
    # analyse_rms_files('lf_prod_lstm+ms_cnn')

if __name__ == '__main__':
    main()