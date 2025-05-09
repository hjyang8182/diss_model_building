import pandas as pd
from data import ApneaDataLoader 
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

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

def save_mfcc_to_np(): 
    data_loader = ApneaDataLoader()
    mfccs = []
    test_data = data_loader.load_full_data(['audio_mfcc'], parse_type = 'audio_feature', batch = False)[2]
    for data in test_data: 
        mfcc, _, _ = data
        mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    np.save('/home/hy381/rds/hpc-work/mfccs.npy', mfccs)
    print('MFCCs saved')

def compute_pca(features): 
    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)

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
    rms = librosa.feature.rms(y = og_audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y = og_audio)[0]
    zcr_mean = np.mean(zcr)
    rms_mean = np.mean(rms)

    return snr, rms_mean, zcr_mean


def analyse_spectral_centroid_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    snr = pd.read_csv('/home/hy381/model_training/eval/audio_snr_new.csv')

    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    
    audio_fused_both_correct_avg = snr.loc[snr['filename'].isin(audio_fused_both_correct), 'avg_spectral_centroid'].values
    audio_incorrect_fused_avg = snr.loc[snr['filename'].isin(audio_incorrect_fused_correct), 'avg_spectral_centroid'].values
    labels = ['Audio, SpO2, Fused All Correct', 'Audio Only Incorrect']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_fused_both_correct_avg)), audio_fused_both_correct_avg, label = 'All Correct', color = 'red')
    plt.scatter(np.repeat(1, len(audio_incorrect_fused_avg)), audio_incorrect_fused_avg, label = 'Audio Incorrect', color = 'blue')
    plt.legend()
    plt.savefig(f'/home/hy381/model_training/eval/avg_sc_plot_{label}.png')


def analyse_snr_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    snr = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')

    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['fused_model_correct'] == 1), 'filename'].values
    
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) , 'filename'].values
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) , 'filename'].values
    
    audio_fused_both_correct_snr = snr.loc[snr['filename'].isin(audio_fused_both_correct), 'snr'].values
    audio_incorrect_fused_snr = snr.loc[snr['filename'].isin(audio_incorrect_fused_correct), 'snr'].values

    labels = ['Audio, SpO2, Fused All Correct', 'Audio Only Incorrect']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_fused_both_correct_snr)), audio_fused_both_correct_snr, label = 'All Correct', color = 'red')
    plt.scatter(np.repeat(1, len(audio_incorrect_fused_snr)), audio_incorrect_fused_snr, label = 'Audio Incorrect', color = 'blue')
    plt.legend()
    plt.savefig(f'/home/hy381/model_training/eval/snr_plot_{label}.png')


def analyse_rms_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    audio_feat_csv = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')

    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    # audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    # audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) , 'filename'].values
    
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['true_label'] == label), 'filename'].values
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['fused_model_correct'] == 1), 'filename'].values
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) , 'filename'].values
    
    audio_fused_both_correct_rms = audio_feat_csv.loc[audio_feat_csv['filename'].isin(audio_fused_both_correct), 'rms'].values
    audio_incorrect_fused_correct_rms = audio_feat_csv.loc[audio_feat_csv['filename'].isin(audio_incorrect_fused_correct), 'rms'].values
    labels = ['Audio Only Correct', 'SpO2 Only Correct']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_fused_both_correct_rms)), audio_fused_both_correct_rms, label = 'All Correct Correct', color = 'red')
    plt.scatter(np.repeat(1, len(audio_incorrect_fused_correct_rms)), audio_incorrect_fused_correct_rms, label = 'Audio InCorrect', color = 'blue')
    plt.legend()
    plt.savefig(f'/home/hy381/model_training/eval/rms_plot_{label}.png')
    plt.close()

def analyse_zcr_files(fused_model_name, label): 
    csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    model_results = pd.read_csv(csv_path)
    snr = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')

    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    # audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    
    audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) , 'filename'].values
    
    # audio_fused_both_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['true_label'] == label), 'filename'].values
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['true_label'] == label), 'filename'].values
    audio_incorrect_fused_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['fused_model_correct'] == 1), 'filename'].values
    
    audio_fused_both_correct_zcr = snr.loc[snr['filename'].isin(audio_fused_both_correct), 'zcr'].values
    audio_incorrect_fused_correct_zcr = snr.loc[snr['filename'].isin(audio_incorrect_fused_correct), 'zcr'].values
    labels = ['Audio Only Correct', 'SpO2 Only Correct']
    plt.figure(figsize=(10, 6))
    plt.scatter(np.repeat(0, len(audio_fused_both_correct_zcr)), audio_fused_both_correct_zcr, label = 'All Correct', color = 'red')
    plt.scatter(np.repeat(1, len(audio_incorrect_fused_correct_zcr)), audio_incorrect_fused_correct_zcr, label = 'Audio Incorrect', color = 'blue')
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

def save_mel_spec_metrics(): 
    audio_csv = pd.read_csv('/home/hy381/model_training/eval/audio_snr.csv')
    data_loader = ApneaDataLoader()
    _, _, test_data = data_loader.prepare_data(['audio_denoised'], parse_type = 'audio_features', batch = False)
    max_centroids = []
    avg_centroids = []
    min_centroids = []
    for data in test_data: 
        audio, label, _ = data
        centroid = librosa.feature.spectral_centroid(y=audio.numpy(), sr = 8000)
        max_centroids.append(np.max(centroid))
        avg_centroids.append(np.mean(centroid))
        min_centroids.append(np.min(centroid))

    audio_csv['max_spectral_centroid'] = max_centroids
    audio_csv['min_spectral_centroid'] = min_centroids
    audio_csv['avg_spectral_centroid'] = avg_centroids

    audio_csv.to_csv('/home/hy381/model_training/eval/audio_snr_new.csv')

def plot_pca_features(): 
    csv_path = f'/home/hy381/model_training/eval/lf_prod_bilstm+mfcc_cnn_lstm.csv'
    model_results = pd.read_csv(csv_path)
    audio_feats = pd.read_csv('/home/hy381/model_training/eval/audio_snr_new.csv')

    mfccs = np.load('/home/hy381/rds/hpc-work/mfccs.npy')
    mfcc_avg = np.mean(mfccs, axis = 2)
    mfcc_std = np.std(mfccs, axis = 2)
    spectral_centroids = audio_feats['avg_spectral_centroid'].values.reshape(-1, 1)
    # print(spectral_centroids.shape)
    mfcc_flat = np.concatenate([mfcc_avg, mfcc_std, spectral_centroids], axis = 1)
    mfcc_pca = compute_pca(mfcc_flat)
    
    label_colors = ['red', 'blue', 'green']
    class_markers = ['o', '^'] 
    classes = ['Normal', 'Hypopnea', 'Apnea']

    audio_model_correct = model_results['audio_model_correct'].values
    fused_model_correct = model_results['fused_model_correct'].values
    true_labels = model_results['true_label'].values
    audio_fused_both_correct = (audio_model_correct == 1) & (fused_model_correct == 1) 
    audio_incorrect_fused_correct = (audio_model_correct == 0) & (fused_model_correct == 1) 
    plt.figure(figsize = (8,6))
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.scatter(mfcc_pca[audio_fused_both_correct,0], mfcc_pca[audio_fused_both_correct,1], color = 'turquoise', label = 'Correct Audio Predictions', alpha = 0.6)
    plt.scatter(mfcc_pca[audio_incorrect_fused_correct,0], mfcc_pca[audio_incorrect_fused_correct,1], color = 'firebrick', label = 'Fusion-Corrected Audio Predictions',  alpha = 0.6)
    plt.legend()
    plt.savefig('/home/hy381/model_training/eval/pca_audio_performance.png')
    plt.close()

    plt.figure(figsize=(8,6))    
    label_colors = ['orange', 'forestgreen', 'lightblue']
    alphas = [0.8, 0.6, 0.2]
    for i in range(3): 
        class_name = classes[i]
        audio_model_correct = model_results['audio_model_correct'].values
        fused_model_correct = model_results['fused_model_correct'].values
        true_labels = model_results['true_label'].values
        audio_fused_both_correct = (true_labels == i)
    
        plt.scatter(mfcc_pca[audio_fused_both_correct,0], mfcc_pca[audio_fused_both_correct,1], color = label_colors[i], label = classes[i], alpha =alphas[i])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig('/home/hy381/model_training/eval/pca_label_distribution.png')
    plt.close()

def plot_tsne_mfcc_features(): 
    csv_path = f'/home/hy381/model_training/eval/lf_prod_bilstm+mfcc_cnn_lstm.csv'
    model_results = pd.read_csv(csv_path)
    audio_feats = pd.read_csv('/home/hy381/model_training/eval/audio_snr_new.csv')

    mfccs = np.load('/home/hy381/rds/hpc-work/mfccs.npy')
    mfcc_avg = np.mean(mfccs, axis = 2)
    mfcc_std = np.std(mfccs, axis = 2)
    spectral_centroids = audio_feats['avg_spectral_centroid'].values.reshape(-1, 1)
    classes = model_results['true_label'].values

    mfcc_flat = np.concatenate([mfcc_avg, mfcc_std, spectral_centroids], axis = 1)
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 300, n_iter= 4000)
    X_tsne = tsne.fit_transform(mfcc_flat)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = classes)

    plt.colorbar(scatter, label='Class Label')
    plt.title('t-SNE Visualization of Iris Dataset')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('tsne.png')


def save_spo2_metrics(): 
    data_loader = ApneaDataLoader()
    csv_dict = {}
    _, _, test_files = data_loader.split_train_valid_test()
    _, _, test_data = data_loader.prepare_data(['spo2'], parse_type = 'default', batch = False)

    csv_dict['filename'] = test_files
    ranges = []
    stds = []
    for data in test_data:
        spo2, _, _ = data 
        range = np.max(spo2) - np.min(spo2)
        std = np.std(spo2)
        ranges.append(range)
        stds.append(std)

    csv_dict['range'] = ranges
    csv_dict['std'] = stds
    df = pd.DataFrame(csv_dict)
    csv_path = f'/home/hy381/model_training/eval/spo2_metrics.csv'
    df.to_csv(csv_path, index = False)


def save_spo2_delta_metrics(): 
    data_loader = ApneaDataLoader()
    csv_dict = pd.read_csv('/home/hy381/model_training/eval/spo2_metrics.csv')

    _, _, test_files = data_loader.split_train_valid_test()
    _, _, test_data = data_loader.prepare_data(['spo2'], parse_type = 'default', apply_delta=True, batch = False)

    max_deltas = []
    min_deltas = []
    avg_deltas = []

    for data in test_data:
        delta_spo2, _, _ = data 
        max_deltas.append(np.max(delta_spo2))
        min_deltas.append(np.min(delta_spo2))
        avg_deltas.append(np.mean(delta_spo2))

    csv_dict['max_delta'] = max_deltas
    csv_dict['min_delta'] = min_deltas
    csv_dict['avg_delta'] = avg_deltas
    df = pd.DataFrame(csv_dict)
    csv_path = f'/home/hy381/model_training/eval/spo2_metrics_new.csv'
    df.to_csv(csv_path, index = False)


def visualise_spo2_range(): 
    csv_path = f'/home/hy381/model_training/eval/lf_prod_bilstm+mfcc_cnn_lstm.csv'
    model_results = pd.read_csv(csv_path)
    spo2_metrics = pd.read_csv('/home/hy381/model_training/eval/spo2_metrics_new.csv')

    spo2_only_incorrect = model_results.loc[(model_results['spo2_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) , 'filename'].values
    baseline_correct = model_results.loc[(model_results['spo2_model_correct'] == 1) & (model_results['fused_model_correct'] == 1) , 'filename'].values

    spo2_only_incorrect_range = spo2_metrics.loc[spo2_metrics['filename'].isin(spo2_only_incorrect), 'range'].values
    baseline_correct_range = spo2_metrics.loc[spo2_metrics['filename'].isin(baseline_correct), 'range'].values

    data_dict = {
        'Correct SpO2 Predictions': baseline_correct_range,
        'Fusion-Corrected SpO2 Predictions': spo2_only_incorrect_range
    }
    flierprops = dict(marker='o', markersize=4, markeredgecolor='gray')
    # maxsize = max(len(baseline_correct_range), len(spo2_only_incorrect_range))
    # baseline_correct_range = np.pad(baseline_correct_range, pad_width=(0,maxsize-len(baseline_correct_range),), mode='constant', constant_values=np.nan)
    # spo2_only_incorrect_range = np.pad(spo2_only_incorrect_range, pad_width=(0,maxsize-len(spo2_only_incorrect_range),), mode='constant', constant_values=np.nan)
    data = [baseline_correct_range, spo2_only_incorrect_range]
    plt.figure(figsize=(8,6))
    box = plt.boxplot(data, tick_labels = ['Correct SpO2 Predictions', 'Fusion-Corrected SpO2 Predictions'], patch_artist = True, flierprops= flierprops)
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')  # Box border color
        patch.set_linewidth(2)       # Border width

    # Customize whisker colors
    for whisker in box['whiskers']:
        whisker.set(color='black', linewidth=2)
    for cap in box['caps']:
        cap.set(color='black', linewidth=2)

    # Customize median line colors
    for median in box['medians']:
        median.set(color='firebrick', linewidth=3)

    plt.grid(True, zorder=0,linestyle=':', alpha = 0.8, axis = 'y')
    plt.ylabel("SpO2 Range (%)", labelpad=10)
    plt.ylim((-1, 35))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/home/hy381/model_training/eval/spo2_range.png')
    plt.close()


def main(): 
    visualise_spo2_range()
    # save_mfcc_to_np()
    # fused_results_to_csv('spo2_bilstm_model', 'audio_mfcc_cnn_lstm', 'prod')
    # save_mel_spec_metrics()
    # spo2_model = 'spo2_lstm_model'
    # audio_model = 'audio_ms_cnn'
    # fused_results_to_csv(spo2_model, audio_model, 'prod')
    # save_audio_snr()
    # fused_model_name = 'lf_prod_lstm+ms_cnn'
    # csv_path = f'/home/hy381/model_training/eval/{fused_model_name}.csv'
    # model_results = pd.read_csv(csv_path)
    # audio_only_correct = model_results.loc[(model_results['audio_model_correct'] == 1) & (model_results['spo2_model_correct'] == 0) & (model_results['fused_model_correct'] == 1) & (model_results['true_label'] == 2), 'filename'].values
    # visualize_audio_only(audio_only_correct)

    # spo2_only_correct = model_results.loc[(model_results['audio_model_correct'] == 0) & (model_results['fused_model_correct'] == 1)  & (model_results['true_label'] == 0), 'filename'].values
    # visualize_audio_only(spo2_only_correct)
    
    # np.savetxt(f'/home/hy381/model_training/eval/audio_only.txt', audio_only_correct, fmt='%s')
    # np.savetxt(f'/home/hy381/model_training/eval/spo2_only.txt', spo2_only_correct, fmt='%s')

    # analyse_snr_files('lf_prod_lstm+ms_cnn')
    # analyse_spectral_centroid_files('lf_prod_bilstm+mfcc_cnn_lstm', 0)
    # analyse_spectral_centroid_files('lf_prod_bilstm+mfcc_cnn_lstm', 1)
    # analyse_spectral_centroid_files('lf_prod_bilstm+mfcc_cnn_lstm', 2)

    # analyse_zcr_files('lf_prod_bilstm+mfcc_cnn_lstm', 0)
    # analyse_zcr_files('lf_prod_bilstm+mfcc_cnn_lstm', 1)
    # analyse_zcr_files('lf_prod_bilstm+mfcc_cnn_lstm', 2)

    # analyse_rms_files('lf_prod_bilstm+mfcc_cnn_lstm', 0)
    # analyse_rms_files('lf_prod_bilstm+mfcc_cnn_lstm', 1)
    # analyse_rms_files('lf_prod_bilstm+mfcc_cnn_lstm', 2)

    # analyse_snr_files('lf_prod_bilstm+mfcc_cnn_lstm', 0)
    # analyse_snr_files('lf_prod_bilstm+mfcc_cnn_lstm', 1)
    # analyse_snr_files('lf_prod_bilstm+mfcc_cnn_lstm', 2)
    # analyse_rms_files('lf_prod_lstm+ms_cnn')

if __name__ == '__main__':
    main()