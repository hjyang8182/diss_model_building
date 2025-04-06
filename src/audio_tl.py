import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import sys
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.multiprocessing as mp
import glob 
import pandas as pd
from data import ApneaDataLoader
# use tl_env 
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, '/home/hy381/rds/hpc-work/OPERA')
from src.benchmark.model_util import extract_opera_feature
torch.cuda.empty_cache()

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

# def save_spectrograms():
# feature_dir = '/home/hy381/rds/hpc-work/opera_features'
# x_data = []
# train_files, valid_files, test_files = data_loader.split_train_valid_test_subset(train_length = 8000, valid_length = 1000, test_length = 1000)
# train_files, valid_files, test_files = np.vectorize(os.path.basename)(train_files), np.vectorize(os.path.basename)(valid_files), np.vectorize(os.path.basename)(test_files)
# train_files, valid_files, test_files = np.char.replace(train_files, '.tfrecord', '.wav'), np.char.replace(valid_files, '.tfrecord', '.wav'), np.char.replace(test_files, '.tfrecord', '.wav')

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
        print(len(files))
        feature_dir = '/home/hy381/rds/hpc-work/opera_features/'
        opera_features = extract_opera_feature(files,  pretrain="operaCT", input_sec=40, dim=768)
        # np.save(feature_dir + f"{data_type[i]}_operaCT_feature.npy", np.array(opera_features))
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

class CNNClassifier(nn.Module):
    def __init__(self, input_dim=768, output_dim=3):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)  # Output: (B, 32, 768)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: (B, 32, 384)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3)  # Output: (B, 64, 384)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Output: (B, 64, 192)

        # Compute FC input size dynamically
        self.fc_input_dim = 64 * 192
        self.fc = nn.Linear(self.fc_input_dim, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 768) - Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)  # (B, 32, 384)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)  # (B, 64, 192)

        x = x.view(x.size(0), -1)  # Flatten for FC
        x = self.fc(x)
        return x
    
class AudioDataset(Dataset):
    def __init__(self, data, labels): 
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        return x, label

class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        
        self.fc1 = nn.Linear(768, 512)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(512, 512)  # Second hidden layer
        self.fc3 = nn.Linear(512, 128)  # Output layer
        self.fc4 = nn.Linear(128, 128)  # Output layer
        # self.fc5 = nn.Linear(64, 16)  # Output layer
        self.fc6 = nn.Linear(128, 3)  # Output layer


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
   
        x = self.fc3(x)  # No activation for output (use softmax in loss if classification)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)

        # x = self.fc5(x)
        # x = self.relu(x)
        x = self.fc6(x)

        return x
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model parameters
    input_dim = 768
    output_dim = 3

    # Create the model
    # model = CNNClassifier()

    from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
    from src.model.models_eval import AudioClassifier


    pretrained_model = initialize_pretrained_model("operaCT")

    encoder_path = get_encoder_path("operaCT")
    print("loading weights from", encoder_path)
    ckpt = torch.load(encoder_path, map_location=device)
    pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)


    net = pretrained_model.encoder
    
    # add fully connected layers

    class CustomHead(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(CustomHead, self).__init__()
            self.fc_layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)  # Final layer for classification
            )

        def forward(self, x):
            return self.fc_layers(x)

    model = AudioClassifier(net=net, head='mlp', classes=3, freeze_encoder="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    wav2vec_feature_dir = '/home/hy381/rds/hpc-work/opera_features/'

    train_features = np.load(wav2vec_feature_dir + 'train_spectrogram.npy')
    train_labels = np.load(wav2vec_feature_dir + 'train_labels.npy')

    valid_features = np.load(wav2vec_feature_dir + 'valid_spectrogram.npy')
    valid_labels = np.load(wav2vec_feature_dir + 'valid_labels.npy')

    test_features = np.load(wav2vec_feature_dir + 'test_spectrogram.npy')
    test_labels = np.load(wav2vec_feature_dir + 'test_labels.npy')

    train_data = AudioDataset(train_features, train_labels)
    valid_data = AudioDataset(valid_features, valid_labels)
    test_data = AudioDataset(test_features, test_labels)

    train_loader = DataLoader(train_data, batch_size = 128,shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = 128, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = 128, shuffle = False)

    num_epochs = 200
    num_training_steps = num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(num_epochs): 
        model.train()
        total_loss = 0
        correct_train = 0 
        total_train = 0 
        print(f"Starting training for {epoch + 1}")
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) # Calculate loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            preds = torch.argmax(outputs, dim=1)  # Predicted class
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            progress_bar.update(1)

        train_acc = correct_train / total_train
        avg_train_loss = total_loss / len(train_loader)

        # Validation Phase
        model.eval()  # Set model to evaluation mode
        correct_val, total_val = 0, 0
        val_loss = 0

        with torch.no_grad():  # No gradient calculation
            for val_inputs, val_labels in valid_loader:
                val_inputs, val_labels = val_inputs.to(device).float(), val_labels.to(device)
                val_outputs = model(val_inputs)

                loss = criterion(val_outputs, val_labels) # Calculate loss
                val_loss += loss.item()

                preds = torch.argmax(val_outputs, dim=1)

                correct_val += (preds == val_labels).sum().item()
                total_val += val_labels.size(0)

        val_acc = correct_val / total_val
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    torch.save(model.state_dict(), 'finetuned_model_weights.pth')

if __name__ == '__main__':
    try:
        # Set the start method to 'spawn'
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # If the start method is already set, just pass
        pass
    main()