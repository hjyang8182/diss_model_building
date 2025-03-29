import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import sys
import os
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.multiprocessing as mp
import glob 

# use tl_env 
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, '/home/hy381/hpc-work/OPERA')

torch.cuda.empty_cache()

# def save_spectrograms():
# feature_dir = '/home/hy381/rds/hpc-work/opera_features'
# from src.util import get_split_signal_librosa
# x_data = []
# train_files, valid_files, test_files = data_loader.split_train_valid_test_subset(train_length = 8000, valid_length = 1000, test_length = 1000)
# train_files, valid_files, test_files = np.vectorize(os.path.basename)(train_files), np.vectorize(os.path.basename)(valid_files), np.vectorize(os.path.basename)(test_files)
# train_files, valid_files, test_files = np.char.replace(train_files, '.tfrecord', '.wav'), np.char.replace(valid_files, '.tfrecord', '.wav'), np.char.replace(test_files, '.tfrecord', '.wav')

data_loader = DataLoader()
os.chdir('/home/hy381/rds/hpc-work/audio_wav')
train_files, valid_files, test_files = data_loader.split_train_valid_test()
train_files, valid_files, test_files = np.vectorize(os.path.basename)(train_files), np.vectorize(os.path.basename)(valid_files), np.vectorize(os.path.basename)(test_files)
train_files, valid_files, test_files = np.char.replace(train_files, '.tfrecord', '.wav'), np.char.replace(valid_files, '.tfrecord', '.wav'), np.char.replace(test_files, '.tfrecord', '.wav')
print(len(train_files))
all_files = [train_files, valid_files, test_files]
data_type = ['train', 'valid', 'test']
for i in range(3): 
    files = all_files[i]
    feature_dir = '/home/hy381/rds/hpc-work/opera_features'
    opera_features = extract_opera_feature(files,  pretrain="operaCT", input_sec=40, dim=768)
    np.save(feature_dir + f"{data_type[i]}_operaCT_feature.npy", np.array(opera_features))

# class Simple3LayerCNN(pl.LightningModule):
#     def __init__(self, n_cls=2):
#         super(Simple3LayerCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(2, 2)

#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.fc1 = nn.Linear(128, 64)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(64, n_cls)

#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.pool2(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.pool3(x)

#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, 1)

#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x
    
#     def training_step(self, batch, batch_idx): 
#         inputs, targets = batch 
#         true_labels = torch.argmax(labels, dim=1)
#         outputs = self(inputs)
#         loss = self.criterion(outputs, true_labels)
    
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=0.001)

# class LinearModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)

# class AudioDataset(Dataset):
#     def __init__(self, data, labels): 
#         self.data = data
#         self.labels = labels
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         label = self.labels[idx]
#         return x, label

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Model parameters
#     input_dim = 768
#     output_dim = 3

#     # Create the model
#     model = LinearModel(input_dim, output_dim)
#     model = model.to(device)
#     learning_rate = 0.001
#     epochs = 16
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     wav2vec_feature_dir = '/home/hy381/rds/hpc-work/opera_features/'

#     train_features = np.load(wav2vec_feature_dir + 'train_operaCT_feature.npy')
#     train_labels = np.load(wav2vec_feature_dir + 'train_labels.npy')

#     valid_features = np.load(wav2vec_feature_dir + 'valid_operaCT_feature.npy')
#     valid_labels = np.load(wav2vec_feature_dir + 'valid_labels.npy')

#     test_features = np.load(wav2vec_feature_dir + 'test_operaCT_feature.npy')
#     test_labels = np.load(wav2vec_feature_dir + 'test_labels.npy')

#     train_data = AudioDataset(train_features, train_labels)
#     valid_data = AudioDataset(valid_features, valid_labels)
#     test_data = AudioDataset(test_features, test_labels)

#     train_loader = DataLoader(train_data, batch_size = 32, num_workers = 2,shuffle = True)
#     valid_loader = DataLoader(valid_data, batch_size = 32, num_workers = 2, shuffle = False)
#     test_loader = DataLoader(test_data, batch_size = 32, num_workers = 2, shuffle = False)

#     num_epochs = 50
#     num_training_steps = num_epochs * len(train_loader)
#     progress_bar = tqdm(range(num_training_steps))
    
#     total_loss = 0
#     val_loss = 0
#     correct_train = 0 
#     total_train = 0 
#     correct_val  = 0 
#     total_val = 0
#     for epoch in range(num_epochs): 
#         print(f"Starting training for {epoch + 1}")
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device).float()
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels) # Calculate loss
#             total_loss += loss.item()

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#             preds = torch.argmax(outputs, dim=1)  # Predicted class
#             correct_train += (preds == labels).sum().item()
#             total_train += labels.size(0)
            
#             progress_bar.update(1)

#         train_acc = correct_train / total_train
#         avg_train_loss = total_loss / len(train_loader)

#         # Validation Phase
#         model.eval()  # Set model to evaluation mode
#         correct_val, total_val = 0, 0
#         val_loss = 0

#         with torch.no_grad():  # No gradient calculation
#             for val_inputs, val_labels in valid_loader:
#                 val_inputs, val_labels = val_inputs.to(device).float(), val_labels.to(device)
#                 val_outputs = model(val_inputs)

#                 loss = criterion(val_outputs, val_labels) # Calculate loss
#                 val_loss += loss.item()

#                 preds = torch.argmax(val_outputs, dim=1)

#                 correct_val += (preds == val_labels).sum().item()
#                 total_val += val_labels.size(0)

#         val_acc = correct_val / total_val
#         avg_val_loss = val_loss / len(valid_loader)
#         print(f"Epoch {epoch+1}/{num_epochs}: "
#             f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
#             f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
#     torch.save(model.state_dict(), 'finetuned_model_weights.pth')

# if __name__ == '__main__':
#     try:
#         # Set the start method to 'spawn'
#         mp.set_start_method('spawn', force=True)
#     except RuntimeError:
#         # If the start method is already set, just pass
#         pass
    
    # Call your main training function
    # main() 
# 