import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ECGPPGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data, abp_labels):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data
        self.abp_labels = abp_labels

    def __len__(self):
        return len(self.abp_labels)

    def __getitem__(self, idx):
        ecg_sample = torch.FloatTensor(self.ecg_data[idx])
        ppg_sample = torch.FloatTensor(self.ppg_data[idx])
        abp_label = torch.FloatTensor([self.abp_labels[idx]])

        return {'ecg': ecg_sample, 'ppg': ppg_sample, 'abp': abp_label}
    
def get_datasets(dataset_path):
    '''Dataset Preparation'''
    # Listing the files in the dataset directory
    files = os.listdir(dataset_path)

    # Printing the file paths
    for file in files:
        file_path = os.path.join(dataset_path, file)
        print(file_path)

    #considering files with .npy extension
    npy_files = [file for file in files if file.endswith('.npy')]

    # Load ECG train data from the .npy file
    ecg_train_data = np.load(dataset_path + 'ecg_train_10sec.npy')
    ecg_test_data = np.load(dataset_path + 'ecg_test_10sec.npy')

    ppg_train_data = np.load(dataset_path + 'ppg_train_10sec.npy')
    ppg_test_data = np.load(dataset_path + 'ppg_test_10sec.npy')

    labels_train_data = np.load(dataset_path + 'labels_train_10sec.npy')
    labels_test_data = np.load(dataset_path + 'labels_test_10sec.npy')

    train_dataset = ECGPPGDataset(ecg_train_data, ppg_train_data, labels_train_data)
    test_dataset = ECGPPGDataset(ecg_test_data, ppg_test_data, labels_test_data)
    
    return train_dataset, test_dataset