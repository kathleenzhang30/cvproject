import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from google.colab import drive
drive.mount('/content/drive')

dir_path = "/content/drive/MyDrive/CT-RATE_NIfTI"
folders = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
# num_folders = len(folders)
# print(f"Number of folders: {num_folders}")

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#pip install iterative-stratification

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

base_dir = "/content/drive/MyDrive/CT-RATE_NIfTI"

train_csv_path = os.path.join(base_dir, "train_labels.csv")
val_csv_path = os.path.join(base_dir, "validation_labels.csv")
train_root = os.path.join(base_dir, "dataset/train")
val_root = os.path.join(base_dir, "dataset/valid")

def get_nii_gz_files(directory):
    nii_gz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_gz_files.append(os.path.join(root, file))
    return nii_gz_files

train_files = get_nii_gz_files(train_root)
val_files = get_nii_gz_files(val_root)

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

train_file_map = {os.path.basename(p): p for p in train_files}
val_file_map = {os.path.basename(p): p for p in val_files}

train_df['filepath'] = train_df['VolumeName'].map(train_file_map)
val_df['filepath'] = val_df['VolumeName'].map(val_file_map)

train_df = train_df.dropna(subset=['filepath'])
val_df = val_df.dropna(subset=['filepath'])

label_columns = [col for col in train_df.columns if col not in ['VolumeName', 'filepath']]

X_train = train_df['filepath'].values
y_train = train_df[label_columns].values

X_val = val_df['filepath'].values
y_val = val_df[label_columns].values

print(f"Matched train samples: {len(X_train)}, label shape: {y_train.shape}")
print(f"Matched val samples: {len(X_val)}, label shape: {y_val.shape}")

def stratified_sample(df, class_names, max_samples_per_class=200):
    sampled_indices = []
    for cls in class_names:
        indices = df[df[cls] == 1].index
        sampled = np.random.choice(indices, min(len(indices), max_samples_per_class), replace=False)
        sampled_indices.extend(sampled)
    return df.loc[list(set(sampled_indices))].reset_index(drop=True)

df_balanced = stratified_sample(train_df, label_columns, max_samples_per_class=200)

df_train, df_val = train_test_split(df_balanced, test_size=0.2, random_state=42)

X_train = df_train['filepath'].values
y_train = df_train[label_columns].values

X_val = df_val['filepath'].values
y_val = df_val[label_columns].values

# print(f"Balanced train samples: {len(X_train)}, label shape: {y_train.shape}")
# print(f"Balanced val samples: {len(X_val)}, label shape: {y_val.shape}")




