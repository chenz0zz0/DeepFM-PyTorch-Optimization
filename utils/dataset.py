# -*- coding: utf-8 -*-
"""
Dataset module for DeepFM.
Original code inspired by: https://github.com/chenxijun1029/DeepFM_with_PyTorch
Modified by: [Your Name/GitHub ID]

Key Optimizations:
1. Log-Transformation: Applied to numerical features to handle heavy-tailed distributions 
   and stabilize gradients.
2. Stable Hashing: Used MD5 to ensure consistent feature mapping across different sessions.
3. Fixed Offset Strategy: Resolved Embedding IndexOutOfRange errors.
"""

import numpy as np
import pandas as pd
import torch
import hashlib
from torch.utils.data import Dataset

# Criteo Dataset Constants
CONT_FEATURES = 13  # I1 - I13
HASH_BUCKET_SIZE = 10000 

def stable_hash(x):
    """Generates a consistent hash value for categorical features using MD5."""
    return int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16) % HASH_BUCKET_SIZE

class CriteoDataset(Dataset):
    """
    Custom Dataset for Criteo Display Ads Challenge.
    Handles missing values, log-normalization, and feature indexing.
    """
    def __init__(self, root, train=True):
        self.root = root
        path = f"{root}/train.txt" if train else f"{root}/test.txt"
        self.data = pd.read_csv(path, sep='\t', header=None)
        
        # 1. Numerical Features: Log-Transformation (Standard practice in CTR prediction)
        for col in range(1, CONT_FEATURES + 1):
            self.data[col] = self.data[col].fillna(0).replace(-1, 0)
            # Log(x+1) compresses the range and helps prevent gradient explosion/large loss
            self.data[col] = self.data[col].apply(lambda x: np.log(x + 1.0) if x > 0 else 0)
        
        # 2. Categorical Features: Stable Hash Bucketing
        for col in range(CONT_FEATURES + 1, self.data.shape[1]):
            self.data[col] = self.data[col].fillna('0').astype(str)
            self.data[col] = self.data[col].apply(stable_hash)
        
        # 3. Feature Sizes: Defines the embedding table dimensions
        # Numerical fields take 1 slot (scalar value), Categorical fields take HASH_BUCKET_SIZE
        self.feature_sizes = [1] * CONT_FEATURES 
        self.feature_sizes += [HASH_BUCKET_SIZE] * (self.data.shape[1] - CONT_FEATURES - 1)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = torch.tensor(float(row[0]), dtype=torch.float32)
        features = row[1:].values
        
        xi = np.zeros(len(features), dtype=int)
        xv = np.zeros(len(features), dtype=float)
        
        # Numerical features: Fixed index 0, actual value passed via Xv
        xi[:CONT_FEATURES] = 0 
        xv[:CONT_FEATURES] = features[:CONT_FEATURES].astype(float)
        
        # Categorical features: Hashed index, value fixed to 1.0 (One-hot logic)
        xi[CONT_FEATURES:] = features[CONT_FEATURES:].astype(int)
        xv[CONT_FEATURES:] = 1.0
        
        return torch.tensor(xi, dtype=torch.long), torch.tensor(xv, dtype=torch.float32), label

    def __len__(self):
        return len(self.data)