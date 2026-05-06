# -*- coding: utf-8 -*-
"""
Training Entrance for DeepFM.
Run this script to start the training and evaluation pipeline.
"""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from utils.dataset import CriteoDataset
from model.DeepFM import DeepFM

# --- Hyperparameters Configuration ---
NUM_TRAIN = 5000
NUM_VAL = 500
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

def main():
    print("Step 1: Loading Data with Log Normalization...")
    # Dataset will handle Log-transformation and stable hashing internally
    dataset = CriteoDataset('./data', train=True)
    
    indices = list(range(len(dataset)))
    train_idx = indices[:NUM_TRAIN]
    val_idx = indices[NUM_TRAIN : NUM_TRAIN + NUM_VAL]
    
    loader_train = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    loader_val = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))

    print("Step 2: Initializing DeepFM Model...")
    model = DeepFM(
        feature_sizes=dataset.feature_sizes,
        embedding_size=8,
        use_cuda=False  # Recommended False for Mac unless MPS is configured
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Step 3: Start Training Loop...")
    model.fit(loader_train, loader_val, optimizer, epochs=EPOCHS)

if __name__ == "__main__":
    main()