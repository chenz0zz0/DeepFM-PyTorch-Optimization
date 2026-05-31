# -*- coding: utf-8 -*-
"""
Data Preprocessing Script for Criteo Dataset.
Original code inspired by: https://github.com/chenxijun1029/DeepFM_with_PyTorch
Modified by: chenz0zz0

Purpose: 
This script performs initial data cleaning, such as handling missing values 
for raw Criteo data files. 
"""

import pandas as pd
import os

# Criteo Dataset feature counts
CONT_FEATURES = 13  # I1-I13: Numerical features
CAT_FEATURES = 26   # C1-C26: Categorical features

def preprocess(input_path, output_path):
    """
    Reads raw .txt files, fills missing values, and saves them in a standardized format.
    
    Args:
        input_path (str): Directory containing raw train.txt and test.txt.
        output_path (str): Directory to save cleaned files.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} does not exist.")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in ['train.txt', 'test.txt']:
        full_input_path = os.path.join(input_path, file_name)
        if not os.path.exists(full_input_path):
            print(f"Warning: {file_name} not found in {input_path}. Skipping.")
            continue
            
        print(f"Processing {file_name}...")
        
        # Load raw data (Criteo raw data uses tab separators and no headers)
        data = pd.read_csv(full_input_path, sep='\t', header=None)
        
        # 1. Handle Numerical Features: Fill missing with 0
        # In industry, sometimes median or mean is used, but 0 is a standard baseline for Criteo.
        for col in range(1, CONT_FEATURES + 1):
            data[col] = data[col].fillna(0)
            
        # 2. Handle Categorical Features: Fill missing with a placeholder string "0"
        for col in range(CONT_FEATURES + 1, CONT_FEATURES + CAT_FEATURES + 1):
            data[col] = data[col].fillna("0")
            
        # Save the cleaned data
        output_file = os.path.join(output_path, file_name)
        data.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"Successfully saved processed file to: {output_file}")

if __name__ == "__main__":
    # Standard project paths
    # Note: If your downloaded data is already in ./data, this script serves as a logic reference.
    raw_data_dir = './data'       
    processed_data_dir = './data' 
    
    preprocess(raw_data_dir, processed_data_dir)
