"""
Data Loader Module for Hugging Face Dataset

This module provides functions to load datasets from Hugging Face Hub
for the Dice Challenge 2025 autocomplete project.
"""

import pandas as pd
from datasets import load_dataset

def load_train_data():
    """
    Load training data from Hugging Face.
    
    Returns:
        pd.DataFrame: Training data with prefix-query pairs
    """
    print("   Loading train data from Hugging Face...")
    train_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="train_data/*.parquet")
    return train_data['train'].to_pandas()

def load_test_prefixes():
    """
    Load test prefixes from Hugging Face.
    
    Returns:
        pd.DataFrame: Test prefixes
    """
    print("   Loading test prefixes from Hugging Face...")
    test_prefixes_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="test_prefixes_data/*.parquet")
    return test_prefixes_data['train'].to_pandas()

def load_query_features():
    """
    Load query features from Hugging Face.
    
    Returns:
        pd.DataFrame: Query features with popularity metrics
    """
    print("   Loading query features from Hugging Face...")
    query_features_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="query_features/*.parquet")
    return query_features_data['train'].to_pandas()

def load_query_pool():
    """
    Load query pool from Hugging Face.
    
    Returns:
        pd.DataFrame: Query pool with all candidate queries
    """
    print("   Loading query pool from Hugging Face...")
    pool_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="pool/*.parquet")
    return pool_data['train'].to_pandas()

def load_all_datasets():
    """
    Load all datasets from Hugging Face.
    
    Returns:
        tuple: (train_df, test_prefixes_df, query_features_df, query_pool_df)
    """
    print("Loading all datasets from Hugging Face...")
    
    train_df = load_train_data()
    test_prefixes_df = load_test_prefixes()
    query_features_df = load_query_features()
    query_pool_df = load_query_pool()
    
    print("✓ All datasets loaded successfully from Hugging Face!\n")
    
    return train_df, test_prefixes_df, query_features_df, query_pool_df

if __name__ == "__main__":
    # Test the data loader
    print("="*80)
    print("TESTING DATA LOADER")
    print("="*80)
    
    train_df, test_prefixes_df, query_features_df, query_pool_df = load_all_datasets()
    
    print(f"\n✓ Train data shape: {train_df.shape}")
    print(f"✓ Test prefixes shape: {test_prefixes_df.shape}")
    print(f"✓ Query features shape: {query_features_df.shape}")
    print(f"✓ Query pool shape: {query_pool_df.shape}")
    
    print("\n✓ Data loader working correctly!")
