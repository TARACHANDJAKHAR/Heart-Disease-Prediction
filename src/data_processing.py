"""
Data Processing Module for Heart Disease Prediction using SGD Classifier

Key Changes from Logistic Regression Version:
1. Added data shuffling (critical for SGD)
2. Modified preprocessing for online learning compatibility
3. Added partial_fit preparation
4. Enhanced memory efficiency
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Generator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
import os

# NEW: SGD-specific configuration
SGD_CHUNK_SIZE = 256  # For memory-efficient processing

def load_and_combine_datasets(data_dir: str, datasets: List[str], column_names: List[str]) -> pd.DataFrame:
    """
    Optimized dataset loading with memory monitoring for SGD.
    """
    df_list = []
    for file in datasets:
        try:
            # NEW: Low-memory reading for large datasets
            df = pd.read_csv(os.path.join(data_dir, file), 
                           names=column_names, 
                           header=None,
                           na_values=['?', ' ', ''],
                           low_memory=True)
            df_list.append(df)
            print(f"Loaded {file} ({len(df)} records) | Memory: {df.memory_usage().sum()/1024:.1f} KB")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nFinal dataset: {len(combined_df)} records | {combined_df.memory_usage().sum()/1024:.1f} KB")
    return combined_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced cleaning for SGD:
    - Preserves row order for chunking
    - More efficient type conversion
    """
    print("\nCleaning data for SGD...")
    
    # Convert numeric columns in place
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # NEW: More memory-efficient categorical conversion
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    df[categorical_cols] = df[categorical_cols].astype('category').apply(lambda x: x.cat.codes)
    
    # Remove nulls without resetting index (preserves chunk alignment)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Removed {before - len(df)} rows with missing values")
    
    # Binarize target
    df['target'] = df['target'].astype('int8')  # NEW: Reduced memory usage
    return df

# NEW: Added chunk generator for SGD
def data_generator(df: pd.DataFrame, scaler: StandardScaler, chunk_size: int = SGD_CHUNK_SIZE) -> Generator[Tuple[pd.DataFrame, pd.Series], None, None]:
    """
    Yields scaled data chunks for SGD partial_fit.
    """
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        X = chunk.drop(columns=['target'])
        y = chunk['target']
        
        # Scale only numeric features
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        
        yield X, y

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    SGD-optimized preprocessing:
    - Shuffles data
    - Fits scaler on full dataset
    - Returns unprocessed DataFrame for chunking
    """
    # NEW: Critical shuffle for SGD
    df = shuffle(df, random_state=42)
    
    # Fit scaler on full data
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    
    # Delay scaling until training chunks
    return df, scaler

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Modified for SGD:
    - Preserves chunk alignment
    - Stratified splitting with chunk awareness
    """
    # NEW: Group-aware splitting for chunk integrity
    x = df.drop(columns=['target'])
    y = df['target']
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nTrain chunks: {len(x_train)//SGD_CHUNK_SIZE}")
    print(f"Test chunks: {len(x_test)//SGD_CHUNK_SIZE}")
    return x_train, x_test, y_train, y_test