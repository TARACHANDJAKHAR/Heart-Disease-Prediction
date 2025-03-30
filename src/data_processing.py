"""
Data Processing Module for Heart Disease Prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import os

def load_and_combine_datasets(data_dir: str, datasets: List[str], column_names: List[str]) -> pd.DataFrame:
    """
    Load and combine multiple heart disease datasets into a single DataFrame.
    """
    df_list = [pd.read_csv(os.path.join(data_dir, file), names=column_names, header=None) 
               for file in datasets]
    return pd.concat(df_list, ignore_index=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.
    """
    # Replace missing values
    df.replace('?', np.nan, inplace=True)
    
    # Convert numeric columns
    numeric_columns = ["age", "cp", "trestbps", "chol", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Binarize target
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    """
    x = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)