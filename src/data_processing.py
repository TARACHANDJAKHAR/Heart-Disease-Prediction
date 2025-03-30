"""
Data Processing Module
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_combine_datasets(data_dir: str, datasets: List[str], column_names: List[str]) -> pd.DataFrame:
    """Load and combine datasets"""
    df_list = [pd.read_csv(os.path.join(data_dir, file), names=column_names, header=None) 
               for file in datasets]
    return pd.concat(df_list, ignore_index=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data"""
    df.replace('?', np.nan, inplace=True)
    numeric_columns = ["age", "cp", "trestbps", "chol", "restecg", 
                      "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df

def split_data_logistic(df: pd.DataFrame, test_size: float, random_state: int):
    """Split data for Logistic Regression (no scaling)"""
    x = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

def split_data_sgd(df: pd.DataFrame, test_size: float, random_state: int):
    """Split data for SGD (with scaling)"""
    x = df.drop(columns=["target"])
    y = df["target"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (x_train, x_test, y_train, y_test), scaler