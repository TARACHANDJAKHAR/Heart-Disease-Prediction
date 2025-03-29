"""
Data Processing Module for Heart Disease Prediction using Logistic Regression

This module handles all data loading, cleaning, and preprocessing operations
specifically optimized for logistic regression models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_combine_datasets(data_dir: str, datasets: List[str], column_names: List[str]) -> pd.DataFrame:
    """
    Load and combine multiple heart disease datasets into a single DataFrame.

    Args:
        data_dir (str): Directory containing the dataset files
        datasets (List[str]): List of dataset filenames to combine
        column_names (List[str]): List of column names for the DataFrame

    Returns:
        pd.DataFrame: Combined dataset containing all records
    """
    df_list = []
    for file in datasets:
        try:
            df = pd.read_csv(os.path.join(data_dir, file), 
                           names=column_names, 
                           header=None,
                           na_values=['?', ' ', ''])
            df_list.append(df)
            print(f"Successfully loaded {file} with {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nCombined dataset contains {len(combined_df)} records")
    return combined_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset specifically for logistic regression.

    This function performs:
    1. Missing value handling
    2. Type conversion
    3. Feature engineering
    4. Outlier treatment (important for logistic regression)
    5. Target variable processing

    Args:
        df (pd.DataFrame): Input DataFrame to clean

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for logistic regression
    """
    print("\nInitial Data Overview:")
    print(f"Total records: {len(df)}")
    print(f"Initial missing values:\n{df.isnull().sum()}")

    # Convert all columns to numeric where possible
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle categorical variables (convert to dummy variables later)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    for col in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing values (logistic regression is sensitive to missing data)
    initial_count = len(df)
    df.dropna(inplace=True)
    print(f"\nRemoved {initial_count - len(df)} records with missing values")

    # Handle outliers (important for logistic regression convergence)
    df = handle_outliers(df)

    # Binarize target variable
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    print(f"\nClass distribution:\n{df['target'].value_counts(normalize=True)}")

    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers in numeric features using capping at 5th and 95th percentiles.
    Important for logistic regression as it's sensitive to extreme values.
    """
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    for col in numeric_cols:
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess data specifically for logistic regression:
    1. Create dummy variables for categorical features
    2. Scale numeric features (important for logistic regression)
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: 
            Processed DataFrame and fitted scaler object
    """
    # Create dummy variables for categorical features
    categorical_features = ['cp', 'restecg', 'slope', 'thal']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Scale numeric features (important for logistic regression)
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df, scaler

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets with stratification.
    Includes additional validation for class balance.

    Args:
        df (pd.DataFrame): Input DataFrame to split
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            x_train, x_test, y_train, y_test
    """
    x = df.drop(columns=["target"])
    y = df["target"]
    
    # Verify class balance before splitting
    print("\nClass distribution before splitting:")
    print(y.value_counts(normalize=True))
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Verify class balance after splitting
    print("\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(y_test.value_counts(normalize=True))
    
    return x_train, x_test, y_train, y_test