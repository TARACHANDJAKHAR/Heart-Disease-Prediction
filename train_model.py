"""
Heart Disease Prediction Model Training Script

This script implements a machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset.
It combines multiple heart disease datasets, preprocesses the data, trains a Random Forest classifier,
evaluates its performance, and saves the trained model.

Dataset Source:
This implementation uses the UCI Heart Disease dataset, which combines data from multiple sources:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- University Hospital, Zurich
- VA Medical Center, Long Beach

Original Dataset Citation:
Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., 
Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new 
probability algorithm for the diagnosis of coronary artery disease. American Journal 
of Cardiology, 64(5), 304-310.

Authors:
Ishat Shivhare
[Group Member 2 Name]
[Group Member 3 Name]
[Group Member 4 Name]

Date: 2025-03-23
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime

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
    df_list = [pd.read_csv(os.path.join(data_dir, file), names=column_names, header=None) 
               for file in datasets]
    return pd.concat(df_list, ignore_index=True)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    This function performs the following cleaning steps:
    1. Replaces missing values (marked as '?') with NaN
    2. Converts all numeric columns to numeric type (except binary categorical variables)
    3. Removes rows with missing values
    4. Binarizes the target variable (0 for no heart disease, 1 for presence)

    Args:
        df (pd.DataFrame): Input DataFrame to clean

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for model training
    """
    # Display initial data types for verification
    print("\nInitial DataFrame Data Types:")
    print(df.dtypes)
    
    # Replace missing values
    df.replace('?', np.nan, inplace=True)
    
    # Define columns that should be numeric
    # Binary categorical variables are excluded
    numeric_columns = [
        "age", "cp", "trestbps", "chol", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    # Convert numeric columns with error handling
    for col in numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"\nConverted {col} column to numeric type")
            print(f"Unique values in {col}: {df[col].unique()}")
        except Exception as e:
            print(f"Error converting {col} to numeric: {str(e)}")
    
    # Display DataFrame info after conversion
    print("\nDataFrame Info After Conversion:")
    print(df.info())
    
    # Remove rows with missing values
    missing_before = df.isnull().sum()
    df.dropna(inplace=True)
    missing_after = df.isnull().sum()
    
    print("\nMissing Values Before and After Cleaning:")
    print("Before:")
    print(missing_before[missing_before > 0])
    print("\nAfter:")
    print(missing_after[missing_after > 0])
    
    # Binarize target variable
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

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
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(x_train: pd.DataFrame, y_train: pd.Series, 
                n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the provided data.

    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_estimators (int): Number of trees in the forest
        random_state (int): Random state for reproducibility

    Returns:
        RandomForestClassifier: Trained Random Forest model
    """
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(x_train, y_train)
    return rf_model

def evaluate_model(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the model's performance using various metrics.

    Args:
        model (RandomForestClassifier): Trained model to evaluate
        x_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model: RandomForestClassifier, filename: str) -> None:
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained model to save
        filename (str): Name of the file to save the model
    """
    joblib.dump(model, filename)
    print(f"\nModel saved successfully as '{filename}'")

def main():
    """Main execution function for the heart disease prediction model training pipeline."""
    # Configuration
    data_dir = "Processed Data"
    datasets = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data"
    ]
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    print("Starting Heart Disease Prediction Model Training Pipeline")
    print("=" * 50)

    # Load and combine datasets
    print("\nLoading and combining datasets...")
    df = load_and_combine_datasets(data_dir, datasets, column_names)
    
    # Clean the data
    print("\nCleaning and preprocessing the data...")
    df = clean_data(df)
    
    # Split the data
    print("\nSplitting data into training and test sets...")
    x_train, x_test, y_train, y_test = split_data(df)
    
    # Train the model
    print("\nTraining Random Forest model...")
    rf_model = train_model(x_train, y_train)
    
    # Evaluate the model
    evaluate_model(rf_model, x_test, y_test)
    
    # Save the model
    save_model(rf_model, "heart_disease_model.pkl")
    
    print("\nPipeline completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
