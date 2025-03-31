"""
Data Processing Module
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from autoviz.AutoViz_Class import AutoViz_Class
from .image_processing import process_medical_image, combine_image_and_tabular_data


def load_and_combine_datasets(
    data_dir: str, datasets: List[str], column_names: List[str]
) -> pd.DataFrame:
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
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Warning: File {file} not found in {data_dir}")
            continue
            
        try:
            # Read the files and handle missing values
            df = pd.read_csv(
                file_path,
                names=column_names,
                header=None,
                sep=',',
                na_values=['?']
            )
            print(f"Successfully loaded {file} with {len(df)} rows")
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    if not df_list:
        raise ValueError("No datasets were successfully loaded")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print("\nDataset Info:")
    print(combined_df.info())
    print("\nMissing Values:")
    print(combined_df.isnull().sum())
    return combined_df

def process_image_data(image_dir: str, patient_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Process medical report images for a list of patients.

    Args:
        image_dir (str): Directory containing medical report images
        patient_ids (List[str]): List of patient IDs to process

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping patient IDs to extracted measurements
    """
    image_data = {}
    
    for patient_id in patient_ids:
        # Look for image files with patient ID in the filename
        for file in os.listdir(image_dir):
            if patient_id in file and any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff']):
                try:
                    image_path = os.path.join(image_dir, file)
                    measurements = process_medical_image(image_path)
                    image_data[patient_id] = measurements
                except Exception as e:
                    print(f"Error processing image for patient {patient_id}: {str(e)}")
    
    return image_data

def clean_data(df: pd.DataFrame, image_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Clean and preprocess the dataset, incorporating image data if available.

    Args:
        df (pd.DataFrame): Input DataFrame to clean
        image_dir (str, optional): Directory containing medical report images

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for model training
    """
    
    # Define columns that should be numeric
    numeric_columns = [
        "age", "cp", "trestbps", "chol", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    # Convert numeric columns with error handling
    for col in numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"\nConverted {col} column to numeric type")
            print(f"Unique values in {col}: {df[col].unique()}")
        except Exception as e:
            print(f"Error converting {col} to numeric: {str(e)}")
    
    # Process image data if available
    if image_dir and os.path.exists(image_dir):
        print("\nProcessing medical report images...")
        patient_ids = df.index.astype(str).tolist()
        image_data = process_image_data(image_dir, patient_ids)
        
        # Combine image data with tabular data
        for patient_id, measurements in image_data.items():
            if patient_id in df.index:
                row_data = df.loc[patient_id].to_dict()
                combined_data = combine_image_and_tabular_data(measurements, row_data)
                for key, value in combined_data.items():
                    if key in df.columns:
                        df.loc[patient_id, key] = value
    
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
    
    print("\nFinal DataFrame Info:")
    print(df.info())
    
    return df

def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    
    # Print class distribution
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("\nSplit Results:")
    print(f"Training set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test

def perform_eda(df: pd.DataFrame, target_col: str = "target", save_dir: str = "EDA_Reports"):
    """
    Perform automated EDA with image outputs.
    Creates directory if it doesn't exist.
    """
    print("\nPerforming Exploratory Data Analysis...")
    
    # Create directory if missing
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Initialize AutoViz
        AV = AutoViz_Class()
        
        # Generate EDA report
        dft = AV.AutoViz(
            filename="",
            sep=",",
            depVar=target_col,
            dfte=df,
            header=0,
            verbose=0,  # Reduce verbosity
            lowess=False,
            chart_format="png",
            max_rows_analyzed=150000,
            max_cols_analyzed=30,
            save_plot_dir=save_dir
        )
        
        print(f"\nEDA reports saved to {save_dir}")
    except Exception as e:
        print(f"Error during EDA: {str(e)}")
        print("Continuing with model training...")
