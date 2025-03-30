"""
Configuration settings for the project.
"""

import os

# Data directories
DATA_DIR = os.path.join("data", "processed")
RAW_DATA_DIR = os.path.join("data", "raw")

# Dataset files
DATASETS = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data"
]

# Column names for the dataset
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Model parameters (Updated for Logistic Regression)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model directory and filename
MODEL_DIR = "models"
MODEL_FILENAME = "logistic_regression_model.pkl"

# Logistic Regression Hyperparameters
LOGISTIC_PARAMS = {
    "C": [0.1, 1.0, 10.0],       # Inverse regularization strength
    "penalty": ["l1", "l2"],      # Regularization type
    "solver": ["liblinear"],      # Solver for small datasets
    "max_iter": [100, 200, 300]   # Maximum iterations
}