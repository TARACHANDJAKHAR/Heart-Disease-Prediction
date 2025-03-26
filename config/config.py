"""
Configuration settings for the project.
"""

import os

# Data directories
DATA_DIR = os.path.join("data", "processed")
RAW_DATA_DIR = os.path.join("data", "raw")
IMAGE_DATA_DIR = os.path.join("data", "images")

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

# Image processing settings
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff']
IMAGE_PREPROCESSING = {
    'resize': (224, 224),
    'normalize': True
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model directory and filename
MODEL_DIR = "models"
MODEL_FILENAME = "heart_disease_model.pkl" 