"""
Configuration for Logistic Regression and SGD
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

# Column names
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
MODEL_DIR = "models"

# Logistic Regression Config
LOGISTIC_FILENAME = "logistic_model.pkl"
LOGISTIC_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "max_iter": [100, 200, 300]
}

# SGD Config
SGD_FILENAME = "sgd_model.pkl"
SGD_PARAMS = {
    "alpha": [0.0001, 0.001, 0.01],
    "penalty": ["l1", "l2", "elasticnet"],
    "learning_rate": ["constant", "optimal", "invscaling"],
    "max_iter": [1000, 2000]
}
MODEL_FILENAME = "heart_disease_model.pkl" 
MODEL_FILENAME_DT = "decision_tree_model.pkl"
