"""
Configuration settings for SGD Classifier
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

# Column names
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# SGD-specific parameters
LOSS = 'log_loss'          # Equivalent to Logistic Regression
PENALTY = 'l2'             # Regularization
ALPHA = 0.0001             # Regularization strength
MAX_ITER = 1000            # Epochs
LEARNING_RATE = 'optimal'  # Adaptive learning rate
TOL = 1e-4                 # Stopping tolerance

# Model persistence
MODEL_DIR = "models"
MODEL_FILENAME = "sgd_classifier.pkl"
SCALER_FILENAME = "sgd_scaler.pkl"