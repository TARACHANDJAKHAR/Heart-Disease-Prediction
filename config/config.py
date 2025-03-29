"""
Configuration settings for the project.
"""

import os # This is inbuilt python's module to handle the file and directory's paths

# Data directories
DATA_DIR = os.path.join("data", "processed") # Path to the processed data
RAW_DATA_DIR = os.path.join("data", "raw") # Path to the raw data
# os.path.join used for the platform independecies , after this use we can use the data for both the windows and linux

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

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

#  Logistic Regression specific parameters
MAX_ITER = 1000           # Maximum number of iterations for convergence
C = 1.0                   # Inverse of regularization strength
SOLVER = 'lbfgs'          # Algorithm to use in optimization problem
PENALTY = 'l2'            # Regularization penalty type
CLASS_WEIGHT = None       # Weights associated with classes

# Model directory and filename
MODEL_DIR = "models" #  Directory for saving the trained model
MODEL_FILENAME = "heart_disease_model.pkl" # Saving the trained model by the name and the .pkl ext.


