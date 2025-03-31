"""
Configuration for Heart Disease Prediction Models
"""

import os

# Data directories
DATA_DIR = os.path.join("data", "processed")
RAW_DATA_DIR = os.path.join("data", "raw")
IMAGE_DATA_DIR = os.path.join("data", "images")
EDA_DIR = "EDA_Reports"

# Dataset files
DATASETS = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
    "processed.new.data"
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
BEST_MODEL_FILENAME = "best_model.pkl"

# Model filenames
MODEL_FILENAMES = {
    "random_forest": "rf_model.pkl",
    "svm": "svm_model.pkl",
    "knn": "knn_model.pkl",
    "logistic": "logistic_model.pkl",
    "sgd": "sgd_model.pkl",
    "decision_tree": "decision_tree_model.pkl"
}

# Cross-validation settings
CV_SPLITS = 5
SCORING_METRICS = {
    "accuracy": "accuracy",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy"
}

# Random Forest Config
RF_PARAMS = {
    "rf__n_estimators": [50, 100, 200, 500],
    "rf__max_depth": [3, 5, 10, 20, None],
    "rf__min_samples_split": [2, 5, 10, 20],
    "rf__min_samples_leaf": [1, 2, 4, 8],
    "rf__max_features": ["sqrt", "log2"]
}

# SVM Config
SVM_PARAMS = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svc__class_weight': [None, 'balanced']
}

# KNN Config
KNN_PARAMS = {
    'knn__n_neighbors': list(range(3, 21, 2)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Logistic Regression Config
LOGISTIC_PARAMS = {
    'logisticregression__C': [0.1, 1.0, 10.0, 100.0, 1000.0],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear', 'saga'],
    'logisticregression__max_iter': [100, 200, 300, 1000],
}

# SGD Config
SGD_PARAMS = {
    'sgdclassifier__loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
    'sgdclassifier__alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'sgdclassifier__penalty': ['l1', 'l2', 'elasticnet', None],
    'sgdclassifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'sgdclassifier__eta0': [0.0001, 0.001, 0.01],
    'sgdclassifier__max_iter': [1000, 2000, 3000],
}

# Decision Tree Config
DT_PARAMS = {
    'decisiontreeclassifier__max_depth': [3, 5, 10, 20, None, 30, 50],
    'decisiontreeclassifier__min_samples_split': [2, 5, 10, 20, 50],
    'decisiontreeclassifier__min_samples_leaf': [1, 2, 4, 8, 16],
    'decisiontreeclassifier__max_features': ['sqrt', 'log2', None],
    'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
    'decisiontreeclassifier__splitter': ['best', 'random'],
    'decisiontreeclassifier__class_weight': ['balanced', None]
}

# Model comparison settings
COMPARISON_METRICS = ["accuracy", "f1_score", "training_time"]
BEST_MODEL_SELECTION_METRIC = "f1_score"
