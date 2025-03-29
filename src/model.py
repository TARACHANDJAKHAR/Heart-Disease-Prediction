"""
Model Training and Evaluation Module for Heart Disease Prediction (Logistic Regression Version)

This module handles model training, hyperparameter tuning, and evaluation
specifically optimized for logistic regression.
"""

import pandas as pd
from typing import Tuple
from sklearn.linear_model import LogisticRegression  # CHANGED: Import LR instead of RF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score  # NEW: Added AUC metric

def train_model(x_train: pd.DataFrame, y_train: pd.Series,
                max_iter: int = 1000, random_state: int = 42,
                C: float = 1.0, solver: str = 'lbfgs',
                penalty: str = 'l2', class_weight: str = None) -> LogisticRegression:
    """
    Train a Logistic Regression model with hyperparameter tuning.

    Args:
        x_train (pd.DataFrame): Training features (should be pre-scaled)
        y_train (pd.Series): Training labels
        max_iter (int): Maximum iterations for convergence
        random_state (int): Random seed
        C (float): Inverse regularization strength
        solver (str): Optimization algorithm
        penalty (str): Regularization type
        class_weight (str): Class weight strategy

    Returns:
        LogisticRegression: Trained model with best parameters
    """
    # Define parameter grid for logistic regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # NEW: Regularization strengths
        'penalty': ['l1', 'l2'],  # NEW: Regularization types
        'solver': ['liblinear', 'saga'],  # NEW: Solvers that support L1/L2
        'class_weight': [None, 'balanced']  # NEW: Class imbalance handling
    }
    
    # Initialize base model with provided defaults
    base_model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        warm_start=True  # NEW: Helps with convergence
    )
    
    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',  # NEW: Using AUC for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    # Perform Grid Search
    print("\nPerforming Grid Search for logistic regression...")
    grid_search.fit(x_train, y_train)
    
    # Print best parameters
    print("\nBest Parameters Found:")
    print(grid_search.best_params_)
    print(f"Best AUC Score: {grid_search.best_score_:.4f}")  # CHANGED: Show AUC
    
    # Retrain with best params and increased max_iter for convergence
    best_model = LogisticRegression(
        **grid_search.best_params_,
        max_iter=5000,  # NEW: Increased for better convergence
        random_state=random_state
    )
    best_model.fit(x_train, y_train)
    
    return best_model

def evaluate_model(model: LogisticRegression, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, dict]:
    """
    Evaluate logistic regression model with additional metrics.

    Args:
        model (LogisticRegression): Trained model
        x_test (pd.DataFrame): Test features (should be pre-scaled)
        y_test (pd.Series): Test labels

    Returns:
        Tuple[float, dict]: Accuracy and comprehensive metrics report
    """
    # Get predictions and probabilities
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]  # NEW: Probability scores
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)  # NEW: AUC metric
    
    print("\nLogistic Regression Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC Score: {auc_score:.4f}")  # NEW: Show AUC
    print("\nDetailed Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Add AUC to report dictionary
    report['auc_score'] = auc_score  # NEW: Include AUC in return dict
    
    return accuracy, report