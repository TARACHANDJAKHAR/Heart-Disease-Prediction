"""
Model Training and Evaluation Module for Heart Disease Prediction (SGD Version)

Key changes from Logistic Regression:
1. Replaced LogisticRegression with SGDClassifier
2. Added partial_fit capability for online learning
3. Modified hyperparameter tuning for SGD
4. Enhanced evaluation with hinge loss metrics
5. Added probability calibration
"""

import pandas as pd
from typing import Tuple, Generator
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                            roc_auc_score, hinge_loss)

def train_model(x_train: pd.DataFrame, y_train: pd.Series,
                loss: str = 'log_loss', 
                penalty: str = 'l2',
                alpha: float = 0.0001,
                max_iter: int = 1000,
                learning_rate: str = 'optimal',
                tol: float = 1e-4,
                random_state: int = 42) -> CalibratedClassifierCV:
    """
    Train an SGD classifier with hyperparameter tuning and probability calibration.
    
    Args:
        x_train: Training features (must be pre-scaled)
        y_train: Training labels
        loss: Loss function ('log_loss', 'hinge', etc.)
        penalty: Regularization type
        alpha: Regularization strength
        max_iter: Maximum epochs
        learning_rate: Learning rate schedule
        tol: Stopping tolerance
        random_state: Random seed
        
    Returns:
        Calibrated SGD model with best parameters
    """
    # SGD-specific parameter grid
    param_grid = {
        'alpha': [1e-5, 1e-4, 1e-3],  # Regularization strengths
        'penalty': ['l1', 'l2', 'elasticnet'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'eta0': [0.01, 0.1]  # Initial learning rate
    }
    
    base_model = SGDClassifier(
        loss=loss,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=10
    )
    
    # Grid search with AUC scoring
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nPerforming SGD hyperparameter tuning...")
    grid.fit(x_train, y_train)
    
    print("\nBest SGD Parameters:")
    print(grid.best_params_)
    print(f"Best Validation AUC: {grid.best_score_:.4f}")
    
    # Calibrate for probability outputs
    calibrated_model = CalibratedClassifierCV(
        grid.best_estimator_,
        cv=5,
        method='sigmoid'
    )
    calibrated_model.fit(x_train, y_train)
    
    return calibrated_model

def evaluate_model(model: CalibratedClassifierCV, 
                 x_test: pd.DataFrame, 
                 y_test: pd.Series) -> Tuple[float, dict]:
    """
    Evaluate SGD model with additional metrics including hinge loss.
    
    Args:
        model: Calibrated SGD model
        x_test: Test features (must be pre-scaled)
        y_test: Test labels
        
    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    # Get both class predictions and probabilities
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    y_decision = model.decision_function(x_test)  # For hinge loss
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'hinge_loss': hinge_loss(y_test, y_decision),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print("\nSGD Model Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Hinge Loss: {metrics['hinge_loss']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics['accuracy'], metrics

# NEW: Online learning support
def partial_fit_model(model: SGDClassifier,
                     data_generator: Generator,
                     n_chunks: int = None) -> SGDClassifier:
    """
    Incremental training for large datasets.
    
    Args:
        model: Initialized SGD model
        data_generator: Yields (X_chunk, y_chunk)
        n_chunks: Number of chunks to process
        
    Returns:
        Partially fitted model
    """
    for i, (X_chunk, y_chunk) in enumerate(data_generator):
        if n_chunks and i >= n_chunks:
            break
        model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
        
    return model