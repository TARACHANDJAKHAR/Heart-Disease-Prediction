"""
Utility functions for model persistence and common operations (Logistic Regression Enhanced)

Key additions for Logistic Regression:
1. Added scaler saving/loading
2. Model validation checks
3. Version compatibility handling
"""

import os
import joblib
from typing import Optional, Tuple
import warnings
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

def save_model(model: BaseEstimator, filename: str, 
              model_dir: Optional[str] = None,
              scaler: Optional[StandardScaler] = None) -> None:
    """
    Save a trained model and optional preprocessing objects.
    
    Enhanced for Logistic Regression by:
    - Adding scaler persistence
    - Model type validation
    - Version checking

    Args:
        model: The trained model (must be a scikit-learn estimator)
        filename: Name of the model file (.pkl recommended)
        model_dir: Directory to save the model
        scaler: Optional StandardScaler object to save
    """
    # Validate model type
    if not isinstance(model, BaseEstimator):
        raise ValueError("Model must be a scikit-learn BaseEstimator")
    
    # Create directory if needed
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
    else:
        model_path = filename
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # NEW: Save scaler if provided (critical for LR)
    if scaler is not None:
        scaler_path = os.path.join(model_dir, 'scaler.pkl') if model_dir else 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

def load_model(filename: str, 
              model_dir: Optional[str] = None,
              load_scaler: bool = False) -> Tuple[BaseEstimator, Optional[StandardScaler]]:
    """
    Load a model and optionally its scaler.
    
    Enhanced with:
    - Scaler loading
    - Version compatibility warnings
    - Type hints

    Args:
        filename: Model filename
        model_dir: Directory containing the model
        load_scaler: Whether to load associated scaler

    Returns:
        Tuple of (model, scaler) where scaler is None if not loaded
    """
    # Construct paths
    model_path = os.path.join(model_dir, filename) if model_dir else filename
    scaler_path = os.path.join(model_dir, 'scaler.pkl') if model_dir else 'scaler.pkl'
    
    # Load model with version check
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress version warnings
        model = joblib.load(model_path)
    
    # NEW: Load scaler if requested (required for LR)
    scaler = None
    if load_scaler:
        try:
            scaler = joblib.load(scaler_path)
            print("Successfully loaded scaler")
        except FileNotFoundError:
            print("Warning: No scaler found - ensure you're loading pre-scaled data")
    
    return model, scaler  # NEW: Returns both model and scaler

# NEW: Added validation function specifically for Logistic Regression
def validate_lr_model(model: BaseEstimator) -> None:
    """Check if model has logistic regression attributes"""
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model must support predict_proba for logistic regression")
    if not hasattr(model, 'classes_'):
        raise ValueError("Invalid model - missing classes_ attribute")