"""
Utility functions for model persistence and operations (SGD Enhanced)

Key additions for SGD:
1. Added partial model saving/loading
2. Online learning checkpoint support
3. Enhanced memory efficiency
4. SGD-specific validation
"""

import os
import joblib
from typing import Optional, Tuple, Union
import warnings
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

def save_model(model: Union[SGDClassifier, CalibratedClassifierCV], 
              filename: str,
              model_dir: Optional[str] = None,
              scaler: Optional[StandardScaler] = None,
              partial: bool = False) -> None:
    """
    Save SGD model with enhanced options for online learning.
    
    Args:
        model: SGD model or calibrated model
        filename: Output filename (.pkl or .joblib)
        model_dir: Target directory
        scaler: Optional scaler object
        partial: If True, saves partial fit progress
    """
    # Validate model type
    if not isinstance(model, (SGDClassifier, CalibratedClassifierCV)):
        raise ValueError("Model must be SGDClassifier or CalibratedClassifierCV")
    
    # Create directory if needed
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
    else:
        model_path = filename
    
    # NEW: Handle partial fit models differently
    if partial and isinstance(model, SGDClassifier):
        checkpoint = {
            'model': model,
            'n_iter': model.n_iter_,
            't_': model.t_  # Learning state
        }
        joblib.dump(checkpoint, model_path)
    else:
        joblib.dump(model, model_path)
    
    print(f"SGD model saved to {model_path}")
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(model_dir, 'sgd_scaler.pkl') if model_dir else 'sgd_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

def load_model(filename: str,
              model_dir: Optional[str] = None,
              load_scaler: bool = False,
              is_partial: bool = False) -> Tuple[Union[SGDClassifier, CalibratedClassifierCV], Optional[StandardScaler]]:
    """
    Load SGD model with online learning support.
    
    Args:
        filename: Model filename
        model_dir: Directory path
        load_scaler: Whether to load associated scaler
        is_partial: If loading a partial fit checkpoint
        
    Returns:
        Tuple of (model, scaler)
    """
    model_path = os.path.join(model_dir, filename) if model_dir else filename
    scaler_path = os.path.join(model_dir, 'sgd_scaler.pkl') if model_dir else 'sgd_scaler.pkl'
    
    # NEW: Handle partial fit checkpoints
    if is_partial:
        checkpoint = joblib.load(model_path)
        model = checkpoint['model']
        model.n_iter_ = checkpoint['n_iter']
        model.t_ = checkpoint['t_']
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_path)
    
    # Load scaler if requested
    scaler = None
    if load_scaler:
        try:
            scaler = joblib.load(scaler_path)
            print("Loaded SGD scaler successfully")
        except FileNotFoundError:
            print("Warning: Scaler not found - using unscaled data")
    
    return model, scaler

# NEW: SGD-specific validation
def validate_sgd_model(model: BaseEstimator) -> None:
    """Check if model has SGD-specific attributes"""
    if not hasattr(model, 'partial_fit'):
        raise ValueError("Model must support partial_fit for online learning")
    if not hasattr(model, 't_'):
        raise ValueError("Invalid SGD model - missing training state")

# NEW: Create learning rate schedule
def create_learning_schedule(initial_rate: float = 0.1, 
                           decay: float = 0.5) -> dict:
    """Generate learning rate schedule for SGD"""
    return {
        'learning_rate': 'optimal',
        'eta0': initial_rate,
        'power_t': decay
    }