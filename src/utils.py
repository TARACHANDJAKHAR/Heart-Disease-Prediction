"""
Utility functions for model persistence.
"""

import os
import joblib
from typing import Optional

def save_model(model, filename: str, model_dir: Optional[str] = None) -> None:
    """
    Save a trained model to a file.
    """
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
    else:
        filepath = filename
    
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename: str, model_dir: Optional[str] = None):
    """
    Load a trained model from a file.
    """
    filepath = os.path.join(model_dir, filename) if model_dir else filename
    return joblib.load(filepath)