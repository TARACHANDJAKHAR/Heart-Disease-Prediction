"""
Utility functions for model persistence and common operations.
"""

import os
import joblib
from typing import Optional

def save_model(model, filename: str, model_dir: Optional[str] = None) -> None:
    """
    Save a trained model to a file.

    Args:
        model: The trained model to save
        filename (str): Name of the file to save the model
        model_dir (Optional[str]): Directory to save the model in
    """
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
    else:
        filepath = filename
    
    joblib.dump(model, filepath)
    print(f"\nModel saved successfully as '{filepath}'")

def load_model(filename: str, model_dir: Optional[str] = None):
    """
    Load a trained model from a file.

    Args:
        filename (str): Name of the file containing the model
        model_dir (Optional[str]): Directory containing the model file

    Returns:
        The loaded model
    """
    if model_dir:
        filepath = os.path.join(model_dir, filename)
    else:
        filepath = filename
    
    return joblib.load(filepath)