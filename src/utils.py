"""
Utility functions
"""

import os
import joblib
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def save_model(model, filename: str, model_dir: Optional[str] = None) -> None:
    """Save model to file"""
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
    else:
        filepath = filename
    
    # Ensure .joblib extension
    if not filepath.endswith('.joblib'):
        filepath = os.path.splitext(filepath)[0] + '.joblib'
    
    # Handle PyTorch models
    if isinstance(model, torch.nn.Module):
        # Save the full model using joblib
        joblib.dump(model, filepath)
    else:
        joblib.dump(model, filepath)
    
    print(f"Model saved to {filepath}")

def load_model(filename: str, model_dir: Optional[str] = None):
    """Load model from file"""
    filepath = os.path.join(model_dir, filename) if model_dir else filename
    
    # Ensure .joblib extension
    if not filepath.endswith('.joblib'):
        filepath = os.path.splitext(filepath)[0] + '.joblib'
    
    # Try to load as joblib
    try:
        model = joblib.load(filepath)
        return model
    except:
        raise ValueError(f"Failed to load model from {filepath}")

def evaluate_model_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, float]:
    """Calculate model evaluation metrics"""
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_pred_proba is not None and isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    
    return metrics

def print_model_metrics(metrics: Dict[str, float]) -> None:
    """Print model evaluation metrics"""
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")