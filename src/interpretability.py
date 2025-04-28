"""
Model Interpretability using LIME

This module provides functions to explain model predictions using LIME
(Local Interpretable Model-agnostic Explanations).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import torch
from typing import Union, Dict, Any, List
import os
from config.config import (
    INTERPRETABILITY_DIR,
    RANDOM_STATE
)

def create_lime_explainer(
    x_train: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    class_names: List[str] = ['No Heart Disease', 'Heart Disease']
) -> lime_tabular.LimeTabularExplainer:
    """
    Create a LIME explainer for tabular data
    
    Args:
        x_train: Training data
        feature_names: List of feature names
        class_names: List of class names
        
    Returns:
        LIME explainer object
    """
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.values
        
    explainer = lime_tabular.LimeTabularExplainer(
        x_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=RANDOM_STATE
    )
    return explainer

def explain_ml_prediction(
    model: Any,
    x_test: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    instance_idx: int = 0,
    num_features: int = 10
) -> Dict[str, Any]:
    """
    Explain a single prediction from an ML model using LIME
    
    Args:
        model: Trained ML model
        x_test: Test data
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        num_features: Number of features to show in explanation
        
    Returns:
        Dictionary containing explanation details
    """
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values
        
    # Get feature names from the model's scaler if available
    if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
        scaler = model.named_steps['scaler']
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_.tolist()
    
    explainer = create_lime_explainer(x_test, feature_names)
    
    # Get prediction probabilities
    def predict_proba_fn(x):
        return model.predict_proba(x)
    
    # Explain instance
    exp = explainer.explain_instance(
        x_test[instance_idx],
        predict_proba_fn,
        num_features=num_features
    )
    
    # Get explanation details
    explanation = {
        'local_pred': exp.local_pred,
        'intercept': exp.intercept,
        'local_exp': exp.local_exp,
        'score': exp.score,
        'feature_importance': exp.as_list()
    }
    
    return explanation

def explain_dl_prediction(
    model: torch.nn.Module,
    x_test: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    instance_idx: int = 0,
    num_features: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Explain a single prediction from a DL model using LIME
    
    Args:
        model: Trained DL model
        x_test: Test data
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        num_features: Number of features to show in explanation
        device: Device to run model on
        
    Returns:
        Dictionary containing explanation details
    """
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values
        
    explainer = create_lime_explainer(x_test, feature_names)
    
    # Get prediction probabilities
    def predict_proba_fn(x):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(1)
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    # Explain instance
    exp = explainer.explain_instance(
        x_test[instance_idx],
        predict_proba_fn,
        num_features=num_features
    )
    
    # Get explanation details
    explanation = {
        'local_pred': exp.local_pred,
        'intercept': exp.intercept,
        'local_exp': exp.local_exp,
        'score': exp.score,
        'feature_importance': exp.as_list()
    }
    
    return explanation

def plot_feature_importance(
    explanation: Dict[str, Any],
    feature_names: List[str],
    save_path: str = None
) -> None:
    """
    Plot feature importance from LIME explanation
    
    Args:
        explanation: LIME explanation dictionary
        feature_names: List of feature names
        save_path: Path to save the plot (optional)
    """
    # Extract feature importance
    features = [x[0] for x in explanation['feature_importance']]
    importance = [x[1] for x in explanation['feature_importance']]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    # Sort by absolute importance
    df['abs_importance'] = df['Importance'].abs()
    df = df.sort_values('abs_importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title('Feature Importance for Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
            plt.close()
    else:
        plt.show()

def save_explanation(
    explanation: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save LIME explanation to file
    
    Args:
        explanation: LIME explanation dictionary
        save_path: Path to save the explanation
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("LIME Explanation Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Local Prediction: {explanation['local_pred']}\n")
            f.write(f"Intercept: {explanation['intercept']}\n")
            f.write(f"Explanation Score: {explanation['score']}\n\n")
            
            f.write("Feature Importance:\n")
            for feature, importance in explanation['feature_importance']:
                f.write(f"{feature}: {importance:.4f}\n")
    except Exception as e:
        print(f"Error saving explanation: {str(e)}")