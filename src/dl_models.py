"""
Deep Learning Models for Heart Disease Prediction

This module implements deep learning models including:
- LSTM
- BiLSTM
- Transformer

The models are designed to work with the same data format as the ML models,
but with additional preprocessing for sequence-based learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Any
import math
import joblib
from config.config import (
    RANDOM_STATE, DL_MODEL_DIR,
    LSTM_PARAMS, BILSTM_PARAMS, TRANSFORMER_PARAMS,
    MODEL_FILENAMES
)
from src.utils import save_model, load_model, evaluate_model_metrics, print_model_metrics

# Set random seed for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class LSTMClassifier(nn.Module):
    """LSTM model for heart disease prediction"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, num_classes: int = 2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM model for heart disease prediction"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, num_classes: int = 2):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerClassifier(nn.Module):
    """Transformer model for heart disease prediction"""
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int, dropout: float, 
                 num_classes: int = 2):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def evaluate_dl_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str
) -> Dict[str, float]:
    """
    Evaluate a deep learning model
    
    Args:
        model: Trained model
        x_test: Test features
        y_test: Test labels
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate metrics using utils
        metrics = evaluate_model_metrics(y_test.cpu().numpy(), predicted.cpu().numpy())
        
        # Print metrics
        print_model_metrics(metrics)
        
        return metrics

def create_model(model_type: str, input_size: int, device: str) -> nn.Module:
    """Create a new model instance"""
    if model_type == 'lstm':
        model = LSTMClassifier(
            input_size=input_size,
            **LSTM_PARAMS
        ).to(device)
    elif model_type == 'bilstm':
        model = BiLSTMClassifier(
            input_size=input_size,
            **BILSTM_PARAMS
        ).to(device)
    elif model_type == 'transformer':
        model = TransformerClassifier(
            input_size=input_size,
            **TRANSFORMER_PARAMS
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def train_dl_model(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    force_retrain: bool = False
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train a deep learning model
    
    Args:
        model_type: Type of model to train ('lstm', 'bilstm', 'transformer')
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        device: Device to train on ('cuda' or 'cpu')
        force_retrain: Whether to force retraining even if model exists
        
    Returns:
        Trained model and evaluation metrics
    """
    # Check if model exists and retrain is not forced
    model_filename = MODEL_FILENAMES.get(model_type, f"{model_type}_model.joblib")
    model_path = os.path.join(DL_MODEL_DIR, model_filename)
    
    # Convert data to PyTorch tensors
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    # Add sequence dimension if needed
    if len(x_train.shape) == 2:
        x_train = x_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
    
    # Initialize model
    input_size = x_train.shape[-1]
    model = create_model(model_type, input_size, device)
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading existing {model_type} model...")
        model = load_model(model_filename, DL_MODEL_DIR)
        model.eval()
        return model, evaluate_dl_model(model, x_test, y_test, device)
    
    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with increased patience and epochs
    num_epochs = 200  # Increased from 100
    best_val_loss = float('inf')
    patience = 20  # Increased from 10
    patience_counter = 0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Early stopping check
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs, y_test).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    model.eval()
    
    # Save final model
    save_model(model, model_filename, DL_MODEL_DIR)
    
    # Evaluate final model
    metrics = evaluate_dl_model(model, x_test, y_test, device)
    
    return model, metrics 