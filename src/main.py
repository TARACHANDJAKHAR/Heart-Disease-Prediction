"""
Heart Disease Prediction Model Training Pipeline (Logistic Regression)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_and_combine_datasets, clean_data, split_data
from src.model import train_model, evaluate_model
from src.utils import save_model
from config.config import (
    DATA_DIR, DATASETS, COLUMN_NAMES,
    RANDOM_STATE, TEST_SIZE,
    MODEL_DIR, MODEL_FILENAME
)

def main():
    print("Starting Heart Disease Prediction (Logistic Regression)")
    print("=" * 50)
    
    # Load and clean data
    print("\nLoading datasets...")
    df = load_and_combine_datasets(DATA_DIR, DATASETS, COLUMN_NAMES)
    df = clean_data(df)
    
    # Split data
    print("\nSplitting data...")
    x_train, x_test, y_train, y_test = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Train model
    print("\nTraining Logistic Regression...")
    model = train_model(x_train, y_train, random_state=RANDOM_STATE)
    
    # Evaluate and save
    accuracy, report = evaluate_model(model, x_test, y_test)
    save_model(model, MODEL_FILENAME, MODEL_DIR)
    
    print("\nPipeline completed!")

if __name__ == "__main__":
    main()