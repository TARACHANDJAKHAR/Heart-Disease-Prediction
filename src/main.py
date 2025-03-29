"""
Heart Disease Prediction Model Training Pipeline

This script implements a machine learning pipeline that:
- Loads and combines multiple datasets
- Preprocesses the data
- Trains a Random Forest classifier
- Evaluates model performance
- Saves the trained model

Dataset Source:
UCI Heart Disease dataset combining data from:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- University Hospital, Zurich
- VA Medical Center, Long Beach

Original Dataset Citation:
Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., 
Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new 
probability algorithm for the diagnosis of coronary artery disease. American Journal 
of Cardiology, 64(5), 304-310.

Authors:
Ishat Shivhare
Parth Parmar
Vedang Dubey
Tarachand Jhakhar

Date: 2025-03-23
"""


import os
import sys
import joblib  # NEW: For saving preprocessing objects

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_and_combine_datasets, 
    clean_data, 
    preprocess_data,  # NEW: Added preprocessing function
    split_data
)
from src.model import train_logistic_regression, evaluate_model  # CHANGED: Specific LR training
from src.utils import save_model
from config.config import (
    DATA_DIR, DATASETS, COLUMN_NAMES,
    RANDOM_STATE, TEST_SIZE,
    MODEL_DIR, MODEL_FILENAME,
    # NEW: Added logistic regression specific parameters
    MAX_ITER, C, SOLVER, PENALTY, CLASS_WEIGHT  
)

def main():
    """Main execution function for the logistic regression training pipeline."""
    print("Starting Heart Disease Prediction Model Training (Logistic Regression)")
    print("=" * 50)

    # Load and combine datasets
    print("\n[1/6] Loading and combining datasets...")
    df = load_and_combine_datasets(DATA_DIR, DATASETS, COLUMN_NAMES)
    
    # Clean the data
    print("\n[2/6] Cleaning and preprocessing the data...")
    df = clean_data(df)
    
    # NEW: Preprocess data (scaling + encoding)
    print("\n[3/6] Applying logistic regression-specific preprocessing...")
    df_processed, scaler = preprocess_data(df)  # Get both data and scaler
    
    # Split the data
    print("\n[4/6] Splitting data into training and test sets...")
    x_train, x_test, y_train, y_test = split_data(
        df_processed, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Train the model (CHANGED to logistic regression)
    print("\n[5/6] Training Logistic Regression model...")
    lr_model = train_logistic_regression(
        x_train, y_train,
        max_iter=MAX_ITER,
        C=C,
        solver=SOLVER,
        penalty=PENALTY,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE
    )
    
    # Evaluate the model
    accuracy, report = evaluate_model(lr_model, x_test, y_test)
    
    # Save the model and preprocessing objects (MODIFIED)
    print("\n[6/6] Saving model and preprocessing artifacts...")
    save_model(lr_model, MODEL_FILENAME, model_dir=MODEL_DIR)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))  # NEW: Save scaler
    
    print("\nPipeline completed successfully!")
    print("=" * 50)
    print(f"Final Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()