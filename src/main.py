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

Key Changes from Logistic Regression:
1. Replaced Logistic Regression with SGDClassifier
2. Added online learning capability
3. Modified preprocessing for chunked data
4. Added calibration for probability outputs
"""

import os
import sys
import joblib
from sklearn.linear_model import SGDClassifier  # CHANGED
from sklearn.calibration import CalibratedClassifierCV  # NEW

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_and_combine_datasets, 
    clean_data, 
    preprocess_data,
    split_data,
    data_generator  # NEW
)
from src.model import train_sgd_model, evaluate_model  # CHANGED
from src.utils import save_model
from config.config import (
    DATA_DIR, DATASETS, COLUMN_NAMES,
    RANDOM_STATE, TEST_SIZE,
    MODEL_DIR, MODEL_FILENAME,
    # SGD-specific parameters
    LOSS, PENALTY, ALPHA, 
    MAX_ITER, LEARNING_RATE, TOL,
    SGD_CHUNK_SIZE  # NEW
)

def main():
    """Main execution function for SGD training pipeline."""
    print("Starting Heart Disease Prediction Model Training (SGD)")
    print("=" * 50)

    # Load and combine datasets
    print("\n[1/6] Loading and combining datasets...")
    df = load_and_combine_datasets(DATA_DIR, DATASETS, COLUMN_NAMES)
    
    # Clean the data
    print("\n[2/6] Cleaning and preprocessing the data...")
    df = clean_data(df)
    
    # Preprocess data (shuffles and creates scaler)
    print("\n[3/6] Applying SGD-specific preprocessing...")
    df_processed, scaler = preprocess_data(df)  # Returns shuffled data
    
    # Split the data
    print("\n[4/6] Splitting data into training and test sets...")
    x_train, x_test, y_train, y_test = split_data(
        df_processed, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Train the model (CHANGED to SGD)
    print("\n[5/6] Training SGD model...")
    sgd_model = train_sgd_model(
        x_train, y_train,
        loss=LOSS,
        penalty=PENALTY,
        alpha=ALPHA,
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        tol=TOL,
        random_state=RANDOM_STATE
    )
    
    # NEW: Calibrate for better probability estimates
    print("Calibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(sgd_model, cv=5)
    calibrated_model.fit(x_train, y_train)
    
    # Evaluate the model
    accuracy, report = evaluate_model(calibrated_model, x_test, y_test)
    
    # Save the model and preprocessing objects
    print("\n[6/6] Saving model and preprocessing artifacts...")
    save_model(calibrated_model, MODEL_FILENAME, model_dir=MODEL_DIR, scaler=scaler)
    
    print("\nPipeline completed successfully!")
    print("=" * 50)
    print(f"Final Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()