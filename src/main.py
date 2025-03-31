"""
Heart Disease Prediction Model Training Pipeline

This script implements a machine learning pipeline that:
- Loads and combines multiple datasets
- Processes medical report images
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
Tarachand Jakhar
Vedang Dubey
Parth Parmar

Date: 2025-03-23
Main Pipeline
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_and_combine_datasets,
    clean_data,
    split_data_logistic,
    split_data_sgd
)
from src.model import (
    train_logistic_regression,
    train_sgd_classifier,
    evaluate_model
)
from src.utils import save_model
from config.config import (
    DATA_DIR, DATASETS, COLUMN_NAMES,
    RANDOM_STATE, TEST_SIZE,
    MODEL_DIR, MODEL_FILENAME,
    IMAGE_DATA_DIR,
    RANDOM_STATE, TEST_SIZE, MODEL_DIR,
    LOGISTIC_FILENAME, SGD_FILENAME
)
from model import train_svm_model, train_knn_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["logistic", "sgd"])
    args = parser.parse_args()

    # Load and clean data
    print("Loading data...")
    df = load_and_combine_datasets(DATA_DIR, DATASETS, COLUMN_NAMES)
    
    # Clean the data and process images if available
    print("\nCleaning and preprocessing the data...")
    df = clean_data(df, image_dir=IMAGE_DATA_DIR)
    
    # Split the data
    print("\nSplitting data into training and test sets...")
    x_train, x_test, y_train, y_test = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Train the model
    # print("\nTraining Random Forest model...")
    # rf_model = train_model(x_train, y_train, random_state=RANDOM_STATE)
    # accuracy, report = evaluate_model(rf_model, x_test, y_test)

    # To use SVM:
    print("\nTraining SVM model...")
    svm_model = train_svm_model(x_train, y_train)
    svm_accuracy, svm_report = evaluate_model(svm_model, x_test, y_test)

    # To use KNN:
    # print("\nTraining KNN model...")
    # knn_model = train_knn_model(x_train, y_train)
    # knn_accuracy, knn_report = evaluate_model(knn_model, x_test, y_test)
    
    # Save the model
    save_model(svm_model, "svm"+MODEL_FILENAME, model_dir=MODEL_DIR)
    
    print("\nPipeline completed successfully!")
    print("=" * 50)
    df = clean_data(df)

    # Model-specific processing
    if args.model == "logistic":
        x_train, x_test, y_train, y_test = split_data_logistic(df, TEST_SIZE, RANDOM_STATE)
        model = train_logistic_regression(x_train, y_train, RANDOM_STATE)
        filename = LOGISTIC_FILENAME
    else:  # sgd
        (x_train, x_test, y_train, y_test), scaler = split_data_sgd(df, TEST_SIZE, RANDOM_STATE)
        model = train_sgd_classifier(x_train, y_train, RANDOM_STATE)
        filename = SGD_FILENAME

    # Evaluate and save
    evaluate_model(model, x_test, y_test)
    save_model(model, filename, MODEL_DIR)

if __name__ == "__main__":
    main()