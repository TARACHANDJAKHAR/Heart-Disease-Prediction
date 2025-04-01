"""
Heart Disease Prediction Model Training Pipeline

This script implements a machine learning pipeline that:
- Loads and combines multiple datasets
- Processes medical report images
- Preprocesses the data
- Trains and compares multiple ML models
- Finds the best performing model
- Generates comprehensive evaluation reports

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

import pandas as pd
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_and_combine_datasets,
    clean_data,
    split_data,
)
from src.model import (
    train_model,
    compare_models
)
from config.config import (
    DATA_DIR, DATASETS, COLUMN_NAMES,
    RANDOM_STATE, TEST_SIZE, MODEL_DIR,
    IMAGE_DATA_DIR, EDA_DIR
)

def main():
    parser = argparse.ArgumentParser(description="Heart Disease Prediction Model Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rf", "svm", "knn", "lr", "sgd", "dt", "all", "compare"],
        help="Model type to train or 'compare' to find best model"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of models even if they exist (useful when data or parameters change)"
    )
    args = parser.parse_args()

    try:
        # Load and clean data
        print("\nLoading data...")
        df = load_and_combine_datasets(DATA_DIR, DATASETS, COLUMN_NAMES)
        
        # Clean the data and process images if available
        print("\nCleaning and preprocessing the data...")
        df = clean_data(df, image_dir=IMAGE_DATA_DIR)
        
        # Split the data
        print("\nSplitting data into training and test sets...")
        x_train, x_test, y_train, y_test = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        if args.model == "compare":
            # Compare all models and find the best one
            print("\nStarting model comparison to find the best model...")
            results = compare_models(x_train, y_train, x_test, y_test, RANDOM_STATE, force_retrain=args.retrain)
            
            # Print detailed comparison results
            print("\nDetailed Model Comparison Results:")
            print("=" * 50)
            for model_name, model_results in results["all_results"].items():
                print(f"\n{model_name.upper()}:")
                print(f"Accuracy: {model_results['accuracy']:.4f}")
                print(f"F1 Score: {model_results['f1_score']:.4f}")
                print("-" * 30)
            
            print("\nBest Model Selected:")
            print(f"Model Type: {results['best_model_name']}")
            print(f"F1 Score: {results['best_score']:.4f}")
            print(f"Accuracy: {results['all_results'][results['best_model_name']]['accuracy']:.4f}")
        else:
            # Train specific model(s)
            if args.model == "all":
                # Basic models
                basic_models = ["rf", "svm", "knn", "lr", "sgd", "dt"]
                
                # Train all basic models
                print("\nTraining all basic models...")
                print("=" * 50)
                for model_name in basic_models:
                    model = train_model(model_name, x_train, y_train, x_test, y_test, RANDOM_STATE, force_retrain=args.retrain)
                    print(f"Completed training {model_name}")
            else:
                model = train_model(args.model, x_train, y_train, x_test, y_test, RANDOM_STATE, force_retrain=args.retrain)
                print(f"\nCompleted training {args.model}")
        
        print("\nPipeline completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()