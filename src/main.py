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
from src.ml_model import (
    train_model,
    compare_models
)
from src.dl_models import (
    train_dl_model
)
from src.utils import save_model
from config.config import (
    DATA_DIR, DATASETS, COLUMN_NAMES,
    RANDOM_STATE, TEST_SIZE, MODEL_DIR,
    IMAGE_DATA_DIR, EDA_DIR, DL_MODEL_DIR,
    ML_MODEL_DIR, MODEL_FILENAMES
)

def train_and_compare_models(x_train, y_train, x_test, y_test, model_type, force_retrain=False):
    """Train and compare models, returning the best model and its metrics"""
    print(f"\nStarting {model_type.upper()} model comparison to find the best model...")
    
    if model_type == "ml":
        results = compare_models(x_train, y_train, x_test, y_test, RANDOM_STATE, force_retrain=force_retrain)
        best_model = results['best_model']
        best_score = results['best_score']
        best_model_name = results['best_model_name']
        best_accuracy = results['best_accuracy']
        model_dir = ML_MODEL_DIR
        best_model_filename = MODEL_FILENAMES["best_ml_model"]
    else:  # DL models
        dl_models = ["lstm", "bilstm", "transformer"]
        results = {}
        best_model = None
        best_score = 0
        best_model_name = None
        best_accuracy = 0
        
        for model_name in dl_models:
            print(f"\nTraining {model_name.upper()}...")
            model, metrics = train_dl_model(
                model_type=model_name,
                x_train=x_train.values,
                y_train=y_train.values,
                x_test=x_test.values,
                y_test=y_test.values,
                force_retrain=force_retrain
            )
            
            results[model_name] = metrics
            print(f"\n{model_name.upper()} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print("-" * 30)
            
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model = model
                best_model_name = model_name
                best_accuracy = metrics['accuracy']
        
        model_dir = DL_MODEL_DIR
        best_model_filename = MODEL_FILENAMES["best_dl_model"]
    
    # Print results
    print(f"\nBest {model_type.upper()} Model Selected:")
    print(f"Model Type: {best_model_name}")
    print(f"F1 Score: {best_score:.4f}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    # Save best model with proper naming
    save_model(best_model, best_model_filename, model_dir)
    
    return best_model, best_score

def main():
    parser = argparse.ArgumentParser(description="Heart Disease Prediction Model Training Pipeline")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["ml", "dl", "all"],
        help="Type of model to train (ml for machine learning, dl for deep learning, all for both)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rf", "svm", "knn", "lr", "sgd", "dt", "lstm", "bilstm", "transformer", "all", "compare"],
        help="Specific model to train or 'compare' to find best model"
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
        
        if args.model_type == "ml" or args.model_type == "all":
            if args.model == "compare":
                best_ml_model, best_ml_score = train_and_compare_models(
                    x_train, y_train, x_test, y_test, "ml", force_retrain=args.retrain
                )
            else:
                # Train specific ML model(s)
                if args.model == "all":
                    # Basic models
                    basic_models = ["rf", "svm", "knn", "lr", "sgd", "dt"]
                    
                    # Train all basic models
                    print("\nTraining all basic ML models...")
                    print("=" * 50)
                    for model_name in basic_models:
                        model = train_model(model_name, x_train, y_train, x_test, y_test, RANDOM_STATE, force_retrain=args.retrain)
                        print(f"Completed training {model_name}")
                else:
                    model = train_model(args.model, x_train, y_train, x_test, y_test, RANDOM_STATE, force_retrain=args.retrain)
                    print(f"\nCompleted training {args.model}")
        
        if args.model_type == "dl" or args.model_type == "all":
            if args.model == "compare":
                best_dl_model, best_dl_score = train_and_compare_models(
                    x_train, y_train, x_test, y_test, "dl", force_retrain=args.retrain
                )
            else:
                # Train specific DL model
                if args.model == "all":
                    # Train all DL models
                    dl_models = ["lstm", "bilstm", "transformer"]
                    print("\nTraining all DL models...")
                    print("=" * 50)
                    for model_name in dl_models:
                        model, metrics = train_dl_model(
                            model_type=model_name,
                            x_train=x_train.values,
                            y_train=y_train.values,
                            x_test=x_test.values,
                            y_test=y_test.values,
                            force_retrain=args.retrain
                        )
                        print(f"Completed training {model_name}")
                else:
                    model, metrics = train_dl_model(
                        model_type=args.model,
                        x_train=x_train.values,
                        y_train=y_train.values,
                        x_test=x_test.values,
                        y_test=y_test.values,
                        force_retrain=args.retrain
                    )
                    print(f"\nCompleted training {args.model}")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # If comparing both ML and DL models, find the overall best model
        if args.model_type == "all" and args.model == "compare":
            print("-" * 50)
            if best_ml_score > best_dl_score:
                print("\nOverall Best Model: ML Model")
                save_model(best_ml_model, MODEL_FILENAMES["best_model"], MODEL_DIR)
            else:
                print("\nOverall Best Model: DL Model")
                save_model(best_dl_model, MODEL_FILENAMES["best_model"], MODEL_DIR)
        
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