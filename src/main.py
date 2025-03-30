"""
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
    RANDOM_STATE, TEST_SIZE, MODEL_DIR,
    LOGISTIC_FILENAME, SGD_FILENAME
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["logistic", "sgd"])
    args = parser.parse_args()

    # Load and clean data
    print("Loading data...")
    df = load_and_combine_datasets(DATA_DIR, DATASETS, COLUMN_NAMES)
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