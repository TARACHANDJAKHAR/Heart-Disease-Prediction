# Heart Disease Prediction Project

A machine learning project for predicting heart disease using various features and comparing multiple ML and DL models to find the optimal solution.

## Project Structure

```
heart-disease-prediction/
├── src/                    # Source code
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── ml_model.py        # ML model definitions and training
│   ├── dl_models.py       # DL model definitions and training
│   ├── image_processing.py # Medical image processing
│   ├── utils.py           # Utility functions
│   └── main.py            # Main application
├── data/                   # Data files
│   ├── raw/               # Original data files
│   ├── processed/         # Processed data files
│   └── images/            # Medical report images
├── models/                 # Saved models
│   ├── best_model.joblib   # Overall best model
│   ├── ml/                # ML models
│   │   ├── rf_model.joblib    # Random Forest model
│   │   ├── svm_model.joblib   # SVM model
│   │   ├── knn_model.joblib   # KNN model
│   │   ├── lr_model.joblib    # Logistic Regression model
│   │   ├── sgd_model.joblib   # SGD model
│   │   ├── dt_model.joblib    # Decision Tree model
│   │   └── best_ml_model.joblib # Best ML model
│   └── dl/                # DL models
│       ├── lstm_model.joblib    # LSTM model
│       ├── bilstm_model.joblib  # BiLSTM model
│       ├── transformer_model.joblib # Transformer model
│       └── best_dl_model.joblib # Best DL model
├── config/                # Configuration files
│   └── config.py         # Model and data configurations
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features

- Data preprocessing and cleaning
- Medical image processing and feature extraction
- Multiple ML model implementations:
  - Random Forest (rf)
  - Support Vector Machine (svm)
  - K-Nearest Neighbors (knn)
  - Logistic Regression (lr)
  - Stochastic Gradient Descent (sgd)
  - Decision Tree (dt)
- Deep Learning model implementations:
  - LSTM
  - BiLSTM
  - Transformer
- Hyperparameter tuning for all models
- Model comparison and selection
- Comprehensive evaluation metrics
- Exploratory Data Analysis (EDA)
- Model persistence and loading
- Force retraining option for model updates

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. Train and compare ML models:
   ```bash
   python src/main.py --model_type ml --model compare
   ```

2. Train and compare DL models:
   ```bash
   python src/main.py --model_type dl --model compare
   ```

3. Train and compare both ML and DL models:
   ```bash
   python src/main.py --model_type all --model compare
   ```

4. Train a specific model:
   ```bash
   # ML models
   python src/main.py --model_type ml --model rf
   python src/main.py --model_type ml --model svm
   python src/main.py --model_type ml --model knn
   python src/main.py --model_type ml --model lr
   python src/main.py --model_type ml --model sgd
   python src/main.py --model_type ml --model dt

   # DL models
   python src/main.py --model_type dl --model lstm
   python src/main.py --model_type dl --model bilstm
   python src/main.py --model_type dl --model transformer
   ```

### Retraining Models

To force retraining of models (useful when data or parameters change):
```bash
# Retrain all models
python src/main.py --model_type all --model compare --retrain

# Retrain specific model type
python src/main.py --model_type ml --model compare --retrain
python src/main.py --model_type dl --model compare --retrain

# Retrain a specific model
python src/main.py --model_type ml --model rf --retrain
python src/main.py --model_type dl --model lstm --retrain
```

### Model Comparison

The comparison process:
1. Evaluates existing models or trains new ones if needed
2. Evaluates models using multiple metrics:
   - Accuracy
   - F1 Score
   - Precision
   - Recall
3. Selects the best model based on F1 Score
4. Saves the best model as `best_ml_model.joblib` or `best_dl_model.joblib`
5. Generates detailed comparison reports

### Output

- Trained models are saved in their respective directories (`models/ml/` or `models/dl/`)
- Best models are saved as `best_ml_model.joblib` and `best_dl_model.joblib`
- EDA reports are saved in `EDA_Reports/` (if --eda flag is used)
- Detailed evaluation metrics and confusion matrices are displayed
- Training time is shown when models are retrained

## Model Selection Criteria

The best model is selected based on:
1. F1 Score (primary metric)
2. Accuracy
3. Model stability and reliability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
