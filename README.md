# Heart Disease Prediction Project

A machine learning project for predicting heart disease using various features and comparing multiple ML models to find the optimal solution.

## Project Structure

```
heart-disease-prediction/
├── src/                    # Source code
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── model.py           # Model definitions and training
│   ├── image_processing.py # Medical image processing
│   ├── utils.py           # Utility functions
│   └── main.py            # Main application
├── data/                   # Data files
│   ├── raw/               # Original data files
│   ├── processed/         # Processed data files
│   └── images/            # Medical report images
├── models/                 # Saved models
│   └── best_model.pkl     # Best performing model
├── config/                # Configuration files
│   └── config.py         # Model and data configurations
├── EDA_Reports/          # Exploratory Data Analysis reports
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features

- Data preprocessing and cleaning
- Medical image processing and feature extraction
- Multiple ML model implementations:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Stochastic Gradient Descent (SGD)
  - Decision Tree
- Hyperparameter tuning for all models
- Model comparison and selection
- Comprehensive evaluation metrics
- Exploratory Data Analysis (EDA)

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

1. Train and compare all models to find the best one:
   ```bash
   python src/main.py --model compare
   ```

2. Train a specific model:
   ```bash
   python src/main.py --model random_forest
   python src/main.py --model svm
   python src/main.py --model knn
   python src/main.py --model logistic
   python src/main.py --model sgd
   python src/main.py --model decision_tree
   ```

3. Train all models:
   ```bash
   python src/main.py --model all
   ```

### Model Comparison

The comparison process:
1. Trains each model with hyperparameter tuning
2. Evaluates models using multiple metrics:
   - Accuracy
   - F1 Score
   - Training Time
3. Selects the best model based on F1 Score
4. Saves the best model as `best_model.pkl`
5. Generates detailed comparison reports

### Output

- Trained models are saved in the `models/` directory
- Best model is saved as `best_model.pkl`
- EDA reports are saved in `EDA_Reports/` (if --eda flag is used)
- Detailed evaluation metrics and confusion matrices are displayed

## Model Selection Criteria

The best model is selected based on:
1. F1 Score (primary metric)
2. Accuracy
3. Training Time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
