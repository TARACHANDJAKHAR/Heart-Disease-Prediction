# Heart Disease Prediction Project

A machine learning project for predicting heart disease using various features.

## Project Structure

```
heart-disease-prediction/
├── src/                    # Source code
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── model.py           # Model definition
│   ├── train_model.py     # Training script
│   ├── utils.py           # Utility functions
│   └── main.py            # Main application
├── data/                   # Data files
│   ├── raw/               # Original data files
│   └── processed/         # Processed data files
├── models/                 # Saved models
├── config/                # Configuration files
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

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

3. Run the training script:
   ```bash
   python src/main.py
   ```

## Usage

The project includes scripts for:
- Data preprocessing and cleaning
- Model training and evaluation
- Prediction on new data

See individual script documentation for more details.
