"""
Model Training and Evaluation Module for Logistic Regression
"""

import pandas as pd
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train_model(x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression model with GridSearchCV.
    """
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [100, 200, 300]
    }
    
    lr_model = LogisticRegression(random_state=random_state)
    grid_search = GridSearchCV(lr_model, param_grid, cv=5, n_jobs=-1, verbose=1)
    
    print("\nTraining Logistic Regression with GridSearch...")
    grid_search.fit(x_train, y_train)
    
    print("\nBest Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, dict]:
    """
    Evaluate model performance.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\nLogistic Regression Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    return accuracy, report