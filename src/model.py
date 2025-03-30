"""
Model Training and Evaluation
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Tuple, Union

def train_logistic_regression(
    x_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> LogisticRegression:
    """Train Logistic Regression with GridSearch"""
    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [100, 200, 300]
    }
    model = LogisticRegression(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    print(f"Best Logistic Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_sgd_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> SGDClassifier:
    """Train SGD Classifier with GridSearch"""
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01],
        "penalty": ["l1", "l2", "elasticnet"],
        "learning_rate": ["constant", "optimal", "invscaling"],
        "max_iter": [1000, 2000]
    }
    model = SGDClassifier(random_state=random_state, early_stopping=True)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    print(f"Best SGD Params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(
    model: Union[LogisticRegression, SGDClassifier],
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, dict]:
    """Evaluate model performance"""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    return accuracy, report