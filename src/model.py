"""
Model Training and Evaluation Module for Heart Disease Prediction

This module handles model training, hyperparameter tuning, and evaluation
for the heart disease prediction model.
"""

import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the provided data using Grid Search CV for hyperparameter tuning.

    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_estimators (int): Number of trees in the forest (default value, will be tuned)
        random_state (int): Random state for reproducibility

    Returns:
        RandomForestClassifier: Trained Random Forest model with best parameters
    """
    # Define parameter grid for Grid Search
    param_grid = {
        "n_estimators": [50, 100, 200, 500],  # defines the number of trees
        "max_depth": [3, 5, 10, 20, None], # Limits how deep each tree can grow. (prevents overfitting)
        "min_samples_split": [2, 5, 10, 20], # Minimum number of samples needed to split an internal node.
        "min_samples_leaf": [1, 2, 4, 8], # Minimum number of samples required to be in a leaf node.
        "max_features": ["sqrt", "log2"], # Controls how many features are randomly selected for each split.

    }

    # Initialize base model
    base_model = RandomForestClassifier(random_state=random_state)
    
    # Ensures each fold has an equal proportion of classes, improving accuracy.
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    
    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        # controls how the dataset is split into training and validation sets during hyperparameter tuning.
        n_jobs=-1,
        # Defines the number of CPU cores used for training.
        verbose=1,
        # Controls the amount of information printed during training.
        # Higher values → More detailed logs.
        scoring="accuracy",
    )

    # Perform Grid Search
    print("\nPerforming Grid Search for hyperparameter tuning...")
    grid_search.fit(x_train, y_train)

    # Print best parameters
    print("\nBest Parameters Found:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(
    model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, dict]:
    """
    Evaluate the model's performance using various metrics.

    Args:
        model (RandomForestClassifier): Trained model to evaluate
        x_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels

    Returns:
        Tuple[float, dict]: Accuracy score and classification report as a dictionary
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    
    # Show the plot (if using a script, this may be necessary)
    plt.show()
    
    return accuracy, report

"""
Decision Tree Training and Evaluation Module for Heart Disease Prediction

This module handles model training, hyperparameter tuning, and evaluation
for the heart disease prediction model using a Decision Tree classifier.
"""

def train_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier on the provided data using Grid Search CV for hyperparameter tuning.

    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        random_state (int): Random state for reproducibility

    Returns:
        DecisionTreeClassifier: Trained Decision Tree model with best parameters
    """
    # Define parameter grid for Grid Search
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, 20, None], 
        "min_samples_split": [2, 5, 10, 20], 
        "min_samples_leaf": [1, 2, 4, 8], 
    }

    # Initialize base model
    base_model = DecisionTreeClassifier(random_state=random_state)
    
    # Ensures each fold has an equal proportion of classes, improving accuracy.
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
        scoring="accuracy",
    )

    # Perform Grid Search
    print("\nPerforming Grid Search for hyperparameter tuning...")
    grid_search.fit(x_train, y_train)

    # Print best parameters
    print("\nBest Parameters Found:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def evaluate_decision_tree(
    model: DecisionTreeClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, dict]:
    """
    Evaluate the Decision Tree model's performance using various metrics.

    Args:
        model (DecisionTreeClassifier): Trained Decision Tree model to evaluate
        x_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels

    Returns:
        Tuple[float, dict]: Accuracy score and classification report as a dictionary
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nDecision Tree Model Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    
    plt.show()
    
    return accuracy, report



