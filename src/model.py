"""
Model Training and Evaluation
"""

import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# Existing Random Forest function remains unchanged
def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with hyperparameter tuning.
    (Existing implementation remains unchanged)
    """
    # Existing Random Forest implementation
    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(random_state=random_state)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
        scoring="accuracy",
    )

    grid_search.fit(x_train, y_train)
    print("\nBest Parameters Found (RF):")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Score (RF): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

# New SVM Model Function
import time
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import math
import time
from tqdm import tqdm

from sklearn.base import clone

def train_svm_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> SVC:
    """
    Final robust SVM training with version-agnostic progress tracking.
    """
    # Create pipeline with proper scaling
    pipe = make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=random_state)
    )

    # Optimized parameter distribution
    param_dist = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto', 0.1],
        'svc__class_weight': [None, 'balanced']
    }

    # Configure search
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    # Initialize progress tracking
    start_time = time.time()
    total_iterations = 20 * 5  # n_iter * cv folds
    pbar = tqdm(total=total_iterations, desc="SVM Training")

    # Store original methods
    original_fit = search.fit
    original_format = search._format_results

    # Define version-agnostic wrapper
    def format_wrapper(*args, **kwargs):
        pbar.update(1)
        if hasattr(search, 'best_score_'):
            pbar.set_postfix({"Best Score": f"{search.best_score_:.2f}"})
        return original_format(*args, **kwargs)

    def wrapped_fit(X, y):
        search._format_results = format_wrapper
        try:
            result = original_fit(X, y)
        finally:
            pbar.close()
        return result

    search.fit = wrapped_fit

    # Execute training
    try:
        search.fit(x_train, y_train)
    except Exception as e:
        pbar.close()
        raise RuntimeError(f"SVM training failed: {str(e)}") from e

    print(f"\nBest Parameters (SVM):")
    print(search.best_params_)
    print(f"Training Time: {time.time()-start_time:.1f}s")
    print(f"Validation Accuracy: {search.best_score_:.4f}")

    return search.best_estimator_

# New KNN Model Function
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

def train_knn_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_jobs: int = -1
) -> KNeighborsClassifier:
    """
    Optimized KNN classifier training with proper pipeline configuration.
    Fixes 0-second iteration issue and improves accuracy.
    """
    # Corrected pipeline order and components
    pipeline = ImbPipeline([
        ('scaler', RobustScaler()),        # Step 1: Scale features
        ('smote', SMOTE(random_state=42)), # Step 2: Balance classes
        ('selector', SelectKBest(f_classif)),  # Step 3: Feature selection
        ('knn', KNeighborsClassifier())    # Step 4: Final classifier
    ])

    # Revised hyperparameter grid
    param_grid = {
        'selector__k': [8, 10],            # Reduced feature selection options
        'knn__n_neighbors': list(range(3, 15, 2)),  # More reasonable neighbor range
        'knn__weights': ['distance'],      # Focus on distance weighting
        'knn__p': [1, 2],                  # Manhattan vs Euclidean
        'knn__leaf_size': [30],            # Single optimized value
        'knn__metric': ['minkowski']       # Standard metric
    }

    # Validation checks
    print("Initial feature count:", x_train.shape[1])
    print("Class distribution:\n", y_train.value_counts())

    # Configure grid search with proper CV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=n_jobs,
        verbose=2
    )

    # Add timing instrumentation
    start_time = time.time()
    grid_search.fit(x_train, y_train)
    print(f"\nTotal training time: {time.time() - start_time:.1f} seconds")

    # Best model evaluation
    best_model = grid_search.best_estimator_
    
    # Pipeline validation
    try:
        transformed_sample = best_model[:-1].transform(x_train[:1])
        print(f"\nPost-pipeline feature count: {transformed_sample.shape[1]}")
    except Exception as e:
        print("\nPipeline transformation error:", str(e))

    print("\nBest parameters:", grid_search.best_params_)
    print(f"Validation score: {grid_search.best_score_:.4f}")
    
    return best_model
def evaluate_model(
    model, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[float, dict]:
    """
    Evaluate model performance.
    (Existing implementation remains unchanged)
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
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
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()
    
    return accuracy, report