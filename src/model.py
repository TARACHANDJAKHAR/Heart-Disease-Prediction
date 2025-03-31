"""
Model Training and Evaluation
"""

import pandas as pd
import time
import os
from typing import Tuple, Union, Any, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay,
                            make_scorer, f1_score, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
from scipy.stats import uniform, loguniform
from config.config import (
    RF_PARAMS, SVM_PARAMS, KNN_PARAMS,
    LOGISTIC_PARAMS, SGD_PARAMS, DT_PARAMS,
    RANDOM_STATE, MODEL_DIR, BEST_MODEL_FILENAME,
    CV_SPLITS, SCORING_METRICS, COMPARISON_METRICS,
    BEST_MODEL_SELECTION_METRIC
)

def create_cv_strategy() -> StratifiedKFold:
    """Create cross-validation strategy"""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def create_scorers():
    """Create scorer dictionary with zero_division handling"""
    return {
        'accuracy': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0)
    }

def train_rf_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train Random Forest classifier with hyperparameter tuning"""
    print("\nTraining Random Forest model...")
    
    pipe = ImbPipeline([
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(f_classif)),
        ('rf', RandomForestClassifier(random_state=random_state, class_weight='balanced'))
    ])
        
    # Use RandomizedSearchCV for more efficient hyperparameter search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=RF_PARAMS,
        cv=create_cv_strategy(),
        verbose=1,
        scoring=create_scorers(),
        refit='f1_score',
        n_jobs=-1,
        random_state=random_state
    )

    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print("\nBest Parameters Found (RF):")
    print(search.best_params_)
    print(f"Best Cross-Validation Score (RF): {search.best_score_:.4f}")
    print(f"Training Time: {training_time:.2f}s")

    return search.best_estimator_

def train_svm_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> SVC:
    """Train SVM classifier with hyperparameter tuning"""
    print("\nTraining SVM model...")
    
    # Create pipeline with robust preprocessing
    pipe = make_pipeline(
        RobustScaler(),  # More robust to outliers
        SVC(probability=True, random_state=random_state, cache_size=1000, class_weight='balanced')  # Added class_weight
    )

    # Use RandomizedSearchCV with more iterations
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=RF_PARAMS,
        n_iter=50,
        cv=create_cv_strategy(),
        verbose=1,
        scoring=create_scorers(),
        refit='f1_score',
        n_jobs=-1,
        random_state=random_state
    )
    
    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nTraining Time: {training_time:.1f}s")
    print("\nBest Parameters (SVM):")
    print(search.best_params_)
    print(f"Validation Score: {search.best_score_:.4f}")

    return search.best_estimator_

def train_knn_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> KNeighborsClassifier:
    """Train KNN classifier with hyperparameter tuning"""
    print("\nTraining KNN model...")
    
    # Create pipeline with feature selection and class weight handling
    pipe = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(f_classif)),
        ('knn', KNeighborsClassifier(weights='distance'))  # Use distance weights
    ])

    # Create scorer dictionary with zero division handling
    scorers = create_scorers()

    # Use RandomizedSearchCV for more efficient search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=RF_PARAMS,
        n_iter=50,
        cv=create_cv_strategy(),
        verbose=1,
        scoring=create_scorers(),
        refit='f1_score',
        n_jobs=-1,
        random_state=random_state
    )

    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nTraining Time: {training_time:.1f}s")
    print("\nBest Parameters (KNN):")
    print(search.best_params_)
    print(f"Validation Score: {search.best_score_:.4f}")
    
    return search.best_estimator_

def train_logistic_regression(
    x_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = RANDOM_STATE
) -> LogisticRegression:
    """Train Logistic Regression with hyperparameter tuning"""
    print("\nTraining Logistic Regression model...")
    
    # Create pipeline with robust preprocessing
    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(random_state=random_state, class_weight='balanced')
    )
    
    # Create scorer dictionary with zero division handling
    scorers = create_scorers()
    
    # Use RandomizedSearchCV with more iterations
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=RF_PARAMS,
        n_iter=50,
        cv=create_cv_strategy(),
        verbose=1,
        scoring=create_scorers(),
        refit='f1_score',
        n_jobs=-1,
        random_state=random_state
    )

    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nTraining Time: {training_time:.1f}s")
    print("\nBest Parameters Found (Logistic):")
    print(search.best_params_)
    print(f"Best Cross-Validation Score (Logistic): {search.best_score_:.4f}")

    return search.best_estimator_

def train_sgd_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> SGDClassifier:
    """Train SGD classifier with hyperparameter tuning"""
    print("\nTraining SGD model...")
    
    # Create pipeline with robust preprocessing
    pipe = make_pipeline(
        RobustScaler(),
        SGDClassifier(random_state=random_state, class_weight='balanced')
    )
    
    # Create scorer dictionary with zero division handling
    scorers = create_scorers()
    
    # Use RandomizedSearchCV for more efficient search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=RF_PARAMS,
        n_iter=50,
        cv=create_cv_strategy(),
        verbose=1,
        scoring=create_scorers(),
        refit='f1_score',
        n_jobs=-1,
        random_state=random_state
    )

    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nTraining Time: {training_time:.1f}s")
    print("\nBest Parameters Found (SGD):")
    print(search.best_params_)
    print(f"Best Cross-Validation Score (SGD): {search.best_score_:.4f}")

    return search.best_estimator_

def train_decision_tree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE
) -> DecisionTreeClassifier:
    """Train Decision Tree with hyperparameter tuning"""
    print("\nTraining Decision Tree model...")
    
    # Create pipeline with robust preprocessing
    pipe = make_pipeline(
        RobustScaler(),
        DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
    )

    # Create scorer dictionary with zero division handling
    scorers = create_scorers()
    
    # Use RandomizedSearchCV with more iterations
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=RF_PARAMS,
        n_iter=50,
        cv=create_cv_strategy(),
        verbose=1,
        scoring=create_scorers(),
        refit='f1_score',
        n_jobs=-1,
        random_state=random_state
    )

    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nTraining Time: {training_time:.1f}s")
    print("\nBest Parameters Found (Decision Tree):")
    print(search.best_params_)
    print(f"Best Cross-Validation Score (Decision Tree): {search.best_score_:.4f}")

    return search.best_estimator_

def train_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int
) -> Any:
    """Train a single model and save it"""
    from src.utils import save_model
    
    # Map model names to training functions
    model_functions = {
        "random_forest": train_rf_model,
        "svm": train_svm_model,
        "knn": train_knn_model,
        "logistic": train_logistic_regression,
        "sgd": train_sgd_classifier,
        "decision_tree": train_decision_tree
    }
    
    # Train the model
    train_func = model_functions[model_name]
    model = train_func(x_train, y_train, random_state)
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, x_test, y_test)
    
    # Save the model
    model_filename = f"{model_name}_model.joblib"
    save_model(model, model_filename, MODEL_DIR)
    
    print(f"\nModel saved as: {model_filename}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {report['1']['f1-score']:.4f}")
    
    return model

def evaluate_model(
    model: Union[RandomForestClassifier, SVC, KNeighborsClassifier, 
                LogisticRegression, SGDClassifier, DecisionTreeClassifier],
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, dict]:
    """Evaluate model performance with comprehensive metrics"""
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix with seaborn
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return accuracy, report

def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Compare all models and find the best performing one.
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing best model info and comparison results
    """
    models = {
        "random_forest": train_rf_model,
        "svm": train_svm_model,
        "knn": train_knn_model,
        "logistic": train_logistic_regression,
        "sgd": train_sgd_classifier,
        "decision_tree": train_decision_tree
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None
    
    print("\nStarting model comparison...")
    print("=" * 50)
    
    for model_name, train_func in tqdm(models.items(), desc="Training Models"):
        try:
            # Train model
            start_time = time.time()
            model = train_func(x_train, y_train, random_state)
            training_time = time.time() - start_time
            
            # Evaluate model
            accuracy, report = evaluate_model(model, x_test, y_test)
            f1 = f1_score(y_test, model.predict(x_test))
            
            results[model_name] = {
                "model": model,
                "accuracy": accuracy,
                "f1_score": f1,
                "training_time": training_time,
                "report": report
            }
            
            # Update best model if current model performs better
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = model_name
            
            print(f"\n{model_name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Training Time: {training_time:.2f}s")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Save best model
    if best_model is not None:
        best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
        joblib.dump(best_model, best_model_path)
        print("\nBest Model Summary:")
        print(f"Model Type: {best_model_name}")
        print(f"F1 Score: {best_score:.4f}")
        print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"Training Time: {results[best_model_name]['training_time']:.2f}s")
        print(f"Saved to: {best_model_path}")
    
    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "all_results": results
    }