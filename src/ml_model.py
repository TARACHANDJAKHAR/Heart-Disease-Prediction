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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix,
                            make_scorer, f1_score, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
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
    RANDOM_STATE, ML_MODEL_DIR, MODEL_FILENAMES,
    CV_SPLITS
)
from src.utils import save_model, load_model, evaluate_model_metrics, print_model_metrics

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
        param_distributions=SVM_PARAMS,
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
        param_distributions=KNN_PARAMS,
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

def train_lr_model(
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

    # Use RandomizedSearchCV with more iterations
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=LOGISTIC_PARAMS,
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

def train_sgd_model(
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
        param_distributions=SGD_PARAMS,
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

def train_dt_model(
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
        param_distributions=DT_PARAMS,
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

def check_model_exists(model_name: str) -> bool:
    """Check if a model file exists in the models directory"""
    model_path = os.path.join(ML_MODEL_DIR, f"{model_name}_model.joblib")
    return os.path.exists(model_path)

def load_saved_model(model_name: str) -> Any:
    """Load a saved model from the models directory"""
    model_path = os.path.join(ML_MODEL_DIR, f"{model_name}_model.joblib")
    return joblib.load(model_path)

def train_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    force_retrain: bool = False
) -> Any:
    """Train a single model and save it, or load if already exists"""
    # Check if model already exists and retrain is not forced
    model_filename = MODEL_FILENAMES.get(model_name, f"{model_name}_model.joblib")
    model_path = os.path.join(ML_MODEL_DIR, model_filename)
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"\nLoading existing {model_name} model...")
        model = load_model(model_filename, ML_MODEL_DIR)
        # Evaluate the loaded model
        accuracy, report = evaluate_model(model, x_test, y_test)
        print(f"\nLoaded model performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {report['1']['f1-score']:.4f}")
        return model
    
    # If model doesn't exist or retrain is forced, train it
    start_time = time.time()
    
    # Map model names to training functions
    model_functions = {
        "rf": train_rf_model,
        "svm": train_svm_model,
        "knn": train_knn_model,
        "lr": train_lr_model,
        "sgd": train_sgd_model,
        "dt": train_dt_model,
        "ensemble": train_ensemble_model
    }
    
    # Train the model
    train_func = model_functions[model_name]
    model = train_func(x_train, y_train, random_state)
    
    # Calculate and display training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, x_test, y_test)
    
    # Save the model
    save_model(model, model_filename, ML_MODEL_DIR)
    
    print(f"\nModel saved as: {model_filename}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {report['1']['f1-score']:.4f}")
    
    return model

def train_ensemble_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: List[str],
    random_state: int = RANDOM_STATE,
    force_retrain: bool = False
) -> Any:
    """Train an ensemble model combining basic models"""
    print(f"\nTraining Ensemble model with {', '.join(model_names)}...")
    
    # Map model names to training functions
    model_functions = {
        "rf": train_rf_model,
        "svm": train_svm_model,
        "knn": train_knn_model,
        "lr": train_lr_model,
        "sgd": train_sgd_model,
        "dt": train_dt_model
    }
    
    # Train or load individual models
    models = []
    for model_name in model_names:
        # Use the same model files as train_model function
        model_filename = MODEL_FILENAMES.get(model_name, f"{model_name}_model.joblib")
        model_path = os.path.join(ML_MODEL_DIR, model_filename)
        
        if os.path.exists(model_path) and not force_retrain:
            print(f"Loading existing {model_name} model for ensemble...")
            model = load_model(model_filename, ML_MODEL_DIR)
        else:
            print(f"Training {model_name} model for ensemble...")
            model = model_functions[model_name](x_train, y_train, random_state)
        models.append(model)
    
    # Create a voting classifier
    from sklearn.ensemble import VotingClassifier
    estimators = [(name, model) for name, model in zip(model_names, models)]
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability predictions
        weights=None  # Equal weights for all models
    )
    
    # Train the ensemble
    start_time = time.time()
    ensemble.fit(x_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nEnsemble Training Time: {training_time:.1f}s")
    
    return ensemble

def evaluate_model(
    model: Union[RandomForestClassifier, SVC, KNeighborsClassifier, 
                LogisticRegression, SGDClassifier, DecisionTreeClassifier],
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, dict]:
    """Evaluate model performance with comprehensive metrics"""
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics using utils
    metrics = evaluate_model_metrics(y_test, y_pred, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Print metrics
    print_model_metrics(metrics)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    return metrics['accuracy'], report

def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Compare all models and find the best performing one.
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        random_state: Random state for reproducibility
        force_retrain: Whether to force retraining of models even if they exist
        
    Returns:
        Dictionary containing best model info and comparison results
    """
    from itertools import combinations
    
    # Basic models
    basic_models = {
        "rf": train_rf_model,
        "svm": train_svm_model,
        "knn": train_knn_model,
        "lr": train_lr_model,
        "sgd": train_sgd_model,
        "dt": train_dt_model
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None
    best_accuracy = 0
    
    print("\nStarting model comparison...")
    print("=" * 50)
    
    # First evaluate basic models
    for model_name in tqdm(basic_models.keys(), desc="Evaluating Basic Models"):
        try:
            # Load or train model
            model = train_model(model_name, x_train, y_train, x_test, y_test, random_state, force_retrain)
            
            # Evaluate model
            accuracy, report = evaluate_model(model, x_test, y_test)
            f1 = f1_score(y_test, model.predict(x_test))
            
            results[model_name] = {
                "model": model,
                "accuracy": accuracy,
                "f1_score": f1,
                "report": report
            }
            
            # Update best model if current model performs better
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = model_name
                best_accuracy = accuracy
            
            print(f"\n{model_name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    # Now evaluate ensemble models with 3 models
    print("\nEvaluating Ensemble Models with 3 models...")
    print("=" * 50)
    
    # Generate all possible combinations of 3 models
    model_names = list(basic_models.keys())
    ensemble_combinations = list(combinations(model_names, 3))
    
    for combo in tqdm(ensemble_combinations, desc="Evaluating Ensemble Models"):
        try:
            ensemble_name = f"ensemble_{'_'.join(combo)}"
            
            # Train ensemble model with the specific combination
            ensemble = train_ensemble_model(x_train, y_train, list(combo), random_state, force_retrain)
            
            # Evaluate ensemble
            accuracy, report = evaluate_model(ensemble, x_test, y_test)
            f1 = f1_score(y_test, ensemble.predict(x_test))
            
            results[ensemble_name] = {
                "model": ensemble,
                "accuracy": accuracy,
                "f1_score": f1,
                "report": report
            }
            
            # Update best model if ensemble performs better
            if f1 > best_score:
                best_score = f1
                best_model = ensemble
                best_model_name = ensemble_name
                best_accuracy = accuracy
            
            print(f"\n{ensemble_name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error evaluating ensemble {combo}: {str(e)}")
            continue
    
    # Save best model
    if best_model is not None:
        save_model(best_model, MODEL_FILENAMES["best_model"], ML_MODEL_DIR)
        print("\nBest Model Summary:")
        print(f"Model Type: {best_model_name}")
        print(f"F1 Score: {best_score:.4f}")
        print(f"Accuracy: {best_accuracy:.4f}")
    
    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "best_accuracy": best_accuracy,
        "all_results": results
    }