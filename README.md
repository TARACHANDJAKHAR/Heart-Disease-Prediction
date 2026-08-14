# CardioPredict: Heart Disease Assessment Tool

A robust machine learning application for clinical heart disease prediction. This project leverages Scikit-Learn pipelines to process tabular clinical data and Flask to serve predictions via a modern, minimalist web interface.

## Overview

CardioPredict serves as an educational and research tool designed to demonstrate the application of machine learning in early diagnosis. It evaluates patient metrics (e.g., resting blood pressure, cholesterol, ECG results) to predict the likelihood of coronary artery disease. A core focus of this project is **model interpretability**, utilizing LIME (Local Interpretable Model-agnostic Explanations) to dynamically explain why the model made a specific prediction for a given patient.

## Features

- **Clinical Assessment**: Web-based form accepting 13 standard clinical features.
- **Robust ML Pipeline**: Implements advanced Scikit-Learn pipelines integrating data scaling, feature selection (`SelectKBest`), and classification.
- **Dynamic Interpretability**: Real-time generation of LIME explanations for every prediction, showing exact feature contributions.
- **Premium UX**: A clean, accessible, and responsive user interface designed with a clinical aesthetic.
- **Production-Ready**: Architected to separate heavy training dependencies from lightweight inference deployment.

## Architecture

The project strictly separates the training environment from the production web application.

```text
User 
 └─> Frontend (HTML/CSS/JS)
      └─> Flask API (/api/predict)
           ├─> Pandas DataFrame
           ├─> Scikit-Learn Inference Pipeline
           │    └─> Model Prediction & Probability
           └─> Flask API (/api/interpretation)
                └─> LIME Explainer (Dynamic Generation)
```

## Machine Learning

- **Dataset**: Combined UCI Heart Disease dataset (Cleveland, Hungarian, Switzerland, VA).
- **Preprocessing**: Robust scaling, missing-value handling, and K-Best feature selection are encapsulated inside an `ImbPipeline` to completely prevent train/test contamination.
- **Model Selection**: The pipeline evaluates Random Forest, SVM, KNN, Logistic Regression, SGD, and Decision Trees using `RandomizedSearchCV` with Stratified K-Fold cross-validation.
- **Deep Learning (Experimental)**: PyTorch implementations of LSTM and Transformers are included in `src/dl_models.py` as a comparative study to demonstrate sequence models on tabular data, though Scikit-Learn is prioritized for production efficiency.

## Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib, LIME
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6)

## Project Structure

```text
.
├── app.py                      # Flask API entry point
├── config/                     # Configuration and hyperparameters
├── data/                       # Raw and processed datasets
├── models/                     # Serialized best_model.joblib
├── src/                        # Model training and data processing logic
├── static/                     # CSS and JS for the frontend
├── templates/                  # HTML templates
├── requirements.txt            # Lightweight production dependencies
└── requirements-dev.txt        # Heavy dependencies for training (PyTorch, OpenCV)
```

## Installation & Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Install production dependencies to run the web app
   pip install -r requirements.txt
   
   # (Optional) If you want to train models, also run:
   # pip install -r requirements-dev.txt
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`.

## API Documentation

### `POST /api/predict`
Predicts the likelihood of heart disease based on patient metrics.
- **Request Body**: JSON object containing 13 clinical features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`).
- **Response**: 
  ```json
  {
    "prediction": 1,
    "probability": 0.85,
    "message": "Heart Disease Detected"
  }
  ```

### `POST /api/interpretation`
Generates a dynamic LIME explanation for the given patient metrics.
- **Request Body**: Same as `/api/predict`.
- **Response**: Returns a JSON object containing a Base64-encoded feature importance plot, a user-friendly summary, and a technical breakdown.

## Model Explainability

We integrate **LIME (Local Interpretable Model-agnostic Explanations)** directly into the inference pipeline. When an assessment is generated, the LIME explainer perturbs the user's input features around their local neighborhood to approximate the complex model with a simple linear model. This reveals exactly which features drove the prediction up or down, fostering trust and transparency.

## Screenshots

*(Add screenshots of your UI here)*
- `![Assessment Form](link-to-image)`
- `![Prediction Result](link-to-image)`
- `![LIME Explanation](link-to-image)`

## Deployment

The application is configured for deployment on platforms like Render or Heroku. 
- The lightweight `requirements.txt` ensures that the build process will not exceed free-tier memory limits.
- The `render.yaml` and `Procfile` are configured to launch the app using `gunicorn app:app`.

## Disclaimer

This application is intended for **educational and research purposes only**. It is a machine learning demonstration and should not be used as a substitute for professional clinical diagnosis or medical advice.

## Future Improvements

- Incorporate automated `pytest` suites for API endpoints and data processing logic.
- Expand input validation using Pydantic.
- Containerize the application using Docker for simpler deployment workflows.
