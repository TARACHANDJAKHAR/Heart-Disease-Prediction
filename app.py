from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import base64
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import os
import numpy as np
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Resolve absolute paths based on the location of this file
BASE_DIR = Path(__file__).resolve().parent

print(f"Project root: {BASE_DIR}")

# Load the best model
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")

try:
    model = joblib.load(MODEL_PATH)
    print("Loaded the best model available.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

# Initialize LIME Explainer
DATA_PATH = BASE_DIR / "data" / "processed" / "processed.cleveland.data"
print(f"Data path: {DATA_PATH}")
print(f"Data exists: {DATA_PATH.exists()}")

try:
    from lime import lime_tabular
    # Load dataset sample to initialize LIME feature statistics
    df = pd.read_csv(DATA_PATH, header=None, na_values=['?'])
    df.dropna(inplace=True)
    X_train = df.iloc[:, :-1].values
    
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                     
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['No Heart Disease', 'Heart Disease'],
        mode='classification',
        random_state=42
    )
    print("LIME Explainer initialized successfully.")
except Exception as e:
    print(f"Error initializing LIME from {DATA_PATH}: {e}")
    explainer = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'No trained model available'}), 500
            
        data = request.get_json()
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        missing_features = [f for f in features if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing required features: {", ".join(missing_features)}'}), 400
            
        # Basic validation
        try:
            for f in features:
                data[f] = float(data[f])
        except ValueError:
            return jsonify({'error': 'All features must be numeric.'}), 400
            
        input_data = pd.DataFrame([data], columns=features)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        })
        
    except Exception as e:
        return jsonify({'error': 'Internal server error during prediction.'}), 500


def generate_user_friendly_explanation(exp_list, prediction, probability):
    """Convert LIME results into user-friendly explanation"""
    try:
        friendly_text = "Here's what influenced the model's prediction:\n\n"
        
        confidence = (probability if prediction == 1 else 1 - probability) * 100
        friendly_text += f"The model is {confidence:.1f}% confident in its prediction.\n\n"
        
        friendly_text += "Top influencing factors:\n"
        
        feature_map = {
            'thal': 'Thalassemia condition',
            'slope': 'ST segment slope',
            'restecg': 'Resting ECG results',
            'chol': 'Cholesterol level',
            'fbs': 'Fasting blood sugar',
            'age': 'Age',
            'oldpeak': 'ST depression',
            'sex': 'Gender',
            'trestbps': 'Resting blood pressure',
            'exang': 'Exercise-induced angina',
            'ca': 'Number of major vessels',
            'cp': 'Chest pain type',
            'thalach': 'Maximum heart rate'
        }
        
        # Sort by absolute impact
        sorted_features = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for name, impact in sorted_features:
            # Replace internal name with friendly name
            friendly_name = name
            for key, val in feature_map.items():
                if key in name:
                    friendly_name = name.replace(key, val)
                    break
            
            if impact > 0:
                friendly_text += f"• {friendly_name} increased the likelihood of heart disease\n"
            else:
                friendly_text += f"• {friendly_name} decreased the likelihood of heart disease\n"
                
        return friendly_text
    except Exception as e:
        return f"Could not generate user-friendly explanation."


@app.route('/api/interpretation', methods=['POST'])
def get_interpretation():
    try:
        if explainer is None or model is None:
            return jsonify({'error': 'Interpretability module is not available.'}), 500

        data = request.get_json()
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                   
        input_array = np.array([float(data[f]) for f in features])
        
        def predict_proba_fn(x):
            df = pd.DataFrame(x, columns=features)
            return model.predict_proba(df)

        # Generate LIME explanation
        exp = explainer.explain_instance(
            input_array, 
            predict_proba_fn, 
            num_features=10
        )
        
        # Get prediction details
        input_df = pd.DataFrame([input_array], columns=features)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        exp_list = exp.as_list()
        user_friendly_explanation = generate_user_friendly_explanation(exp_list, prediction, probability)
        
        # Generate raw technical interpretation
        tech_interpretation = f"Prediction: {prediction}\nProbability: {probability:.4f}\n\nFeature Contributions:\n"
        for name, impact in exp_list:
            tech_interpretation += f"{name}: {impact:.4f}\n"

        # Create visualization
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight')
        plt.close(fig)
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')

        return jsonify({
            'technical_interpretation': tech_interpretation,
            'user_friendly_interpretation': user_friendly_explanation,
            'plot_image': img_str
        })

    except Exception as e:
        return jsonify({'error': 'Internal server error during interpretation.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)