from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from src.utils import load_model
from config.config import MODEL_DIR, MODEL_FILENAMES
import base64
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)

# Load the best model
try:
    # Try loading best ML model first
    model = load_model(MODEL_FILENAMES["best_model"], model_dir=MODEL_DIR)
    print("Loaded the best model available.")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response with 204 status

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
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400
            
        input_data = pd.DataFrame([data], columns=features)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_user_friendly_explanation(lime_results):
    """Convert LIME results into user-friendly explanation"""
    try:
        # Extract prediction from LIME results more safely
        for line in lime_results.split('\n'):
            if 'Local Prediction:' in line:
                # Extract number between square brackets
                pred_str = line.split('[')[1].split(']')[0]
                prediction = float(pred_str)
                break
        else:
            prediction = 0.5  # default if not found
        
        # Generate friendly explanation
        friendly_text = "Here's what influenced the model's prediction:\n\n"
        
        # Calculate confidence - if prediction < 0.5, invert it for proper confidence calculation
        confidence = (prediction if prediction >= 0.5 else 1 - prediction) * 100
        friendly_text += f"The model is {confidence:.1f}% confident in its prediction.\n"
        friendly_text += f"(Prediction value: {prediction:.3f})\n\n"
        
        friendly_text += "Top influencing factors:\n"
        
        # Map feature names to friendly names
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
        
        # Show top 5 most influential features
        features = []
        feature_section_started = False
        for line in lime_results.split('\n'):
            if 'Feature Importance:' in line:
                feature_section_started = True
                continue
            if feature_section_started and line.strip():
                try:
                    name, value = line.strip().split(':')
                    features.append((name.strip(), float(value)))
                except ValueError:
                    continue  # Skip lines that don't match expected format

        for name, impact in sorted(features, key=lambda x: abs(x[1]), reverse=True)[:5]:
            # Clean feature name
            for key, friendly_name in feature_map.items():
                if key in name:
                    name = friendly_name
                    break
            
            # Add impact description
            if impact > 0:
                friendly_text += f"• {name} increased the likelihood of heart disease\n"
            else:
                friendly_text += f"• {name} decreased the likelihood of heart disease\n"
        
        return friendly_text
    
    except Exception as e:
        return f"Could not generate explanation due to error: {str(e)}\n\nRaw results:\n{lime_results}"

@app.route('/api/interpretation', methods=['GET'])
def get_interpretation():
    try:
        # Read the technical interpretation text
        with open('interpretation/best_model_explanation.txt', 'r') as f:
            tech_interpretation = f.read()
        
        # Generate user-friendly explanation
        user_friendly_explanation = generate_user_friendly_explanation(tech_interpretation)
        
        # Create visualization
        img_buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        # Create plot based on the feature importance data
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')

        return jsonify({
            'technical_interpretation': tech_interpretation,
            'user_friendly_interpretation': user_friendly_explanation,
            'plot_image': img_str
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)