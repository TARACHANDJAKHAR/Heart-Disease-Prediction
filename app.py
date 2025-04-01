from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from src.utils import load_model
from config.config import MODEL_DIR, BEST_MODEL_FILENAME

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model(BEST_MODEL_FILENAME, model_dir=MODEL_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response with 204 status

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)