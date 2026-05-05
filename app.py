import os
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract features in the correct order
        # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 0)),
            float(data.get('chol', 0)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 0)),
            float(data.get('thalach', 0)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('slope', 0)),
            float(data.get('ca', 0)),
            float(data.get('thal', 0))
        ]
        
        # Prepare data for prediction
        input_data = np.array([features])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        
        try:
            prob = model.predict_proba(input_data)[0]
            prob_high_risk = float(prob[1]) * 100
        except:
            prob_high_risk = 100.0 if prediction[0] == 1 else 0.0
            
        result = int(prediction[0])
        
        return jsonify({
            'success': True,
            'prediction': result, # 1 for high risk, 0 for low risk
            'probability': prob_high_risk,
            'message': 'High Risk of Heart Disease' if result == 1 else 'Low Risk of Heart Disease'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)