from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Global variables to store model and encoders
model = None
label_encoders = {}

def load_model():
    global model
    try:
        model_path = os.path.join('model', 'model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
        else:
            print("Model file not found")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "House Price Prediction API",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        load_model()
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get input data from request
        data = request.get_json()
        
        # For this simple example, we'll use dummy encoding
        # In a real application, you would save and load the label encoders
        area = float(data.get('area', 0))
        bedrooms = float(data.get('bedrooms', 0))
        baths = float(data.get('baths', 0))
        city = 0  # Dummy encoding
        province_name = 0  # Dummy encoding
        
        # Create feature array
        features = [area, bedrooms, baths, city, province_name]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        return jsonify({
            "predicted_price": float(prediction),
            "currency": "PKR"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)