from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
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
        
        # Extract features (in a real scenario, you would need to preprocess these)
        features = [
            data.get('area', 0),
            data.get('bedrooms', 0),
            data.get('baths', 0),
            data.get('city', 0),  # This would need encoding in a real scenario
            data.get('province_name', 0)  # This would need encoding in a real scenario
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        return jsonify({
            "predicted_price": prediction,
            "currency": "PKR"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)