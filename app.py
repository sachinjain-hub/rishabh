from flask import Flask, request, jsonify
import os
import pickle
import numpy as np

app = Flask(__name__)

# Load your model (ensure 'model.pkl' is in the same directory)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "🏡 House Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Ensure all keys are present
        expected_keys = ['area', 'bedrooms', 'bathrooms', 'stories',
                         'mainroad', 'guestroom', 'basement',
                         'hotwaterheating', 'airconditioning',
                         'parking', 'prefarea', 'furnishingstatus']
        
        if not all(key in data for key in expected_keys):
            return jsonify({"error": "Missing input values"}), 400

        features = [data[key] for key in expected_keys]
        prediction = model.predict([np.array(features)])

        return jsonify({'predicted_price': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # for Render deployment
    app.run(host='0.0.0.0', port=port)
