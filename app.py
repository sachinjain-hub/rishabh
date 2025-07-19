from flask import Flask, request, render_template, jsonify
import os
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # this renders your form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch data from form
        features = [
            float(request.form['area']),
            int(request.form['bedrooms']),
            int(request.form['bathrooms']),
            int(request.form['stories']),
            int(request.form['mainroad']),
            int(request.form['guestroom']),
            int(request.form['basement']),
            int(request.form['hotwaterheating']),
            int(request.form['airconditioning']),
            int(request.form['parking']),
            int(request.form['prefarea']),
            int(request.form['furnishingstatus'])
        ]

        prediction = model.predict([np.array(features)])
        output = int(prediction[0])

        return render_template('index.html', prediction_text=f'Predicted House Price: ₹{output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
