
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Create the templates directory if it doesn't exist
#if not os.path.exists('templates'):
   # os.makedirs('templates')

# Load the scaler and model
try:
    with open(os.path.join(BASE_DIR, 'minmax.pkl'), 'rb') as f:
       minmax_scaler = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
       loaded_scaler = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'random_forest_model.pkl'), 'rb') as f:
       loaded_model = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model file missing: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        input_data = pd.DataFrame([{
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }])

        minmax_scaled_data = minmax_scaler.transform(input_data)
        scaled_input_data = loaded_scaler.transform(minmax_scaled_data)

        predicted_crop = loaded_model.predict(scaled_input_data)

        return jsonify({'predicted_crop': predicted_crop[0]})

    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
