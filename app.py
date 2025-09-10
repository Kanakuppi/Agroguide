
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os

app = Flask(__name__)

# Create the templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Load the scaler and model
try:
    with open('minmax.pkl', 'rb') as f:
        minmax_scaler = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    with open('random_forest_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    print("Error: pickle files not found. Please make sure 'scaler.pkl' and 'random_forest_model.pkl' are in the same directory.")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Create a DataFrame from the received data
        input_data = pd.DataFrame([data], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        minmax_scaled_data=minmax_scaler.transform(input_data)
        # Scale the input data using the loaded scaler
        scaled_input_data = loaded_scaler.transform(minmax_scaled_data)

        # Predict the crop using the loaded model
        predicted_crop = loaded_model.predict(scaled_input_data)

        # Return the predicted crop label
        return jsonify({"predicted_crop": predicted_crop[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)