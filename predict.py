from flask import Flask, request, jsonify
import pickle
import numpy as np

# Creating the Flask application
app = Flask(__name__)

# Load the trained Random Forest model
model_filename = 'random_forest_model.pkl'  

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Mapping dictionary for AQI Category
aqi_category_mapping = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy for Sensitive Groups",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}    

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    input_features = [
        data['Country'],
        data['City'],
        data['AQI_Value'],
        data['CO_AQI_Value'],
        data['CO_AQI_Category'],
        data['Ozone_AQI_Value'],
        data['Ozone_AQI_Category'],
        data['NO2_AQI_Value'],
        data['NO2_AQI_Category'],
        data['PM2.5_AQI_Value'],
        data['PM2.5_AQI_Category']
    ]

    # Make prediction
    prediction = model.predict([input_features])

    # Convert numerical prediction to string label
    predicted_label = aqi_category_mapping[prediction[0]]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)