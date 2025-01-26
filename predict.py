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
    try:
        data = request.json
        
        
        input_features = [
            int(data['Country']),           # Country code
            int(data['City']),              # City 
            float(data['AQI_Value']),       # AQI Value
            float(data['CO_AQI_Value']),    # CO AQI Value
            int(data['CO_AQI_Category']),   # CO AQI Category
            float(data['Ozone_AQI_Value']), # Ozone AQI Value
            int(data['Ozone_AQI_Category']),# Ozone AQI Category
            float(data['NO2_AQI_Value']),   # NO2 AQI Value
            int(data['NO2_AQI_Category']),  # NO2 AQI Category
            float(data['PM2.5_AQI_Value']), # PM2.5 AQI Value
            int(data['PM2.5_AQI_Category']) # PM2.5 AQI Category
        ]
        
        # Make prediction
        prediction = model.predict([input_features])
        
        # Convert prediction to category label
        predicted_label = aqi_category_mapping[prediction[0]]
        
        return jsonify({'prediction': predicted_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
