import requests


url = 'http://localhost:9696/predict'

input_data = {
    'Country': 1,  # Example value
    'City': 1,    
    'AQI_Value': 41,
    'CO_AQI_Value': 1,
    'CO_AQI_Category': 1,
    'Ozone_AQI_Value': 5,
    'Ozone_AQI_Category': 1,
    'NO2_AQI_Value': 1,
    'NO2_AQI_Category': 1,
    'PM2.5_AQI_Value': 41,
    'PM2.5_AQI_Category': 1
}

# Send POST request to the API
response = requests.post(url, json=input_data)
print(response.json())

