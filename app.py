import streamlit as st
import requests

st.title('AQI Prediction Dashboard')

country = st.number_input('Country Code', min_value=0)
city = st.number_input('City Code', min_value=0)
aqi_value = st.number_input('AQI Value', min_value=0)
co_aqi_value = st.number_input('CO AQI Value', min_value=0)
co_aqi_category = st.number_input('CO AQI Category', min_value=0)
ozone_aqi_value = st.number_input('Ozone AQI Value', min_value=0)
ozone_aqi_category = st.number_input('Ozone AQI Category', min_value=0)
no2_aqi_value = st.number_input('NO2 AQI Value', min_value=0)
no2_aqi_category = st.number_input('NO2 AQI Category', min_value=0)
pm25_aqi_value = st.number_input('PM2.5 AQI Value', min_value=0)
pm25_aqi_category = st.number_input('PM2.5 AQI Category', min_value=0)

if st.button('Predict AQI Category'):
    input_data = {
        'Country': country,
        'City': city,
        'AQI_Value': aqi_value,
        'CO_AQI_Value': co_aqi_value,
        'CO_AQI_Category': co_aqi_category,
        'Ozone_AQI_Value': ozone_aqi_value,
        'Ozone_AQI_Category': ozone_aqi_category,
        'NO2_AQI_Value': no2_aqi_value,
        'NO2_AQI_Category': no2_aqi_category,
        'PM2.5_AQI_Value': pm25_aqi_value,
        'PM2.5_AQI_Category': pm25_aqi_category
    }
    
    response = requests.post('http://localhost:9696/predict', json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.write(f"Predicted AQI Category: {result['prediction']}")
    else:
        st.write("Error in prediction")
